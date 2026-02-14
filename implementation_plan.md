# Implementation Plan: VMM-Based Partial Component Sharing

## Goal

Share layer 2 attention's K projection between two models (Model A and Model B)
while each model keeps its own Q and V projections. True zero-copy via CUDA VMM.

## Toy Example

- **Model A** (owner): loads DeepSeek-Math-7B, allocates layer 2 QKV via VMM
  (3 physical chunks), exports K's physical handle to registry
- **Model B** (consumer): loads same model on a different port, imports K from
  Model A, allocates its own Q and V, maps all three contiguously

Both models see a contiguous `qkv_proj.weight [12288, 4096]` tensor.
Forward pass is identical — single GEMM, no kernel changes.

---

## Files to Create/Modify

| File | Action | Lines (est.) |
|------|--------|-------------|
| `vllm/distributed/device_communicators/cuda_wrapper.py` | Add `CudaDriverLibrary` class | ~100 |
| `vllm/model_executor/model_loader/vmm_utils.py` | New file — `VMMCompositeWeight` | ~80 |
| `vllm/model_executor/model_loader/default_loader.py` | Modify `load_weights()` | ~50 |
| `sharing_spec.json` | New file — toy example spec | ~15 |

**No changes to:** `llama.py`, `linear.py`, any model code, forward pass, kernels,
`cumem_allocator.cpp`.

---

## Step 1: Add CUDA Driver API Wrapper

**File:** `vllm/distributed/device_communicators/cuda_wrapper.py`

Add a new `CudaDriverLibrary` class below the existing `CudaRTLibrary`, following
the same ctypes pattern. Wraps `libcuda.so` (driver API) instead of `libcudart.so`.

**Functions to wrap (7 total):**

```
cuMemGetAllocationGranularity(size_t* granularity, CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option)
cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, CUmemAllocationProp* prop, unsigned long long flags)
cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
cuMemSetAccess(CUdeviceptr ptr, size_t size, CUmemAccessDesc* desc, size_t count)
cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags)
cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType handleType)
```

**ctypes structures needed:**

```python
class CUmemAllocationProp(ctypes.Structure):
    # type, location (type + id), allocFlags, etc.
    # Only need: type=CU_MEM_ALLOCATION_TYPE_PINNED (1),
    #            location.type=CU_MEM_LOCATION_TYPE_DEVICE (1),
    #            location.id=device_id
    # Total struct size: 128 bytes (pad the rest)

class CUmemAccessDesc(ctypes.Structure):
    # location (type + id), flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE (3)
```

**Python methods to expose:**

```python
class CudaDriverLibrary:
    def get_allocation_granularity(self, device: int) -> int:
        """Returns minimum physical allocation alignment in bytes (typically 2MB)."""

    def reserve_va(self, size: int) -> int:
        """Reserve virtual address range. Returns VA base pointer."""

    def create_physical(self, device: int, size: int) -> int:
        """Allocate physical memory chunk. Returns CUmemGenericAllocationHandle."""

    def map_physical(self, va_base: int, offset: int, size: int, handle: int):
        """Map a physical chunk at va_base+offset."""

    def set_access(self, device: int, va_base: int, total_size: int):
        """Set read/write access on the VA range."""

    def export_to_fd(self, handle: int) -> int:
        """Export physical handle as POSIX fd. Returns fd number."""

    def import_from_fd(self, pid: int, fd: int) -> int:
        """Import physical handle from another process. Returns handle."""
        # Internally: opens /proc/{pid}/fd/{fd}, then calls cuMemImportFromShareableHandle
```

**How to find libcuda.so:** Use the existing `find_loaded_library("libcuda")` helper.

### Verification

After implementing, run a standalone test:

```python
# test_vmm_wrapper.py
import torch
torch.cuda.init()  # ensure CUDA context

from vllm.distributed.device_communicators.cuda_wrapper import CudaDriverLibrary

drv = CudaDriverLibrary()
gran = drv.get_allocation_granularity(0)
print(f"Granularity: {gran} bytes ({gran / 1024 / 1024} MB)")

# Reserve 4MB VA, create 2x2MB physical chunks, map them
va = drv.reserve_va(4 * 1024 * 1024)
h1 = drv.create_physical(0, 2 * 1024 * 1024)
h2 = drv.create_physical(0, 2 * 1024 * 1024)
drv.map_physical(va, 0, 2 * 1024 * 1024, h1)
drv.map_physical(va, 2 * 1024 * 1024, 2 * 1024 * 1024, h2)
drv.set_access(0, va, 4 * 1024 * 1024)

# Wrap as torch tensor and verify read/write works
import cupy as cp
mem = cp.cuda.UnownedMemory(va, 4 * 1024 * 1024, owner=None)
ptr = cp.cuda.MemoryPointer(mem, 0)
arr = cp.ndarray((1024, 1024), dtype=cp.float32, memptr=ptr)
t = torch.as_tensor(arr, device="cuda:0")
t.fill_(42.0)
assert t.mean().item() == 42.0
print("VMM wrapper works!")
```

```bash
cd /data/vgupta345/vllm_merged_model
python test_vmm_wrapper.py
```

---

## Step 2: Create VMMCompositeWeight Helper

**File:** `vllm/model_executor/model_loader/vmm_utils.py` (new)

```python
class VMMCompositeWeight:
    """
    Allocates a contiguous virtual address backed by N separate
    physical memory chunks. Each chunk can be independently
    owned (freshly allocated) or imported (from another process).
    """

    def __init__(self, device, sub_component_sizes, dtype):
        """
        Args:
            device: CUDA device index
            sub_component_sizes: list of sizes in ELEMENTS per sub-component
                e.g. [q_rows * hidden, k_rows * hidden, v_rows * hidden]
            dtype: torch dtype (e.g. torch.bfloat16)
        """
        # 1. Query granularity
        # 2. Compute byte sizes, align each to granularity
        # 3. cuMemAddressReserve(sum of aligned sizes)
        # 4. For each sub-component: cuMemCreate + cuMemMap at cumulative offset
        # 5. cuMemSetAccess on full range
        # 6. Wrap VA as torch.Tensor via CuPy (same technique as
        #    default_loader.py:386-393)
        # 7. Store: handles[], aligned_sizes[], offsets[], tensor

    def export_chunk(self, index: int) -> tuple[int, int]:
        """Export physical chunk at index as (pid, fd)."""
        # calls drv.export_to_fd(self.handles[index])
        # returns (os.getpid(), fd)

    def import_chunk(self, index: int, pid: int, fd: int):
        """
        Replace physical chunk at index with imported memory.
        Must be called BEFORE writing any data to this chunk.
        """
        # 1. cuMemUnmap the existing chunk at this offset (if any)
        # 2. cuMemRelease the existing physical handle
        # 3. Import: drv.import_from_fd(pid, fd) -> new handle
        # 4. cuMemMap new handle at the same offset
        # 5. cuMemSetAccess
        # 6. Update self.handles[index]

    @property
    def tensor(self) -> torch.Tensor:
        """The contiguous tensor wrapping the full VA range."""
        return self._tensor
```

### Verification

Extend the Step 1 test to create a VMMCompositeWeight with 3 chunks,
write different values to each chunk, verify the tensor is contiguous:

```python
# test_vmm_composite.py
import torch
torch.cuda.init()
from vllm.model_executor.model_loader.vmm_utils import VMMCompositeWeight

# Simulate QKV for DeepSeek-Math-7B layer 2: each 4096 * 4096 elements
sizes = [4096 * 4096, 4096 * 4096, 4096 * 4096]  # Q, K, V in elements
w = VMMCompositeWeight(device=0, sub_component_sizes=sizes, dtype=torch.bfloat16)

t = w.tensor.view(12288, 4096)  # same shape as qkv_proj.weight
print(f"Shape: {t.shape}, contiguous: {t.is_contiguous()}, device: {t.device}")

# Write different values to each region
t[0:4096, :].fill_(1.0)      # Q region
t[4096:8192, :].fill_(2.0)   # K region
t[8192:12288, :].fill_(3.0)  # V region

# Verify
assert t[0, 0].item() == 1.0
assert t[4096, 0].item() == 2.0
assert t[8192, 0].item() == 3.0
print("VMMCompositeWeight works!")
```

```bash
python test_vmm_composite.py
```

---

## Step 3: Create Sharing Spec File

**File:** `sharing_spec.json`

```json
{
  "components": [
    {
      "layer": 2,
      "module": "self_attn.qkv_proj",
      "shards": {
        "q": "own",
        "k": {"owner": "model_a"},
        "v": "own"
      }
    }
  ]
}
```

Semantics:
- `"own"` → this model allocates and loads its own physical memory
- `{"owner": "model_a"}` → import physical memory from model_a's registry

Every model reads the same spec. The `--model-id` flag tells each process
which role it plays. If `model_id == owner`, it exports. Otherwise, it imports.

**File:** `vmm_registry.jsonl` (written at runtime by owner, read by consumers)

Each line written by the owner after weight loading:
```json
{"owner": "model_a", "layer": 2, "component": "self_attn.qkv_proj", "shard": "k", "pid": 12345, "fd": 7, "aligned_size": 33554432, "dtype": "torch.bfloat16"}
```

---

## Step 4: Modify default_loader.py

**File:** `vllm/model_executor/model_loader/default_loader.py`

Modify the `load_weights` method. The new flow replaces the existing
`store_weight_pointers`/`load_weight_pointers` mechanism entirely.

### New flow inside `load_weights()`:

```
BEFORE model.load_weights():
    1. Parse sharing_spec.json
    2. For each spec entry (e.g., layer 2 qkv_proj):
       a. Look up the module: model.layers.2.self_attn.qkv_proj
       b. Read sub-component sizes from module.output_sizes
          → [4096*128, 4096*128, 4096*128] for DeepSeek-Math-7B
          (or use _get_shard_size_mapping for Q/K/V)
       c. Create VMMCompositeWeight(sizes, dtype)
       d. If this model is an importer for any shard:
          - Read registry.jsonl, wait for owner's entry
          - Call vmm_weight.import_chunk(shard_index, pid, fd)
       e. Replace module.weight = nn.Parameter(vmm_weight.tensor)
       f. Build skip_set: {(2, "k")} for imported shards

    3. Wrap the weight iterator to skip imported shards:

        def filtered_weights(weights_iter, skip_set):
            stacked = {".q_proj": ("qkv_proj", "q"),
                       ".k_proj": ("qkv_proj", "k"),
                       ".v_proj": ("qkv_proj", "v")}
            for name, weight in weights_iter:
                layer_idx = extract_layer_index(name)  # parse "layers.2"
                for shard_name, (_, shard_id) in stacked.items():
                    if shard_name in name and (layer_idx, shard_id) in skip_set:
                        break  # skip this weight
                else:
                    yield name, weight  # pass through

CALL model.load_weights(filtered_weights(...)):
    - Layer 2's k_proj never reaches the weight_loader
    - Layer 2's q_proj and v_proj load normally into VMM-backed tensor
    - All other layers load normally into standard PyTorch tensors

AFTER model.load_weights():
    4. If this model owns any shards:
       - For each owned VMM chunk: call vmm_weight.export_chunk(index)
       - Write (pid, fd, size) to vmm_registry.jsonl
       - Write sentinel: {"owner": "model_a", "status": "ready"}
```

### Key detail: weight_loader compatibility

The `QKVParallelLinear.weight_loader` (linear.py:1023) does:
```python
param_data = param.data.narrow(output_dim, shard_offset, shard_size)
param_data.copy_(loaded_weight)
```

This works unchanged because `param.data` is the VMM-backed contiguous tensor.
`narrow()` creates a view at the correct offset. `copy_()` writes into the
VMM physical chunk backing that offset. The weight_loader has no idea the
physical backing is VMM — it just sees a contiguous tensor.

### Config additions

Add `--model-id` to `engine/arg_utils.py` and `config/model.py`:
```python
model_id: Optional[str] = None  # e.g., "model_a" or "model_b"
```

Reuse existing `--shared-layers-spec-path` for the spec file path.
Reuse existing `--shared-layers-ptrs-path` for the registry file path.

---

## Step 5: Test the Toy Example

### 5a. Start Model A (owner)

```bash
CUDA_VISIBLE_DEVICES=7 vllm serve \
    /scratch/shared_dir/unified_models/deepseek-math-7b-instruct/ \
    --port 12301 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --shared-layers-spec-path sharing_spec.json \
    --shared-layers-ptrs-path vmm_registry.jsonl \
    --model-id model_a \
    --gpu-memory-utilization 0.4
```

Expected behavior:
- Loads all weights normally
- Layer 2 qkv_proj allocated via VMM (3 physical chunks)
- K chunk exported to `vmm_registry.jsonl`
- Server starts listening on port 12301

### 5b. Start Model B (consumer)

```bash
CUDA_VISIBLE_DEVICES=7 vllm serve \
    /scratch/shared_dir/unified_models/deepseek-math-7b-instruct/ \
    --port 12305 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --shared-layers-spec-path sharing_spec.json \
    --shared-layers-ptrs-path vmm_registry.jsonl \
    --model-id model_b \
    --gpu-memory-utilization 0.4
```

Expected behavior:
- Reads `vmm_registry.jsonl`, finds Model A's K handle for layer 2
- Layer 2 qkv_proj allocated via VMM: imports K from Model A, allocates own Q, V
- Skips loading `layers.2.self_attn.k_proj.weight` from checkpoint
- Loads own Q, V for layer 2 and all weights for other layers normally
- Server starts listening on port 12305

### 5c. Verify correctness

Send identical prompts to both models and compare outputs:

```bash
# Query Model A
curl -s http://localhost:12301/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "deepseek-math-7b-instruct", "prompt": "What is 2+2?", "max_tokens": 50, "temperature": 0}' \
    | python -m json.tool

# Query Model B
curl -s http://localhost:12305/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "deepseek-math-7b-instruct", "prompt": "What is 2+2?", "max_tokens": 50, "temperature": 0}' \
    | python -m json.tool
```

Since both models use the same weights (same checkpoint, shared K, own Q/V
loaded from same checkpoint), outputs should be identical with temperature=0.

### 5d. Verify memory sharing

```bash
# Check GPU memory — Model B should use less memory than standalone
# The saving is 1 layer's K weight: 4096 * 4096 * 2 bytes = 32 MB
# Small for a toy example, but proves the mechanism works
nvidia-smi
```

### 5e. Verify K is truly shared (not copied)

Add a debug check in the loader: after both models are running, the K region
of Model B's qkv_proj.weight should have the same device pointer (modulo VA
mapping) as Model A's K region. We can verify by modifying Model A's K in-place
and checking if Model B sees the change:

```python
# From a debug script that connects to both servers
# Modify Model A's K weight:
model_a_layers[2].self_attn.qkv_proj.weight.data[4096, 0] = 999.0
# Read Model B's K weight at the same position:
val = model_b_layers[2].self_attn.qkv_proj.weight.data[4096, 0].item()
assert val == 999.0  # Same physical memory!
```

---

## Implementation Order

1. **Step 1** — CudaDriverLibrary wrapper (can be tested independently)
2. **Step 2** — VMMCompositeWeight helper (depends on Step 1, testable independently)
3. **Step 3** — Spec file format (just a JSON file, no code)
4. **Step 4** — default_loader.py changes (depends on Steps 1-3)
5. **Step 5** — End-to-end test

Steps 1 and 2 are self-contained and can be developed + tested in isolation
before touching any vLLM serving code.

---

## Important: Zero C Changes, Zero Rebuilds

All CUDA driver API calls are made via **Python ctypes** wrapping `libcuda.so`.
This is the exact same technique used by the existing `CudaRTLibrary` class
in `cuda_wrapper.py` (which wraps `libcudart.so` via ctypes). No C extension
changes, no vLLM rebuild required.

The context mismatch issue noted in `cumem.py:4-10` was specific to the
**pluggable allocator callback path** — where C code calls back into Python
during `cudaMalloc`, potentially from a different CUDA context. Our use case
is completely different: we make direct, synchronous calls from the main Python
thread during weight loading, with an already-initialized CUDA context
(ensured by `torch.cuda.init()` or the existing `lib.cudaSetDevice(0)` call
at `default_loader.py:278`). This is identical to how `CudaRTLibrary` already
calls `cudaIpcGetMemHandle` successfully from Python.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Sub-component size not aligned to 2MB granularity | For DeepSeek-Math-7B: Q=32MB, K=32MB, V=32MB — all clean multiples. Add assertion to fail early if misaligned. |
| Model B starts before Model A finishes exporting | Registry polling with timeout: Model B reads registry in a loop until owner's sentinel appears. |
| Owner process (Model A) dies | All imported mappings become invalid. Operational constraint: keep owner alive. Can add health check later. |
| `weights_not_loaded` validation at default_loader.py:296 fails for skipped weights | The filtered iterator means `load_weights()` never reports the skipped weight names in `loaded_params`. Need to pre-add skipped names to `loaded_params` so the validation passes. |

---

## Future: Scaling to N Models

The same spec file supports N models with no code changes:

```json
{
  "components": [
    {"layer": 2, "module": "self_attn.qkv_proj",
     "shards": {"q": "own", "k": {"owner": "model_a"}, "v": "own"}},
    {"layer": 5, "module": "mlp.gate_up_proj",
     "shards": {"gate": {"owner": "model_a"}, "up": "own"}}
  ]
}
```

Each new model process gets `--model-id model_c` and reads the same spec.
The loader builds the skip set and import map from the spec + model_id.
No model code changes, no hardcoding.
