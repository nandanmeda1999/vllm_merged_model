# VMM Sub-Component Sharing: Integration Plan

## Contradictions with `implementation_plan.md`

The original plan (Steps 1-2) is **unchanged** — CudaDriverLibrary and VMMCompositeWeight
are implemented and tested. Steps 3-4 are **superseded** by this plan:

| Topic | Old Plan | This Plan |
|-------|----------|-----------|
| Spec format | Internal shard names (`"q"`, `"k"`, `"v"`) | Checkpoint-level names (`"k_proj"`, `"gate_proj"`) — general across models |
| Filtering | Hardcoded mapping `{".k_proj": ("qkv_proj", "k")}` | Spec-driven string matching against checkpoint weight names |
| Synchronization | Polling/retry with sentinel file | Sequential loading — registry guaranteed to exist |
| fd passing | Unix socket SCM_RIGHTS | `/proc/<pid>/fd/<fd>` (simpler, no socket server) |
| VMMCompositeWeight | Allocates all chunks, replaces on import | `skip_indices` param — no wasted physical allocation |
| Role identification | `--model-id` CLI flag | `VMM_MODEL_ID` env var (zero config changes) |

---

## Prerequisites (DONE)

- [x] `CudaDriverLibrary` — VMM ctypes wrapper (cuda_wrapper.py)
- [x] `VMMCompositeWeight` — multi-chunk contiguous tensor (vmm_utils.py)
- [x] Bug fix: `import_from_fd` uses `c_void_p(fd)` not `byref(c_int(fd))`
- [x] Bug fix: `recv_fd` starts with empty array
- [x] Cross-process sharing test passes

---

## Step 1: Add `skip_indices` to VMMCompositeWeight

**File:** `vllm/model_executor/model_loader/vmm_utils.py`

Add a `skip_indices` parameter to the constructor. For indices in this set,
the constructor reserves VA space but does NOT call `cuMemCreate` or `cuMemMap`.
These "holes" are filled later via `import_chunk_from_fd()`.

```python
def __init__(self, device, sub_component_sizes_bytes, shape, dtype,
             exportable=True, skip_indices=None):
    # ... existing code ...
    for i in range(self.n_chunks):
        if skip_indices and i in skip_indices:
            self.handles[i] = None
            self.owned[i] = False
            continue  # leave hole — VA reserved but unmapped
        h = self.drv.create_physical(...)
        # ... existing allocation + map code ...
```

**Important:** `set_access` and tensor wrapping must be deferred until ALL
holes are filled (after all imports). Add a `finalize()` method:

```python
def finalize(self):
    """Call after all imports are done. Sets access and wraps as tensor."""
    self.drv.set_access(self.physical_device, self.va_base, self.total_aligned)
    self._tensor = self._wrap_as_tensor()
```

For the owner (no holes), `finalize()` is called immediately in `__init__`.
For the consumer, `finalize()` is called after all `import_chunk_from_fd()` calls.

### Verification

```python
# Same as test_vmm_composite.py but with skip_indices=[1]
w = VMMCompositeWeight(..., skip_indices={1})
# w.tensor not available yet — holes exist
w.import_chunk_from_fd(1, received_fd)
w.finalize()
# Now w.tensor is usable
```

---

## Step 2: Test `/proc/<pid>/fd/<fd>` for DMA-BUF fds

Before integrating into the loader, verify that opening `/proc/<pid>/fd/<fd>`
works for CUDA DMA-BUF fds (alternative to Unix socket SCM_RIGHTS).

```python
# In owner process:
pid, fd = vmm_weight.export_chunk(1)
# Write pid, fd to a file

# In consumer process:
import_fd = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR)
vmm_weight.import_chunk_from_fd(1, import_fd)
os.close(import_fd)
```

If this fails, fall back to Unix socket (already tested and working).

---

## Step 3: Create Sharing Spec Format

**File:** `sharing_spec.json` (static, created by user before runtime)

```json
{
  "owner": "model_a",
  "components": [
    {
      "layer": 2,
      "module": "self_attn.qkv_proj",
      "shared": ["k_proj"]
    }
  ]
}
```

**Semantics:**
- `owner` — the `VMM_MODEL_ID` of the process that allocates and exports
- `components[].layer` — layer index
- `components[].module` — merged module name (matches `model.named_modules()` path)
- `components[].shared` — list of checkpoint-level weight names that are shared
  (e.g., `k_proj` from `qkv_proj`, `gate_proj` from `gate_up_proj`)

**Generalizes to any model/component:**
```json
{
  "owner": "model_a",
  "components": [
    {"layer": 2, "module": "self_attn.qkv_proj", "shared": ["k_proj"]},
    {"layer": 5, "module": "self_attn.qkv_proj", "shared": ["k_proj", "v_proj"]},
    {"layer": 3, "module": "mlp.gate_up_proj", "shared": ["gate_proj"]}
  ]
}
```

---

## Step 4: VMM Registry Format

**File:** `vmm_registry.jsonl` (written by owner at runtime, read by consumers)

Each line written by the owner after weight loading:
```json
{"layer": 2, "module": "self_attn.qkv_proj", "shard": "k_proj", "shard_index": 1, "pid": 12345, "fd": 66, "aligned_size": 33554432}
```

Consumers read this file to get `(pid, fd)` for each shared shard.
Sequential loading guarantees this file exists when consumers start.

---

## Step 5: Modify `default_loader.py`

**File:** `vllm/model_executor/model_loader/default_loader.py`

### Overview of new `load_weights()` flow

```
BEFORE model.load_weights():
  1. Parse sharing_spec.json
  2. Read VMM_MODEL_ID from env
  3. Determine role: owner (model_id == spec.owner) or consumer
  4. For each spec entry:
     a. Look up module: model.layers.{layer}.{module}
     b. Read module.output_sizes + module.weight.shape/dtype
     c. Use stacked_params_mapping to map shared checkpoint names → shard indices
     d. Owner: create VMMCompositeWeight(all chunks, exportable=True)
        Consumer: create VMMCompositeWeight(skip_indices=shared shard indices)
     e. Consumer: read registry, import shared chunks via /proc/pid/fd/
     f. Finalize VMM tensor
     g. Swap: module.weight = nn.Parameter(vmm.tensor, requires_grad=False)
  5. Build skip_names: set of checkpoint weight name patterns to filter
     e.g., {"model.layers.2.self_attn.k_proj"}

  6. Wrap weight iterator with filter:
     def filtered_weights(weights_iter, skip_names):
         for name, tensor in weights_iter:
             if any(skip in name for skip in skip_names):
                 continue
             yield name, tensor

CALL model.load_weights(filtered_weights(...))
  - Shared shards never reach the weight_loader
  - Owned shards load directly into VMM physical memory via narrow().copy_()
  - All non-shared layers load normally into standard PyTorch tensors

AFTER model.load_weights():
  7. Owner: export shared chunks, write registry
     for each spec entry:
         pid, fd = vmm_weight.export_chunk(shard_index)
         append {layer, module, shard, shard_index, pid, fd, aligned_size} to registry
```

### How shard index is determined (general, no model-specific code)

```python
# Read from model — every vLLM model defines this
stacked_mapping = model.stacked_params_mapping
# e.g., [(".qkv_proj", ".q_proj", "q"), (".qkv_proj", ".k_proj", "k"), ...]

# For spec entry: module="self_attn.qkv_proj", shared=["k_proj"]
# Find all entries for qkv_proj, in order:
#   index 0: q_proj
#   index 1: k_proj  ← shared
#   index 2: v_proj
# So k_proj → shard_index = 1
```

This works for ANY model because the mapping is always defined.

### Sub-component sizes (general)

```python
module = dict(model.named_modules())[f"model.layers.{layer}.{module_name}"]
output_sizes = module.output_sizes  # e.g., [4096, 4096, 4096] for QKV
input_size = module.weight.shape[1]  # e.g., 4096
dtype = module.weight.dtype
element_size = dtype.itemsize  # e.g., 2 for bfloat16

sub_component_sizes_bytes = [s * input_size * element_size for s in output_sizes]
```

### Weight validation

The `weights_not_loaded` check at line 295 compares `weights_to_load`
(parameter names) against `loaded_weights` (returned by `model.load_weights()`).

For partial sharing: the parameter `model.layers.2.self_attn.qkv_proj.weight`
will still be in `loaded_weights` because the non-shared shards (q_proj, v_proj)
ARE loaded from checkpoint. The weight_loader adds the parameter name to
`loaded_params` when ANY shard is loaded. So validation passes automatically.

### Filtering is spec-driven (adapts to any components)

The filter doesn't know about Q/K/V semantics. It just builds a set of
checkpoint weight name patterns from the spec:

```python
skip_names = set()
for entry in spec["components"]:
    layer = entry["layer"]
    for shard_name in entry["shared"]:
        skip_names.add(f"model.layers.{layer}.self_attn.{shard_name}")
        # or f"model.layers.{layer}.mlp.{shard_name}" depending on module
```

Different models share different components → different spec → different filter.
Same code.

---

## Step 6: End-to-End Test

### 6a. Create sharing_spec.json

```json
{
  "owner": "model_a",
  "components": [
    {"layer": 2, "module": "self_attn.qkv_proj", "shared": ["k_proj"]}
  ]
}
```

### 6b. Start Model A (owner)

```bash
VMM_MODEL_ID=model_a CUDA_VISIBLE_DEVICES=7 vllm serve \
    /scratch/shared_dir/unified_models/deepseek-math-7b-instruct/ \
    --port 12301 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --shared-layers-spec-path sharing_spec.json \
    --shared-layers-ptrs-path vmm_registry.jsonl \
    --gpu-memory-utilization 0.4
```

### 6c. Start Model B (consumer, after owner is ready)

```bash
VMM_MODEL_ID=model_b CUDA_VISIBLE_DEVICES=7 vllm serve \
    /scratch/shared_dir/unified_models/deepseek-math-7b-instruct/ \
    --port 12305 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --shared-layers-spec-path sharing_spec.json \
    --shared-layers-ptrs-path vmm_registry.jsonl \
    --gpu-memory-utilization 0.4
```

### 6d. Verify

```bash
# Same prompt, temperature=0 → identical outputs (same checkpoint, shared K)
curl -s http://localhost:12301/v1/completions \
    -d '{"model":"deepseek-math-7b-instruct","prompt":"What is 2+2?","max_tokens":50,"temperature":0}'

curl -s http://localhost:12305/v1/completions \
    -d '{"model":"deepseek-math-7b-instruct","prompt":"What is 2+2?","max_tokens":50,"temperature":0}'
```

---

## Implementation Order

1. **Step 1** — Add `skip_indices` + `finalize()` to VMMCompositeWeight (~15 lines)
2. **Step 2** — Test `/proc/pid/fd/` with DMA-BUF fds (quick test script)
3. **Step 3** — Create `sharing_spec.json` (static file)
4. **Step 5** — Modify `default_loader.py` (~60 lines in `load_weights()`)
5. **Step 6** — End-to-end test

Steps 1-2 are isolated and testable independently.
Step 5 is the main integration work.

---

## Files Changed

| File | Change | Lines (est.) |
|------|--------|-------------|
| `vmm_utils.py` | Add `skip_indices`, `finalize()` | ~15 |
| `default_loader.py` | Modify `load_weights()` | ~60 |
| `sharing_spec.json` | New static config | ~10 |

**No changes to:** model files (llama.py, etc.), linear.py, kernels, forward pass,
arg_utils.py, config.py.

---

## Coexistence with Existing cudaIpc Sharing

The existing `store_weight_pointers` / `load_weight_pointers` (whole-module sharing
via cudaIpc) remains available for whole-module sharing. The new VMM code path is
triggered only when `sharing_spec.json` contains partial sharing entries.

A future spec format could support both in one file:
```json
{
  "components": [
    {"layer": 2, "module": "self_attn.qkv_proj", "shared": ["k_proj"]},
    {"layer": 3, "module": "self_attn.o_proj", "shared": "whole"}
  ]
}
```

For now, the toy example only uses partial sharing.
