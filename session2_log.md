# Session 2 Log: VMM Sub-Component Sharing — Bug Fixes & Integration

## Date: 2025-02-14

## Summary

Fixed two critical bugs blocking cross-process VMM sharing, built infrastructure
(skip_indices, fd server/client), and wrote the default_loader.py integration.
End-to-end vllm serve test not yet run.

---

## Bugs Fixed

### Bug 1: `cuMemImportFromShareableHandle` wrong calling convention

**File:** `vllm/distributed/device_communicators/cuda_wrapper.py:452-463`

**Problem:** `import_from_fd()` passed the fd as `ctypes.byref(ctypes.c_int(fd))` —
a pointer to an int. The CUDA driver API for POSIX fd expects the fd value CAST
to `void*`, not a pointer to the fd.

**Root cause:** Asymmetric calling convention between export and import:
- Export: `cuMemExportToShareableHandle(&fd, ...)` — pointer to int (writes fd)
- Import: `cuMemImportFromShareableHandle(&handle, (void*)(uintptr_t)fd, ...)` — fd AS the pointer

**Fix:** Changed `ctypes.byref(ctypes.c_int(fd))` → `ctypes.c_void_p(fd)`

**Error before fix:** "CUDA Driver error: invalid device ordinal"

### Bug 2: `recv_fd` returning wrong fd

**File:** `vllm/model_executor/model_loader/vmm_utils.py:205-216`

**Problem:** `array.array("i", [0])` initialized with a dummy `0` element. When
`frombytes()` appended the received fd, `fds[0]` returned `0` (the dummy), not
the actual received fd.

**Fix:** Changed to `array.array("i")` (empty array).

**Symptom:** Consumer received `fd=0` instead of the actual fd from owner.

---

## New Features Added

### 1. `skip_indices` parameter for VMMCompositeWeight

**File:** `vllm/model_executor/model_loader/vmm_utils.py`

- Constructor accepts `skip_indices: Optional[set[int]]`
- For indices in skip_indices: VA is reserved but no physical chunk is allocated (hole)
- Holes are filled via `import_chunk_from_fd()`
- `finalize()` method sets access permissions and wraps VA as tensor
- If no skip_indices, `finalize()` is called automatically in `__init__`

### 2. `start_fd_server()` / `request_fd()` helpers

**File:** `vllm/model_executor/model_loader/vmm_utils.py`

- `start_fd_server(socket_path, fd_map)` — background thread serving DMA-BUF fds
  via Unix socket SCM_RIGHTS. `fd_map` maps string keys to file descriptors.
- `request_fd(socket_path, key)` — client that connects, sends key, receives fd.
- Keys use format: `"{layer}.{module}.{shard}"` e.g., `"2.self_attn.qkv_proj.k_proj"`

### 3. `sharing_spec.json` format

```json
{
  "owner": "model_a",
  "socket_path": "/tmp/vmm_sharing.sock",
  "components": [
    {
      "layer": 2,
      "module": "self_attn.qkv_proj",
      "shard_order": ["q_proj", "k_proj", "v_proj"],
      "shared": ["k_proj"]
    }
  ]
}
```

- `shard_order` maps checkpoint names to VMM chunk indices (no model code dependency)
- `shared` lists which shards are imported from owner
- Fully general: different specs for different models, no code changes needed

### 4. `_setup_vmm_sharing()` in default_loader.py

**File:** `vllm/model_executor/model_loader/default_loader.py`

New method added before `load_weights()`. Flow:

**Owner (VMM_MODEL_ID=model_a):**
1. Parse spec
2. For each component: create VMMCompositeWeight (all chunks, exportable)
3. Swap module.weight with VMM tensor
4. Export shared chunks, start fd server on socket_path
5. `model.load_weights()` runs — writes directly into VMM physical chunks

**Consumer (VMM_MODEL_ID=model_b):**
1. Parse spec
2. For each component: create VMMCompositeWeight with skip_indices (holes)
3. Request fds from owner's fd server
4. Import shared chunks, finalize
5. Swap module.weight with VMM tensor
6. `model.load_weights()` with filtered iterator — skips shared checkpoint weights

**Filter logic:**
- For spec entry `{layer: 2, module: "self_attn.qkv_proj", shared: ["k_proj"]}`:
- Adds `"model.layers.2.self_attn.k_proj"` to skip_names
- Any checkpoint weight matching this substring is skipped
- Spec-driven, adapts to whatever components are shared

**Coexists with old cudaIpc code:** detects VMM spec (JSON with `{`) vs old
handles.jsonl (JSONL starting with `{` per line but different structure).

---

## Test Results

### test_vmm_wrapper.py — PASSES
Basic VMM operations (unchanged from session 1).

### test_vmm_composite.py — PASSES
Multi-chunk contiguous tensor (unchanged from session 1).

### test_vmm_cross_process.py — PASSES (was FAILING in session 1)
Cross-process sharing via Unix socket SCM_RIGHTS. Both bugs fixed.

### test_procfd.py — PASSES
Tests `skip_indices` + `finalize()` + `start_fd_server()`/`request_fd()` across
spawned processes. Confirms the full infrastructure works.

### /proc/pid/fd/ — DOES NOT WORK for DMA-BUF fds
Opening `/proc/<pid>/fd/<fd>` for a CUDA DMA-BUF fd fails with "invalid device
ordinal". Opening via procfs doesn't properly duplicate the DMA-BUF context.
`pidfd_getfd` syscall also not viable (ptrace_scope=1, requires parent-child).
Solution: Unix socket SCM_RIGHTS (tested, works).

---

## What's Left: End-to-End Test

The `default_loader.py` integration is written but NOT yet tested with `vllm serve`.
The background process launch didn't produce output (may have crashed silently).

### To test:

**Terminal 1 (owner):**
```bash
rm -f /tmp/vmm_sharing.sock
VMM_MODEL_ID=model_a CUDA_VISIBLE_DEVICES=7 vllm serve \
    /scratch/shared_dir/unified_models/deepseek-math-7b-instruct/ \
    --port 12301 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --shared-layers-spec-path sharing_spec.json \
    --gpu-memory-utilization 0.4
```

**Terminal 2 (consumer, after owner is serving):**
```bash
VMM_MODEL_ID=model_b CUDA_VISIBLE_DEVICES=7 vllm serve \
    /scratch/shared_dir/unified_models/deepseek-math-7b-instruct/ \
    --port 12305 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --shared-layers-spec-path sharing_spec.json \
    --gpu-memory-utilization 0.4
```

**Verify:**
```bash
curl -s http://localhost:12301/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-math-7b-instruct","prompt":"What is 2+2?","max_tokens":50,"temperature":0}'

curl -s http://localhost:12305/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-math-7b-instruct","prompt":"What is 2+2?","max_tokens":50,"temperature":0}'
```

### Likely issues to debug:
1. Owner process may crash during VMM setup — need to check stderr
2. `output_sizes` attribute may not exist on the module before load_weights
3. Weight validation may fail if skipped weights aren't accounted for
4. The `shared_layers_spec_path` flag may need `shared_layers_ptrs_path` too (existing code checks both)

---

## Files Modified/Created

| File | Status |
|------|--------|
| `vllm/distributed/device_communicators/cuda_wrapper.py` | Modified — fixed `import_from_fd()` |
| `vllm/model_executor/model_loader/vmm_utils.py` | Modified — added `skip_indices`, `finalize()`, `start_fd_server()`, `request_fd()`, fixed `recv_fd()` |
| `vllm/model_executor/model_loader/default_loader.py` | Modified — added `_setup_vmm_sharing()`, modified `load_weights()` |
| `sharing_spec.json` | New — toy example spec |
| `vmm_integration_plan.md` | New — detailed integration plan |
| `test_procfd.py` | New — tests skip_indices + fd server |

---

## Design Decisions

1. **Spec uses checkpoint-level names** (`k_proj`) not internal shard IDs (`k`).
   Filter is simple string matching. No model-specific code.

2. **`shard_order` in spec** maps checkpoint names to VMM chunk indices.
   Avoids dependency on `stacked_params_mapping` (which is a local variable
   inside model's `load_weights()`, not a class attribute).

3. **Unix socket SCM_RIGHTS** for fd passing. `/proc/pid/fd/` doesn't work for
   DMA-BUF fds. `pidfd_getfd` blocked by ptrace_scope.

4. **Sequential loading assumed.** Owner starts first, consumer starts after.
   No polling/retry logic. Fail fast if fd server not available.

5. **Coexists with existing cudaIpc sharing.** VMM path triggered only when
   spec file is JSON with `components` key.

## Environment

- GPU: NVIDIA H200
- CUDA: 12.8
- Python: 3.12
- Linux: 6.8.0-90-generic
- vLLM branch: `users/vima/shared-components`
- venv: `/data/vgupta345/vllm_merged_model/.venv`
- Test GPU: device 7 (`CUDA_VISIBLE_DEVICES=7`)
