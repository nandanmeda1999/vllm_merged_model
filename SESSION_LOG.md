# Session Log: VMM-Based Partial Component Sharing in vLLM

## Date: 2025-02-13

## Goal

Enable partial sharing of sub-components within merged weight tensors (e.g., share
K but not Q/V from a QKV projection) across multiple vLLM model-serving processes,
with zero-copy memory sharing and no kernel/forward-pass changes.

## Background

vLLM merges Q, K, V projections into a single `qkv_proj` weight tensor `[Q+K+V rows, hidden]`
for efficient single-GEMM execution. Similarly, gate and up projections are merged into
`gate_up_proj`. The existing sharing mechanism (`cudaIpcGetMemHandle`) operates on whole
tensors — you either share the entire merged QKV or nothing.

The challenge: share individual sub-components (e.g., just K) while keeping the merged
tensor contiguous for the GEMM kernel.

## Approach: CUDA Virtual Memory Management (VMM)

Use CUDA Driver API VMM functions to decouple physical memory from virtual addressing:

1. Allocate Q, K, V as **3 separate physical chunks** (`cuMemCreate`)
2. Map all 3 **contiguously into one VA range** (`cuMemAddressReserve` + `cuMemMap`)
3. Each physical chunk can be independently shared across processes
4. The GEMM kernel sees a single contiguous tensor — no changes needed

## Key Existing Code

- **`vllm/model_executor/models/llama.py:152-160`** — `QKVParallelLinear` creates merged QKV
- **`vllm/model_executor/layers/linear.py:849-1194`** — `QKVParallelLinear` weight loading with shard offsets
- **`vllm/model_executor/layers/linear.py:191-223`** — `UnquantizedLinearMethod.create_weights()` does the actual `torch.empty()` allocation
- **`vllm/model_executor/model_loader/default_loader.py:268-300`** — `load_weights()` orchestrates weight loading
- **`vllm/model_executor/model_loader/default_loader.py:301-397`** — Current IPC sharing (store/load weight pointers via `cudaIpcGetMemHandle`)
- **`vllm/distributed/device_communicators/cuda_wrapper.py`** — ctypes wrappers for CUDA runtime
- **`csrc/cumem_allocator.cpp`** — Existing VMM C extension (used for sleep/wake, NOT for our purpose)
- **`vllm/device_allocator/cumem.py`** — Python wrapper for the C extension

## What Was Implemented

### 1. `CudaDriverLibrary` class (WORKING)
**File:** `vllm/distributed/device_communicators/cuda_wrapper.py`

Pure Python ctypes wrapper for `libcuda.so` (driver API). Wraps:
- `cuCtxGetDevice` — get physical device ordinal
- `cuMemGetAllocationGranularity` — query 2MB page size
- `cuMemAddressReserve` / `cuMemAddressFree` — reserve/free VA ranges
- `cuMemCreate` / `cuMemRelease` — allocate/free physical chunks
- `cuMemMap` / `cuMemUnmap` — map/unmap physical to VA
- `cuMemSetAccess` — set read/write permissions
- `cuMemExportToShareableHandle` — export physical chunk as POSIX fd
- `cuMemImportFromShareableHandle` — import physical chunk from POSIX fd

**Note:** `find_loaded_library("libcuda")` matches `libcudart` because "libcuda" is a
substring. Fixed with custom `_find_libcuda()` that checks for "libcuda.so" while
excluding "libcudart".

### 2. `VMMCompositeWeight` class (WORKING for single-process)
**File:** `vllm/model_executor/model_loader/vmm_utils.py` (NEW)

Creates a contiguous tensor backed by N separate VMM physical chunks.
- Constructor: allocates N physical chunks, maps contiguously, wraps as PyTorch tensor
- `export_chunk(index)` → returns `(pid, fd)` for cross-process sharing
- `import_chunk_from_fd(index, fd)` → replaces a chunk with imported memory
- Includes `send_fd()` / `recv_fd()` helpers for Unix socket SCM_RIGHTS fd passing

### 3. ctypes structures for CUDA driver API
Added to `cuda_wrapper.py`:
- `CUmemLocation`, `CUmemAllocFlags`, `CUmemAllocationProp`, `CUmemAccessDesc`
- Constants: `CU_MEM_ALLOCATION_TYPE_PINNED`, `CU_MEM_LOCATION_TYPE_DEVICE`, etc.

## Test Results

### test_vmm_wrapper.py — PASSES
Basic VMM operations: reserve VA, create physical chunks, map, read/write, export.
```
CUDA_VISIBLE_DEVICES=7 python test_vmm_wrapper.py
```

### test_vmm_composite.py — PASSES
Multi-chunk contiguous tensor: 3 chunks (Q/K/V), write different values per region,
verify contiguity, run GEMM, split output. Proves the merged tensor works for the
forward pass.
```
CUDA_VISIBLE_DEVICES=7 python test_vmm_composite.py
```

### test_vmm_cross_process.py — FAILS
Cross-process sharing: owner exports K fd, consumer imports.
**Status: BLOCKED** on `cuMemImportFromShareableHandle` returning "invalid device ordinal".

## The Blocker: `cuMemImportFromShareableHandle`

`cuMemExportToShareableHandle` succeeds and returns a valid POSIX fd.
`cuMemImportFromShareableHandle` fails with "invalid device ordinal" — even **in the
same process** (not a cross-process issue).

Tested on:
- Device 0 (no CUDA_VISIBLE_DEVICES) → fails
- Device 7 (no CUDA_VISIBLE_DEVICES) → fails
- Device 0 via CUDA_VISIBLE_DEVICES=7 → fails

The issue is likely in how the fd is passed to the ctypes function. The CUDA API
signature is:
```c
CUresult cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle *handle,  // output
    void *osHandle,                        // for POSIX: pointer to int containing fd
    CUmemAllocationHandleType handleType   // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=1
)
```

Current code passes `ctypes.byref(fd_val)` where `fd_val = ctypes.c_int(fd)`.
This SHOULD be correct (pointer to int), but the CUDA driver rejects it.

**Possible causes to investigate:**
1. The `void*` parameter might need different ctypes casting (tried `ctypes.byref`, not yet tried `ctypes.pointer` or `ctypes.c_void_p(ctypes.addressof(fd_val))`)
2. Maybe the fd needs to be passed as a raw `void*` value (i.e., `ctypes.c_void_p(fd)`) rather than a pointer to the fd
3. The `requestedHandleTypes` prop field might need a specific combination of flags
4. There could be a driver version or GPU architecture issue (H200, CUDA 12.8, driver ?)
5. Maybe `cuInit(0)` needs to be called explicitly before import

**Recommended next step:** Write a minimal C program that calls `cuMemCreate` +
`cuMemExportToShareableHandle` + `cuMemImportFromShareableHandle` to verify the
correct argument format, then mirror it in ctypes. Alternatively, check if the existing
C extension (`cumem_allocator.cpp`) can be extended with just one function for import.

## Alternative Approach If VMM Import Stays Broken

Fall back to `cudaIpcGetMemHandle` (which DOES NOT work on VMM memory — confirmed
"invalid argument") combined with a **copy-compose** strategy:

1. Both models allocate merged QKV normally via PyTorch
2. Owner exports per-sub-component IPC handles with byte offsets
3. Consumer opens the IPC handle, uses `cudaMemcpy` to copy just the shared
   sub-component (K) from owner's tensor into its own tensor's K region
4. Not zero-copy, but saves the complexity of VMM import/export
5. Memory saving: none (both models allocate full QKV), but achieves weight sharing semantics

## Implementation Plan

Full plan is in `implementation_plan.md`. The remaining steps (after fixing the blocker):

1. Fix `cuMemImportFromShareableHandle` ctypes call
2. Verify cross-process test passes
3. Modify `default_loader.py` to use VMMCompositeWeight for shareable layers
4. Add spec file parsing and weight iterator filtering
5. End-to-end test with two vLLM serve processes sharing layer 2 K

## Files Created/Modified

| File | Status |
|------|--------|
| `vllm/distributed/device_communicators/cuda_wrapper.py` | Modified — added `CudaDriverLibrary` |
| `vllm/model_executor/model_loader/vmm_utils.py` | New — `VMMCompositeWeight` |
| `implementation_plan.md` | New — full implementation plan |
| `test_vmm_wrapper.py` | New — passes |
| `test_vmm_composite.py` | New — passes |
| `test_vmm_cross_process.py` | New — fails (blocker) |
| `test_vmm_ipc.py` | New — diagnostic test |

## Environment

- GPU: NVIDIA H200
- CUDA: 12.8
- Python: 3.12
- vLLM: custom branch `users/vima/shared-components`
- venv: `/data/vgupta345/vllm_merged_model/.venv`
- Test GPU: device 7 (`CUDA_VISIBLE_DEVICES=7`)
- CuPy: 13.6.0
