"""Test: Can cudaIpcGetMemHandle work on VMM-backed memory?
If so, we can use the existing IPC mechanism instead of cuMemExport/Import."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

import torch
torch.cuda.init()
torch.cuda.set_device(0)
_ = torch.empty(1, device="cuda:0")

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary, CudaDriverLibrary)
import cupy as cp

print("=== Test: cudaIpcGetMemHandle on VMM memory ===\n")

drv = CudaDriverLibrary()
rt = CudaRTLibrary()

phys_dev = drv.get_physical_device()
print(f"Physical device: {phys_dev}")

gran = drv.get_allocation_granularity(phys_dev)
print(f"Granularity: {gran}")

# Allocate VMM memory: 2MB physical chunk
size = 2 * 1024 * 1024
va = drv.reserve_va(size)
handle = drv.create_physical(phys_dev, size, exportable=False)
drv.map_physical(va, 0, size, handle)
drv.set_access(phys_dev, va, size)

print(f"VMM VA: 0x{va:x}")

# Try cudaIpcGetMemHandle on the VMM pointer
try:
    import ctypes
    ipc_handle = rt.cudaIpcGetMemHandle(ctypes.c_void_p(va))
    print(f"[OK] cudaIpcGetMemHandle succeeded on VMM pointer!")

    # Encode it like the existing code does
    import base64
    handle_bytes = ctypes.string_at(ctypes.addressof(ipc_handle), 128)
    handle_b64 = base64.b64encode(handle_bytes).decode("ascii")
    print(f"[OK] IPC handle: {handle_b64[:40]}...")

except RuntimeError as e:
    print(f"[FAIL] cudaIpcGetMemHandle failed: {e}")

# Also test cuMemExportToShareableHandle directly
print("\n--- Testing cuMemExportToShareableHandle ---")
handle_exp = drv.create_physical(phys_dev, size, exportable=True)
try:
    fd = drv.export_to_fd(handle_exp)
    print(f"[OK] cuMemExportToShareableHandle succeeded, fd={fd}")

    # Now try import in the SAME process
    imported = drv.import_from_fd(fd)
    print(f"[OK] cuMemImportFromShareableHandle succeeded in same process, handle={imported}")
except RuntimeError as e:
    print(f"[FAIL] Export/import failed: {e}")

# Cleanup
drv.unmap(va, size)
drv.release_physical(handle)
drv.free_va(va, size)

print("\n=== Done ===")
