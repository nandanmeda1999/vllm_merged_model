"""Test 1: CudaDriverLibrary VMM wrapper — basic operations."""

import torch
# Ensure CUDA context exists before loading driver API
torch.cuda.init()
torch.cuda.set_device(0)
# Allocate a small tensor to force context creation
_ = torch.empty(1, device="cuda:0")

from vllm.distributed.device_communicators.cuda_wrapper import CudaDriverLibrary
import cupy as cp

print("=== Test 1: CudaDriverLibrary VMM Wrapper ===\n")

drv = CudaDriverLibrary()

# 1. Query granularity
gran = drv.get_allocation_granularity(0)
print(f"[OK] Allocation granularity: {gran} bytes ({gran / 1024 / 1024:.1f} MB)")

# 2. Reserve VA (4 MB)
va_size = 4 * 1024 * 1024
va = drv.reserve_va(va_size)
print(f"[OK] Reserved VA: 0x{va:x} ({va_size} bytes)")

# 3. Create two physical chunks (2 MB each)
h1 = drv.create_physical(0, 2 * 1024 * 1024)
h2 = drv.create_physical(0, 2 * 1024 * 1024)
print(f"[OK] Created physical chunk 1: handle={h1}")
print(f"[OK] Created physical chunk 2: handle={h2}")

# 4. Map them contiguously
drv.map_physical(va, 0, 2 * 1024 * 1024, h1)
drv.map_physical(va, 2 * 1024 * 1024, 2 * 1024 * 1024, h2)
print(f"[OK] Mapped both chunks into contiguous VA")

# 5. Set access
drv.set_access(0, va, va_size)
print(f"[OK] Set read/write access")

# 6. Wrap as torch tensor and verify read/write
n_elements = va_size // 4  # float32
mem = cp.cuda.UnownedMemory(va, va_size, owner=None)
ptr = cp.cuda.MemoryPointer(mem, 0)
arr = cp.ndarray((n_elements,), dtype=cp.float32, memptr=ptr)
t = torch.as_tensor(arr, device="cuda:0")

# Write to first chunk region
t[:n_elements // 2].fill_(42.0)
# Write to second chunk region
t[n_elements // 2:].fill_(99.0)

assert t[0].item() == 42.0, f"Expected 42.0, got {t[0].item()}"
assert t[n_elements // 2].item() == 99.0, f"Expected 99.0, got {t[n_elements // 2].item()}"
print(f"[OK] Read/write across two physical chunks works")

# 7. Verify contiguity
assert t.is_contiguous(), "Tensor should be contiguous"
print(f"[OK] Tensor is contiguous: shape={t.shape}, stride={t.stride()}")

# 8. Test export (create an exportable chunk)
h3 = drv.create_physical(0, 2 * 1024 * 1024, exportable=True)
print(f"[OK] Created exportable physical chunk: handle={h3}")
fd = drv.export_to_fd(h3)
print(f"[OK] Exported to fd={fd}")

# Cleanup: release the exportable chunk
drv.release_physical(h3)

# Cleanup: unmap and release the VA
drv.unmap(va, 2 * 1024 * 1024)
drv.unmap(va + 2 * 1024 * 1024, 2 * 1024 * 1024)
drv.release_physical(h1)
drv.release_physical(h2)
drv.free_va(va, va_size)
print(f"[OK] Cleanup complete")

print("\n=== All CudaDriverLibrary tests passed! ===")
