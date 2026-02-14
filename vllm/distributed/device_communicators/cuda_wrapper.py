# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is a pure Python wrapper for the cudart and cuda driver libraries.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Any, Optional

# this line makes it possible to directly load `libcudart.so` using `ctypes`
import torch  # noqa

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# === export types and functions from cudart to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """ # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), \
        f"Unexpected filename: {filename} for library {lib_name}"
    return path


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t 	cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),

        # const char* 	cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),

        # ​cudaError_t 	cudaMalloc ( void** devPtr, size_t size )
        Function("cudaMalloc", cudaError_t,
                 [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        # ​cudaError_t 	cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function("cudaMemset", cudaError_t,
                 [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function("cudaMemcpy", cudaError_t, [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind
        ]),

        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function("cudaIpcGetMemHandle", cudaError_t,
                 [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p]),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function("cudaIpcOpenMemHandle", cudaError_t, [
            ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint
        ]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            if so_file is None:
                so_file = envs.VLLM_CUDART_SO_PATH  # fallback to env var
            assert so_file is not None, \
                (
                    "libcudart is not loaded in the current process, "
                    "try setting VLLM_CUDART_SO_PATH"
                )
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int,
                   count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p,
                   count: int) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self,
                            devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(self.funcs["cudaIpcGetMemHandle"](
            ctypes.byref(handle), devPtr))
        return handle

    def cudaIpcOpenMemHandle(self,
                             handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaIpcOpenMemHandle"](
            ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess))
        return devPtr


# === CUDA Driver API types and structures for VMM ===
# For the original driver API definition, please check
# https://docs.nvidia.com/cuda/cuda-driver-api/index.html

CUresult = ctypes.c_uint
CUdeviceptr = ctypes.c_ulonglong
CUmemGenericAllocationHandle = ctypes.c_ulonglong

# Enum constants
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1


class CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),   # CUmemLocationType
        ("id", ctypes.c_int),
    ]


class CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),                    # CUmemAllocationType
        ("requestedHandleTypes", ctypes.c_uint),    # CUmemAllocationHandleType
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", CUmemAllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_uint),  # CUmemAccessFlags
    ]


class CudaDriverLibrary:
    """Pure Python ctypes wrapper for the CUDA driver API (libcuda.so).
    Provides VMM (Virtual Memory Management) functions for allocating
    physical memory separately from virtual address space, enabling
    cross-process memory sharing at sub-tensor granularity.
    """

    exported_functions = [
        # CUresult cuGetErrorString(CUresult error, const char** pStr)
        Function("cuGetErrorString", CUresult,
                 [CUresult, ctypes.POINTER(ctypes.c_char_p)]),

        # CUresult cuCtxGetDevice(CUdevice* device)
        # CUdevice is just an int
        Function("cuCtxGetDevice", CUresult,
                 [ctypes.POINTER(ctypes.c_int)]),

        # CUresult cuMemGetAllocationGranularity(
        #     size_t* granularity, const CUmemAllocationProp* prop,
        #     CUmemAllocationGranularity_flags option)
        Function("cuMemGetAllocationGranularity", CUresult,
                 [ctypes.POINTER(ctypes.c_size_t),
                  ctypes.POINTER(CUmemAllocationProp),
                  ctypes.c_uint]),

        # CUresult cuMemAddressReserve(
        #     CUdeviceptr* ptr, size_t size, size_t alignment,
        #     CUdeviceptr addr, unsigned long long flags)
        Function("cuMemAddressReserve", CUresult,
                 [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t,
                  ctypes.c_size_t, CUdeviceptr, ctypes.c_ulonglong]),

        # CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
        Function("cuMemAddressFree", CUresult,
                 [CUdeviceptr, ctypes.c_size_t]),

        # CUresult cuMemCreate(
        #     CUmemGenericAllocationHandle* handle, size_t size,
        #     const CUmemAllocationProp* prop, unsigned long long flags)
        Function("cuMemCreate", CUresult,
                 [ctypes.POINTER(CUmemGenericAllocationHandle),
                  ctypes.c_size_t,
                  ctypes.POINTER(CUmemAllocationProp),
                  ctypes.c_ulonglong]),

        # CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
        Function("cuMemRelease", CUresult,
                 [CUmemGenericAllocationHandle]),

        # CUresult cuMemMap(
        #     CUdeviceptr ptr, size_t size, size_t offset,
        #     CUmemGenericAllocationHandle handle, unsigned long long flags)
        Function("cuMemMap", CUresult,
                 [CUdeviceptr, ctypes.c_size_t, ctypes.c_size_t,
                  CUmemGenericAllocationHandle, ctypes.c_ulonglong]),

        # CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
        Function("cuMemUnmap", CUresult,
                 [CUdeviceptr, ctypes.c_size_t]),

        # CUresult cuMemSetAccess(
        #     CUdeviceptr ptr, size_t size,
        #     const CUmemAccessDesc* desc, size_t count)
        Function("cuMemSetAccess", CUresult,
                 [CUdeviceptr, ctypes.c_size_t,
                  ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t]),

        # CUresult cuMemExportToShareableHandle(
        #     void* shareableHandle, CUmemGenericAllocationHandle handle,
        #     CUmemAllocationHandleType handleType, unsigned long long flags)
        Function("cuMemExportToShareableHandle", CUresult,
                 [ctypes.c_void_p, CUmemGenericAllocationHandle,
                  ctypes.c_uint, ctypes.c_ulonglong]),

        # CUresult cuMemImportFromShareableHandle(
        #     CUmemGenericAllocationHandle* handle, void* osHandle,
        #     CUmemAllocationHandleType handleType)
        Function("cuMemImportFromShareableHandle", CUresult,
                 [ctypes.POINTER(CUmemGenericAllocationHandle),
                  ctypes.c_void_p, ctypes.c_uint]),
    ]

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _find_libcuda() -> Optional[str]:
        """Find libcuda.so (driver API), not libcudart.so (runtime API)."""
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcuda.so" in line and "libcudart" not in line:
                    start = line.index("/")
                    return line[start:].strip()
        return None

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = self._find_libcuda()
            if so_file is None:
                # Try common paths
                for path in ["/usr/lib/x86_64-linux-gnu/libcuda.so",
                             "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                             "/usr/lib64/libcuda.so",
                             "/usr/local/cuda/lib64/stubs/libcuda.so"]:
                    if os.path.exists(path):
                        so_file = path
                        break
            assert so_file is not None, \
                "libcuda.so not found. Ensure NVIDIA driver is installed."
        if so_file not in CudaDriverLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaDriverLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaDriverLibrary.path_to_library_cache[so_file]

        if so_file not in CudaDriverLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaDriverLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaDriverLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaDriverLibrary.path_to_dict_mapping[so_file]

    def CUDRV_CHECK(self, result: CUresult) -> None:
        if result != 0:
            error_str = ctypes.c_char_p()
            self.funcs["cuGetErrorString"](result, ctypes.byref(error_str))
            msg = error_str.value.decode("utf-8") if error_str.value else \
                f"Unknown error {result}"
            raise RuntimeError(f"CUDA Driver error: {msg}")

    def get_physical_device(self) -> int:
        """Get the physical device ordinal from the current CUDA context.
        The CUDA driver API uses physical ordinals (unaffected by
        CUDA_VISIBLE_DEVICES), so this must be used for location.id.
        """
        dev = ctypes.c_int()
        self.CUDRV_CHECK(self.funcs["cuCtxGetDevice"](ctypes.byref(dev)))
        return dev.value

    def _make_prop(self, device: int, exportable: bool = False
                   ) -> CUmemAllocationProp:
        """Build a CUmemAllocationProp for pinned device memory.
        device should be the physical device ordinal (from get_physical_device).
        """
        prop = CUmemAllocationProp()
        ctypes.memset(ctypes.byref(prop), 0, ctypes.sizeof(prop))
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        if exportable:
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        return prop

    def get_allocation_granularity(self, device: int) -> int:
        """Returns minimum physical allocation alignment in bytes."""
        prop = self._make_prop(device)
        granularity = ctypes.c_size_t()
        self.CUDRV_CHECK(self.funcs["cuMemGetAllocationGranularity"](
            ctypes.byref(granularity), ctypes.byref(prop),
            CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        return granularity.value

    def reserve_va(self, size: int) -> int:
        """Reserve virtual address range. Returns VA base as int."""
        ptr = CUdeviceptr()
        self.CUDRV_CHECK(self.funcs["cuMemAddressReserve"](
            ctypes.byref(ptr), size, 0, CUdeviceptr(0), 0))
        return ptr.value

    def free_va(self, va: int, size: int) -> None:
        """Free a previously reserved VA range."""
        self.CUDRV_CHECK(self.funcs["cuMemAddressFree"](
            CUdeviceptr(va), size))

    def create_physical(self, device: int, size: int,
                        exportable: bool = False) -> int:
        """Allocate a physical memory chunk. Returns handle as int."""
        prop = self._make_prop(device, exportable=exportable)
        handle = CUmemGenericAllocationHandle()
        self.CUDRV_CHECK(self.funcs["cuMemCreate"](
            ctypes.byref(handle), size, ctypes.byref(prop), 0))
        return handle.value

    def release_physical(self, handle: int) -> None:
        """Release a physical memory allocation."""
        self.CUDRV_CHECK(self.funcs["cuMemRelease"](
            CUmemGenericAllocationHandle(handle)))

    def map_physical(self, va: int, offset: int, size: int,
                     handle: int) -> None:
        """Map a physical chunk at va+offset."""
        self.CUDRV_CHECK(self.funcs["cuMemMap"](
            CUdeviceptr(va + offset), size, 0,
            CUmemGenericAllocationHandle(handle), 0))

    def unmap(self, va: int, size: int) -> None:
        """Unmap a VA range."""
        self.CUDRV_CHECK(self.funcs["cuMemUnmap"](CUdeviceptr(va), size))

    def set_access(self, device: int, va: int, size: int) -> None:
        """Set read/write access on a VA range."""
        desc = CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.CUDRV_CHECK(self.funcs["cuMemSetAccess"](
            CUdeviceptr(va), size, ctypes.byref(desc), 1))

    def export_to_fd(self, handle: int) -> int:
        """Export a physical handle as a POSIX file descriptor."""
        fd = ctypes.c_int()
        self.CUDRV_CHECK(self.funcs["cuMemExportToShareableHandle"](
            ctypes.byref(fd), CUmemGenericAllocationHandle(handle),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0))
        return fd.value

    def import_from_fd(self, fd: int) -> int:
        """Import a physical handle from a POSIX file descriptor.
        The fd must be valid in this process (e.g., opened via
        /proc/<pid>/fd/<fd> or dup'd via Unix socket).
        Returns the CUmemGenericAllocationHandle as int.

        NOTE: For POSIX fd, the CUDA API takes the fd value cast to void*
        (not a pointer to the fd). This matches NVIDIA sample code:
            cuMemImportFromShareableHandle(&h, (void*)(uintptr_t)fd, ...)
        """
        handle = CUmemGenericAllocationHandle()
        self.CUDRV_CHECK(self.funcs["cuMemImportFromShareableHandle"](
            ctypes.byref(handle), ctypes.c_void_p(fd),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))
        return handle.value
