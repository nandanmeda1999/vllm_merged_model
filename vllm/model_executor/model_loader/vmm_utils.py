"""VMM (Virtual Memory Management) utilities for partial weight sharing.

Allocates a contiguous virtual address range backed by N separate
physical memory chunks. Each chunk can be independently owned (freshly
allocated) or imported from another process for zero-copy sharing.
"""

import array
import os
import socket
from typing import Optional

import cupy as cp
import torch

from vllm.distributed.device_communicators.cuda_wrapper import CudaDriverLibrary
from vllm.logger import init_logger

logger = init_logger(__name__)


def _align_up(size: int, granularity: int) -> int:
    """Round size up to the nearest multiple of granularity."""
    return ((size + granularity - 1) // granularity) * granularity


def _torch_to_cupy_dtype(dtype: torch.dtype):
    """Map torch dtype to cupy dtype. bfloat16 uses uint16 reinterpretation."""
    mapping = {
        torch.float16: cp.float16,
        torch.bfloat16: cp.uint16,   # reinterpret later
        torch.float32: cp.float32,
        torch.int8: cp.int8,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for VMM: {dtype}")
    return mapping[dtype]


class VMMCompositeWeight:
    """A contiguous tensor backed by N separate VMM physical chunks.

    Each physical chunk can be independently:
    - Owned: freshly allocated on this device
    - Imported: mapped from another process's exported handle

    The full VA range appears as a single contiguous tensor for GEMM kernels.

    Args:
        device: CUDA device index
        sub_component_sizes_bytes: list of sizes in BYTES for each sub-component
            e.g. [q_bytes, k_bytes, v_bytes]
        total_elements: total number of elements in the merged tensor
            e.g. 12288 * 4096 for qkv_proj
        shape: shape of the merged tensor (e.g. [12288, 4096])
        dtype: torch dtype (e.g. torch.bfloat16)
        exportable: if True, physical chunks can be exported for cross-process
            sharing (sets CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    """

    def __init__(
        self,
        device: int,
        sub_component_sizes_bytes: list[int],
        shape: list[int],
        dtype: torch.dtype,
        exportable: bool = True,
        skip_indices: Optional[set[int]] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.shape = shape
        self.drv = CudaDriverLibrary()
        self._skip_indices = skip_indices or set()

        # Get the physical device ordinal (driver API ignores CUDA_VISIBLE_DEVICES)
        self.physical_device = self.drv.get_physical_device()
        logger.info("VMM using physical device: %d (runtime device: %d)",
                    self.physical_device, device)

        # Query allocation granularity
        self.granularity = self.drv.get_allocation_granularity(self.physical_device)
        logger.info("VMM allocation granularity: %d bytes (%.1f MB)",
                    self.granularity, self.granularity / 1024 / 1024)

        # Align each sub-component size
        self.n_chunks = len(sub_component_sizes_bytes)
        self.raw_sizes = list(sub_component_sizes_bytes)
        self.aligned_sizes = [_align_up(s, self.granularity)
                              for s in sub_component_sizes_bytes]
        self.total_aligned = sum(self.aligned_sizes)

        # Compute cumulative offsets within the VA range
        self.offsets = []
        offset = 0
        for s in self.aligned_sizes:
            self.offsets.append(offset)
            offset += s

        # Reserve contiguous VA
        self.va_base = self.drv.reserve_va(self.total_aligned)
        logger.info("VMM reserved VA: 0x%x, total %d bytes (%d chunks, %d holes)",
                    self.va_base, self.total_aligned, self.n_chunks,
                    len(self._skip_indices))

        # Allocate physical chunks and map them (skip holes)
        self.handles: list[Optional[int]] = [None] * self.n_chunks
        self.owned: list[bool] = [False] * self.n_chunks
        for i in range(self.n_chunks):
            if i in self._skip_indices:
                continue  # hole — VA reserved but no physical allocation
            h = self.drv.create_physical(self.physical_device,
                                         self.aligned_sizes[i],
                                         exportable=exportable)
            self.handles[i] = h
            self.owned[i] = True
            self.drv.map_physical(self.va_base, self.offsets[i],
                                  self.aligned_sizes[i], h)

        # If no holes, finalize immediately (set access + wrap tensor)
        self._tensor: Optional[torch.Tensor] = None
        if not self._skip_indices:
            self.finalize()

    def _wrap_as_tensor(self) -> torch.Tensor:
        """Wrap the contiguous VA as a PyTorch tensor via CuPy."""
        cp_dtype = _torch_to_cupy_dtype(self.dtype)
        total_elements = 1
        for s in self.shape:
            total_elements *= s
        nbytes = cp.dtype(cp_dtype).itemsize * total_elements

        mem = cp.cuda.UnownedMemory(self.va_base, nbytes, owner=self)
        ptr = cp.cuda.MemoryPointer(mem, 0)
        cupy_arr = cp.ndarray(self.shape, dtype=cp_dtype, memptr=ptr)

        tensor = torch.as_tensor(cupy_arr, device=f"cuda:{self.device}")
        if self.dtype == torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        return tensor

    def finalize(self) -> None:
        """Set access permissions and wrap VA as tensor.

        Must be called after all holes are filled via import_chunk_from_fd().
        Called automatically in __init__ if there are no skip_indices.
        """
        assert all(h is not None for h in self.handles), \
            "Cannot finalize: some chunks are still holes (not imported)"
        self.drv.set_access(self.physical_device, self.va_base,
                            self.total_aligned)
        self._tensor = self._wrap_as_tensor()

    @property
    def tensor(self) -> torch.Tensor:
        """The contiguous tensor wrapping the full VA range."""
        assert self._tensor is not None, \
            "Tensor not ready — call finalize() after importing all holes"
        return self._tensor

    def export_chunk(self, index: int) -> tuple[int, int]:
        """Export physical chunk at index for cross-process sharing.

        Returns:
            (pid, fd) tuple. The fd is a POSIX file descriptor valid in
            this process. Other processes can import it via
            /proc/<pid>/fd/<fd>.
        """
        assert 0 <= index < self.n_chunks
        assert self.handles[index] is not None
        fd = self.drv.export_to_fd(self.handles[index])
        return (os.getpid(), fd)

    def import_chunk_from_fd(self, index: int, fd: int) -> None:
        """Import a physical chunk from a file descriptor into a hole or
        replace an existing chunk.

        The fd must be valid in THIS process (e.g., opened via
        /proc/<pid>/fd/<fd> or received via Unix socket SCM_RIGHTS).

        Call finalize() after all imports are done.

        Args:
            index: which sub-component to import (0=Q, 1=K, 2=V)
            fd: file descriptor valid in THIS process
        """
        assert 0 <= index < self.n_chunks

        # Unmap and release any existing chunk at this offset
        if self.handles[index] is not None:
            self.drv.unmap(self.va_base + self.offsets[index],
                           self.aligned_sizes[index])
            if self.owned[index]:
                self.drv.release_physical(self.handles[index])

        # Import the physical handle from the local fd
        imported_handle = self.drv.import_from_fd(fd)

        # Map at the same offset
        self.drv.map_physical(self.va_base, self.offsets[index],
                              self.aligned_sizes[index], imported_handle)

        self.handles[index] = imported_handle
        self.owned[index] = False

        logger.info("VMM imported chunk %d from fd=%d at offset=%d",
                    index, fd, self.offsets[index])


def send_fd(sock: socket.socket, fd: int) -> None:
    """Send a file descriptor over a Unix domain socket using SCM_RIGHTS."""
    fds = array.array("i", [fd])
    sock.sendmsg(
        [b"\x00"],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)]
    )


def recv_fd(sock: socket.socket) -> int:
    """Receive a file descriptor from a Unix domain socket using SCM_RIGHTS."""
    fds = array.array("i")
    _msg, ancdata, _flags, _addr = sock.recvmsg(
        1,
        socket.CMSG_SPACE(4)  # space for one int (fd)
    )
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            return fds[0]
    raise RuntimeError("No fd received via SCM_RIGHTS")


def start_fd_server(socket_path: str, fd_map: dict[str, int]) -> None:
    """Start a background thread serving DMA-BUF fds via Unix socket.

    Args:
        socket_path: path for the Unix domain socket
        fd_map: maps "layer.module.shard" keys to file descriptors
            e.g., {"2.self_attn.qkv_proj.k_proj": 66}
    """
    import json
    import threading

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(8)

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                break
            try:
                # Client sends a JSON request with the key
                data = conn.recv(4096).decode()
                request = json.loads(data)
                key = request["key"]
                if key in fd_map:
                    send_fd(conn, fd_map[key])
                else:
                    conn.sendall(b"ERROR: key not found")
            except Exception as e:
                logger.error("fd server error: %s", e)
            finally:
                conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    logger.info("VMM fd server started on %s with %d fds", socket_path, len(fd_map))


def request_fd(socket_path: str, key: str) -> int:
    """Request a DMA-BUF fd from the owner's fd server.

    Args:
        socket_path: path to the owner's Unix domain socket
        key: "layer.module.shard" key matching the fd_map on the server

    Returns:
        A file descriptor valid in this process.
    """
    import json

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    client.sendall(json.dumps({"key": key}).encode())
    fd = recv_fd(client)
    client.close()
    return fd
