"""External KV cache allocator using shared VRAM pool.

This module provides an allocator that connects to an external GPU memory pool
broker (gpu_poold) to allocate KV cache from shared VRAM. This enables multiple
vLLM instances to share GPU memory efficiently.
"""
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

try:
    from vmm_pool_py import VmmPoolClient, tensor_from_device_ptr
    VMM_POOL_AVAILABLE = True
except ImportError:
    VMM_POOL_AVAILABLE = False
    VmmPoolClient = None  # type: ignore
    tensor_from_device_ptr = None  # type: ignore


class ExternalKVSlice:
    """Represents a slice of memory allocated from the external pool.

    Args:
        offset: Offset within the pool in bytes.
        length: Length of the allocation in bytes.
        ptr: Device pointer (CUdeviceptr cast to int).
    """
    __slots__ = ("offset", "length", "ptr")

    def __init__(self, offset: int, length: int, ptr: int):
        self.offset = offset
        self.length = length
        self.ptr = ptr


class ExternalKVAllocator:
    """Allocator that uses external shared VRAM pool for KV cache.

    This allocator connects to a GPU memory pool broker via Unix domain socket
    and allocates KV cache blocks from the shared pool using CUDA VMM.

    Args:
        endpoint: Unix domain socket path to gpu_poold broker
                  (e.g., "/tmp/gpu_pool.sock").
        device_id: CUDA device ID to use.

    Raises:
        RuntimeError: If VMM pool client is not available or connection fails.
    """

    def __init__(self, endpoint: str, device_id: int):
        if not VMM_POOL_AVAILABLE:
            raise RuntimeError(
                "VMM pool client is not available. "
                "Please build vmm_pool_py extension first:\n"
                "  cd vllm/tools/gpu_pool && mkdir -p build && cd build\n"
                "  cmake .. && make -j")

        self.cli = VmmPoolClient(endpoint, device_id)
        self.device = device_id
        # Get pool granularity (page size)
        _, _, gran = self.cli.stats()
        self.page_bytes = int(gran)

    def allocate(self, nbytes: int) -> ExternalKVSlice:
        """Allocate a slice from the external pool.

        Args:
            nbytes: Number of bytes to allocate.

        Returns:
            ExternalKVSlice representing the allocated memory.

        Raises:
            RuntimeError: If allocation fails (pool out of memory).
        """
        off, length = self.cli.allocate(int(nbytes))
        if length == 0 or off == (1 << 64) - 1:
            raise RuntimeError(
                "ExternalKVAllocator: pool out of memory. "
                f"Requested {nbytes} bytes.")
        # Map the allocated slice to get a device pointer
        ptr = int(self.cli.map(off, length))
        return ExternalKVSlice(off, length, ptr)

    def free(self, s: ExternalKVSlice) -> None:
        """Free a previously allocated slice.

        Args:
            s: The slice to free.

        Raises:
            RuntimeError: If free operation fails.
        """
        # Unmap first, then free in the pool
        self.cli.unmap(s.offset, s.length)
        ok = self.cli.free(s.offset, s.length)
        if not ok:
            raise RuntimeError("ExternalKVAllocator: free failed")

    def make_torch_tensor(self, s: ExternalKVSlice,
                          nbytes: int) -> torch.Tensor:
        """Create a PyTorch tensor wrapping the allocated device memory.

        Args:
            s: The allocated slice.
            nbytes: Number of bytes for the tensor.

        Returns:
            PyTorch tensor backed by the device memory.
        """
        if tensor_from_device_ptr is None:
            raise RuntimeError("tensor_from_device_ptr not available")
        return tensor_from_device_ptr(s.ptr, int(nbytes), self.device)

    def get_stats(self) -> tuple[int, int, int]:
        """Get pool usage statistics.

        Returns:
            Tuple of (used_bytes, total_bytes, granularity_bytes).
        """
        return self.cli.stats()
