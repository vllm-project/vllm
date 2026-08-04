# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)


class BufferAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, device: str = "cpu") -> None:
        """
        :param str device: The device of the buffer memory.
        """
        self.device = device

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
        allocator_type: Optional[str] = None,
    ) -> BytesBufferMemoryObj:
        """Allocate one byte-buffer memory object.

        Args:
            shapes: Buffer shape. The first dimension is used as byte length.
            dtypes: Unused dtype argument kept for allocator interface parity.
            fmt: Memory format for the allocation.
            allocator_type: Optional allocator type string.

        Returns:
            A bytes-backed memory object.
        """
        if isinstance(shapes, list):
            n = shapes[0][0]
        else:
            n = shapes[0]
        byte_array = bytearray(n)
        return BytesBufferMemoryObj(byte_array)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
        allocator_type: Optional[str] = None,
    ) -> List[BytesBufferMemoryObj]:
        """Allocate multiple byte-buffer memory objects.

        Args:
            shapes: Buffer shape. The first dimension is used as byte length.
            dtypes: Unused dtype argument kept for allocator interface parity.
            batch_size: Number of buffers to allocate.
            fmt: Memory format for each allocation.
            allocator_type: Optional allocator type string.

        Returns:
            Bytes-backed memory objects.
        """
        if isinstance(shapes, list):
            n = shapes[0][0]
        else:
            n = shapes[0]
        # TODO(Jiayi): Optimize the following loop.
        byte_arrays = [bytearray(n) for _ in range(batch_size)]
        return [BytesBufferMemoryObj(byte_array) for byte_array in byte_arrays]

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """Release a byte-buffer memory object.

        Args:
            memory_obj: Memory object to release.
            allocator_type: Optional allocator type string.
        """
        return

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Release byte-buffer memory objects.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        return

    def __str__(self) -> str:
        return "BufferAllocator"

    def memcheck(self) -> bool:
        """Return whether allocator state is consistent."""
        return True
