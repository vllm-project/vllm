# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import deque
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
    logger,
)


class PagedAddressManager:
    """
    A lightweight address manager for PagedTensorMemoryAllocator.
    Provides get_free_size() and get_heap_size() by reading the
    paged allocator's state.
    """

    def __init__(self, paged_allocator: "PagedTensorMemoryAllocator") -> None:
        self._allocator = paged_allocator

    def get_heap_size(self) -> int:
        """Get the total size of the paged address space in bytes."""
        return self._allocator.buffer_size

    def get_free_size(self) -> int:
        """Get the total free size in bytes."""
        return len(self._allocator.free_blocks) * self._allocator.align_bytes


class PagedTensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a paged memory allocator.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> None:
        self.buffer = tensor.view(torch.uint8).flatten()
        self.buffer_size = self.buffer.numel() * self.buffer.element_size()
        self.buffer_ptr = self.buffer.data_ptr()

        self.shapes = shapes
        self.dtypes = dtypes
        self.fmt = fmt

        # full chunk size bytes
        self.align_bytes = get_size_bytes(shapes, dtypes)

        assert self.buffer_size % self.align_bytes == 0, (
            f"Buffer size {self.buffer_size} must be a"
            f" multiple of align bytes {self.align_bytes}"
            " in paged memory allocator."
        )

        self.paged_buffers = torch.split(self.buffer, self.align_bytes, dim=0)

        # NOTE: deque is used since thread-safety is not a concern here as
        # is implemented in C under the hood (in CPython), and operations
        # on deque are atomic.
        self.free_blocks: deque[TensorMemoryObj] = deque()

        for idx, buf in enumerate(self.paged_buffers):
            # NOTE: idx is the paged index
            # NOTE: the last unfull chunk's shape needs to be
            # adjusted during allocation.
            metadata = MemoryObjMetadata(
                self.shapes[0],
                self.dtypes[0],
                idx,
                self.align_bytes,  # 1 page
                1,  # ref_count=1
                0,  # pin_count=0
                self.fmt,
                shapes=self.shapes,
                dtypes=self.dtypes,
            )
            mem_obj = TensorMemoryObj(
                raw_data=buf,
                metadata=metadata,
                parent_allocator=self,
            )
            self.free_blocks.append(mem_obj)

        # Address manager for memory usage tracking
        self.address_manager = PagedAddressManager(self)

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        logger.info(
            "Paged tensor memory allocator initialized, "
            "shapes: %s, dtypes: %s, align bytes: %s",
            self.shapes,
            self.dtypes,
            self.align_bytes,
        )

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[TensorMemoryObj]:
        """Allocate one page-backed memory object.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            A page-backed tensor memory object, or ``None`` if no page is free.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        try:
            free_block = self.free_blocks.popleft()
        except IndexError:
            logger.debug(
                f"Failed to allocate memory for "
                f"tensor({shapes}, {dtypes}) because "
                "no free blocks is available"
            )
            return None

        # TODO (Jiayi): This is a bit redundant.
        free_block.meta.shape = shapes[0]
        free_block.meta.dtype = dtypes[0]
        free_block.meta.shapes = shapes
        free_block.meta.dtypes = dtypes
        free_block.meta.fmt = fmt
        free_block.meta.ref_count = 1
        # Reset any narrowed-size override left over from the previous
        # owner of this block, so get_size() returns the layout-derived
        # size for the fresh allocation.
        free_block._used_size_override = None

        if shapes != self.shapes:
            size_in_bytes = get_size_bytes(shapes, dtypes)
            free_block.raw_data = free_block.raw_data[:size_in_bytes]

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.num_active_allocations += 1
        self.total_allocated_size += self.align_bytes
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return free_block

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """
        Batched allocate tensor memory objs with pre-defined equal sizes.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        allocated_blocks: list[TensorMemoryObj] = []
        for i in range(batch_size):
            try:
                free_block = self.free_blocks.popleft()
            except IndexError:
                logger.debug(
                    f"Failed to allocate memory for "
                    f"tensor({shapes}, {dtypes}) because "
                    "no free blocks is available"
                )
                self.batched_free(allocated_blocks, update_stats=False)
                return None

            # FIXME: think about whether parent_allocator
            # should be updated here.
            free_block.meta.shape = shapes[0]
            free_block.meta.dtype = dtypes[0]
            free_block.meta.shapes = shapes
            free_block.meta.dtypes = dtypes
            free_block.meta.fmt = fmt
            free_block.meta.ref_count = 1
            # Reset narrowed-size override (see notes in ``allocate``).
            free_block._used_size_override = None

            if shapes != self.shapes:
                size_in_bytes = get_size_bytes(shapes, dtypes)
                free_block.raw_data = free_block.raw_data[:size_in_bytes]

            allocated_blocks.append(free_block)

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.num_active_allocations += batch_size
        self.total_allocated_size = self.num_active_allocations * self.align_bytes
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return allocated_blocks

    @_lmcache_nvtx_annotate
    def free(
        self, memory_obj: TensorMemoryObj, allocator_type: Optional[str] = None
    ) -> None:
        """Free one page-backed memory object.

        Args:
            memory_obj: Memory object to return to the free-page pool.
            allocator_type: Optional allocator type string.
        """
        if not memory_obj.is_valid():
            return
        if memory_obj.meta.shapes != self.shapes:
            page_idx = memory_obj.meta.address
            memory_obj.raw_data = self.paged_buffers[page_idx]

        self.free_blocks.append(memory_obj)

        # memory_obj.invalidate()

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.total_allocated_size -= self.align_bytes
        self.num_active_allocations -= 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[TensorMemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free multiple page-backed memory objects.

        Unlike `batched_allocate`, this function does not
        assume that the memory objs are equal-sized.

        Args:
            memory_objs: Memory objects to return to the free-page pool.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        if not memory_objs:
            return

        for memory_obj in memory_objs:
            if not memory_obj.is_valid():
                logger.warning("Trying to free an invalidated MemoryObj")
                continue
            # memory_obj.invalidate()
            if memory_obj.meta.shapes != self.shapes:
                page_idx = memory_obj.meta.address
                memory_obj.raw_data = self.paged_buffers[page_idx]

            self.free_blocks.append(memory_obj)

        if update_stats:
            num_freed_blocks = len(memory_objs)
            # TODO (Jiayi): need a flag to drop these debug ops
            # NOTE (Jiayi): the following code is not thread-safe but
            # is tolerable as this is only used for debugging purposes.
            # Update debug status
            self.total_allocated_size -= self.align_bytes * num_freed_blocks
            self.num_active_allocations -= num_freed_blocks
            self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
            self.stats_monitor.update_active_memory_objs_count(
                self.num_active_allocations
            )

    def memcheck(self) -> bool:
        """Check allocator consistency for debugging.

        Returns:
            True if allocator accounting is internally consistent, otherwise
            False.
        """

        logger.info("Checking memory allocator consistency")
        logger.info(" - Total active allocations: %d", self.num_active_allocations)
        logger.info(
            " - Total allocated size: %f MB", self.total_allocated_size / 1048576
        )

        # Check the real total free size
        total_free_size = len(self.free_blocks) * self.align_bytes
        logger.info(" - Total free size: %f MB", total_free_size / 1048576)

        # Check if the numbers are consistent
        if total_free_size + self.total_allocated_size != self.buffer.numel():
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            return False

        return True

    def __str__(self) -> str:
        return "PagedTensorMemoryAllocator"

    def get_paged_buffers(self) -> tuple[torch.Tensor, ...]:
        """
        Get the paged buffers for fixed buffer registration.

        Returns:
            Tuple of paged buffer tensors that can be registered with io_uring
            for true zero copy operations.
        """
        return self.paged_buffers

    def __del__(self) -> None:
        # FIXME: NIXL-related memory leak should be handled somewhere (else).
        del self.buffer
