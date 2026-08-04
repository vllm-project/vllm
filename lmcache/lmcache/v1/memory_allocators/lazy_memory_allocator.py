# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Union
import ctypes
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform import current_device_spec
from lmcache.v1.system_detection import NUMAMapping
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Helper functions
def get_numa_id(numa_mapping: NUMAMapping) -> int:
    """
    Get the NUMA ID for the current GPU

    Args:
        numa_mapping (NUMAMapping): The NUMA mapping object.

    Returns:
        int: The NUMA ID for the current GPU.

    Raises:
        KeyError: If GPU id is not detected in the numa mapping.
    """
    gpu_id = torch_dev.current_device() if torch_dev.is_available() else 0
    return numa_mapping.gpu_to_numa_mapping[gpu_id]


def align_to(size: int, align_size: int) -> int:
    """
    Align the given size to the nearest multiple of align_size.

    Args:
        size (int): The size to align.
        align_size (int): The alignment size, MUST BE a power of two.

    Returns:
        int: The aligned size.
    """
    return (size + align_size - 1) & (~(align_size - 1))


# Main class
class LazyMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates CPU (numa) pinned memory with a initial size and expand
    the size to the required size in the background.

    Background expansion logic:
    - After registering X GB memory, we call sbrk and updates _curr_size
    - Once everything is registered, the background thread stops
    """

    PIN_CHUNK_SIZE = 1 << 26  # 64 MB pin chunk
    COMMIT_SIZE = 1 << 30  # Do a commit every 1 GB
    LOG_INTERVAL = 10 << 30  # Log expansion progress every 10 GB

    def __init__(
        self,
        init_size: int,
        final_size: int,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        numa_mapping: NUMAMapping | None = None,
    ) -> None:
        """
        Args:
            init_size (int): Initial size of the memory allocation in bytes.
            final_size (int): Final size of the memory allocation in bytes.
            align_bytes (int, optional): Alignment in for the underlying allocations
        """
        # Whether using NUMA allocation
        self._use_numa = numa_mapping is not None
        # Currently pinned size, only accessed by the expansion thread
        self._curr_size = align_to(init_size, self.PIN_CHUNK_SIZE)
        # Final size of the allocation, only accessed by the expansion thread
        self._final_size = align_to(final_size, self.PIN_CHUNK_SIZE)
        # Underlying buffer for the memory allocation
        self._buffer: torch.Tensor
        if not current_device_spec.is_pin_supported:
            raise RuntimeError(
                f"Backend '{torch_device_type}' does not support memory "
                "pinning. LazyMemoryAllocator requires pinned memory."
            )

        # List of (ptr, size) for pinned memory chunks
        self._pin_record: list[tuple[int, int]] = []

        # Detect numa mapping
        if numa_mapping is not None:
            numa_id = get_numa_id(numa_mapping)
            ptr = lmc_ops.alloc_numa_ptr(self._final_size, numa_id)
            arr_type = ctypes.c_uint8 * self._final_size
            buf = arr_type.from_address(ptr)
            self._buffer = torch.frombuffer(buf, dtype=torch.uint8)
        else:
            self._buffer = torch.empty(
                self._final_size, dtype=torch.uint8, device="cpu", pin_memory=False
            )

        # Pin the first `curr_size` bytes (aligned to the internal chunk size)
        self._pin_memory_chunk(0, self._curr_size)

        # Create the tensor memory allocator
        self._allocator = TensorMemoryAllocator(
            tensor=self._buffer,
            align_bytes=align_bytes,
            init_address_space=self._curr_size,
        )

        # Get the address manager
        # NOTE(ApostaC): this assumes the tensor memory allocator owns the address
        # manager, which creates extra coupling in the code.
        # NOTE(ApostaC): this also assumes that the behavior of the allocation is
        # completely determined by the address manager.
        self._address_manager = self._allocator.address_manager

        # Launch the background expansion thread
        self._stop_expand = threading.Event()
        self._expand_thread = threading.Thread(
            target=self._expand_worker, daemon=True, name="lazy-mem-expand-thread"
        )
        self._expand_thread.start()

    # Public methods
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one object from the lazily pinned memory pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            A memory object, or ``None`` if the committed address space is full.
        """
        obj = self._allocator.allocate(shapes, dtypes, fmt, allocator_type)
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        if obj is not None:
            obj.parent_allocator = self
        return obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate a batch of objects from the lazily pinned memory pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate for each object.
            dtypes: Logical tensor dtype or dtypes to allocate for each object.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            Memory objects for the batch, or ``None`` if allocation fails.
        """
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        ret = self._allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )

        if ret is None:
            return ret

        for obj in ret:
            obj.parent_allocator = self
        return ret

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Free one memory object back to the lazy allocator.

        Args:
            memory_obj: The memory object to free.
            allocator_type: Optional allocator type string.
        """
        self._allocator.free(memory_obj, allocator_type)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free a batch of memory objects back to the lazy allocator.

        Args:
            memory_objs: Memory objects to free.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        self._allocator.batched_free(memory_objs, allocator_type, update_stats)

    def close(self) -> None:
        """Stop background expansion and release pinned or NUMA memory."""
        # Stop the background expansion thread
        self._stop_expand.set()
        self._expand_thread.join()

        # Unpin all pinned memory chunks
        for ptr, size in self._pin_record:
            current_device_spec.unpin_memory(ptr)
        self._pin_record.clear()

        # Free the underlying buffer if using NUMA allocation
        if self._use_numa:
            lmc_ops.free_numa_ptr(self._buffer.data_ptr(), self._final_size)

    def memcheck(self) -> bool:
        """Return whether the delegated tensor allocator is consistent."""
        return self._allocator.memcheck()

    def get_underlying_buffer(self) -> torch.Tensor:
        """
        Get the underlying buffer tensor. Will be used by RDMA registrations.
        """
        return self._buffer

    def get_address_manager(self) -> AddressManager:
        """
        Get the address manager used by this allocator.
        """
        return self._address_manager

    # Helper functions
    def _pin_memory_chunk(self, offset: int, size: int) -> None:
        """
        Pin a chunk of memory.

        Args:
            offset (int): Offset in the buffer to pin.
            size (int): Size of the memory chunk in bytes.
        """
        assert offset & (self.PIN_CHUNK_SIZE - 1) == 0, (
            "Offset must be aligned to PIN_CHUNK_SIZE"
        )
        assert size & (self.PIN_CHUNK_SIZE - 1) == 0, (
            "Size must be aligned to PIN_CHUNK_SIZE"
        )
        assert offset + size <= self._final_size, "Pinning exceeds buffer size"

        ptr = self._buffer.data_ptr() + offset
        # Use flag: cudaHostRegisterMapped (0x02)
        if not current_device_spec.pin_memory(ptr, size, 2):
            logger.warning(
                "pin_memory failed for chunk at ptr=%#x size=%d; "
                "DMA performance may be degraded",
                ptr,
                size,
            )
        else:
            self._pin_record.append((ptr, size))

    def _commit_expansion(self, expand_size: int) -> None:
        """
        Call sbrk in the address manager to commit the expansion.
        """
        self._address_manager.sbrk(expand_size)

    def _log_expansion_progress(self, expanded_since_last_log: int) -> None:
        """
        Log the cumulative expansion progress since the last log.
        """
        percent = 100.0 * self._curr_size / self._final_size
        logger.info(
            "LazyMemoryAllocator: Expanded %s MB pinned memory, "
            "now total is %s MB / %s MB (%.1f%%)",
            expanded_since_last_log >> 20,
            self._curr_size >> 20,
            self._final_size >> 20,
            percent,
        )

    def _expand_worker(self) -> None:
        """
        Background worker to expand the pinned memory.
        """
        last_commit_size = self._curr_size
        last_log_size = self._curr_size
        while self._curr_size < self._final_size and not self._stop_expand.is_set():
            # Expand chunk by chunk and commit
            for i in range(self.COMMIT_SIZE // self.PIN_CHUNK_SIZE):
                if self._curr_size >= self._final_size:
                    break
                self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
                self._curr_size += self.PIN_CHUNK_SIZE

            expand_size = self._curr_size - last_commit_size
            self._commit_expansion(expand_size)
            last_commit_size = self._curr_size

            # Log every LOG_INTERVAL bytes, and always on the final commit.
            expanded_since_last_log = self._curr_size - last_log_size
            if (
                expanded_since_last_log >= self.LOG_INTERVAL
                or self._curr_size >= self._final_size
            ):
                self._log_expansion_progress(expanded_since_last_log)
                last_log_size = self._curr_size
