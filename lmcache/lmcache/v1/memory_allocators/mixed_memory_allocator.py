# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from typing import List, Optional, Union
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
)
import lmcache.v1.memory_management as memory_management


class MixedMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates (1) memory in the pre-allocated pinned memory.
              (2) byte_array buffer memory.
    """

    def __init__(
        self, size: int, use_paging: bool = False, use_hugepages: bool = False, **kwargs
    ) -> None:
        """
        :param int size: The size of the pinned memory in bytes.
        :param bool use_hugepages: Whether to use hugepages.
        """

        self.numa_mapping = kwargs.get("numa_mapping", None)
        self.use_hugepages = use_hugepages
        self.align_bytes = kwargs.get("align_bytes", AddressManager.ALIGN_BYTES)
        if self.align_bytes <= 0 or self.align_bytes & (self.align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        # Extract shm_name from config.extra_config if available
        config = kwargs.get("config", None)
        if config is not None:
            self.shm_name: Optional[str] = config.get_extra_config_value(
                "shm_name", None
            )
        else:
            self.shm_name = kwargs.get("shm_name", None)

        self.size = size

        self.buffer = memory_management._allocate_cpu_memory(
            size, self.numa_mapping, self.shm_name, use_hugepages=use_hugepages
        )

        self._unregistered = False

        self.pin_allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.pin_allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.pin_allocator = TensorMemoryAllocator(
                self.buffer, align_bytes=self.align_bytes
            )

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

        self.buffer_allocator = BufferAllocator("cpu")

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one object from the mixed pinned/buffer allocator.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format to allocate.
            allocator_type: Optional allocator type string.

        Returns:
            A memory object, or ``None`` if the selected allocator is full.

        Raises:
            ValueError: If ``fmt`` is unsupported.
        """
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shapes, dtypes, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
            MemoryFormat.HS_TD,
        ]:
            with self.host_mem_lock:
                obj = self.pin_allocator.allocate(shapes, dtypes, fmt, str(self))
                if isinstance(obj, TensorMemoryObj):
                    obj.parent_allocator = self
                return obj
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate multiple objects from the mixed pinned/buffer allocator.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format to allocate.
            allocator_type: Optional allocator type string.

        Returns:
            Memory objects, or ``None`` if the selected allocator is full.

        Raises:
            ValueError: If ``fmt`` is unsupported.
        """
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
            MemoryFormat.HS_TD,
        ]:
            with self.host_mem_lock:
                objs = self.pin_allocator.batched_allocate(
                    shapes, dtypes, batch_size, fmt, str(self)
                )
                if objs is not None:
                    for obj in objs:
                        if isinstance(obj, TensorMemoryObj):
                            obj.parent_allocator = self
                return objs
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """Free one mixed-format memory object.

        Args:
            memory_obj: Memory object to release.
            allocator_type: Optional allocator type string.

        Raises:
            ValueError: If the object's memory format is unsupported.
        """
        fmt = memory_obj.meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
            MemoryFormat.HS_TD,
        ]:
            with self.host_mem_lock:
                self.pin_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free multiple mixed-format memory objects.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.

        Raises:
            ValueError: If the objects' memory format is unsupported.
        """
        if not memory_objs:
            return

        # NOTE: fmts of all memory_objs should be the same
        fmt = memory_objs[0].meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.batched_free(memory_objs)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
            MemoryFormat.HS_TD,
        ]:
            with self.host_mem_lock:
                self.pin_allocator.batched_free(memory_objs)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def memcheck(self) -> bool:
        """Return whether allocator state is consistent."""
        with self.host_mem_lock:
            return self.pin_allocator.memcheck()

    def close(self) -> None:
        """Release the owned pinned CPU arena."""
        if not self._unregistered:
            if torch_dev.is_available():
                torch_dev.synchronize()
            if self.buffer.numel() == 0:
                return
            memory_management._free_cpu_memory(
                self.buffer,
                self.size,
                self.numa_mapping,
                self.shm_name,
                use_hugepages=self.use_hugepages,
            )
            self._unregistered = True

    def get_paged_buffers(self) -> Optional[tuple[torch.Tensor, ...]]:
        """
        Get the paged buffers for fixed buffer registration.

        Returns:
            Tuple of paged buffer tensors if using paged allocator, None otherwise.
            These buffers can be registered with io_uring for true zero copy operations.
        """
        if isinstance(self.pin_allocator, PagedTensorMemoryAllocator):
            return self.pin_allocator.get_paged_buffers()
        return None

    def __str__(self) -> str:
        return "MixedMemoryAllocator"
