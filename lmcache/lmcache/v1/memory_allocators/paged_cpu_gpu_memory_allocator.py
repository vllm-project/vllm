# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.system_detection import NUMAMapping
import lmcache.v1.memory_management as memory_management


class PagedCpuGpuMemoryAllocator(MemoryAllocatorInterface):
    """
    Paged Memory Allocator for both CPU and GPU memory.
    This is a paged memory allocator for PD and P2P sharing
    when NIXL is enabled as NIXL relies on the paging abstraction.
    """

    def __init__(self) -> None:
        self.cpu_buffer: torch.Tensor | None = None
        self.cpu_size = 0
        self.cpu_numa_mapping: Optional[NUMAMapping] = None

    def init_gpu_memory_allocator(
        self,
        size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        device: str = torch_device_type,
    ) -> None:
        """Initialize the GPU paged allocator.

        Args:
            size: GPU arena size in bytes.
            shapes: Logical tensor shapes served by each page.
            dtypes: Logical tensor dtypes served by each page.
            fmt: Memory format served by each page.
            device: Torch device string for the GPU arena.
        """
        self.gpu_buffer = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
        )
        self.gpu_allocator = PagedTensorMemoryAllocator(
            self.gpu_buffer,
            shapes,
            dtypes,
            fmt,
        )

    def init_cpu_memory_allocator(
        self,
        size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> None:
        """Initialize the CPU paged allocator.

        Args:
            size: CPU arena size in bytes.
            shapes: Logical tensor shapes served by each page.
            dtypes: Logical tensor dtypes served by each page.
            fmt: Memory format served by each page.
            numa_mapping: Optional NUMA placement for pinned CPU allocation.
        """
        self.cpu_buffer = memory_management._allocate_cpu_memory(size, numa_mapping)
        self.cpu_size = size
        self.cpu_numa_mapping = numa_mapping
        self.cpu_allocator = PagedTensorMemoryAllocator(
            self.cpu_buffer,
            shapes,
            dtypes,
            fmt,
        )
        self.align_bytes = self.cpu_allocator.align_bytes

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "cpu",
    ) -> Optional[MemoryObj]:
        """Allocate from the selected CPU or GPU paged allocator.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: ``"cpu"`` or ``"gpu"``.

        Returns:
            A memory object, or ``None`` when the selected allocator is full.

        Raises:
            ValueError: If ``allocator_type`` is unsupported.
        """
        if allocator_type == "gpu":
            return self.gpu_allocator.allocate(shapes, dtypes, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.allocate(shapes, dtypes, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "gpu",
    ) -> Optional[List[MemoryObj]]:
        """Allocate a batch from the selected CPU or GPU paged allocator.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in each returned object's metadata.
            allocator_type: ``"cpu"`` or ``"gpu"``.

        Returns:
            Memory objects, or ``None`` when the selected allocator is full.

        Raises:
            ValueError: If ``allocator_type`` is unsupported.
        """
        if allocator_type == "gpu":
            return self.gpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def free(
        self, memory_obj: MemoryObj, allocator_type: Optional[str] = "cpu"
    ) -> None:
        """Free one CPU or GPU paged memory object.

        Args:
            memory_obj: Memory object to release.
            allocator_type: ``"cpu"`` or ``"gpu"``.

        Raises:
            ValueError: If ``allocator_type`` is unsupported.
        """
        if allocator_type == "gpu":
            self.gpu_allocator.free(memory_obj)
        elif allocator_type == "cpu":
            self.cpu_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free multiple CPU or GPU paged memory objects.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: ``"cpu"`` or ``"gpu"``.
            update_stats: Whether to update allocator statistics.

        Raises:
            ValueError: If ``allocator_type`` is unsupported.
        """
        if allocator_type == "gpu":
            self.gpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        elif allocator_type == "cpu":
            self.cpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def close(self) -> None:
        """Release the owned pinned CPU arena, if initialized."""
        if self.cpu_buffer is not None and self.cpu_buffer.numel() > 0:
            memory_management._free_cpu_memory(
                self.cpu_buffer,
                self.cpu_size,
                self.cpu_numa_mapping,
            )
            self.cpu_buffer = torch.empty(0, dtype=torch.uint8)
            self.cpu_size = 0
            self.cpu_numa_mapping = None

    def __str__(self) -> str:
        return "PDMemoryAllocator"
