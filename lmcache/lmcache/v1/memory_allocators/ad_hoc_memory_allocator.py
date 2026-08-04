# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


class AdHocMemoryAllocator(MemoryAllocatorInterface):
    """
    AdHocMemoryAllocator is a simple allocator that does not actually
    allocate memory. It is used for testing purposes only.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        :param str device: The device of the ad hoc memory allocator.
        """
        if not torch_dev.is_available():
            self.device = "cpu"
        else:
            self.device = device

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Returns a dummy MemoryObj for testing purposes.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        size = get_size_bytes(shapes, dtypes)

        # Return a dummy object with no actual memory allocation
        return TensorMemoryObj(
            raw_data=torch.empty(
                torch.Size([size]), dtype=torch.uint8, device=self.device
            ),
            metadata=MemoryObjMetadata(
                shape=shapes[0],
                dtype=dtypes[0],
                address=0,
                phy_size=0,
                ref_count=1,
                pin_count=0,
                fmt=fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate a batch of test memory objects.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in each object's metadata.
            allocator_type: Optional allocator type string.

        Raises:
            NotImplementedError: Always raised because ad-hoc batched
                allocation is not supported.
        """
        raise NotImplementedError(
            "Batched allocation is not supported in AdHocMemoryAllocator"
        )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """Release a test memory object.

        Args:
            memory_obj: Memory object to release.
            allocator_type: Optional allocator type string.
        """
        pass

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Release a batch of test memory objects.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        pass

    def ref_count_up(self, memory_obj: MemoryObj) -> None:
        """Increment the reference count for a test memory object.

        Args:
            memory_obj: Memory object whose reference count would be incremented.
        """
        pass

    def ref_count_down(self, memory_obj: MemoryObj) -> None:
        """Decrement the reference count for a test memory object.

        Args:
            memory_obj: Memory object whose reference count would be decremented.
        """
        pass

    def get_ref_count(self, memory_obj: MemoryObj) -> int:
        """Return the reference count for a test memory object.

        Args:
            memory_obj: Memory object to inspect.

        Returns:
            Always returns 0 because this test allocator does not track counts.
        """
        return 0

    def memcheck(self) -> bool:
        """Return whether allocator state is consistent."""
        return True

    def __str__(self) -> str:
        return "AdHocMemoryAllocator"
