# SPDX-License-Identifier: Apache-2.0
"""GDS slab-file L1 memory manager."""

# Standard
from typing import Optional

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import (
    AddressManager,
    GDSMemoryObject,
    MemoryObj,
    MemoryObjMetadata,
)

logger = init_logger(__name__)


class GDSL1MemoryManager:
    """L1 memory manager for the GDS slab-file tier.

    A peer of
    :class:`~lmcache.v1.distributed.memory_manager.l1_memory_manager.L1MemoryManager`
    (both satisfy
    :class:`~lmcache.v1.distributed.memory_manager.l1_manager_protocol.L1ManagerProtocol`).
    It owns an :class:`AddressManager` over the slab's byte-offset space and
    hands out :class:`GDSMemoryObject` chunks; the actual GPU<->slab DMA is
    performed by the global
    :class:`~lmcache.v1.gpu_connector.gds_context.GDSContext`, reached from the
    ``gpu_ops`` dispatch.

    There is no on-disk index: the slab is created and cleared at startup
    (treated like DRAM), so allocations do not survive a restart.
    """

    def __init__(self, config: GdsL1Config) -> None:
        """Create the manager.

        Args:
            config: The GDS tier config. ``size_in_bytes`` sizes the slab
                address space, and ``align_bytes`` sets the allocation alignment
                (cuFile/O_DIRECT require 4 KiB). The same ``GdsL1Config`` drives
                the :class:`GDSContext` that preallocates the slab file, so the
                address space and the file match by construction. The CPU-tier
                ``memory_config`` is not referenced on the GDS path.
        """
        self._address_manager = AddressManager(config.size_in_bytes, config.align_bytes)

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Reserve ``count`` slab regions for the given layout.

        All-or-nothing: on the first slab OOM, frees what was reserved and
        returns ``(L1Error.OUT_OF_MEMORY, [])``.

        Args:
            layout_desc: Layout descriptor; all ``count`` chunks share its
                shape/dtype, and its byte size sets each chunk's size.
            count: Number of chunks to reserve.

        Returns:
            ``(L1Error.SUCCESS, objects)`` on success, otherwise
            ``(L1Error.OUT_OF_MEMORY, [])``.
        """
        chunk_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        shape = layout_desc.shapes[0]
        dtype = layout_desc.dtypes[0]
        objects: list[MemoryObj] = []
        for _ in range(count):
            try:
                address, allocated = self._address_manager.allocate(chunk_bytes)
            except RuntimeError:
                for obj in objects:
                    self._address_manager.free(
                        obj.metadata.address, obj.get_physical_size()
                    )
                return L1Error.OUT_OF_MEMORY, []
            meta = MemoryObjMetadata(
                shape=shape,
                dtype=dtype,
                address=address,
                phy_size=allocated,
                ref_count=0,
            )
            objects.append(GDSMemoryObject(meta))
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Return the slab regions of the given objects to the address manager.

        Args:
            mem_objs: Objects to free (slab-anchored :class:`GDSMemoryObject`s
                handed out by :meth:`allocate`).

        Returns:
            ``L1Error.SUCCESS``.
        """
        for mo in mem_objs:
            self._address_manager.free(mo.metadata.address, mo.get_physical_size())
        return L1Error.SUCCESS

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the storage medium backing ``memory_obj``.

        Args:
            memory_obj: An object allocated by this manager.

        Returns:
            ``L1BackendType.GDS`` — this tier's only medium is the slab file.
        """
        return L1BackendType.GDS

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` of the slab."""
        free_size = self._address_manager.get_free_size()
        total_size = self._address_manager.get_heap_size()
        return total_size - free_size, total_size

    def get_l1_memory_desc(self) -> Optional[L1MemoryDesc]:
        """Return ``None``: the GDS L1 medium is the slab file, not a buffer.

        The only registerable memory on the GDS path is the GPU staging buffer,
        not an L1 pool, so there is no descriptor to hand to L2 adapters (which
        must be disabled when GDS L1 is enabled).
        """
        return None

    def close(self) -> None:
        """No-op: the GDSContext owning the slab is closed at server shutdown."""
        return

    def memcheck(self) -> bool:
        """For debug purposes; logs allocator state and checks consistency.

        Mirrors ``TensorMemoryAllocator.memcheck`` for the GDS slab address
        space: logs the allocated / free sizes, then verifies the free and
        allocated bytes add up to the slab size and that free blocks are
        coalesced. Returns ``True`` when consistent, ``False`` otherwise.
        """
        clear = True
        logger.info("Checking memory allocator consistency")
        logger.info(
            " - Total allocated size: %s MB",
            self._address_manager.total_allocated_size / 1048576,
        )

        total_free_size = self._address_manager.get_free_size()
        logger.info(" - Total free size: %s MB", total_free_size / 1048576)

        if (
            total_free_size + self._address_manager.total_allocated_size
            != self._address_manager.get_heap_size()
        ):
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        if not self._address_manager.check_consistency():
            logger.error("Memory allocator has non-coalesced blocks")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        return clear
