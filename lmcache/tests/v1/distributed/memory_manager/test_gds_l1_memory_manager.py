# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``GDSL1MemoryManager``.

These are pure: the manager sits on the in-memory ``AddressManager``, so no
CUDA / cuFile / GDS hardware is required (the actual slab DMA lives in
``GDSContext`` and is covered separately).
"""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
from lmcache.v1.memory_management import GDSMemoryObject

_ALIGN = 4096


def _config(size_bytes: int) -> GdsL1Config:
    return GdsL1Config(file_location="/unused", size_in_bytes=size_bytes)


def _layout(nbytes: int) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([nbytes])], dtypes=[torch.uint8])


class TestAllocate:
    def test_returns_distinct_gds_objects(self):
        mgr = GDSL1MemoryManager(_config(1 << 20))
        err, objs = mgr.allocate(_layout(4096), 3)
        assert err == L1Error.SUCCESS
        assert len(objs) == 3
        assert all(isinstance(o, GDSMemoryObject) for o in objs)
        # Non-overlapping slab regions.
        assert len({o.slab_offset for o in objs}) == 3

    def test_oom_is_all_or_nothing(self):
        # Slab fits exactly two 4 KiB chunks; asking for three must fail and
        # leave nothing reserved.
        mgr = GDSL1MemoryManager(_config(2 * _ALIGN))
        err, objs = mgr.allocate(_layout(4096), 3)
        assert err == L1Error.OUT_OF_MEMORY
        assert objs == []
        assert mgr.get_memory_usage()[0] == 0

    def test_chunk_size_rounded_up_to_alignment(self):
        # 5000-byte chunk rounds up to 8192 (next 4 KiB multiple).
        mgr = GDSL1MemoryManager(_config(1 << 20))
        _, objs = mgr.allocate(_layout(5000), 1)
        assert objs[0].get_physical_size() == 8192


class TestFreeAndUsage:
    def test_free_returns_all_space(self):
        mgr = GDSL1MemoryManager(_config(1 << 20))
        _, objs = mgr.allocate(_layout(4096), 2)
        assert mgr.get_memory_usage()[0] == 2 * _ALIGN
        assert mgr.free(objs) == L1Error.SUCCESS
        assert mgr.get_memory_usage()[0] == 0

    def test_total_is_slab_size(self):
        used, total = GDSL1MemoryManager(_config(1 << 20)).get_memory_usage()
        assert used == 0
        assert total == (1 << 20)

    def test_memcheck_consistent_through_cycles(self):
        mgr = GDSL1MemoryManager(_config(1 << 20))
        assert mgr.memcheck() is True
        _, a = mgr.allocate(_layout(4096), 2)
        _, b = mgr.allocate(_layout(8192), 1)
        assert mgr.memcheck() is True
        mgr.free(a)
        mgr.free(b)
        assert mgr.memcheck() is True


class TestMisc:
    def test_get_l1_memory_desc_is_none(self):
        assert GDSL1MemoryManager(_config(1 << 20)).get_l1_memory_desc() is None

    def test_close_does_not_raise(self):
        GDSL1MemoryManager(_config(1 << 20)).close()
