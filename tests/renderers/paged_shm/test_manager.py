# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.renderers.paged_shm.manager import AllocatedItem, Item, PagedSHMManager


class TestInit:
    def test_init(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        assert manager.block_size == 256
        assert manager.n_block == 4
        assert manager.size == 1024
        assert len(manager.free_blocks) == 4
        assert manager.n_free_block == 4
        assert manager.all_items == {}
        assert manager.lru_cache.capacity == 4

    def test_init2(self):
        manager = PagedSHMManager(size=1025, block_size=256)
        assert manager.block_size == 256
        assert manager.n_block == 4
        assert manager.size == 1024
        assert len(manager.free_blocks) == 4
        assert manager.n_free_block == 4
        assert manager.all_items == {}
        assert manager.lru_cache.capacity == 4

    def test_init_invalid_size(self):
        with pytest.raises(AssertionError):
            PagedSHMManager(size=128, block_size=256)

        with pytest.raises(AssertionError):
            PagedSHMManager(size=0, block_size=256)

        with pytest.raises(AssertionError):
            PagedSHMManager(size=128, block_size=0)


class TestAllocate:
    def test_allocate_single_cached(self):
        manager = PagedSHMManager(size=1024, block_size=256)

        item = Item(uuid="a", size=200, cached=True)
        allocated = manager.allocate([item])
        assert len(allocated) == 1
        a = allocated[0]
        assert a.uuid == "a"
        assert a.size == 200
        assert a.cached is True
        assert a.ref_count == -1
        # Ceil division: (200+255)//256 = 1 block
        assert a.n_block() == 1
        assert len(a.blocks) == 1
        assert manager.all_items["a"] is a
        assert len(manager.free_blocks) == 3

    def test_allocate_multiple_mixed(self):
        manager = PagedSHMManager(size=1024, block_size=256)

        items = [
            Item(uuid="x", size=256, cached=True),
            Item(uuid="y", size=300, cached=False),
            Item(uuid="z", size=100, cached=True),
        ]
        allocated = manager.allocate(items)
        assert len(allocated) == 3
        # x: 1 block, y: 2 blocks, z: 1 block -> total 4 blocks
        assert allocated[0].n_block() == 1
        assert allocated[1].n_block() == 2
        assert allocated[2].n_block() == 1
        assert len(manager.free_blocks) == 0
        assert all(item.ref_count == -1 for item in allocated)

    def test_allocate_zero_size(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        with pytest.raises(ValueError):
            item = Item(uuid="zero", size=0, cached=False)
            manager.allocate([item])
        assert len(manager.free_blocks) == 4

    def test_allocate_uuid_conflict(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        manager.allocate([Item(uuid="dup", size=100, cached=False)])
        with pytest.raises(ValueError, match="already exists"):
            manager.allocate([Item(uuid="dup", size=100, cached=False)])

    def test_allocate_exceeds_total_capacity(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        item = Item(uuid="big", size=1025, cached=False)  # (1025+255)//256 = 5
        with pytest.raises(MemoryError):
            manager.allocate([item])

    def test_allocate_eviction_impossible_memory_error(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        manager.allocate([Item(uuid="full", size=1024, cached=True)])
        assert len(manager.free_blocks) == 0
        with pytest.raises(MemoryError):
            manager.allocate([Item(uuid="fail", size=100, cached=False)])


class TestBorrow:
    def test_borrow_cached_item_in_lru(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        item = AllocatedItem(uuid="b", size=256, cached=True, blocks=[0], ref_count=0)
        manager.all_items["b"] = item
        manager.lru_cache["b"] = item
        manager.borrow("b")
        assert item.ref_count == 1

    def test_borrow_nonexistent_uuid(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        with pytest.raises(ValueError, match="not found"):
            manager.borrow("nonexistent")

    def test_borrow_item_being_written(self):
        manager = PagedSHMManager(size=1024, block_size=256)

        manager.allocate([Item(uuid="w", size=256, cached=True)])
        assert manager.all_items["w"].ref_count == -1

        with pytest.raises(ValueError, match="being written"):
            manager.borrow("w")

    def test_borrow_increases_ref_and_touches(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        manager.allocate([Item(uuid="w", size=256, cached=True)])

        item = AllocatedItem(uuid="r2", size=256, cached=True, blocks=[0], ref_count=2)
        manager.all_items["r2"] = item
        manager.lru_cache["r2"] = item
        manager.borrow("r2")
        assert item.ref_count == 3


class TestRestore:
    def test_restore_uncached_item(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        manager.allocate([Item(uuid="a", size=256, cached=False)])
        assert "a" in manager.all_items
        assert len(manager.free_blocks) == 3
        manager.restore("a")
        assert "a" not in manager.all_items
        assert len(manager.free_blocks) == 4  # Blocks have been returned

    def test_restore_cached_item_after_write(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        manager.allocate([Item(uuid="c", size=256, cached=True)])
        item = manager.all_items["c"]
        assert item.ref_count == -1
        manager.restore("c")
        # Write completed, ref_count becomes 0
        assert item.ref_count == 0
        # But is it removed from all_items? cached items are not removed
        assert "c" in manager.all_items

    def test_restore_cached_item_with_readers(self):
        manager = PagedSHMManager(size=1024, block_size=256)

        manager.allocate([Item(uuid="r", size=256, cached=True)])
        item = manager.all_items["r"]
        assert item.ref_count == -1

        manager.restore("r")
        assert item.ref_count == 0

        manager.borrow("r")
        assert item.ref_count == 1

        manager.restore("r")
        assert item.ref_count == 0

    def test_restore_nonexistent_uuid(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        with pytest.raises(ValueError, match="not found"):
            manager.restore("nonexistent")

    def test_restore_uncached_item_with_readers(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        manager.allocate([Item(uuid="u", size=256, cached=False)])
        item = manager.all_items["u"]
        item.ref_count = 3  # Simulate active readers
        manager.restore("u")
        assert "u" not in manager.all_items
        assert len(manager.free_blocks) == 4


class TestEvict:
    def test_evict(self):
        manager = PagedSHMManager(size=1024, block_size=256)
        assert manager.n_free_block == 4

        manager.allocate([Item(uuid="i1", size=256, cached=True)])
        assert manager.n_free_block == 3

        manager.allocate([Item(uuid="i2", size=512, cached=True)])
        assert manager.n_free_block == 1

        with pytest.raises(MemoryError):
            manager.allocate([Item(uuid="i3", size=512, cached=True)])

        manager.restore("i1")
        assert manager.n_free_block == 2

        manager.allocate([Item(uuid="i3", size=512, cached=True)])

        manager.restore("i2")
        assert manager.n_free_block == 2

        manager.borrow("i2")
        assert manager.n_free_block == 0

        with pytest.raises(MemoryError):
            manager.allocate([Item(uuid="i4", size=512, cached=True)])

        manager.restore("i2")
        assert manager.n_free_block == 2
        manager.allocate([Item(uuid="i4", size=512, cached=True)])
