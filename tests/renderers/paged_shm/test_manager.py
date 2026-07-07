# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from vllm.renderers.paged_shm.manager import Item, PagedSHMManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create a small pool: 4 blocks of 256 bytes each (total 1024 bytes)."""
    return PagedSHMManager(size=1024, block_size=256)


@pytest.fixture
def item_small():
    """An item that fits in 1 block."""
    return Item(uuid="small", size=200, use_cache=True)


@pytest.fixture
def item_large():
    """An item that needs 2 blocks."""
    return Item(uuid="large", size=400, use_cache=True)


@pytest.fixture
def item_nocache():
    """An item that should not be cached after writing."""
    return Item(uuid="nocache", size=200, use_cache=False)


# ---------------------------------------------------------------------------
# Basic allocation & write lifecycle
# ---------------------------------------------------------------------------


class TestBasicWriteLifecycle:
    def test_open_write_allocates_blocks(self, manager, item_small):
        [alloc] = manager.open_write([item_small])
        assert alloc.uuid == "small"
        assert len(alloc.blocks) == 1
        assert alloc.ref_count == -1
        assert manager._total_available_blocks == 3  # 4 - 1

    def test_close_write_moves_to_cache(self, manager, item_small):
        [alloc] = manager.open_write([item_small])
        manager.close_write("small")
        item = manager._all_items["small"]
        assert item.ref_count == 0
        # Available blocks increase because item is now evictable
        assert manager._total_available_blocks == 4
        assert "small" in manager._lru_cache

    def test_close_write_does_not_cache_if_use_cache_false(self, manager, item_nocache):
        [alloc] = manager.open_write([item_nocache])
        manager.close_write("nocache")
        assert "nocache" not in manager._lru_cache
        # Available blocks remain unchanged (still 4 - 1 = 3)
        assert manager._total_available_blocks == 3

    def test_close_write_on_non_writing_item_raises(self, manager, item_small):
        [alloc] = manager.open_write([item_small])
        manager.close_write("small")
        with pytest.raises(ValueError, match="not being written"):
            manager.close_write("small")

    def test_double_close_write(self, manager, item_small):
        [alloc] = manager.open_write([item_small])
        manager.close_write("small")

        with pytest.raises(ValueError, match="not being written"):
            manager.close_write("small")


# ---------------------------------------------------------------------------
# Read lifecycle (reference counting & cache interaction)
# ---------------------------------------------------------------------------


class TestReadLifecycle:
    def test_open_read_removes_from_cache(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        # Now in cache, total_available_blocks = 4
        manager.open_read("small")
        item = manager._all_items["small"]
        assert item.ref_count == 1
        assert "small" not in manager._lru_cache
        assert manager._total_available_blocks == 3  # taken out of evictable pool

    def test_close_read_returns_to_cache(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.open_read("small")
        manager.close_read("small")
        item = manager._all_items["small"]
        assert item.ref_count == 0
        assert "small" in manager._lru_cache
        assert manager._total_available_blocks == 4

    def test_multiple_readers(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.open_read("small")
        manager.open_read("small")
        item = manager._all_items["small"]
        assert item.ref_count == 2
        # First close_read should not put it back
        manager.close_read("small")
        assert item.ref_count == 1
        assert "small" not in manager._lru_cache
        # Second close_read returns it to cache
        manager.close_read("small")
        assert item.ref_count == 0
        assert "small" in manager._lru_cache

    def test_cannot_open_read_while_writing(self, manager, item_small):
        manager.open_write([item_small])
        with pytest.raises(ValueError, match="is being written"):
            manager.open_read("small")

    def test_cannot_open_read_nonexistent(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.open_read("ghost")

    def test_double_close_read(self, manager, item_small):
        [alloc] = manager.open_write([item_small])
        manager.close_write("small")

        manager.open_read("small")
        manager.close_read("small")

        with pytest.raises(ValueError, match="not being written"):
            manager.close_read("small")


# ---------------------------------------------------------------------------
# Pin / unpin
# ---------------------------------------------------------------------------


class TestPinUnpin:
    def test_pin_removes_from_cache(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.pin("small")
        assert "small" not in manager._lru_cache
        # Pinned item is not evictable, so available blocks drop
        assert manager._total_available_blocks == 3

    def test_unpin_returns_to_cache(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.pin("small")
        manager.unpin("small")
        assert "small" in manager._lru_cache
        assert manager._total_available_blocks == 4

    def test_pin_during_read_stays_out_of_cache(self, manager, item_small):
        """Pinning while item is being read keeps it pinned after reads finish."""
        manager.open_write([item_small])
        manager.close_write("small")
        manager.open_read("small")
        manager.pin("small")
        # ref_count=1, so not in cache anyway
        manager.close_read("small")
        # ref_count=0, but pinned -> still not in cache
        assert "small" not in manager._lru_cache
        assert manager._total_available_blocks == 3  # remains unavailable
        manager.unpin("small")
        assert "small" in manager._lru_cache
        assert manager._total_available_blocks == 4

    def test_pin_non_cacheable(self, manager, item_nocache):
        manager.open_write([item_nocache])

        manager.pin("nocache")
        manager.unpin("nocache")
        assert "nocache" in manager._all_items

        manager.close_write("nocache")
        manager.pin("nocache")
        manager.unpin("nocache")
        assert "nocache" not in manager._all_items


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_idle_cacheable_item(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.delete("small")
        assert "small" not in manager._all_items
        assert "small" not in manager._lru_cache
        # Blocks returned to free list
        assert len(manager._free_blocks) == 4
        # Available blocks unchanged (was 4, now all free)
        assert manager._total_available_blocks == 4

    def test_delete_idle_noncacheable_item(self, manager, item_nocache):
        manager.open_write([item_nocache])
        manager.close_write("nocache")
        manager.delete("nocache")
        assert "nocache" not in manager._all_items
        # total_available_blocks should increase by 1 (block was previously
        # tied up in a non‑evictable item)
        assert manager._total_available_blocks == 4
        assert len(manager._free_blocks) == 4

    def test_delete_pinned_item(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.pin("small")
        manager.delete("small")
        assert "small" not in manager._all_items
        assert len(manager._free_blocks) == 4
        assert manager._total_available_blocks == 4

    def test_delete_not_in_lru_does_not_raise(self, manager, item_nocache):
        """Regression test: deleting an item not in the LRU cache must not raise KeyError."""
        manager.open_write([item_nocache])
        manager.close_write("nocache")
        # 'nocache' is not in the LRU; delete should succeed silently
        manager.delete("nocache")
        assert "nocache" not in manager._all_items

    def test_delete_busy_item_raises(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.open_read("small")
        with pytest.raises(ValueError, match="busy now"):
            manager.delete("small")


# ---------------------------------------------------------------------------
# Batch allocation & memory pressure
# ---------------------------------------------------------------------------


class TestBatchAllocation:
    def test_batch_allocates_all_items(self, manager):
        items = [
            Item(uuid="a", size=300, use_cache=True),  # needs 2 blocks
            Item(uuid="b", size=100, use_cache=True),  # needs 1 block
            Item(uuid="c", size=100, use_cache=True),  # needs 1 block
        ]
        allocs = manager.open_write(items)
        assert len(allocs) == 3
        assert {len(a.blocks) for a in allocs} == {2, 1}
        # total blocks used = 4, so _free_blocks is empty
        assert len(manager._free_blocks) == 0
        assert manager._total_available_blocks == 0

    def test_memory_error_when_not_enough_space(self, manager):
        items = [
            Item(uuid="x", size=800, use_cache=True),  # needs 4 blocks
            Item(uuid="y", size=10, use_cache=True),  # needs 1 block -> total 5 > 4
        ]
        with pytest.raises(MemoryError, match="No sufficient space"):
            manager.open_write(items)

    def test_batch_duplicate_uuid_raises(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        with pytest.raises(ValueError, match="already exists"):
            manager.open_write([item_small])

    def test_invalid_size_raises(self, manager):
        with pytest.raises(ValueError, match="must be greater than zero"):
            manager.open_write([Item(uuid="z", size=0, use_cache=True)])
        with pytest.raises(ValueError, match="must be greater than zero"):
            manager.open_write([Item(uuid="z", size=-1, use_cache=True)])


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_evicts_oldest_cached_item(self, manager):
        # Fill the pool with two items of 2 blocks each (total 4 blocks)
        item1 = Item(uuid="old", size=400, use_cache=True)
        item2 = Item(uuid="new", size=400, use_cache=True)
        manager.open_write([item1, item2])
        manager.close_write("old")
        manager.close_write("new")
        # Both in cache; total_available_blocks = 4, free_blocks = 0
        assert len(manager._free_blocks) == 0

        # Request a 1‑block item → must evict the LRU item ("old")
        [new_alloc] = manager.open_write([Item(uuid="extra", size=100, use_cache=True)])
        assert len(new_alloc.blocks) == 1
        # "old" should have been evicted
        assert "old" not in manager._all_items
        assert "new" in manager._all_items
        # The freed blocks from "old" (2) were used: 1 for the new item,
        # the other is now in the free list
        assert len(manager._free_blocks) == 1

    def test_pinned_item_not_evicted(self, manager):
        # Same setup, but pin the older item
        item1 = Item(uuid="pinned_old", size=400, use_cache=True)
        item2 = Item(uuid="unpinned", size=400, use_cache=True)
        manager.open_write([item1, item2])
        manager.close_write("pinned_old")
        manager.close_write("unpinned")
        manager.pin("pinned_old")

        # Now only "unpinned" is in cache. total_available_blocks = 2 (from unpinned)
        # free_blocks = 0.  Requesting a 1‑block item should evict "unpinned".
        manager.open_write([Item(uuid="extra", size=100, use_cache=True)])
        assert "pinned_old" in manager._all_items
        assert "unpinned" not in manager._all_items

    def test_eviction_respects_order_of_close_write(self, manager):
        """LRU order follows close_write, not open_write."""
        # Allocate three items: two small, one large (but large uses 2 blocks)
        items = [
            Item(uuid="first", size=200, use_cache=True),  # 1 block
            Item(uuid="second", size=200, use_cache=True),  # 1 block
            Item(uuid="third", size=400, use_cache=True),  # 2 blocks
        ]
        manager.open_write(items)
        # Close them in a specific order: third, first, second
        manager.close_write("third")
        manager.close_write("first")
        manager.close_write("second")
        # Cache now contains all three; LRU order from oldest to newest:
        # third (2 blk), first (1 blk), second (1 blk)  total 4 blocks
        # A new request of 1 block must evict the LRU item (third).
        [alloc] = manager.open_write([Item(uuid="new", size=100, use_cache=True)])
        assert "third" not in manager._all_items
        assert "first" in manager._all_items
        assert "second" in manager._all_items
        # After eviction of third (2 blocks) and allocation of new (1 block),
        # one block remains free.
        assert len(manager._free_blocks) == 1