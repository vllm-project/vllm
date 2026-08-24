# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from vllm.multimodal.paged_shm.manager import PagedShmManager
from vllm.multimodal.paged_shm.types import ShmWriteRequest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create a small pool: 4 blocks of 256 bytes each (total 1024 bytes)."""
    return PagedShmManager(size=1024, block_size=256)


@pytest.fixture
def item_small():
    """An item that fits in 1 block."""
    return ShmWriteRequest(uuid="small", size=200, use_cache=True)


@pytest.fixture
def item_large():
    """An item that needs 2 blocks."""
    return ShmWriteRequest(uuid="large", size=400, use_cache=True)


@pytest.fixture
def item_nocache():
    """An item that should not be cached after writing."""
    return ShmWriteRequest(uuid="nocache", size=200, use_cache=False)


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

    def test_close_write_auto_deletes_non_cacheable(self, manager, item_nocache):
        """
        For use_cache=False and no open reads, close_write deletes the item
        immediately, freeing its blocks.
        """
        [alloc] = manager.open_write([item_nocache])
        manager.close_write("nocache")
        assert "nocache" not in manager._all_items
        assert "nocache" not in manager._lru_cache
        # Blocks are returned to the pool
        assert manager._total_available_blocks == 4
        assert len(manager._free_blocks) == 4

    def test_close_write_keeps_non_cacheable_if_open_reads(self, manager, item_nocache):
        """
        If open_n_reads > 0, the non-cacheable item is kept with the given
        reference count. After all reads are closed, it is automatically deleted.
        """
        [alloc] = manager.open_write([item_nocache])
        manager.close_write("nocache", open_n_reads=1)
        item = manager._all_items["nocache"]
        assert item.ref_count == 1
        assert "nocache" not in manager._lru_cache
        # Blocks are not available (still owned)
        assert manager._total_available_blocks == 3

        # Close the read – ref_count becomes 0, and because use_cache=False,
        # the item should be deleted automatically (manager.close_read does this).
        manager.close_read("nocache")
        assert "nocache" not in manager._all_items
        assert manager._total_available_blocks == 4
        assert len(manager._free_blocks) == 4

    def test_close_write_with_open_read(self, manager, item_small):
        """When open_read, the item is automatically opened for reading."""
        [alloc] = manager.open_write([item_small])
        manager.close_write("small", open_n_reads=1)
        item = manager._all_items["small"]
        assert item.ref_count == 1
        assert "small" not in manager._lru_cache
        assert manager._total_available_blocks == 3

        manager.close_read("small")
        assert item.ref_count == 0
        assert "small" in manager._lru_cache
        assert manager._total_available_blocks == 4

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

        with pytest.raises(ValueError, match="not being read"):
            manager.close_read("small")

    def test_non_cacheable_read_and_auto_delete(self, manager, item_nocache):
        """
        For non-cacheable items, they must be kept alive by an open read
        reference (via close_write with open_n_reads).  After close_read,
        the item is auto-deleted.
        """
        manager.open_write([item_nocache])
        manager.close_write("nocache", open_n_reads=1)
        # Now we have one read reference; reading again increases it.
        item = manager.open_read("nocache")
        assert item.ref_count == 2
        assert "nocache" not in manager._lru_cache
        assert manager._total_available_blocks == 3

        # First close_read: ref_count becomes 1, still alive
        manager.close_read("nocache")
        assert "nocache" in manager._all_items
        assert manager._total_available_blocks == 3

        # Second close_read: ref_count becomes 0, auto-delete
        manager.close_read("nocache")
        assert "nocache" not in manager._all_items
        assert manager._total_available_blocks == 4
        assert len(manager._free_blocks) == 4


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
        """
        Non-cacheable items are auto-deleted on close_write, so we cannot
        have an idle non-cacheable item. We test force-deleting a busy one.
        """
        manager.open_write([item_nocache])
        manager.close_write("nocache", open_n_reads=1)  # keep alive
        # force delete while read is active
        manager.delete("nocache", force=True)
        assert "nocache" not in manager._all_items
        assert manager._total_available_blocks == 4
        assert len(manager._free_blocks) == 4

    def test_delete_busy_item_raises(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        manager.open_read("small")
        with pytest.raises(ValueError, match="busy now"):
            manager.delete("small")

    def test_delete_while_writing(self, manager, item_small):
        """Deleting an item that is still being written should raise."""
        manager.open_write([item_small])
        with pytest.raises(ValueError, match="busy now"):
            manager.delete("small")


# ---------------------------------------------------------------------------
# Batch allocation & memory pressure
# ---------------------------------------------------------------------------


class TestBatchAllocation:
    def test_batch_allocates_all_items(self, manager):
        items = [
            ShmWriteRequest(uuid="a", size=300, use_cache=True),  # needs 2 blocks
            ShmWriteRequest(uuid="b", size=100, use_cache=True),  # needs 1 block
            ShmWriteRequest(uuid="c", size=100, use_cache=True),  # needs 1 block
        ]
        allocs = manager.open_write(items)
        assert len(allocs) == 3
        assert {len(a.blocks) for a in allocs} == {2, 1}
        # total blocks used = 4, so _free_blocks is empty
        assert len(manager._free_blocks) == 0
        assert manager._total_available_blocks == 0

    def test_memory_error_when_not_enough_space(self, manager):
        items = [
            ShmWriteRequest(uuid="x", size=800, use_cache=True),  # needs 4 blocks
            ShmWriteRequest(
                uuid="y", size=10, use_cache=True
            ),  # needs 1 block -> total 5 > 4
        ]
        with pytest.raises(MemoryError, match="Not enough blocks"):
            manager.open_write(items)

    def test_batch_duplicate_uuid_raises(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        with pytest.raises(ValueError, match="already exists"):
            manager.open_write([item_small])

    def test_invalid_size_raises(self, manager):
        with pytest.raises(ValueError, match="must be greater than zero"):
            manager.open_write([ShmWriteRequest(uuid="z", size=0, use_cache=True)])
        with pytest.raises(ValueError, match="must be greater than zero"):
            manager.open_write([ShmWriteRequest(uuid="z", size=-1, use_cache=True)])

    def test_open_write_empty_list(self, manager):
        """Calling open_write with an empty list should be a no-op."""
        allocs = manager.open_write([])
        assert allocs == []
        assert manager._total_available_blocks == manager.n_block
        assert len(manager._free_blocks) == manager.n_block


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_evicts_oldest_cached_item(self, manager):
        # Fill the pool with two items of 2 blocks each (total 4 blocks)
        item1 = ShmWriteRequest(uuid="old", size=400, use_cache=True)
        item2 = ShmWriteRequest(uuid="new", size=400, use_cache=True)
        manager.open_write([item1, item2])
        manager.close_write("old")
        manager.close_write("new")
        # Both in cache; total_available_blocks = 4, free_blocks = 0
        assert len(manager._free_blocks) == 0

        # Request a 1‑block item → must evict the LRU item ("old")
        [new_alloc] = manager.open_write(
            [ShmWriteRequest(uuid="extra", size=100, use_cache=True)]
        )
        assert len(new_alloc.blocks) == 1
        # "old" should have been evicted
        assert "old" not in manager._all_items
        assert "new" in manager._all_items
        # The freed blocks from "old" (2) were used: 1 for the new item,
        # the other is now in the free list
        assert len(manager._free_blocks) == 1

    def test_eviction_respects_order_of_close_write(self, manager):
        """LRU order follows close_write, not open_write."""
        # Allocate three items: two small, one large (but large uses 2 blocks)
        items = [
            ShmWriteRequest(uuid="first", size=200, use_cache=True),  # 1 block
            ShmWriteRequest(uuid="second", size=200, use_cache=True),  # 1 block
            ShmWriteRequest(uuid="third", size=400, use_cache=True),  # 2 blocks
        ]
        manager.open_write(items)
        # Close them in a specific order: third, first, second
        manager.close_write("third")
        manager.close_write("first")
        manager.close_write("second")
        # Cache now contains all three; LRU order from oldest to newest:
        # third (2 blk), first (1 blk), second (1 blk)  total 4 blocks
        # A new request of 1 block must evict the LRU item (third).
        [alloc] = manager.open_write(
            [ShmWriteRequest(uuid="new", size=100, use_cache=True)]
        )
        assert "third" not in manager._all_items
        assert "first" in manager._all_items
        assert "second" in manager._all_items
        # After eviction of third (2 blocks) and allocation of new (1 block),
        # one block remains free.
        assert len(manager._free_blocks) == 1

    def test_evict_multiple_items(self, manager):
        """Eviction should pop only the minimum number of cache entries needed."""
        # Fill pool with three 1-block items (total 3 blocks)
        items = [
            ShmWriteRequest(uuid="a", size=100, use_cache=True),
            ShmWriteRequest(uuid="b", size=100, use_cache=True),
            ShmWriteRequest(uuid="c", size=100, use_cache=True),
        ]
        manager.open_write(items)
        manager.close_write("a")
        manager.close_write("b")
        manager.close_write("c")
        # All in cache; free_blocks = 1 (because 4 blocks total, 3 used)
        assert len(manager._free_blocks) == 1
        # Request 2 blocks -> need to evict at least one 1-block item
        [new] = manager.open_write(
            [ShmWriteRequest(uuid="new", size=300, use_cache=True)]
        )
        # new needs 2 blocks (300/256 = 2)
        # Eviction order: a (oldest) -> freed 1 block, plus existing 1 = 2 free
        # allocate 2 -> leaves 0 free blocks
        assert len(manager._free_blocks) == 0
        assert "a" not in manager._all_items  # evicted
        assert "b" in manager._all_items  # not evicted
        assert "c" in manager._all_items  # not evicted
        assert "new" in manager._all_items

    def test_evict_no_eviction_when_free_blocks_sufficient(self, manager):
        """If free_blocks already >= needed, _evict should do nothing."""
        # Allocate one 2-block item and close it -> goes to cache
        item = ShmWriteRequest(uuid="big", size=400, use_cache=True)
        manager.open_write([item])
        manager.close_write("big")
        # free_blocks = 2, cached blocks = 2
        # Request 1 block, no eviction needed
        [new] = manager.open_write(
            [ShmWriteRequest(uuid="small", size=100, use_cache=True)]
        )
        # "big" should still be in cache and all items exist
        assert "big" in manager._all_items
        assert "small" in manager._all_items
        # free_blocks should be 1 (2 initial -1 allocated)
        assert len(manager._free_blocks) == 1
        # total_available_blocks = free(1) + cached(2) = 3
        assert manager._total_available_blocks == 3


# ---------------------------------------------------------------------------
# Info and state reporting
# ---------------------------------------------------------------------------


class TestInfoAndState:
    def test_get_info(self, manager, item_small):
        manager.open_write([item_small])
        manager.close_write("small")
        info = manager.get_info("small")
        assert info["uuid"] == "small"
        assert info["size"] == 200
        assert info["use_cache"] is True
        assert info["ref_count"] == 0

    def test_get_info_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.get_info("ghost")

    def test_get_manager_state(self, manager, item_small, item_large):
        state = manager.get_manager_state()
        assert state["size"] == 1024
        assert state["block_size"] == 256
        assert state["n_block"] == 4
        assert state["free_blocks_count"] == 4
        assert state["total_available_blocks"] == 4
        assert state["cached_items_count"] == 0
        assert state["cached_blocks_count"] == 0
        assert state["total_items_count"] == 0

        manager.open_write([item_small, item_large])
        manager.close_write("small")
        state = manager.get_manager_state()
        assert state["free_blocks_count"] == 1  # 4 - (1+2) = 1
        # total_available_blocks = free_blocks(1) + cached_blocks(1) = 2
        assert state["total_available_blocks"] == 2
        assert state["cached_items_count"] == 1  # small
        assert state["cached_blocks_count"] == 1  # small's block
        assert state["total_items_count"] == 2

    def test_get_manager_state_with_noncacheable(self, manager):
        """Non-cacheable items with open reads are counted as reading."""
        item = ShmWriteRequest(uuid="nc", size=200, use_cache=False)
        manager.open_write([item])
        manager.close_write("nc", open_n_reads=1)
        state = manager.get_manager_state()
        assert state["total_items_count"] == 1
        assert state["reading_items_count"] == 1
        assert state["idle_items_count"] == 0
        assert state["writing_items_count"] == 0
        # Close the read -> item will be deleted
        manager.close_read("nc")
        state = manager.get_manager_state()
        assert state["total_items_count"] == 0
        assert state["free_blocks_count"] == 4
