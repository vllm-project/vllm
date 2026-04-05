# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for priority-based KV-cache eviction.

Tests the PriorityEvictionQueue and BlockPool integration with
retention directives.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.priority_eviction_queue import PriorityEvictionQueue

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# PriorityEvictionQueue unit tests
# ---------------------------------------------------------------------------


class TestPriorityEvictionQueue:
    def _make_block(
        self,
        block_id: int,
        priority: int,
        freed_time: float = 0.0,
        expiry: float | None = None,
    ) -> KVCacheBlock:
        block = KVCacheBlock(block_id)
        block.priority = priority
        block.last_freed_time = freed_time
        block.priority_expiry = expiry
        return block

    def test_empty_queue(self):
        pq = PriorityEvictionQueue()
        assert len(pq) == 0
        assert pq.pop_lowest() is None

    def test_insert_and_pop_single(self):
        pq = PriorityEvictionQueue()
        block = self._make_block(0, priority=50)
        pq.insert(block)
        assert len(pq) == 1

        result = pq.pop_lowest()
        assert result is block
        assert len(pq) == 0

    def test_eviction_order_by_priority(self):
        """Lowest priority evicted first."""
        pq = PriorityEvictionQueue()
        high = self._make_block(0, priority=90)
        low = self._make_block(1, priority=10)
        mid = self._make_block(2, priority=50)

        pq.insert(high)
        pq.insert(low)
        pq.insert(mid)

        assert pq.pop_lowest() is low
        assert pq.pop_lowest() is mid
        assert pq.pop_lowest() is high

    def test_eviction_order_tiebreak_by_time(self):
        """Same priority: oldest freed_time evicted first (LRU)."""
        pq = PriorityEvictionQueue()
        older = self._make_block(0, priority=50, freed_time=1.0)
        newer = self._make_block(1, priority=50, freed_time=2.0)

        pq.insert(newer)
        pq.insert(older)

        assert pq.pop_lowest() is older
        assert pq.pop_lowest() is newer

    def test_lazy_remove(self):
        """remove() marks a block for lazy deletion."""
        pq = PriorityEvictionQueue()
        b0 = self._make_block(0, priority=10)
        b1 = self._make_block(1, priority=20)

        pq.insert(b0)
        pq.insert(b1)
        assert len(pq) == 2

        pq.remove(b0)
        assert len(pq) == 1

        result = pq.pop_lowest()
        assert result is b1
        assert len(pq) == 0

    def test_remove_nonexistent_is_noop(self):
        pq = PriorityEvictionQueue()
        block = self._make_block(0, priority=10)
        pq.remove(block)  # should not raise
        assert len(pq) == 0

    def test_ttl_expiry(self):
        """Expired TTL demotes block to priority 0."""
        pq = PriorityEvictionQueue()
        now = time.monotonic()

        # Block with high priority but already-expired TTL.
        expired = self._make_block(0, priority=100, expiry=now - 1.0)
        # Block with low priority, no TTL.
        low = self._make_block(1, priority=10)

        pq.insert(expired)
        pq.insert(low)

        # Expired block should be demoted to priority 0, evicted first.
        result = pq.pop_lowest()
        assert result is expired
        # After expiry, priority should be cleared on the block.
        assert expired.priority is None
        assert expired.priority_expiry is None

    def test_ttl_not_expired(self):
        """Non-expired TTL preserves priority."""
        pq = PriorityEvictionQueue()
        now = time.monotonic()

        high_ttl = self._make_block(0, priority=80, expiry=now + 3600)
        low = self._make_block(1, priority=10)

        pq.insert(high_ttl)
        pq.insert(low)

        # Low priority evicted first; high-TTL block retained.
        result = pq.pop_lowest()
        assert result is low

        result = pq.pop_lowest()
        assert result is high_ttl
        assert high_ttl.priority == 80  # Not expired, priority intact.


# ---------------------------------------------------------------------------
# BlockPool integration tests
# ---------------------------------------------------------------------------


class TestBlockPoolPriorityEviction:
    def _make_pool(
        self, num_blocks: int = 10, enable_caching: bool = True
    ) -> BlockPool:
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=enable_caching,
            hash_block_size=16,
        )

    def test_free_unprioritized_goes_to_lru(self):
        """Blocks without priority go to the LRU free list."""
        pool = self._make_pool()
        blocks = pool.get_new_blocks(3)
        assert (
            pool.free_block_queue.num_free_blocks == pool.num_gpu_blocks - 1 - 3
        )  # -1 for null block

        pool.free_blocks(blocks)
        assert pool.priority_eviction_queue.num_blocks == 0
        assert (
            pool.free_block_queue.num_free_blocks == pool.num_gpu_blocks - 1
        )  # all back in LRU

    def test_free_prioritized_goes_to_priority_queue(self):
        """Blocks with priority go to the priority eviction queue."""
        pool = self._make_pool()
        blocks = pool.get_new_blocks(3)

        # Annotate one block with priority.
        blocks[1].priority = 50

        pool.free_blocks(blocks)
        assert pool.priority_eviction_queue.num_blocks == 1
        # 2 unprioritized + the rest that were already free.
        assert (
            pool.free_block_queue.num_free_blocks == pool.num_gpu_blocks - 1 - 1
        )  # -1 null, -1 in priority queue

    def test_eviction_drains_lru_before_priority(self):
        """get_new_blocks drains LRU free list before priority queue."""
        pool = self._make_pool(num_blocks=6)
        # Pool starts with 5 free blocks (6 - 1 null).
        # Allocate all 5.
        blocks = pool.get_new_blocks(5)

        # Free them: 2 with priority, 3 without.
        blocks[0].priority = 80
        blocks[1].priority = 20
        pool.free_blocks(blocks)

        assert pool.free_block_queue.num_free_blocks == 3
        assert pool.priority_eviction_queue.num_blocks == 2

        # Allocate 4 blocks: should take 3 from LRU, then 1 from priority
        # queue (lowest priority = 20 first).
        new_blocks = pool.get_new_blocks(4)
        assert len(new_blocks) == 4
        assert pool.free_block_queue.num_free_blocks == 0
        assert pool.priority_eviction_queue.num_blocks == 1

        # The remaining priority block should be the high-priority one.
        last = pool.get_new_blocks(1)
        assert last[0].block_id == blocks[0].block_id  # priority=80

    def test_touch_removes_from_priority_queue(self):
        """Touching a prioritized block removes it from the priority queue."""
        pool = self._make_pool()
        blocks = pool.get_new_blocks(1)
        blocks[0].priority = 50
        pool.free_blocks(blocks)
        assert pool.priority_eviction_queue.num_blocks == 1

        pool.touch(blocks)
        assert pool.priority_eviction_queue.num_blocks == 0
        assert blocks[0].ref_cnt == 1

    def test_touch_removes_from_lru(self):
        """Touching an unprioritized block removes it from the LRU list."""
        pool = self._make_pool()
        initial_free = pool.free_block_queue.num_free_blocks
        blocks = pool.get_new_blocks(1)
        pool.free_blocks(blocks)
        assert pool.free_block_queue.num_free_blocks == initial_free

        pool.touch(blocks)
        assert pool.free_block_queue.num_free_blocks == initial_free - 1
        assert blocks[0].ref_cnt == 1

    def test_get_num_free_blocks_sums_both(self):
        """get_num_free_blocks includes both LRU and priority queue."""
        pool = self._make_pool(num_blocks=6)
        blocks = pool.get_new_blocks(4)
        blocks[0].priority = 50
        blocks[1].priority = 80
        pool.free_blocks(blocks)

        expected = (
            pool.free_block_queue.num_free_blocks
            + pool.priority_eviction_queue.num_blocks
        )
        assert pool.get_num_free_blocks() == expected
        # 1 free initially (6 - 1 null - 4 allocated) + 4 freed = 5
        assert pool.get_num_free_blocks() == 5

    def test_reset_prefix_cache_clears_priority_queue(self):
        """reset_prefix_cache should clear the priority queue."""
        pool = self._make_pool(num_blocks=4)
        blocks = pool.get_new_blocks(2)
        blocks[0].priority = 50
        pool.free_blocks(blocks)
        assert pool.priority_eviction_queue.num_blocks == 1

        result = pool.reset_prefix_cache()
        # Should succeed since all blocks are free now.
        assert result is True
        assert pool.priority_eviction_queue.num_blocks == 0


# ---------------------------------------------------------------------------
# Retention directive application tests
# ---------------------------------------------------------------------------


class TestRetentionDirectives:
    def _make_pool(self) -> BlockPool:
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=4,
            enable_caching=True,
            hash_block_size=16,
        )

    def test_apply_retention_to_block_single_match(self):
        pool = self._make_pool()
        block = KVCacheBlock(1)
        directives = [
            {"start": 0, "end": 32, "priority": 70, "duration": None},
        ]
        pool._apply_retention_to_block(
            block, directives, token_start=0, token_end=16, scope="s1"
        )
        assert block.priority == 70
        assert block.priority_expiry is None
        assert block.priority_scope == "s1"

    def test_apply_retention_no_match(self):
        pool = self._make_pool()
        block = KVCacheBlock(1)
        directives = [
            {"start": 100, "end": 200, "priority": 70, "duration": None},
        ]
        pool._apply_retention_to_block(block, directives, token_start=0, token_end=16)
        assert block.priority is None

    def test_apply_retention_highest_priority_wins(self):
        pool = self._make_pool()
        block = KVCacheBlock(1)
        directives = [
            {"start": 0, "end": 32, "priority": 30},
            {"start": 8, "end": 24, "priority": 90},
            {"start": 0, "end": 16, "priority": 50},
        ]
        pool._apply_retention_to_block(
            block, directives, token_start=0, token_end=16, scope="s1"
        )
        assert block.priority == 90

    def test_apply_retention_with_duration(self):
        pool = self._make_pool()
        block = KVCacheBlock(1)
        directives = [
            {"start": 0, "end": 32, "priority": 60, "duration": 120.0},
        ]
        before = time.monotonic()
        pool._apply_retention_to_block(
            block, directives, token_start=0, token_end=16, scope="s1"
        )
        after = time.monotonic()

        assert block.priority == 60
        assert block.priority_expiry is not None
        assert before + 120.0 <= block.priority_expiry <= after + 120.0

    def test_apply_retention_open_ended_range(self):
        """Directive with end=None matches all blocks from start onward."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        directives = [
            {"start": 0, "priority": 40},  # No 'end' key
        ]
        pool._apply_retention_to_block(
            block, directives, token_start=1000, token_end=1016, scope="s1"
        )
        assert block.priority == 40

    # --- Scoped ownership tests ---

    def test_escalation_from_different_scope(self):
        """A different scope can escalate (raise) priority."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        block.priority = 50
        block.priority_scope = "s1"

        directives = [{"start": 0, "end": 32, "priority": 90}]
        pool._apply_retention_to_block(
            block, directives, token_start=0, token_end=16, scope="s2"
        )
        assert block.priority == 90
        assert block.priority_scope == "s2"

    def test_downgrade_blocked_from_different_scope(self):
        """A different scope cannot downgrade priority."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        block.priority = 90
        block.priority_scope = "s1"

        directives = [{"start": 0, "end": 32, "priority": 50}]
        pool._apply_retention_to_block(
            block, directives, token_start=0, token_end=16, scope="s2"
        )
        assert block.priority == 90
        assert block.priority_scope == "s1"

    def test_owner_can_downgrade(self):
        """The owning scope can downgrade priority."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        block.priority = 90
        block.priority_scope = "s1"

        directives = [{"start": 0, "end": 32, "priority": 30}]
        pool._apply_retention_to_block(
            block, directives, token_start=0, token_end=16, scope="s1"
        )
        assert block.priority == 30
        assert block.priority_scope == "s1"

    def test_owner_clear_on_no_match(self):
        """When no directive matches and scope is the owner, clear priority."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        block.priority = 70
        block.priority_scope = "s1"

        # Empty directives with same scope → clear
        pool._apply_retention_to_block(
            block, [], token_start=0, token_end=16, scope="s1"
        )
        assert block.priority is None
        assert block.priority_scope is None

    def test_non_owner_no_clear_on_no_match(self):
        """When no directive matches and scope differs, keep priority."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        block.priority = 70
        block.priority_scope = "s1"

        pool._apply_retention_to_block(
            block, [], token_start=0, token_end=16, scope="s2"
        )
        assert block.priority == 70
        assert block.priority_scope == "s1"

    def test_no_scope_no_clear(self):
        """When scope is None, no-match does not clear existing priority."""
        pool = self._make_pool()
        block = KVCacheBlock(1)
        block.priority = 70
        block.priority_scope = "s1"

        pool._apply_retention_to_block(
            block, [], token_start=0, token_end=16, scope=None
        )
        assert block.priority == 70
        assert block.priority_scope == "s1"
