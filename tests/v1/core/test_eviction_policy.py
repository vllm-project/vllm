# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPU KV cache eviction policies.

Tests cover:
- LRUGPUCachePolicy: parity with the original FreeKVCacheBlockQueue behaviour.
- TwoQueueGPUCachePolicy: hot/cold separation, scan-pollution resistance.
- ARCGPUCachePolicy: T1/T2 routing, ghost-hit detection and p adjustment,
  ARC eviction rule, scan-pollution resistance, full ARC cycle.
- BlockPool with eviction_policy="arc": integration smoke test.
- make_gpu_eviction_policy: factory validation.
"""

import pytest

from vllm.v1.core.eviction_policy import (
    ARCGPUCachePolicy,
    LRUGPUCachePolicy,
    TwoQueueGPUCachePolicy,
    make_gpu_eviction_policy,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_blocks(n: int, start: int = 0) -> list[KVCacheBlock]:
    """Create *n* fresh KVCacheBlock objects with consecutive IDs."""
    return [KVCacheBlock(i + start) for i in range(n)]


# ---------------------------------------------------------------------------
# LRUGPUCachePolicy tests
# ---------------------------------------------------------------------------


class TestLRUGPUCachePolicy:
    def test_insert_and_evict_fifo_order(self):
        policy = LRUGPUCachePolicy()
        blocks = make_blocks(5)
        for b in blocks:
            policy.insert(b)

        evicted = policy.evict_n(3)
        assert [b.block_id for b in evicted] == [0, 1, 2]
        assert len(policy) == 2

    def test_insert_n_preserves_order(self):
        policy = LRUGPUCachePolicy()
        blocks = make_blocks(4)
        policy.insert_n(blocks)
        evicted = policy.evict_n(4)
        assert [b.block_id for b in evicted] == [0, 1, 2, 3]

    def test_remove_then_reinsert(self):
        policy = LRUGPUCachePolicy()
        blocks = make_blocks(3)
        policy.insert_n(blocks)

        # Remove the middle block (block_id=1)
        policy.remove(blocks[1])
        assert len(policy) == 2

        # Reinsert at tail
        policy.insert(blocks[1])
        evicted = policy.evict_n(3)
        assert [b.block_id for b in evicted] == [0, 2, 1]

    def test_touch_is_noop(self):
        """LRU touch() should not raise and should not change eviction order."""
        policy = LRUGPUCachePolicy()
        blocks = make_blocks(3)
        policy.insert_n(blocks)
        policy.touch(blocks[0])  # no-op
        evicted = policy.evict_n(3)
        assert [b.block_id for b in evicted] == [0, 1, 2]

    def test_len_tracking(self):
        policy = LRUGPUCachePolicy()
        blocks = make_blocks(10)
        policy.insert_n(blocks)
        assert len(policy) == 10
        policy.evict_n(4)
        assert len(policy) == 6
        policy.remove(blocks[4])
        assert len(policy) == 5

    def test_factory_returns_lru(self):
        policy = make_gpu_eviction_policy("lru")
        assert isinstance(policy, LRUGPUCachePolicy)


# ---------------------------------------------------------------------------
# TwoQueueGPUCachePolicy tests
# ---------------------------------------------------------------------------


class TestTwoQueueGPUCachePolicy:
    def test_new_blocks_go_to_cold_queue(self):
        policy = TwoQueueGPUCachePolicy()
        blocks = make_blocks(5)
        policy.insert_n(blocks)
        assert policy._cold.num_free_blocks == 5
        assert policy._hot.num_free_blocks == 0

    def test_touch_promotes_to_hot_on_next_insert(self):
        policy = TwoQueueGPUCachePolicy()
        blocks = make_blocks(3)
        policy.insert_n(blocks)

        # Simulate a prefix-cache hit on block 1:
        # 1. remove from cold queue (block is now in use)
        policy.remove(blocks[1])
        assert policy._cold.num_free_blocks == 2
        # 2. touch() marks for promotion
        policy.touch(blocks[1])
        assert blocks[1].block_id in policy._hot_set
        # 3. insert() routes to hot queue
        policy.insert(blocks[1])
        assert policy._cold.num_free_blocks == 2
        assert policy._hot.num_free_blocks == 1

    def test_eviction_drains_cold_first(self):
        """Hot blocks must not be evicted while cold blocks remain."""
        policy = TwoQueueGPUCachePolicy()

        hot_block = KVCacheBlock(99)
        cold_blocks = make_blocks(5, start=0)

        # Promote hot_block
        policy.insert(hot_block)
        policy.remove(hot_block)
        policy.touch(hot_block)
        policy.insert(hot_block)

        # Insert cold blocks
        policy.insert_n(cold_blocks)

        assert policy._hot.num_free_blocks == 1
        assert policy._cold.num_free_blocks == 5

        # Evict 5 — should drain cold queue entirely
        evicted = policy.evict_n(5)
        evicted_ids = {b.block_id for b in evicted}
        assert 99 not in evicted_ids, "Hot block evicted before cold blocks!"
        assert len(evicted_ids) == 5

        # hot_block should still be in the hot queue
        assert policy._hot.num_free_blocks == 1
        assert policy._cold.num_free_blocks == 0

    def test_eviction_falls_through_to_hot_when_cold_exhausted(self):
        policy = TwoQueueGPUCachePolicy()

        hot_blocks = make_blocks(3, start=100)
        for b in hot_blocks:
            policy.insert(b)
            policy.remove(b)
            policy.touch(b)
            policy.insert(b)

        assert policy._hot.num_free_blocks == 3
        assert policy._cold.num_free_blocks == 0

        evicted = policy.evict_n(2)
        assert len(evicted) == 2
        assert policy._hot.num_free_blocks == 1

    def test_demotion_on_eviction_from_hot(self):
        """Blocks evicted from hot queue should lose their hot status."""
        policy = TwoQueueGPUCachePolicy()

        block = KVCacheBlock(42)
        policy.insert(block)
        policy.remove(block)
        policy.touch(block)
        policy.insert(block)
        assert block.block_id in policy._hot_set

        # Evict from hot (cold is empty)
        policy.evict_n(1)
        assert block.block_id not in policy._hot_set, (
            "Block should be demoted after hot eviction"
        )

    def test_scan_pollution_resistance(self):
        """
        Reproduce the RFC scenario:
        - 64 hot "system-prompt" blocks are frequently reused.
        - 1984 cold one-time blocks flood the queue.
        - Under LRU the hot blocks would be evicted first; under TwoQueue
          they must survive until all cold blocks are exhausted.
        """
        N_HOT = 64
        N_COLD = 200  # smaller than full scenario but still demonstrates

        policy = TwoQueueGPUCachePolicy()

        hot_blocks = make_blocks(N_HOT, start=0)
        cold_blocks = make_blocks(N_COLD, start=N_HOT)

        # 1. Insert hot blocks and promote them all via touch()
        policy.insert_n(hot_blocks)
        for b in hot_blocks:
            policy.remove(b)
            policy.touch(b)
            policy.insert(b)

        # 2. Flood with cold blocks (simulating burst of unique prompts)
        policy.insert_n(cold_blocks)

        assert policy._hot.num_free_blocks == N_HOT
        assert policy._cold.num_free_blocks == N_COLD

        # 3. Evict all cold blocks — hot blocks must remain untouched
        evicted = policy.evict_n(N_COLD)
        assert len(evicted) == N_COLD
        hot_ids = {b.block_id for b in hot_blocks}
        for b in evicted:
            assert b.block_id not in hot_ids, (
                f"Hot block {b.block_id} was evicted before cold blocks!"
            )

        # 4. Only after cold exhaustion can hot blocks be evicted
        assert policy._hot.num_free_blocks == N_HOT
        assert policy._cold.num_free_blocks == 0

    def test_len_is_sum_of_both_queues(self):
        policy = TwoQueueGPUCachePolicy()

        cold = make_blocks(4, start=0)
        hot = make_blocks(3, start=10)

        policy.insert_n(cold)
        for b in hot:
            policy.insert(b)
            policy.remove(b)
            policy.touch(b)
            policy.insert(b)

        assert len(policy) == 7

    def test_factory_returns_two_queue(self):
        policy = make_gpu_eviction_policy("two_queue")
        assert isinstance(policy, TwoQueueGPUCachePolicy)


# ---------------------------------------------------------------------------
# Helpers shared by ARC tests
# ---------------------------------------------------------------------------


def _make_fake_hash(seed: int) -> BlockHashWithGroupId:
    """Create a deterministic fake block hash for testing."""
    return BlockHashWithGroupId(seed.to_bytes(32, "big") + (0).to_bytes(4, "big"))


def _set_block_hash(block: KVCacheBlock, seed: int) -> None:
    """Assign a synthetic hash to *block* so ghost-hit logic can fire."""
    block._block_hash = _make_fake_hash(seed)


# ---------------------------------------------------------------------------
# ARCGPUCachePolicy tests
# ---------------------------------------------------------------------------


class TestARCGPUCachePolicy:
    CAP = 100

    def _make(self, cap: int = CAP) -> ARCGPUCachePolicy:
        return ARCGPUCachePolicy(capacity=cap)

    # ------------------------------------------------------------------
    # Basic routing
    # ------------------------------------------------------------------

    def test_new_blocks_go_to_t1(self):
        p = self._make()
        p.insert_n(make_blocks(5))
        assert p.t1_size == 5
        assert p.t2_size == 0

    def test_touch_routes_to_t2_on_next_insert(self):
        p = self._make()
        block = KVCacheBlock(10)
        p.insert(block)
        p.remove(block)
        p.touch(block)
        p.insert(block)
        assert p.t1_size == 0
        assert p.t2_size == 1

    def test_remove_from_t1(self):
        p = self._make()
        blocks = make_blocks(3)
        p.insert_n(blocks)
        p.remove(blocks[1])
        assert p.t1_size == 2

    def test_remove_from_t2(self):
        p = self._make()
        block = KVCacheBlock(7)
        p.insert(block)
        p.remove(block)
        p.touch(block)
        p.insert(block)
        assert p.t2_size == 1
        p.remove(block)
        assert p.t2_size == 0

    # ------------------------------------------------------------------
    # ARC eviction rule: T1 vs T2 selection
    # ------------------------------------------------------------------

    def test_eviction_prefers_t1_when_t1_gte_p(self):
        """When |T1| >= max(1, p), should evict from T1."""
        p = self._make()
        # p == 0.0 → max(1, p) == 1; any non-empty T1 triggers T1 eviction.
        t1_block = KVCacheBlock(1)
        t2_block = KVCacheBlock(2)
        p.insert(t1_block)
        p.insert(t2_block)
        p.remove(t2_block)
        p.touch(t2_block)
        p.insert(t2_block)  # → T2

        evicted = p.evict_n(1)
        assert evicted[0].block_id == t1_block.block_id

    def test_eviction_falls_back_to_t2_when_t1_empty(self):
        p = self._make()
        block = KVCacheBlock(42)
        p.insert(block)
        p.remove(block)
        p.touch(block)
        p.insert(block)
        assert p.t1_size == 0
        assert p.t2_size == 1
        evicted = p.evict_n(1)
        assert evicted[0].block_id == 42

    def test_eviction_from_t1_records_hash_in_b1(self):
        p = self._make()
        block = KVCacheBlock(5)
        _set_block_hash(block, seed=5)
        p.insert(block)
        p.evict_n(1)
        assert p.b1_size == 1
        assert p.b2_size == 0

    def test_eviction_from_t2_records_hash_in_b2(self):
        p = self._make()
        block = KVCacheBlock(6)
        _set_block_hash(block, seed=6)
        p.insert(block)
        p.remove(block)
        p.touch(block)
        p.insert(block)
        p._p = float(p._capacity)  # force T2 eviction
        p.evict_n(1)
        assert p.b2_size == 1
        assert p.b1_size == 0

    def test_eviction_from_t2_demotes_block(self):
        p = self._make()
        block = KVCacheBlock(9)
        p.insert(block)
        p.remove(block)
        p.touch(block)
        p.insert(block)
        assert block.block_id in p._t2_ids
        p._p = float(p._capacity)
        p.evict_n(1)
        assert block.block_id not in p._t2_ids

    # ------------------------------------------------------------------
    # Ghost hit: B1
    # ------------------------------------------------------------------

    def test_b1_ghost_hit_increases_p(self):
        cap = 50
        p = self._make(cap)
        block = KVCacheBlock(1)
        h = _make_fake_hash(1)
        block._block_hash = h

        p.insert(block)
        p.evict_n(1)
        assert p.b1_size == 1
        p_before = p.p

        # Another block with the same hash is freed (content was recomputed)
        block2 = KVCacheBlock(99)
        block2._block_hash = h
        p.insert(block2)

        assert p.p > p_before, "p must increase on B1 ghost hit"
        assert p.b1_size == 0, "hash must be removed from B1"
        assert p.t2_size == 1, "ghost-hit block must go to T2"

    def test_b2_ghost_hit_decreases_p(self):
        cap = 50
        p = self._make(cap)
        block = KVCacheBlock(1)
        h = _make_fake_hash(2)
        block._block_hash = h

        p.insert(block)
        p.remove(block)
        p.touch(block)
        p.insert(block)
        p._p = 0.0  # force T2 eviction
        p.evict_n(1)
        assert p.b2_size == 1

        p._p = 10.0
        p_before = p.p
        block2 = KVCacheBlock(88)
        block2._block_hash = h
        p.insert(block2)

        assert p.p < p_before, "p must decrease on B2 ghost hit"
        assert p.b2_size == 0, "hash must be removed from B2"
        assert p.t2_size == 1, "ghost-hit block must go to T2"

    def test_ghost_hit_without_hash_does_not_adjust_p(self):
        p = self._make()
        block = KVCacheBlock(3)  # no hash
        p.insert(block)
        p_before = p.p
        p.evict_n(1)
        p.insert(block)  # still no hash
        assert p.p == p_before

    # ------------------------------------------------------------------
    # Ghost-list trimming
    # ------------------------------------------------------------------

    def test_ghost_list_bounded_to_capacity(self):
        cap = 5
        p = self._make(cap)
        blocks = [KVCacheBlock(i) for i in range(cap + 3)]
        for i, b in enumerate(blocks):
            _set_block_hash(b, seed=i + 100)
            p.insert(b)
        p.evict_n(cap + 3)
        assert p.b1_size <= cap

    # ------------------------------------------------------------------
    # Adaptive self-tuning
    # ------------------------------------------------------------------

    def test_p_non_decreasing_on_repeated_b1_hits(self):
        p = self._make(200)
        p_prev = p.p
        for seed in range(10):
            block = KVCacheBlock(seed)
            _set_block_hash(block, seed=seed)
            p.insert(block)
            p.evict_n(1)  # → B1
            block2 = KVCacheBlock(seed + 100)
            block2._block_hash = _make_fake_hash(seed)
            p.insert(block2)
            assert p.p >= p_prev
            p_prev = p.p

    # ------------------------------------------------------------------
    # Scan-pollution resistance
    # ------------------------------------------------------------------

    def test_scan_pollution_resistance(self):
        """T2 (hot) blocks must survive a T1 (cold) flood.

        ARC with p == 0 means max(1, p) == 1: T1 is always preferred as long
        as it is non-empty, so T1 drains completely before any T2 block is
        touched.  This is the state ARC naturally converges to when the
        frequently-used prefix blocks (in T2) keep generating B2 ghost hits
        that push p down toward 0.
        """
        N_HOT = 20
        N_COLD = 50
        p = self._make(N_HOT + N_COLD + 10)

        hot_blocks = make_blocks(N_HOT, start=0)
        cold_blocks = make_blocks(N_COLD, start=N_HOT)

        for b in hot_blocks:
            p.insert(b)
            p.remove(b)
            p.touch(b)
            p.insert(b)

        p.insert_n(cold_blocks)
        assert p.t2_size == N_HOT
        assert p.t1_size == N_COLD

        # p == 0: max(1, p) == 1 → T1 preferred whenever T1 is non-empty
        p._p = 0.0
        evicted = p.evict_n(N_COLD)
        hot_ids = {b.block_id for b in hot_blocks}
        for b in evicted:
            assert b.block_id not in hot_ids, (
                f"Hot block {b.block_id} evicted before cold blocks!"
            )
        assert p.t2_size == N_HOT

    # ------------------------------------------------------------------
    # Full ARC cycle
    # ------------------------------------------------------------------

    def test_full_arc_cycle(self):
        """Simulate the complete ghost-hit learning cycle."""
        p = self._make(50)
        h = _make_fake_hash(77)

        # Block enters T1
        b1 = KVCacheBlock(1)
        b1._block_hash = h
        p.insert(b1)
        assert p.t1_size == 1

        # Evict from T1 → H in B1
        p.evict_n(1)
        assert p.b1_size == 1
        p_after_eviction = p.p

        # Recomputed block with same hash → B1 ghost hit
        b2 = KVCacheBlock(2)
        b2._block_hash = h
        p.insert(b2)
        assert p.p > p_after_eviction
        assert p.t2_size == 1
        assert p.b1_size == 0

        # Another hit on b2 (prefix-cache hit): stays in T2
        p.remove(b2)
        p.touch(b2)
        b2._block_hash = h
        p.insert(b2)
        assert p.t2_size == 1

    # ------------------------------------------------------------------
    # Factory & len
    # ------------------------------------------------------------------

    def test_factory_returns_arc(self):
        policy = make_gpu_eviction_policy("arc", capacity=100)
        assert isinstance(policy, ARCGPUCachePolicy)

    def test_len_is_t1_plus_t2(self):
        p = self._make()
        blocks = make_blocks(6)
        p.insert_n(blocks[:3])  # → T1
        for b in blocks[3:]:
            p.insert(b)
            p.remove(b)
            p.touch(b)
            p.insert(b)  # → T2
        assert len(p) == 6


# ---------------------------------------------------------------------------
# make_gpu_eviction_policy validation
# ---------------------------------------------------------------------------


class TestMakeGpuEvictionPolicy:
    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown GPU eviction policy"):
            make_gpu_eviction_policy("unknown_policy")

    def test_all_known_policies_instantiate(self):
        for name in ("lru", "two_queue", "arc"):
            policy = make_gpu_eviction_policy(name, capacity=50)
            assert isinstance(
                policy,
                (LRUGPUCachePolicy, TwoQueueGPUCachePolicy, ARCGPUCachePolicy),
            )


# ---------------------------------------------------------------------------
# BlockPool integration smoke test (all three policies)
# ---------------------------------------------------------------------------


class TestBlockPoolWithEvictionPolicies:
    """Smoke tests verifying BlockPool works end-to-end with all three policies."""

    def _make_pool(self, num_blocks: int, eviction_policy: str):
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=eviction_policy,
        )

    @pytest.mark.parametrize("policy", ["lru", "two_queue", "arc"])
    def test_allocate_and_free(self, policy: str):
        pool = self._make_pool(20, policy)
        assert pool.get_num_free_blocks() == 19  # null_block takes 1

        blocks = pool.get_new_blocks(5)
        assert pool.get_num_free_blocks() == 14

        pool.free_blocks(reversed(blocks))
        assert pool.get_num_free_blocks() == 19

    @pytest.mark.parametrize("policy", ["lru", "two_queue", "arc"])
    def test_touch_promotes_block(self, policy: str):
        pool = self._make_pool(20, policy)
        blocks = pool.get_new_blocks(3)

        pool.free_blocks(list(reversed(blocks)))
        assert pool.get_num_free_blocks() == 19

        pool.touch([blocks[0]])
        assert pool.get_num_free_blocks() == 18

        blocks[0].ref_cnt -= 1
        pool.free_blocks([blocks[0]])
        assert pool.get_num_free_blocks() == 19

    def test_arc_t2_block_survives_t1_flood(self):
        """ARC T2 block must not be evicted before T1 blocks (p at target)."""
        pool = self._make_pool(60, "arc")
        # null_block → 59 free

        # Promote one block to T2
        hot = pool.get_new_blocks(1)[0]
        pool.free_blocks([hot])   # → T1
        pool.touch([hot])         # remove from T1, mark for T2
        hot.ref_cnt -= 1
        pool.free_blocks([hot])   # → T2

        # Flood T1
        n_cold = 30
        cold_blocks = pool.get_new_blocks(n_cold)
        pool.free_blocks(list(reversed(cold_blocks)))

        policy: ARCGPUCachePolicy = pool._policy  # type: ignore[assignment]
        # p == 0: max(1, p) == 1 → T1 drained before T2 is touched
        policy._p = 0.0

        evicted = pool._policy.evict_n(n_cold)
        for b in evicted:
            assert b.block_id != hot.block_id, "T2 block evicted before T1 blocks!"

