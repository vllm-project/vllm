# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Tail-Optimized LRU (T-LRU) eviction policy.

T-LRU is described in the companion paper and implemented across:
  - KVCacheBlock.is_tel_safe
  - BlockPool.tel_safe_queue
  - BlockPool.free_blocks_tlru()
  - BlockPool.get_new_blocks() (drains tel_safe_queue first)
  - SingleTypeKVCacheManager.free() (routes blocks to correct queue)
"""

from collections import deque

import pytest

from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers to build a minimal BlockPool without standing up the full vLLM stack.
# ---------------------------------------------------------------------------

def _make_pool(
    num_blocks: int,
    tlru_xi_blocks: int | None,
    tlru_qhat_blocks: int = 0,
    enable_caching: bool = False,
    hash_block_size: int = 16,
    enable_kv_cache_events: bool = False,
):
    """Return a BlockPool configured with T-LRU parameters."""
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=enable_kv_cache_events,
        metrics_collector=None,
        tlru_xi_blocks=tlru_xi_blocks,
        tlru_qhat_blocks=tlru_qhat_blocks,
    )
    return pool


# ---------------------------------------------------------------------------
# Tests for KVCacheBlock.is_tel_safe field
# ---------------------------------------------------------------------------

class TestKVCacheBlockTelSafe:
    """Verify the is_tel_safe field behaves correctly on KVCacheBlock."""

    def test_default_is_false(self):
        block = KVCacheBlock(block_id=0)
        assert block.is_tel_safe is False

    def test_can_be_set_true(self):
        block = KVCacheBlock(block_id=0)
        block.is_tel_safe = True
        assert block.is_tel_safe is True

    def test_can_be_reset(self):
        block = KVCacheBlock(block_id=0)
        block.is_tel_safe = True
        block.is_tel_safe = False
        assert block.is_tel_safe is False


# ---------------------------------------------------------------------------
# Tests for BlockPool T-LRU initialisation
# ---------------------------------------------------------------------------

class TestBlockPoolTlruInit:
    """Verify that BlockPool sets up T-LRU state correctly."""

    def test_tlru_disabled_by_default(self):
        pool = _make_pool(num_blocks=10, tlru_xi_blocks=None)
        assert pool.tlru_xi_blocks is None
        assert isinstance(pool.tel_safe_queue, deque)
        assert len(pool.tel_safe_queue) == 0

    def test_tlru_enabled_stores_parameters(self):
        pool = _make_pool(num_blocks=10, tlru_xi_blocks=5, tlru_qhat_blocks=2)
        assert pool.tlru_xi_blocks == 5
        assert pool.tlru_qhat_blocks == 2

    def test_tel_safe_queue_is_empty_initially(self):
        pool = _make_pool(num_blocks=10, tlru_xi_blocks=4, tlru_qhat_blocks=1)
        assert len(pool.tel_safe_queue) == 0


# ---------------------------------------------------------------------------
# Tests for BlockPool.free_blocks_tlru()
# ---------------------------------------------------------------------------

class TestFreeBlocksTlru:
    """Test the T-LRU routing logic in free_blocks_tlru()."""

    def _free_blocks_via_tlru(
        self,
        num_blocks: int,
        xi: int,
        qhat: int,
        req_total_blocks: int,
    ):
        """
        Allocate `num_blocks` from a fresh pool, then free them using
        free_blocks_tlru().  Returns (pool, freed_blocks).

        NOTE: BlockPool always reserves block-0 as the null_block, so a pool
        created with num_gpu_blocks=N has only N-1 usable blocks.  We
        therefore create the pool with num_blocks+1 so that exactly
        `num_blocks` can be allocated.
        """
        pool = _make_pool(
            num_blocks=num_blocks + 1,  # +1 for null_block
            tlru_xi_blocks=xi,
            tlru_qhat_blocks=qhat,
        )
        blocks = pool.get_new_blocks(num_blocks)
        pool.free_blocks_tlru(blocks, req_total_blocks)
        return pool, blocks

    def test_most_blocks_tel_safe_when_history_long_and_xi_tight(self):
        """
        When H is long and xi (SLA threshold) is tight, only the tail blocks
        beyond the prefix cap B are TEL-safe.

        TEL-safe cap  B = max(0, H + Q_hat - xi)
        H=10, qhat=2, xi=5  =>  B = max(0, 10+2-5) = 7

        Blocks at positions 7..9 (3 blocks) are TEL-safe (the suffix beyond B).
        Blocks at positions 0..6 (7 blocks) are TEL-unsafe (the critical prefix).
        """
        H = 10  # total blocks for the request
        xi = 5
        qhat = 2
        pool, blocks = self._free_blocks_via_tlru(H, xi, qhat, H)

        threshold = max(0, H + qhat - xi)  # = 7
        tel_safe_count = sum(1 for b in blocks if b.is_tel_safe)
        expected_tel_safe = H - threshold  # = 3
        assert tel_safe_count == expected_tel_safe
        assert len(pool.tel_safe_queue) == expected_tel_safe

    def test_all_blocks_tel_safe_when_xi_loose(self):
        """
        When xi (SLA threshold) is very large, the prefix cap B=max(0,H+Q-xi)
        is 0, meaning ALL blocks are TEL-safe (the next turn is short enough
        that even recomputing everything stays within the SLA).

        H=2, qhat=1, xi=10  =>  B = max(0, 2+1-10) = 0
        All 2 blocks are at position >= 0 == B, so all are TEL-safe.
        """
        H = 2
        xi = 10
        qhat = 1
        pool, blocks = self._free_blocks_via_tlru(H, xi, qhat, H)

        threshold = max(0, H + qhat - xi)  # = 0
        expected_tel_safe = H - threshold   # = 2
        tel_safe_count = sum(1 for b in blocks if b.is_tel_safe)
        assert tel_safe_count == expected_tel_safe
        assert len(pool.tel_safe_queue) == expected_tel_safe

    def test_mixed_routing_correct_split(self):
        """Verify the exact split between tel_safe_queue and normal queue."""
        H = 8
        xi = 6
        qhat = 2
        pool, blocks = self._free_blocks_via_tlru(H, xi, qhat, H)

        # threshold = max(0, 8+2-6) = 4
        threshold = 4
        expected_tel_safe = H - threshold  # 4
        expected_normal = threshold         # 4

        assert len(pool.tel_safe_queue) == expected_tel_safe
        # Count blocks NOT marked tel_safe
        normal_count = sum(1 for b in blocks if not b.is_tel_safe)
        assert normal_count == expected_normal

    def test_tlru_disabled_all_go_to_normal_queue(self):
        """When T-LRU is disabled (xi=None), free_blocks_tlru raises an
        AssertionError (callers should use free_blocks() instead)."""
        pool = _make_pool(num_blocks=9, tlru_xi_blocks=None)  # 9 total, 8 usable
        blocks = pool.get_new_blocks(8)
        # free_blocks_tlru must raise when T-LRU is disabled
        with pytest.raises(AssertionError):
            pool.free_blocks_tlru(blocks, req_total_blocks=8)


# ---------------------------------------------------------------------------
# Tests for BlockPool.get_new_blocks() priority ordering
# ---------------------------------------------------------------------------

class TestGetNewBlocksPriority:
    """Verify that get_new_blocks() drains tel_safe_queue before normal queue."""

    def test_tel_safe_blocks_allocated_first(self):
        """
        Seed the tel_safe_queue manually, then call get_new_blocks().
        The returned blocks should be the TEL-safe ones.
        """
        # 11 total blocks, 10 usable (null_block takes 1)
        pool = _make_pool(
            num_blocks=11,
            tlru_xi_blocks=5,
            tlru_qhat_blocks=0,
        )
        # H=10, qhat=0, xi=5  => B=max(0,10+0-5)=5, num_tel_safe=5
        # blocks[5..9] (suffix) are TEL-safe
        blocks = pool.get_new_blocks(10)
        pool.free_blocks_tlru(blocks, req_total_blocks=10)

        assert len(pool.tel_safe_queue) == 5

        # Now allocate 5 blocks – they should come from tel_safe_queue
        new_blocks = pool.get_new_blocks(5)
        # TEL-safe queue should be drained
        assert len(pool.tel_safe_queue) == 0
        # Returned blocks should have been is_tel_safe reset to False by pool
        assert all(not b.is_tel_safe for b in new_blocks)

    def test_free_block_count_includes_tel_safe(self):
        """get_num_free_blocks() must count both queues."""
        # 13 total blocks, 12 usable (null_block takes 1)
        pool = _make_pool(
            num_blocks=13,
            tlru_xi_blocks=6,
            tlru_qhat_blocks=0,
        )
        # Initially all 12 usable blocks are in free_block_queue
        assert pool.get_num_free_blocks() == 12

        # Drain and re-free with TLRU routing
        blocks = pool.get_new_blocks(12)
        pool.free_blocks_tlru(blocks, req_total_blocks=12)

        # Total should still be 12 across both queues
        assert pool.get_num_free_blocks() == 12


# ---------------------------------------------------------------------------
# Regression test: touch() must not crash for TEL-safe blocks
# ---------------------------------------------------------------------------

class TestTouchTelSafeBlock:
    """touch() must correctly handle blocks sitting in tel_safe_queue.

    Before the fix, touch() always called free_block_queue.remove(block),
    which raises RuntimeError for TEL-safe blocks because they are held in
    tel_safe_queue (a plain deque) and have no linked-list pointers.
    """

    def test_touch_tel_safe_block_does_not_crash(self):
        """Cache hit on a TEL-safe block must not raise RuntimeError."""
        # 6 total blocks, 5 usable (null_block takes 1).
        # xi=3, qhat=0  =>  B = max(0, 5+0-3) = 2, num_tel_safe = 3
        pool = _make_pool(num_blocks=6, tlru_xi_blocks=3, tlru_qhat_blocks=0)
        blocks = pool.get_new_blocks(5)
        pool.free_blocks_tlru(blocks, req_total_blocks=5)

        tel_safe_blocks = [b for b in blocks if b.is_tel_safe]
        assert len(tel_safe_blocks) > 0, "Pre-condition: need at least one TEL-safe block"

        # Simulate a prefix-cache hit: touch should NOT raise RuntimeError.
        pool.touch(tel_safe_blocks[:1])

        # After touch the block is no longer in tel_safe_queue.
        assert tel_safe_blocks[0] not in pool.tel_safe_queue
        # The TEL-safe tag must be cleared.
        assert tel_safe_blocks[0].is_tel_safe is False
        # ref_cnt must have been incremented.
        assert tel_safe_blocks[0].ref_cnt == 1

    def test_touch_normal_block_still_works(self):
        """Sanity-check that touch() still works for normal (non-TEL-safe) blocks."""
        pool = _make_pool(num_blocks=6, tlru_xi_blocks=1, tlru_qhat_blocks=0)
        blocks = pool.get_new_blocks(5)
        # xi=1 is very tight: B = max(0, 5+0-1) = 4, num_tel_safe = 1
        # Most blocks are normal (non-TEL-safe).
        pool.free_blocks_tlru(blocks, req_total_blocks=5)

        normal_blocks = [b for b in blocks if not b.is_tel_safe]
        assert len(normal_blocks) > 0, "Pre-condition: need at least one non-TEL-safe block"

        # touch() must work as before for normal blocks.
        pool.touch(normal_blocks[:1])
        assert normal_blocks[0].ref_cnt == 1


# ---------------------------------------------------------------------------
# Regression: re-touched TEL-safe block must not be silently discarded from
# get_new_blocks() — it should remain tracked until freed normally.
# ---------------------------------------------------------------------------

class TestGetNewBlocksRetouchedTelSafe:
    """
    When a TEL-safe block gets a cache hit (touch()) while it is sitting in
    tel_safe_queue, its ref_cnt > 0.  If get_new_blocks() later iterates
    over tel_safe_queue and encounters that block, the old code silently
    discarded it, shrinking the effective free-block count and eventually
    causing an under-count.  The fix clears is_tel_safe and leaves the block
    in-use (ref_cnt > 0); it will re-enter a free queue when freed normally.
    """

    def test_retouched_block_not_lost(self):
        """
        After touching a TEL-safe block, get_new_blocks() must not count
        it as a newly allocated block and should not reduce free-block count
        by more than the number of blocks actually handed out.
        """
        # 7 total: 1 null + 6 usable.
        # xi=3, qhat=0  =>  B = max(0, 6+0-3) = 3, num_tel_safe = 3
        pool = _make_pool(num_blocks=7, tlru_xi_blocks=3, tlru_qhat_blocks=0)
        blocks = pool.get_new_blocks(6)
        pool.free_blocks_tlru(blocks, req_total_blocks=6)

        tel_safe = [b for b in blocks if b.is_tel_safe]
        assert len(tel_safe) == 3

        # Simulate a cache hit on one TEL-safe block.
        pool.touch(tel_safe[:1])
        touched_block = tel_safe[0]
        assert touched_block.ref_cnt == 1
        assert touched_block.is_tel_safe is False  # cleared by touch()
        assert touched_block not in pool.tel_safe_queue

        # Now ask for 2 blocks. tel_safe_queue has 2 remaining free blocks.
        # The re-touched block was already removed by touch(), so this is safe.
        free_before = pool.get_num_free_blocks()
        new_blocks = pool.get_new_blocks(2)
        assert len(new_blocks) == 2
        assert pool.get_num_free_blocks() == free_before - 2


# ---------------------------------------------------------------------------
# Regression: reset_prefix_cache() must drain tel_safe_queue.
# ---------------------------------------------------------------------------

class TestResetPrefixCacheWithTelSafe:
    """reset_prefix_cache() must clear tel_safe_queue and restore is_tel_safe=False."""

    def test_reset_drains_tel_safe_queue(self):
        """After a cache reset, tel_safe_queue must be empty."""
        pool = _make_pool(num_blocks=7, tlru_xi_blocks=3, tlru_qhat_blocks=0,
                          enable_caching=False)
        blocks = pool.get_new_blocks(6)
        pool.free_blocks_tlru(blocks, req_total_blocks=6)

        assert len(pool.tel_safe_queue) > 0

        ok = pool.reset_prefix_cache()
        assert ok
        assert len(pool.tel_safe_queue) == 0
        # All blocks should have is_tel_safe cleared.
        for b in blocks:
            assert b.is_tel_safe is False

    def test_free_block_count_correct_after_reset(self):
        """After reset, get_num_free_blocks() must equal all usable blocks."""
        pool = _make_pool(num_blocks=7, tlru_xi_blocks=3, tlru_qhat_blocks=0,
                          enable_caching=False)
        blocks = pool.get_new_blocks(6)
        pool.free_blocks_tlru(blocks, req_total_blocks=6)

        # Before reset: 6 free (2 in free_block_queue + 3 in tel_safe_queue +
        # 1 was mistakenly labelled null at init, but actually 6 usable).
        assert pool.get_num_free_blocks() == 6

        pool.reset_prefix_cache()
        # After reset all 6 usable blocks should be free in one queue.
        assert pool.get_num_free_blocks() == 6


# ---------------------------------------------------------------------------
# Tests for CacheConfig T-LRU fields
# ---------------------------------------------------------------------------

class TestCacheConfigTlruFields:
    """Confirm CacheConfig exposes tlru_xi_tokens and tlru_qhat_tokens."""

    def test_default_values(self):
        from vllm.config.cache import CacheConfig
        cfg = CacheConfig()
        assert cfg.tlru_xi_tokens is None
        assert cfg.tlru_qhat_tokens == 200

    def test_custom_values(self):
        from vllm.config.cache import CacheConfig
        cfg = CacheConfig(tlru_xi_tokens=1024, tlru_qhat_tokens=300)
        assert cfg.tlru_xi_tokens == 1024
        assert cfg.tlru_qhat_tokens == 300

    def test_tlru_not_in_compute_hash(self):
        """T-LRU knobs must NOT affect CacheConfig.compute_hash()."""
        from vllm.config.cache import CacheConfig
        base = CacheConfig()
        with_tlru = CacheConfig(tlru_xi_tokens=512, tlru_qhat_tokens=100)
        assert base.compute_hash() == with_tlru.compute_hash()
