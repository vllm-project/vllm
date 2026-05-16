# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TieringOffloadingManager and ExampleSecondaryTier.

These tests verify:
1. Basic tiered offloading operations (store, load, lookup)
2. Cascade behavior (blocks stored to all secondary tiers)
3. Promotion behavior (blocks loaded from secondary to primary to GPU)
4. ref_cnt management (blocks protected during async transfers)
5. Eviction coordination between tiers
"""

from collections.abc import Iterable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.example import ExampleSecondaryTier
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="test")
_MOCK_VLLM_CONFIG = MagicMock()


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    """Create a mock SharedOffloadRegion for testing."""
    mock = MagicMock()
    view = memoryview(torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy())
    mock.create_kv_memoryview.return_value = view
    return mock


def to_keys(int_ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def count_hits(manager, keys: list[OffloadKey]) -> int | None:
    """Count consecutive lookup hits from the start of keys.

    Returns the count of leading True results, or None if any lookup
    returns None (retry-later signal).
    """
    count = 0
    for key in keys:
        result = manager.lookup(key, _CTX)
        if result is None:
            return None
        if not result:
            break
        count += 1
    return count


class TestExampleSecondaryTier:
    """Tests for ExampleSecondaryTier implementation."""

    def test_basic_store_and_lookup(self):
        """Test basic store and lookup operations."""
        mock_view = memoryview(torch.zeros((10, 16), dtype=torch.int8).numpy())
        tier = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG, primary_kv_view=mock_view, max_blocks=10
        )

        # Initially empty
        blocks = to_keys(range(3))
        assert tier.lookup(blocks[0], _CTX) is False

        # Store blocks (simulate with direct insertion for testing)
        tier.blocks[blocks[0]] = True
        tier.blocks[blocks[1]] = True

        # Lookup should find first two blocks
        assert tier.lookup(blocks[0], _CTX) is True
        assert tier.lookup(blocks[1], _CTX) is True

        # Third block not present
        assert tier.lookup(blocks[2], _CTX) is False

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        mock_view = memoryview(torch.zeros((4, 16), dtype=torch.int8).numpy())
        tier = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG, primary_kv_view=mock_view, max_blocks=3
        )

        # Fill tier to capacity
        blocks = to_keys(range(3))
        for block in blocks:
            tier.blocks[block] = True

        assert tier.get_num_blocks() == 3

        # Touch first block (make it most recently used)
        tier.touch([blocks[0]], _CTX)

        # Store new block should evict blocks[1] (least recently used)
        new_block = to_keys([3])[0]

        tier.submit_store(
            JobMetadata(
                job_id=1,
                keys=[new_block],
                block_ids=np.array([0], dtype=np.int64),
                is_promotion=False,
                req_context=_CTX,
            )
        )

        # Complete the job
        tier.get_finished()

        # Verify new block is stored and blocks[1] was evicted (LRU)
        assert new_block in tier.blocks
        assert blocks[1] not in tier.blocks
        # blocks[0] and blocks[2] should still be present
        assert blocks[0] in tier.blocks
        assert blocks[2] in tier.blocks

    def test_async_simulation(self):
        """Test simulated async behavior."""
        mock_view = memoryview(torch.zeros((10, 16), dtype=torch.int8).numpy())
        tier = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            max_blocks=10,
            simulate_async=True,
        )

        blocks = to_keys(range(2))

        # Submit store job
        tier.submit_store(
            JobMetadata(
                job_id=1,
                keys=blocks,
                block_ids=np.array([0, 1], dtype=np.int64),
                is_promotion=False,
                req_context=_CTX,
            )
        )

        # Blocks should not yet be stored (pending async completion)
        assert tier.get_num_blocks() == 0

        # First get_finished() should complete the job
        completed = list(tier.get_finished())
        assert len(completed) == 1
        assert completed[0].job_id == 1
        assert completed[0].success is True

        # Blocks should now be stored
        assert tier.get_num_blocks() == 2


class TestTieringOffloadingManager:
    """Tests for TieringOffloadingManager."""

    @pytest.fixture
    def manager_setup(self):
        # Create primary tier (CPU-based)
        mock_region = _mock_mmap_region(5)
        self.primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=5, mmap_region=mock_region
        )

        mock_view = mock_region.create_kv_memoryview()

        # Create secondary tiers with the primary view
        self.secondary_tier1 = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG, primary_kv_view=mock_view, max_blocks=10
        )
        self.secondary_tier2 = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG, primary_kv_view=mock_view, max_blocks=10
        )

        # Create tiered manager
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary_tier,
            secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
        )

    def test_basic_store_to_primary(self, manager_setup):
        """Test basic store operation to primary tier."""
        blocks = to_keys(range(3))

        # Prepare store
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 3

        # Complete store
        self.manager.complete_store(blocks, _CTX, success=True)

        # Blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

    def test_cascade_to_all_secondary_tiers(self, manager_setup):
        """Test that blocks are cascaded to ALL secondary tiers."""
        blocks = to_keys(range(3))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Store to primary
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store (triggers cascade via submit_store on each tier)
        self.manager.complete_store(blocks, _CTX, success=True)

        # submit_store was called once per secondary tier
        self.secondary_tier1.submit_store.assert_called_once()
        self.secondary_tier2.submit_store.assert_called_once()

        # Blocks should be in both secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3

        # Verify blocks are present
        assert all(self.secondary_tier1.lookup(b, _CTX) for b in blocks)
        assert all(self.secondary_tier2.lookup(b, _CTX) for b in blocks)

    def test_ref_cnt_protection_during_cascade(self, manager_setup):
        """Test that ref_cnt protects blocks during cascade."""
        blocks = to_keys(range(3))

        # Store to primary
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)

        # After complete_store, blocks should have ref_cnt > 0
        # (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            # ref_cnt should be 2 (one for each secondary tier)
            assert block.ref_cnt == 2

        # End of step 1: _maybe_process_finished_jobs() was already called by
        # prepare_store() above (setting the per-step flag), so take_events()
        # does NOT poll get_finished() again — cascade completions remain
        # unprocessed until the next step.
        list(self.manager.take_events())

        # ref_cnt still held: cascade jobs finished (sync tier) but haven't
        # been polled yet because the per-step guard skipped the second call.
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 2

        # Secondary tiers have completed jobs waiting to be drained
        assert len(self.secondary_tier1.completed_jobs) > 0
        assert len(self.secondary_tier2.completed_jobs) > 0

        # End of step 2: flag was reset, so _maybe_process_finished_jobs()
        # runs and processes the cascade completions (complete_read → ref_cnt--)
        list(self.manager.take_events())

        # After cascade completes, ref_cnt should be 0
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 0

        # All completed jobs have been drained
        assert len(self.secondary_tier1.completed_jobs) == 0
        assert len(self.secondary_tier2.completed_jobs) == 0

    def test_lookup_from_primary(self, manager_setup):
        """Test lookup when blocks are in primary tier."""
        blocks = to_keys(range(3))

        # Store blocks
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)

        # Lookup should find all blocks in primary
        assert count_hits(self.manager, blocks) == 3

    def test_promotion_from_secondary(self, manager_setup):
        """Test promotion of blocks from secondary to primary tier."""
        blocks = to_keys(range(3))

        # Manually add blocks to secondary tier (simulate previous cascade)
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        # Lookup each block to initiate promotion for all of them
        for block in blocks:
            result = self.manager.lookup(block, _CTX)
            assert result is None  # Retry later (promotion initiated)

        # End of step 1: flushes deferred submit_load() calls
        list(self.manager.take_events())

        # End of step 2: processes the completed promotion jobs
        list(self.manager.take_events())

        # Now blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

        # Next lookup should succeed
        assert count_hits(self.manager, blocks) == 3

    def test_partial_lookup(self, manager_setup):
        """Test lookup with partial hits."""
        blocks = to_keys(range(5))

        # Store first 3 blocks to primary
        self.manager.prepare_store(blocks[:3], _CTX)
        self.manager.complete_store(blocks[:3], _CTX, success=True)

        # Lookup all 5 blocks should return 3 (first 3 found)
        assert count_hits(self.manager, blocks) == 3

    def test_eviction_in_primary_tier(self, manager_setup):
        """Test eviction in primary tier when capacity is exceeded."""
        # Primary tier has capacity of 5 blocks
        # First, fill the primary tier
        blocks = to_keys(range(5))
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 5
        self.manager.complete_store(blocks, _CTX, success=True)

        # End of step: release ref_cnt from cascade
        list(self.manager.take_events())

        # Now try to store 2 more blocks (should trigger eviction)
        more_blocks = to_keys(range(5, 7))
        result = self.manager.prepare_store(more_blocks, _CTX)

        # Should evict 2 blocks from primary tier
        assert result is not None
        assert len(result.evicted_keys) == 2
        assert len(result.keys_to_store) == 2

    def test_touch_propagates_to_all_tiers(self, manager_setup):
        """Test that touch() propagates to all tiers."""
        blocks = to_keys(range(3))

        # Store blocks
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        list(self.manager.take_events())

        # Touch blocks
        self.manager.touch(blocks, _CTX)

        # Verify touch was called on primary tier (check LRU order)
        # In LRU, touched blocks should be at the end
        primary_keys = list(self.primary_tier._policy.blocks.keys())
        assert primary_keys[-3:] == list(reversed(blocks))

        # Verify touch was called on all secondary tiers
        secondary1_keys = list(self.secondary_tier1.blocks.keys())
        assert secondary1_keys[-3:] == list(reversed(blocks))

        secondary2_keys = list(self.secondary_tier2.blocks.keys())
        assert secondary2_keys[-3:] == list(reversed(blocks))

    def test_failed_store_no_cascade(self, manager_setup):
        """Test that failed GPU→primary store doesn't cascade."""
        blocks = to_keys(range(3))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Prepare store
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store with failure — cascade must not happen
        self.manager.complete_store(blocks, _CTX, success=False)

        # submit_store was never called on either secondary tier
        self.secondary_tier1.submit_store.assert_not_called()
        self.secondary_tier2.submit_store.assert_not_called()

    def test_multiple_secondary_tiers_independent_eviction(self):
        """Test that secondary tiers manage their own evictions."""
        mock_region = _mock_mmap_region(10)
        mock_view = mock_region.create_kv_memoryview()

        # Create tier with small capacity
        small_tier = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            max_blocks=5,
            simulate_async=False,
        )
        large_tier = ExampleSecondaryTier(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            max_blocks=10,
            simulate_async=False,
        )

        # Create a fresh primary tier for this test
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=10, mmap_region=mock_region
        )

        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[small_tier, large_tier],
        )

        # First, store 5 blocks to fill the small tier
        blocks1 = to_keys(range(5))
        result = manager.prepare_store(blocks1, _CTX)
        assert result is not None
        manager.complete_store(blocks1, _CTX, success=True)
        list(manager.take_events())

        # Both tiers should have 5 blocks
        assert small_tier.get_num_blocks() == 5
        assert large_tier.get_num_blocks() == 5

        # Now store 3 more blocks - small tier should evict 3 blocks
        blocks2 = to_keys(range(5, 8))
        result = manager.prepare_store(blocks2, _CTX)
        assert result is not None
        manager.complete_store(blocks2, _CTX, success=True)
        list(manager.take_events())

        # Small tier should still have 5 blocks (evicted 3, added 3)
        assert small_tier.get_num_blocks() == 5

        # Large tier should have all 8 blocks
        assert large_tier.get_num_blocks() == 8

    def test_lookup_batches_submit_load_per_request(self, manager_setup):
        """lookup() defers submit_load until take_events(), one call per request.

        Blocks from different requests each get their own submit_load call, each
        carrying the correct req_context.
        """
        blocks = to_keys(range(4))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        ctx_a = ReqContext(req_id="req_a")
        ctx_b = ReqContext(req_id="req_b")

        # All lookups return None: secondary hit triggers promotion (in-flight)
        assert self.manager.lookup(blocks[0], ctx_a) is None
        assert self.manager.lookup(blocks[1], ctx_a) is None
        assert self.manager.lookup(blocks[2], ctx_b) is None
        assert self.manager.lookup(blocks[3], ctx_b) is None

        # submit_load must not fire during lookup - only at end of step
        self.secondary_tier1.submit_load.assert_not_called()

        # simulate end of step
        list(self.manager.take_events())

        assert self.secondary_tier1.submit_load.call_count == 2
        calls = self.secondary_tier1.submit_load.call_args_list
        jm_a = calls[0].args[0]
        jm_b = calls[1].args[0]
        assert set(jm_a.keys) == {blocks[0], blocks[1]}
        assert jm_a.req_context is ctx_a
        assert set(jm_b.keys) == {blocks[2], blocks[3]}
        assert jm_b.req_context is ctx_b

    def test_lookup_shared_block_no_duplicate_promotion(self, manager_setup):
        """A block looked up by two requests in the same step is promoted once.

        The first lookup initiates promotion (returns None via secondary hit).
        The second lookup sees ref_cnt=-1 on the primary slot and returns None
        via the primary in-flight path — without triggering a second promotion.
        """
        shared_block = to_keys([0])[0]
        self.secondary_tier1.blocks[shared_block] = True

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        ctx_a = ReqContext(req_id="req_a")
        ctx_b = ReqContext(req_id="req_b")

        result_a = self.manager.lookup(shared_block, ctx_a)
        result_b = self.manager.lookup(shared_block, ctx_b)

        # Both see None (in-flight), but promotion is only queued once
        assert result_a is None
        assert result_b is None

        list(self.manager.take_events())

        # Only one submit_load call despite two lookups
        self.secondary_tier1.submit_load.assert_called_once()
        job_metadata = self.secondary_tier1.submit_load.call_args.args[0]
        assert list(job_metadata.keys) == [shared_block]
        assert job_metadata.req_context is ctx_a

    def test_complete_store_forwards_req_context_to_submit_store(self, manager_setup):
        """complete_store cascades to secondary tiers with the correct req_context."""
        blocks = to_keys(range(2))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )

        ctx = ReqContext(req_id="req_ctx", kv_transfer_params={"key": "value"})

        self.manager.prepare_store(blocks, ctx)
        self.manager.complete_store(blocks, ctx, success=True)

        assert self.secondary_tier1.submit_store.call_count == 1
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert job_metadata.req_context is ctx


class TestTieringOffloadingWithoutSecondaryTiers:
    """Test TieringOffloadingManager with no secondary tiers (backward compat)."""

    def test_works_without_secondary_tiers(self):
        """Test that manager works with empty secondary_tiers list."""
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=5, mmap_region=_mock_mmap_region(5)
        )

        # Create manager with no secondary tiers
        manager = TieringOffloadingManager(
            primary_tier=primary_tier, secondary_tiers=[]
        )

        blocks = to_keys(range(3))

        # Should work like a regular OffloadingManager
        result = manager.prepare_store(blocks, _CTX)
        assert result is not None
        manager.complete_store(blocks, _CTX, success=True)

        assert count_hits(manager, blocks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
