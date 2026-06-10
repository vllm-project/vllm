# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TieringOffloadingManager and ExampleSecondaryTierManager.

These tests verify:
1. Basic tiered offloading operations (store, load, lookup)
2. Cascade behavior (blocks stored to all secondary tiers)
3. Promotion behavior (blocks loaded from secondary to primary to GPU)
4. ref_cnt management (blocks protected during async transfers)
5. Eviction coordination between tiers
"""

from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="test")
_MOCK_OFFLOADING_SPEC = MagicMock()


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


class TestExampleSecondaryTierManager:
    """Tests for ExampleSecondaryTierManager implementation."""

    def test_basic_store_and_lookup(self):
        """Test basic store and lookup operations."""
        mock_view = memoryview(torch.zeros((10, 16), dtype=torch.int8).numpy())
        tier = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
            custom_param=67,
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
        self.secondary_tier1 = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )
        self.secondary_tier2 = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )

        # Create tiered manager
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary_tier,
            secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
        )

    def _simulate_on_schedule_end(self):
        """Simulate end of scheduler step: lifecycle flush + drain events."""
        self.manager.on_schedule_end()
        list(self.manager.take_events())

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
        # prepare_store() above (setting the per-step flag), so on_schedule_end()
        # does NOT poll get_finished_jobs() again — cascade completions remain
        # unprocessed until the next step.
        self._simulate_on_schedule_end()

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
        self._simulate_on_schedule_end()

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
        self._simulate_on_schedule_end()

        # End of step 2: processes the completed promotion jobs
        self._simulate_on_schedule_end()

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
        self._simulate_on_schedule_end()

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
        self._simulate_on_schedule_end()

        self.secondary_tier1.touch = MagicMock(wraps=self.secondary_tier1.touch)
        self.secondary_tier2.touch = MagicMock(wraps=self.secondary_tier2.touch)

        # Touch blocks
        self.manager.touch(blocks, _CTX)

        # Verify touch was called on primary tier (check LRU order)
        primary_keys = list(self.primary_tier._policy.blocks.keys())
        assert primary_keys[-3:] == list(reversed(blocks))

        # Verify touch was propagated to all secondary tiers
        self.secondary_tier1.touch.assert_called_once_with(blocks, _CTX)
        self.secondary_tier2.touch.assert_called_once_with(blocks, _CTX)

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

    def test_lookup_batches_submit_load_per_request(self, manager_setup):
        """lookup() defers submit_load until on_schedule_end(), one per request.

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
        self._simulate_on_schedule_end()

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

        self._simulate_on_schedule_end()

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

    def test_on_new_request_lifecycle(self, manager_setup):
        """Policy defaults to BLOCK_LEVEL, escalates when a tier requests it,
        and is cleaned up on on_request_finished."""
        # Default: all tiers return BLOCK_LEVEL
        ctx = ReqContext(req_id="req_policy_lifecycle")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.BLOCK_LEVEL
        self.manager.on_request_finished(ctx)

        # Escalate: tier1 requests REQUEST_LEVEL
        self.secondary_tier1.on_new_request = (
            lambda req_context: RequestOffloadingContext(
                policy=OffloadPolicy.REQUEST_LEVEL
            )
        )

        ctx = ReqContext(req_id="req_policy_lifecycle_2")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.REQUEST_LEVEL
        assert ctx.req_id in self.manager._request_level_tiers

        # Cleanup
        self.manager.on_request_finished(ctx)
        assert ctx.req_id not in self.manager._request_level_tiers

    def test_prepare_store_cascades_existing_blocks_to_request_level_tiers(
        self, manager_setup
    ):
        """prepare_store cascades hit blocks to request-level tiers only."""
        # Store some blocks to primary first
        existing_blocks = to_keys(range(3))
        result = self.manager.prepare_store(existing_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(existing_blocks, _CTX, success=True)
        # Drain cascade completions
        self._simulate_on_schedule_end()

        # Make tier1 request-level, tier2 stays block-level
        self.secondary_tier1.on_new_request = (
            lambda req_context: RequestOffloadingContext(
                policy=OffloadPolicy.REQUEST_LEVEL
            )
        )

        ctx = ReqContext(req_id="req_cascade")
        self.manager.on_new_request(ctx)

        # Spy on submit_store
        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Call prepare_store with existing + new blocks
        new_blocks = to_keys(range(3, 5))
        all_blocks = existing_blocks + new_blocks
        result = self.manager.prepare_store(all_blocks, ctx)
        assert result is not None
        assert set(result.keys_to_store) == set(new_blocks)

        # Only tier1 (request-level) should get existing blocks cascaded now.
        # New blocks are cascaded to ALL tiers later via complete_store().
        self.secondary_tier1.submit_store.assert_called_once()
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert set(job_metadata.keys) == set(existing_blocks)

        # tier2 (block-level) does not get existing blocks here.
        self.secondary_tier2.submit_store.assert_not_called()


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
