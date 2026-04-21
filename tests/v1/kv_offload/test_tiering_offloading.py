# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TieringOffloadingManager and DummySecondaryTier.

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

from vllm.v1.kv_offload.abstract import (
    JobMetadata,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec
from vllm.v1.kv_offload.secondary_tiers.dummy import DummySecondaryTier
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext()


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


class TestDummySecondaryTier:
    """Tests for DummySecondaryTier implementation."""

    def test_basic_store_and_lookup(self):
        """Test basic store and lookup operations."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=10)

        # Initially empty
        blocks = to_keys(range(3))
        assert tier.lookup(blocks) == 0

        # Store blocks (simulate with direct insertion for testing)
        tier.blocks[blocks[0]] = True
        tier.blocks[blocks[1]] = True

        # Lookup should find 2 blocks
        assert tier.lookup(blocks) == 2

        # Third block not present
        assert tier.lookup([blocks[2]]) == 0

    def test_in_flight_blocks_return_none(self):
        """Test that in-flight blocks cause lookup to return None."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=10)

        blocks = to_keys(range(3))

        # Mark first block as in-flight
        tier.in_flight[blocks[0]] = 1

        # Lookup should return None (retry later)
        assert tier.lookup(blocks) is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=3)

        # Fill tier to capacity
        blocks = to_keys(range(3))
        for block in blocks:
            tier.blocks[block] = True

        assert tier.get_num_blocks() == 3

        # Touch first block (make it most recently used)
        tier.touch([blocks[0]])

        # Store new block should evict blocks[1] (least recently used)
        new_block = to_keys([3])[0]

        mock_tensor = torch.zeros((4, 16), dtype=torch.float32)
        tier.set_primary_view(memoryview(mock_tensor.numpy()))

        tier.submit_store(
            JobMetadata(
                job_id=1,
                keys=[new_block],
                spec=CPULoadStoreSpec([0]),
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
        tier = DummySecondaryTier(tier_name="Test", max_blocks=10, simulate_async=True)

        blocks = to_keys(range(2))

        mock_tensor = torch.zeros((10, 16), dtype=torch.float32)
        tier.set_primary_view(memoryview(mock_tensor.numpy()))

        # Submit store job
        tier.submit_store(
            JobMetadata(
                job_id=1,
                keys=blocks,
                spec=CPULoadStoreSpec([0, 1]),
            )
        )

        # Blocks should be in-flight
        assert tier.get_num_in_flight() == 2
        assert tier.get_num_blocks() == 0

        # First get_finished() should complete the job
        completed = list(tier.get_finished())
        assert len(completed) == 1
        assert completed[0].job_id == 1
        assert completed[0].success is True

        # Blocks should now be stored
        assert tier.get_num_blocks() == 2
        assert tier.get_num_in_flight() == 0


class TestTieringOffloadingManager:
    """Tests for TieringOffloadingManager."""

    @pytest.fixture
    def manager_setup(self):
        # Create primary tier (CPU-based)
        self.primary_tier = CPUPrimaryTierOffloadingManager(num_blocks=5)

        mock_arr = torch.zeros((5, 16), dtype=torch.int8).numpy()
        self.primary_tier.create_kv_memoryview = lambda: memoryview(mock_arr)

        # Create secondary tiers
        self.secondary_tier1 = DummySecondaryTier(tier_name="Storage", max_blocks=10)
        self.secondary_tier2 = DummySecondaryTier(tier_name="Network", max_blocks=10)

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
        self.manager.complete_store(blocks, success=True)

        # Blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

    def test_cascade_to_all_secondary_tiers(self, manager_setup):
        """Test that blocks are cascaded to ALL secondary tiers."""
        blocks = to_keys(range(3))

        # Store to primary
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store (triggers cascade)
        self.manager.complete_store(blocks, success=True)

        # Process finished jobs to complete cascade
        self.manager._process_finished_jobs()

        # Blocks should be in both secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3

        # Verify blocks are present
        assert self.secondary_tier1.lookup(blocks) == 3
        assert self.secondary_tier2.lookup(blocks) == 3

    def test_ref_cnt_protection_during_cascade(self, manager_setup):
        """Test that ref_cnt protects blocks during cascade."""
        blocks = to_keys(range(3))

        # Store to primary
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, success=True)

        # After complete_store, blocks should have ref_cnt > 0
        # (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            # ref_cnt should be 2 (one for each secondary tier)
            assert block.ref_cnt == 2

        # Process finished jobs to complete cascade
        self.manager._process_finished_jobs()

        # After cascade completes, ref_cnt should be 0
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 0

    def test_lookup_from_primary(self, manager_setup):
        """Test lookup when blocks are in primary tier."""
        blocks = to_keys(range(3))

        # Store blocks
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, success=True)

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

        # Process finished jobs to complete promotion
        self.manager._process_finished_jobs()

        # Now blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

        # Next lookup should succeed
        assert count_hits(self.manager, blocks) == 3

    def test_partial_lookup(self, manager_setup):
        """Test lookup with partial hits."""
        blocks = to_keys(range(5))

        # Store first 3 blocks to primary
        self.manager.prepare_store(blocks[:3], _CTX)
        self.manager.complete_store(blocks[:3], success=True)

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
        self.manager.complete_store(blocks, success=True)

        # Process finished jobs to release ref_cnt from cascade
        self.manager._process_finished_jobs()

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
        self.manager.complete_store(blocks, success=True)
        self.manager._process_finished_jobs()

        # Touch blocks
        self.manager.touch(blocks)

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

        # Prepare store
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store with failure
        self.manager.complete_store(blocks, success=False)

        # Process finished jobs
        self.manager._process_finished_jobs()

        # Blocks should NOT be in secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 0
        assert self.secondary_tier2.get_num_blocks() == 0

    def test_multiple_secondary_tiers_independent_eviction(self):
        """Test that secondary tiers manage their own evictions."""
        # Create tier with small capacity
        small_tier = DummySecondaryTier(
            tier_name="SmallStorage", max_blocks=5, simulate_async=False
        )
        large_tier = DummySecondaryTier(
            tier_name="LargeStorage", max_blocks=10, simulate_async=False
        )

        # Create a fresh primary tier for this test
        primary_tier = CPUPrimaryTierOffloadingManager(num_blocks=10)

        mock_arr = torch.zeros((10, 16), dtype=torch.int8).numpy()
        primary_tier.create_kv_memoryview = lambda: memoryview(mock_arr)

        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[small_tier, large_tier],
        )

        # First, store 5 blocks to fill the small tier
        blocks1 = to_keys(range(5))
        result = manager.prepare_store(blocks1, _CTX)
        assert result is not None
        manager.complete_store(blocks1, success=True)
        manager._process_finished_jobs()

        # Both tiers should have 5 blocks
        assert small_tier.get_num_blocks() == 5
        assert large_tier.get_num_blocks() == 5

        # Now store 3 more blocks - small tier should evict 3 blocks
        blocks2 = to_keys(range(5, 8))
        result = manager.prepare_store(blocks2, _CTX)
        assert result is not None
        manager.complete_store(blocks2, success=True)
        manager._process_finished_jobs()

        # Small tier should still have 5 blocks (evicted 3, added 3)
        assert small_tier.get_num_blocks() == 5

        # Large tier should have all 8 blocks
        assert large_tier.get_num_blocks() == 8

    def test_prepare_store_processes_finished_jobs_first(self, manager_setup):
        """Test that prepare_store() calls _process_finished_jobs() first."""
        blocks = to_keys(range(3))

        # Store blocks
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, success=True)

        # Blocks should have ref_cnt = 2 (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 2

        # Call prepare_store again (should process finished jobs first)
        more_blocks = to_keys(range(3, 5))
        self.manager.prepare_store(more_blocks, _CTX)

        # Original blocks should now have ref_cnt = 0
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 0

    def test_req_context_propagated_to_submit_load(self, manager_setup):
        """Test that req_context from lookup() is forwarded to submit_load."""
        block = to_keys([0])[0]
        self.secondary_tier1.blocks[block] = True  # simulate prior cascade

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )
        ctx = ReqContext(kv_transfer_params={"priority": "high"})
        self.manager.lookup(block, ctx)

        self.secondary_tier1.submit_load.assert_called_once()
        job_metadata = self.secondary_tier1.submit_load.call_args[0][0]
        assert job_metadata.req_context.kv_transfer_params == {"priority": "high"}


class TestTieringOffloadingWithoutSecondaryTiers:
    """Test TieringOffloadingManager with no secondary tiers (backward compat)."""

    def test_works_without_secondary_tiers(self):
        """Test that manager works with empty secondary_tiers list."""
        primary_tier = CPUPrimaryTierOffloadingManager(num_blocks=5)

        mock_arr = torch.zeros((5, 16), dtype=torch.int8).numpy()
        primary_tier.create_kv_memoryview = lambda: memoryview(mock_arr)

        # Create manager with no secondary tiers
        manager = TieringOffloadingManager(
            primary_tier=primary_tier, secondary_tiers=[]
        )

        blocks = to_keys(range(3))

        # Should work like a regular OffloadingManager
        result = manager.prepare_store(blocks, _CTX)
        assert result is not None
        manager.complete_store(blocks, success=True)

        assert count_hits(manager, blocks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
