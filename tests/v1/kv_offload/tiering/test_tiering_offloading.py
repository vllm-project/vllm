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

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingCounterMetadata,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

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

    Returns the count of leading HIT results, or None if any lookup
    returns HIT_PENDING or RETRY.
    """
    count = 0
    for key in keys:
        result = manager.lookup(key, _CTX)
        if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
            return None
        if result is not LookupResult.HIT:
            break
        count += 1
    return count


class MetricsSecondaryTierManager(SecondaryTierManager):
    """Test-only secondary tier that declares and emits one labeled metric."""

    MY_TIER_METRIC = "my_tier_metric"

    @classmethod
    def build_metric_definitions(cls, extra_config):
        return {
            cls.MY_TIER_METRIC: OffloadingCounterMetadata(
                documentation="Number of bytes served by the test tier.",
                labelnames=("tier",),
            )
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats: OffloadingConnectorStats | None = None

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return False

    def submit_store(self, job_metadata: JobMetadata) -> None:
        return

    def submit_load(self, job_metadata: JobMetadata) -> None:
        return

    def get_finished_jobs(self) -> Iterable[JobResult]:
        return ()

    def drain_jobs(self) -> None:
        return

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self.stats
        self.stats = None
        return stats


def test_tiering_spec_collects_secondary_metric_definitions(monkeypatch):
    monkeypatch.setitem(
        SecondaryTierFactory._registry,
        "test_metrics",
        lambda: MetricsSecondaryTierManager,
    )

    metrics = TieringOffloadingSpec.build_metric_definitions(
        {"secondary_tiers": [{"type": "test_metrics"}]}
    )

    metadata = metrics[MetricsSecondaryTierManager.MY_TIER_METRIC]
    assert metadata.documentation == "Number of bytes served by the test tier."
    assert metadata.labelnames == ("tier",)


def test_tiering_manager_aggregates_secondary_stats():
    mock_region = _mock_mmap_region(5)
    primary_tier = CPUPrimaryTierOffloadingManager(
        num_blocks=5, mmap_region=mock_region
    )
    secondary_tier = MetricsSecondaryTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_region.create_kv_memoryview(),
        tier_type="test_metrics",
    )
    secondary_stats = OffloadingConnectorStats()
    secondary_stats.increase_counter(
        MetricsSecondaryTierManager.MY_TIER_METRIC, 7, ("test_metrics",)
    )
    secondary_tier.stats = secondary_stats
    manager = TieringOffloadingManager(
        primary_tier=primary_tier,
        secondary_tiers=[secondary_tier],
    )

    stats = manager.get_stats()

    assert stats is not None
    assert (
        stats.data["data"][MetricsSecondaryTierManager.MY_TIER_METRIC][
            ("test_metrics",)
        ]
        == 7
    )

    # The primary tier's cache-usage gauge is always reported, so get_stats()
    # never returns None, but the secondary tier has nothing new to report
    # once its stats have been consumed.
    second_stats = manager.get_stats()
    assert second_stats is not None
    assert MetricsSecondaryTierManager.MY_TIER_METRIC not in second_stats.data["data"]


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
        assert tier.lookup(blocks[0], _CTX) is LookupResult.MISS

        # Store blocks (simulate with direct insertion for testing)
        tier.blocks[blocks[0]] = True
        tier.blocks[blocks[1]] = True

        # Lookup should find first two blocks
        assert tier.lookup(blocks[0], _CTX) is LookupResult.HIT
        assert tier.lookup(blocks[1], _CTX) is LookupResult.HIT

        # Third block not present
        assert tier.lookup(blocks[2], _CTX) is LookupResult.MISS


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

    def _start_request(self, req_context: ReqContext = _CTX):
        if req_context.req_id not in self.manager._req_state:
            self.manager.on_new_request(req_context)

    def test_basic_store_to_primary(self, manager_setup):
        """Test basic store operation to primary tier."""
        blocks = to_keys(range(3))

        # Prepare store
        self._start_request()
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
        self._start_request()
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
        assert all(
            self.secondary_tier1.lookup(b, _CTX) is LookupResult.HIT for b in blocks
        )
        assert all(
            self.secondary_tier2.lookup(b, _CTX) is LookupResult.HIT for b in blocks
        )

    def test_ref_cnt_protection_during_cascade(self, manager_setup):
        """Test that ref_cnt protects blocks during cascade."""
        blocks = to_keys(range(3))

        # Store to primary
        self._start_request()
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
        self._start_request()
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
            assert result is LookupResult.RETRY  # promotion initiated

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
        self._start_request()
        self.manager.prepare_store(blocks[:3], _CTX)
        self.manager.complete_store(blocks[:3], _CTX, success=True)

        # Lookup all 5 blocks should return 3 (first 3 found)
        assert count_hits(self.manager, blocks) == 3

    def test_eviction_in_primary_tier(self, manager_setup):
        """Test eviction in primary tier when capacity is exceeded."""
        # Primary tier has capacity of 5 blocks
        # First, fill the primary tier
        blocks = to_keys(range(5))
        self._start_request()
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
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        self._simulate_on_schedule_end()
        # for secondary tiers to drain jobs, so primary tier's blocks are evictable.
        self._simulate_on_schedule_end()

        self.secondary_tier1.touch = MagicMock(wraps=self.secondary_tier1.touch)
        self.secondary_tier2.touch = MagicMock(wraps=self.secondary_tier2.touch)

        # Touch blocks
        self.manager.touch(blocks, _CTX)

        # Verify touch was called on primary tier (check LRU order)
        primary_keys = list(self.primary_tier._policy.evictable_blocks.keys())
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
        self._start_request()
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

        # All lookups return RETRY: secondary hit triggers promotion
        assert self.manager.lookup(blocks[0], ctx_a) is LookupResult.RETRY
        assert self.manager.lookup(blocks[1], ctx_a) is LookupResult.RETRY
        assert self.manager.lookup(blocks[2], ctx_b) is LookupResult.RETRY
        assert self.manager.lookup(blocks[3], ctx_b) is LookupResult.RETRY

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

        # First lookup triggers promotion (RETRY), second finds block
        # already in primary with write in-flight (HIT_PENDING).
        assert result_a is LookupResult.RETRY
        assert result_b is LookupResult.HIT_PENDING

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

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.complete_store(blocks, ctx, success=True)

        assert self.secondary_tier1.submit_store.call_count == 1
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert job_metadata.req_context is ctx

    def test_on_request_finished_delays_secondary_until_store_submitted(
        self, manager_setup
    ):
        """Manager hook is eager; secondary hooks wait for cascade submission."""
        blocks = to_keys(range(2))
        ctx = ReqContext(req_id="req_delayed_secondary")
        calls: list[tuple[str, str]] = []

        self.primary_tier.on_request_finished = MagicMock(
            side_effect=lambda req_context: calls.append(
                ("primary_finish", req_context.req_id)
            )
        )

        original_submit_store1 = self.secondary_tier1.submit_store
        original_submit_store2 = self.secondary_tier2.submit_store

        def submit_store1(job_metadata):
            calls.append(("submit_store_1", job_metadata.req_context.req_id))
            return original_submit_store1(job_metadata)

        def submit_store2(job_metadata):
            calls.append(("submit_store_2", job_metadata.req_context.req_id))
            return original_submit_store2(job_metadata)

        self.secondary_tier1.submit_store = MagicMock(side_effect=submit_store1)
        self.secondary_tier2.submit_store = MagicMock(side_effect=submit_store2)
        self.secondary_tier1.on_request_finished = MagicMock(
            side_effect=lambda req_context: calls.append(
                ("secondary_finish_1", req_context.req_id)
            )
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            side_effect=lambda req_context: calls.append(
                ("secondary_finish_2", req_context.req_id)
            )
        )

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.on_request_finished(ctx)

        assert calls == [("primary_finish", ctx.req_id)]
        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.complete_store(blocks, ctx, success=True)

        assert calls == [
            ("primary_finish", ctx.req_id),
            ("submit_store_1", ctx.req_id),
            ("submit_store_2", ctx.req_id),
            ("secondary_finish_1", ctx.req_id),
            ("secondary_finish_2", ctx.req_id),
        ]

    def test_failed_store_finalizes_finished_request(self, manager_setup):
        """Failed primary stores still unblock secondary finalization."""
        blocks = to_keys(range(2))
        ctx = ReqContext(req_id="req_failed_store_finalize")

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )
        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.complete_store(blocks, ctx, success=False)

        self.secondary_tier1.submit_store.assert_not_called()
        self.secondary_tier2.submit_store.assert_not_called()
        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_zero_store_request_finalizes_immediately(self, manager_setup):
        """Requests with no pending stores finalize secondary tiers immediately."""
        ctx = ReqContext(req_id="req_zero_store_finalize")

        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_reset_cache_finalizes_delayed_secondary_request(self, manager_setup):
        """reset_cache abandons pending primary stores and finalizes secondaries."""
        blocks = to_keys(range(2))
        ctx = ReqContext(req_id="req_reset_finalize_secondary")

        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.reset_cache()

        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert self.manager._req_state == {}

    def test_reset_cache_clears_pending_primary_stores_for_active_request(
        self, manager_setup
    ):
        """reset_cache drops active pending stores so resumed requests finalize."""
        initial_blocks = to_keys(range(2))
        resumed_blocks = to_keys(range(2, 4))
        ctx = ReqContext(req_id="req_reset_resume")

        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.prepare_store(initial_blocks, ctx)
        assert self.manager._req_state[ctx.req_id].pending_primary_stores == 1

        self.manager.reset_cache()

        assert ctx.req_id in self.manager._req_state
        assert self.manager._req_state[ctx.req_id].pending_primary_stores == 0
        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.prepare_store(resumed_blocks, ctx)
        self.manager.complete_store(resumed_blocks, ctx, success=True)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_on_new_request_lifecycle(self, manager_setup):
        """Policy defaults to BLOCK_LEVEL, escalates when a tier requests it,
        and is cleaned up on on_request_finished."""
        # Default: all tiers return BLOCK_LEVEL
        ctx = ReqContext(req_id="req_policy_lifecycle")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.BLOCK_LEVEL
        assert self.manager._req_state[ctx.req_id].request_level_tiers is None
        self.manager.on_request_finished(ctx)
        assert ctx.req_id not in self.manager._req_state

        # Escalate: tier1 requests REQUEST_LEVEL
        self.secondary_tier1.on_new_request = lambda req_context: (
            RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        )

        ctx = ReqContext(req_id="req_policy_lifecycle_2")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.REQUEST_LEVEL
        assert self.manager._req_state[ctx.req_id].request_level_tiers == {
            self.secondary_tier1
        }

        # Cleanup
        self.manager.on_request_finished(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_prepare_store_cascades_existing_blocks_to_request_level_tiers(
        self, manager_setup
    ):
        """prepare_store cascades hit blocks to request-level tiers only."""
        # Store some blocks to primary first
        existing_blocks = to_keys(range(3))
        self._start_request()
        result = self.manager.prepare_store(existing_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(existing_blocks, _CTX, success=True)
        # Drain cascade completions
        self._simulate_on_schedule_end()

        # Make tier1 request-level, tier2 stays block-level
        self.secondary_tier1.on_new_request = lambda req_context: (
            RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
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

    def test_reset_cache_clears_orchestrator_state(self, manager_setup):
        """reset_cache wipes every kind of orchestrator state and resets
        primary tier; pending submissions are dropped without being sent
        to the secondary tier. Active request state is retained."""
        # Cascade — populates primary blocks and leaves cascade jobs
        # in _transfer_jobs (the synchronous example tier has already
        # queued completions); reset_cache's drain loop will pick them up.
        blocks = to_keys(range(3))
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        assert self.manager._transfer_jobs

        # Pending promotion submission (deferred — no on_schedule_end after
        # the lookup that staged it).
        promo_block = to_keys([99])[0]
        self.secondary_tier1.blocks[promo_block] = True
        assert (
            self.manager.lookup(promo_block, ReqContext(req_id="pending"))
            is LookupResult.RETRY
        )
        assert self.manager._pending_load_submissions

        # Request-level tier registration.
        self.secondary_tier1.on_new_request = lambda req_context: (
            RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        )
        rl_ctx = ReqContext(req_id="rl")
        self.manager.on_new_request(rl_ctx)
        assert self.manager._req_state[rl_ctx.req_id].request_level_tiers == {
            self.secondary_tier1
        }

        # Mark this step as already polled (reset_cache must clear it).
        self.manager._processed_jobs_this_step = True

        # Spy: pending submission must NOT reach the tier.
        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        self.manager.reset_cache()

        # Orchestrator state cleared.
        assert self.manager._transfer_jobs == {}
        assert self.manager._pending_load_submissions == {}
        assert set(self.manager._req_state) == {_CTX.req_id, rl_ctx.req_id}
        assert self.manager._processed_jobs_this_step is False

        # Primary tier reset to a fresh state.
        assert self.primary_tier._num_allocated_blocks == 0
        assert self.primary_tier._free_list == []
        for block in blocks:
            assert self.primary_tier.lookup(block, _CTX) is LookupResult.MISS

        # Pending submission was dropped, not submitted.
        self.secondary_tier1.submit_load.assert_not_called()

    def test_reset_cache_drains_all_tiers(self, manager_setup):
        """reset_cache must drain each secondary tier before resetting
        the primary tier so no tier I/O is touching primary memory.
        Without the drain, an in-flight transfer could write into, or
        read junk from, a primary slot that the post-reset path has
        reallocated.
        """
        self.secondary_tier1.drain_jobs = MagicMock(
            wraps=self.secondary_tier1.drain_jobs
        )
        self.secondary_tier2.drain_jobs = MagicMock(
            wraps=self.secondary_tier2.drain_jobs
        )

        # Drive a cascade so a job lands in _transfer_jobs.
        blocks = to_keys(range(3))
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        assert self.manager._transfer_jobs

        self.manager.reset_cache()

        self.secondary_tier1.drain_jobs.assert_called_once()
        self.secondary_tier2.drain_jobs.assert_called_once()
        assert self.manager._transfer_jobs == {}


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
        manager.on_new_request(_CTX)
        result = manager.prepare_store(blocks, _CTX)
        assert result is not None
        manager.complete_store(blocks, _CTX, success=True)

        assert count_hits(manager, blocks) == 3


class TestTieringOffloadingMultiGroup:
    """Regression tests for cascade paths when num_groups > 1.

    Both complete_store and _cascade_existing_blocks_to_request_level_tiers
    call prepare_read() which returns one spec per KV group. These tests
    verify that block_ids are reassembled in key order (not group order) before
    being passed to submit_store().
    """

    @pytest.fixture
    def manager_setup_2_groups(self):
        mock_region = _mock_mmap_region(num_blocks=10)
        self.primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=10, mmap_region=mock_region, num_groups=2
        )
        mock_view = mock_region.create_kv_memoryview()
        self.secondary_tier = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary_tier,
            secondary_tiers=[self.secondary_tier],
        )

    def _simulate_on_schedule_end(self):
        self.manager.on_schedule_end()
        list(self.manager.take_events())

    def test_complete_store_cascade_multi_group_key_order(self, manager_setup_2_groups):
        """complete_store cascade: block_ids length matches keys for multi-group stores.

        Interleaved keys from two groups exercise _specs_to_key_order_block_ids.
        ExampleSecondaryTierManager.submit_store asserts len(keys)==len(block_ids),
        catching any misalignment.
        """
        keys_g0 = [make_offload_key(f"hash{i}".encode(), 0) for i in range(3)]
        keys_g1 = [make_offload_key(f"hash{i}".encode(), 1) for i in range(3)]
        # Interleave groups to expose ordering bugs
        all_keys = [
            keys_g0[0],
            keys_g1[0],
            keys_g0[1],
            keys_g1[1],
            keys_g0[2],
            keys_g1[2],
        ]

        result = self.manager.prepare_store(all_keys, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 6

        self.secondary_tier.submit_store = MagicMock(
            wraps=self.secondary_tier.submit_store
        )
        self.manager.complete_store(all_keys, _CTX, success=True)

        self.secondary_tier.submit_store.assert_called_once()
        job = self.secondary_tier.submit_store.call_args.args[0]
        assert len(job.keys) == 6
        assert len(job.block_ids) == 6

    def test_cascade_existing_blocks_request_level_multi_group(
        self, manager_setup_2_groups
    ):
        """Request-level cascade of existing primary blocks: block_ids align with
        keys for multi-group key sets.

        ExampleSecondaryTierManager.submit_store asserts len(keys)==len(block_ids),
        catching any misalignment from _cascade_existing_blocks_to_request_level_tiers.
        """
        keys_g0 = [make_offload_key(f"hash{i}".encode(), 0) for i in range(2)]
        keys_g1 = [make_offload_key(f"hash{i}".encode(), 1) for i in range(2)]
        existing_keys = keys_g0 + keys_g1

        # Pre-populate primary tier
        result = self.manager.prepare_store(existing_keys, _CTX)
        assert result is not None
        self.manager.complete_store(existing_keys, _CTX, success=True)
        # Two schedule-end calls: first resets gate, second drains cascade ref_cnts
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        # Configure secondary tier as request-level
        self.secondary_tier.on_new_request = (
            lambda req_context: RequestOffloadingContext(
                policy=OffloadPolicy.REQUEST_LEVEL
            )
        )
        ctx = ReqContext(req_id="req_multigroup")
        self.manager.on_new_request(ctx)

        self.secondary_tier.submit_store = MagicMock(
            wraps=self.secondary_tier.submit_store
        )

        # All keys already in primary → prepare_store cascades existing blocks
        result = self.manager.prepare_store(existing_keys, ctx)
        assert result is not None
        assert len(result.keys_to_store) == 0

        self.secondary_tier.submit_store.assert_called_once()
        job = self.secondary_tier.submit_store.call_args.args[0]
        assert len(job.keys) == 4
        assert len(job.block_ids) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
