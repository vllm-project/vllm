# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for KV cache observability utilities (PR #4).

This test suite verifies the read-only KV cache metrics extraction
functions using minimal fakes. Tests are independent of scheduler
implementation and do not require GPU.

Design principles:
- Use fake objects that expose only the interfaces being tested
- No real scheduler/KV cache instantiation (except optional smoke test)
- Deterministic, stable assertions (no heuristics)
- Fast execution, no flakiness
"""
from __future__ import annotations

import pytest

from vllm.v1.core.kv_cache_observability import (
    PerRequestKVMetrics,
    StepKVSummary,
    get_per_request_kv_metrics,
    get_step_kv_summary,
)


# ============================================================================
# Fake objects for testing (stable, minimal interfaces)
# ============================================================================


class FakeRequest:
    """Minimal fake Request with only fields needed for KV metrics."""

    def __init__(
        self,
        request_id: str = "test-req-1",
        num_prompt_tokens: int = 100,
        num_cached_tokens: int = -1,
    ):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_cached_tokens = num_cached_tokens


class FakeSingleTypeManager:
    """Minimal fake for SingleTypeKVCacheManager."""

    def __init__(
        self,
        req_to_blocks: dict[str, list[int]] | None = None,
        num_cached_block: dict[str, int] | None = None,
    ):
        self.req_to_blocks = req_to_blocks or {}
        self.num_cached_block = num_cached_block or {}


class FakeCoordinator:
    """Minimal fake for KVCacheCoordinator."""

    def __init__(self, single_type_managers: tuple | None = None):
        self.single_type_managers = single_type_managers or ()


class FakeKVCacheManager:
    """Minimal fake for KVCacheManager."""

    def __init__(self, coordinator: FakeCoordinator | None = None):
        self.coordinator = coordinator or FakeCoordinator()


class FakeBlockPool:
    """Minimal fake for BlockPool.

    Can selectively include/exclude methods to test fallback behavior.
    """

    def __init__(
        self,
        num_gpu_blocks: int = 101,
        free_blocks: int = 25,
        usage: float = 0.75,
        include_get_usage: bool = True,
        include_get_free: bool = True,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self._free_blocks = free_blocks
        self._usage = usage

        # Conditionally add methods (cleaner than raising AttributeError)
        if include_get_free:
            self.get_num_free_blocks = lambda: self._free_blocks  # type: ignore
        if include_get_usage:
            self.get_usage = lambda: self._usage  # type: ignore


# ============================================================================
# Tests for get_per_request_kv_metrics
# ============================================================================


class TestPerRequestKVMetrics:
    """Unit tests for per-request KV metrics extraction."""

    def test_allocated_blocks_from_req_to_blocks_map(self):
        """Test allocated blocks counted from req_to_blocks map length."""
        # Setup fake manager with known block allocation
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0, 1, 2], "req-2": [3, 4]},
            num_cached_block={"req-1": 1},
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        request = FakeRequest(request_id="req-1")

        # Execute
        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        # Verify
        assert isinstance(metrics, PerRequestKVMetrics)
        assert metrics.blocks_allocated_gpu == 3  # len([0, 1, 2])
        assert metrics.blocks_cached_gpu == 1

    def test_cached_blocks_from_num_cached_block_map(self):
        """Test cached blocks counted from num_cached_block map."""
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0, 1, 2, 3, 4]},
            num_cached_block={"req-1": 3},  # 3 blocks cached
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        request = FakeRequest(request_id="req-1")

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        assert metrics.blocks_allocated_gpu == 5
        assert metrics.blocks_cached_gpu == 3

    def test_aggregates_across_multiple_kv_cache_groups(self):
        """Test aggregation across multiple KV cache groups (e.g., MLA)."""
        # Setup: 2 KV cache groups with different allocations
        manager1 = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0, 1, 2]},  # 3 blocks
            num_cached_block={"req-1": 2},
        )
        manager2 = FakeSingleTypeManager(
            req_to_blocks={"req-1": [10, 11]},  # 2 blocks
            num_cached_block={"req-1": 1},
        )
        coordinator = FakeCoordinator(single_type_managers=(manager1, manager2))
        manager = FakeKVCacheManager(coordinator=coordinator)

        request = FakeRequest(request_id="req-1")

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        # Should aggregate: 3+2=5 allocated, 2+1=3 cached
        assert metrics.blocks_allocated_gpu == 5
        assert metrics.blocks_cached_gpu == 3

    def test_missing_request_id_returns_zeros(self):
        """Test request not in tracking maps returns zero metrics."""
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"other-req": [0, 1, 2]},
            num_cached_block={"other-req": 1},
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        request = FakeRequest(request_id="not-found")

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        # Request not found, should return zeros
        assert metrics.blocks_allocated_gpu == 0
        assert metrics.blocks_cached_gpu == 0

    def test_missing_coordinator_returns_zeros(self):
        """Test missing coordinator returns zero metrics (defensive)."""

        class ManagerNoCoordinator:
            pass

        manager = ManagerNoCoordinator()
        request = FakeRequest()

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        assert metrics.blocks_allocated_gpu == 0
        assert metrics.blocks_cached_gpu == 0

    def test_missing_single_managers_returns_zeros(self):
        """Test missing single_type_managers returns zero metrics."""

        class CoordinatorNoManagers:
            pass

        class ManagerWithBadCoordinator:
            def __init__(self):
                self.coordinator = CoordinatorNoManagers()

        manager = ManagerWithBadCoordinator()
        request = FakeRequest()

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        assert metrics.blocks_allocated_gpu == 0
        assert metrics.blocks_cached_gpu == 0

    def test_effective_prompt_len_with_cache_hit(self):
        """Test effective_prompt_len computed when cache hit occurred."""
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0, 1, 2]},
            num_cached_block={"req-1": 1},
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        # Request with 100 tokens, 16 cached
        request = FakeRequest(
            request_id="req-1", num_prompt_tokens=100, num_cached_tokens=16
        )

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        # Effective prompt = 100 - 16 = 84
        assert metrics.effective_prompt_len == 84

    def test_effective_prompt_len_no_cache_hit(self):
        """Test effective_prompt_len when no cache hit (num_cached_tokens=0)."""
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0, 1, 2]},
            num_cached_block={"req-1": 0},
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        request = FakeRequest(
            request_id="req-1", num_prompt_tokens=100, num_cached_tokens=0
        )

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        # No cache hit, effective = full prompt
        assert metrics.effective_prompt_len == 100

    def test_effective_prompt_len_uninitialized(self):
        """Test effective_prompt_len is None when num_cached_tokens=-1."""
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0, 1, 2]},
            num_cached_block={"req-1": 0},
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        # num_cached_tokens=-1 (uninitialized)
        request = FakeRequest(request_id="req-1", num_cached_tokens=-1)

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        # Uninitialized, should be None
        assert metrics.effective_prompt_len is None

    def test_offload_fields_are_none(self):
        """Test CPU/disk offload fields are None (not implemented in v1)."""
        single_manager = FakeSingleTypeManager(
            req_to_blocks={"req-1": [0]},
            num_cached_block={"req-1": 0},
        )
        coordinator = FakeCoordinator(single_type_managers=(single_manager,))
        manager = FakeKVCacheManager(coordinator=coordinator)

        request = FakeRequest(request_id="req-1")

        metrics = get_per_request_kv_metrics(request, manager)  # type: ignore

        assert metrics.blocks_cpu_offload is None
        assert metrics.blocks_disk_offload is None


# ============================================================================
# Tests for get_step_kv_summary
# ============================================================================


class TestStepKVSummary:
    """Unit tests for step-level KV cache summary."""

    def test_blocks_and_usage_from_fake_pool(self):
        """Test metrics extracted from fake BlockPool with all methods."""
        block_pool = FakeBlockPool(
            num_gpu_blocks=101,
            free_blocks=25,
            usage=0.75,
            include_get_usage=True,
            include_get_free=True,
        )

        summary = get_step_kv_summary(block_pool)  # type: ignore

        assert isinstance(summary, StepKVSummary)
        # Total = 101 - 1 (null block) = 100
        assert summary.blocks_total_gpu == 100
        assert summary.blocks_free_gpu == 25
        assert summary.usage_gpu_ratio == 0.75

    def test_usage_fallback_when_get_usage_missing(self):
        """Test usage computed from free blocks when get_usage missing."""
        block_pool = FakeBlockPool(
            num_gpu_blocks=101,
            free_blocks=25,
            usage=0.75,  # This won't be used
            include_get_usage=False,  # Simulate missing get_usage
            include_get_free=True,
        )

        summary = get_step_kv_summary(block_pool)  # type: ignore

        # Usage computed: 1.0 - (25 / 100) = 0.75
        assert summary.blocks_total_gpu == 100
        assert summary.blocks_free_gpu == 25
        assert summary.usage_gpu_ratio == 0.75

    def test_none_block_pool_returns_zeros(self):
        """Test None block_pool returns zero metrics (defensive)."""
        summary = get_step_kv_summary(None)  # type: ignore

        assert summary.blocks_total_gpu == 0
        assert summary.blocks_free_gpu == 0
        assert summary.usage_gpu_ratio == 0.0

    def test_missing_methods_returns_conservative_defaults(self):
        """Test missing methods returns conservative defaults (0.0 usage)."""
        # BlockPool with only num_gpu_blocks, no methods
        class MinimalBlockPool:
            def __init__(self):
                self.num_gpu_blocks = 50

        minimal_pool = MinimalBlockPool()

        summary = get_step_kv_summary(minimal_pool)  # type: ignore

        # Total blocks available
        assert summary.blocks_total_gpu == 49  # 50 - 1
        # No methods, conservative defaults
        assert summary.blocks_free_gpu == 0
        # Conservative: return 0.0 when unmeasurable (not 1.0)
        assert summary.usage_gpu_ratio == 0.0

    def test_negative_blocks_clamped_to_zero(self):
        """Test negative num_gpu_blocks clamped to zero."""

        class ZeroBlockPool:
            def __init__(self):
                self.num_gpu_blocks = 0  # Edge case

        zero_pool = ZeroBlockPool()

        summary = get_step_kv_summary(zero_pool)  # type: ignore

        # 0 - 1 = -1, but clamped to 0
        assert summary.blocks_total_gpu == 0
        assert summary.blocks_free_gpu == 0
        assert summary.usage_gpu_ratio == 0.0

    def test_malformed_num_gpu_blocks_handled(self):
        """Test malformed num_gpu_blocks handled defensively."""

        class MalformedBlockPool:
            def __init__(self):
                self.num_gpu_blocks = "not-a-number"

        malformed_pool = MalformedBlockPool()

        summary = get_step_kv_summary(malformed_pool)  # type: ignore

        # Should fall back to default 1, then 1-1=0
        assert summary.blocks_total_gpu == 0
        assert summary.blocks_free_gpu == 0
        assert summary.usage_gpu_ratio == 0.0

    def test_offload_fields_are_none(self):
        """Test CPU/disk offload fields are None (not implemented in v1)."""
        block_pool = FakeBlockPool()

        summary = get_step_kv_summary(block_pool)  # type: ignore

        assert summary.blocks_cpu_offload is None
        assert summary.blocks_disk_offload is None


# ============================================================================
# Optional: Single smoke test with real scheduler
# ============================================================================


class TestSmokeWithRealScheduler:
    """Optional smoke test with real scheduler (kept minimal)."""

    def test_functions_work_with_real_scheduler(self):
        """Smoke test: functions work with real scheduler, no crashes."""
        # This is the ONE allowed integration test to verify
        # the helpers work with actual vLLM infrastructure
        from tests.v1.core.utils import create_requests, create_scheduler

        scheduler = create_scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            num_blocks=50,
            block_size=16,
        )

        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]

        scheduler.add_request(request)
        scheduler.schedule()

        # Test per-request metrics (should not crash)
        req_metrics = get_per_request_kv_metrics(request, scheduler.kv_cache_manager)
        assert isinstance(req_metrics, PerRequestKVMetrics)
        assert req_metrics.blocks_allocated_gpu >= 0  # No heuristics

        # Test step summary (should not crash)
        block_pool = scheduler.kv_cache_manager.block_pool
        step_summary = get_step_kv_summary(block_pool)
        assert isinstance(step_summary, StepKVSummary)
        assert step_summary.blocks_total_gpu > 0  # Non-zero pool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
