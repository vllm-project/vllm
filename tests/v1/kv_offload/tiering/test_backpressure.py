# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for back-pressure detection in TieringOffloadingManager.

These tests use a DelayedSecondaryTierManager that holds completed jobs
until explicitly released, allowing precise control over when the manager
observes store completions and their apparent latency.
"""

import time
from collections.abc import Iterable
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.backpressure import (
    DropStorePolicy,
    EMABackpressureDetector,
)
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
    TieringOffloadingMetrics,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_BP_EMA_ALPHA = EMABackpressureDetector.DEFAULT_ALPHA
_BP_WARMUP = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
# Water marks in s/MiB, scaled for the test's tiny 16-byte blocks.
_BP_HIGH_WATER_S = 1000.0
_BP_LOW_WATER_S = 500.0

_CTX = ReqContext(req_id="test")
_MOCK_OFFLOADING_SPEC = MagicMock()


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    mock = MagicMock()
    view = memoryview(torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy())
    mock.create_kv_memoryview.return_value = view
    return mock


def to_keys(int_ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


class DelayedSecondaryTierManager(SecondaryTierManager):
    """Secondary tier that holds completed store jobs until released.

    Jobs are stored immediately but their completion is not reported
    by get_finished_jobs() until release_jobs() is called. This lets
    tests control the apparent latency of store operations by adjusting
    submit_time on the JobMetadata before triggering the completion poll.
    """

    def __init__(
        self, offloading_spec, primary_kv_view, tier_type, backpressure_detector=None
    ):
        super().__init__(
            offloading_spec,
            primary_kv_view,
            tier_type,
            backpressure_detector=backpressure_detector,
        )
        self.blocks: dict[OffloadKey, bool] = {}
        self._held_jobs: list[JobResult] = []
        self._released_jobs: list[JobResult] = []
        self._request_policy: OffloadPolicy | None = None

    def lookup(self, key, req_context):
        return LookupResult.HIT if key in self.blocks else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        for key in job_metadata.keys:
            self.blocks[key] = True
        self._held_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def submit_load(self, job_metadata: TransferJob) -> None:
        self._released_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def get_finished_jobs(self) -> Iterable[JobResult]:
        result = self._released_jobs
        self._released_jobs = []
        return result

    def release_jobs(self):
        self._released_jobs.extend(self._held_jobs)
        self._held_jobs.clear()

    def on_new_request(self, req_context):
        if self._request_policy is not None:
            return RequestOffloadingContext(policy=self._request_policy)
        return RequestOffloadingContext()

    def drain_jobs(self):
        self.release_jobs()

    def has_pending_work(self):
        return bool(self._held_jobs)

    def get_num_blocks(self):
        return len(self.blocks)


class TestDefaultConfig:
    def test_fs_gets_local_watermarks(self):
        cfg = EMABackpressureDetector.default_config("fs")
        assert cfg["high_water_s"] == EMABackpressureDetector.LOCAL_HIGH_WATER_S
        assert cfg["low_water_s"] == EMABackpressureDetector.LOCAL_LOW_WATER_S

    def test_obj_gets_network_watermarks(self):
        cfg = EMABackpressureDetector.default_config("obj")
        assert cfg["high_water_s"] == EMABackpressureDetector.NETWORK_HIGH_WATER_S
        assert cfg["low_water_s"] == EMABackpressureDetector.NETWORK_LOW_WATER_S

    def test_p2p_gets_local_watermarks(self):
        cfg = EMABackpressureDetector.default_config("p2p")
        assert cfg["high_water_s"] == EMABackpressureDetector.LOCAL_HIGH_WATER_S
        assert cfg["low_water_s"] == EMABackpressureDetector.LOCAL_LOW_WATER_S

    def test_fs_with_remote_locality_gets_network_watermarks(self):
        cfg = EMABackpressureDetector.default_config("fs", locality="REMOTE")
        assert cfg["high_water_s"] == EMABackpressureDetector.NETWORK_HIGH_WATER_S
        assert cfg["low_water_s"] == EMABackpressureDetector.NETWORK_LOW_WATER_S

    def test_fs_with_local_locality_gets_local_watermarks(self):
        cfg = EMABackpressureDetector.default_config("fs", locality="LOCAL")
        assert cfg["high_water_s"] == EMABackpressureDetector.LOCAL_HIGH_WATER_S
        assert cfg["low_water_s"] == EMABackpressureDetector.LOCAL_LOW_WATER_S


class TestIdleDecay:
    """Tests for EMA idle decay — recovery from pressure without completions."""

    def _make_detector(self, **kwargs):
        defaults = dict(
            high_water_s=_BP_HIGH_WATER_S,
            low_water_s=_BP_LOW_WATER_S,
            decay_half_life_s=2.0,
        )
        defaults.update(kwargs)
        return EMABackpressureDetector(**defaults)

    def test_ema_decays_while_under_pressure(self):
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._ema = _BP_HIGH_WATER_S * 2
        bp._under_pressure = True
        bp._last_update = time.monotonic()

        with patch("time.monotonic", return_value=bp._last_update + 4.1):
            # ~2.05 half-lives → EMA * 0.24, well below low water (500)
            assert bp.is_under_pressure() is False
            assert bp.store_latency_ema < _BP_LOW_WATER_S

    def test_ema_stays_pressured_if_not_enough_decay(self):
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._ema = _BP_HIGH_WATER_S * 4
        bp._under_pressure = True
        bp._last_update = time.monotonic()

        with patch("time.monotonic", return_value=bp._last_update + 2.0):
            # 2s = 1 half-life → EMA * 0.5 = HIGH * 2, still above LOW
            assert bp.is_under_pressure() is True
            assert bp.store_latency_ema == pytest.approx(_BP_HIGH_WATER_S * 2, rel=0.01)

    def test_no_decay_before_warmup(self):
        bp = self._make_detector()
        bp._ema = _BP_HIGH_WATER_S * 2
        bp._under_pressure = True
        bp._last_update = time.monotonic()

        with patch("time.monotonic", return_value=bp._last_update + 10.0):
            # Still in warmup (_completions=0 < _warmup_completions),
            # so decay should not be applied in is_under_pressure.
            assert bp.is_under_pressure() is True
            assert bp.store_latency_ema == _BP_HIGH_WATER_S * 2

    def test_no_decay_applied_during_new_sample(self):
        """on_store_completed applies the new sample directly without
        idle decay — decay only runs in is_under_pressure()."""
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._ema = _BP_HIGH_WATER_S * 2
        bp._under_pressure = True
        t0 = time.monotonic()
        bp._last_update = t0

        block_bytes = 16
        with patch("time.monotonic", return_value=t0 + 4.0):
            bp.on_store_completed(0.0, block_bytes)
            # No decay in on_store_completed: EMA = alpha*0 + (1-alpha)*2000
            expected = (1 - _BP_EMA_ALPHA) * _BP_HIGH_WATER_S * 2
            assert bp.store_latency_ema == pytest.approx(expected, rel=0.01)

    def test_reset_clears_last_update(self):
        bp = self._make_detector()
        bp._last_update = time.monotonic()
        bp.reset()
        assert bp._last_update == 0.0


class TestBackpressure:
    @pytest.fixture
    def setup(self):
        mock_region = _mock_mmap_region(20)
        self.primary = CPUPrimaryTierOffloadingManager(
            num_blocks=20, mmap_region=mock_region
        )
        mock_view = mock_region.create_kv_memoryview()
        self.tier = DelayedSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="delayed",
            backpressure_detector=EMABackpressureDetector(
                high_water_s=_BP_HIGH_WATER_S,
                low_water_s=_BP_LOW_WATER_S,
            ),
        )
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary,
            secondary_tiers=[self.tier],
        )

    def _start_request(self, ctx=_CTX):
        if ctx.req_id not in self.manager._req_state:
            self.manager.on_new_request(ctx)

    def _store_blocks(self, keys, ctx=_CTX):
        self._start_request(ctx)
        result = self.manager.prepare_store(keys, ctx)
        assert result is not None
        self.manager.complete_store(keys, ctx, success=True)

    def _simulate_on_schedule_end(self):
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        self.manager.on_schedule_end(ctx)

    def _backdate_held_jobs(self, age_s: float):
        """Set submit_time on all in-flight store jobs so they appear
        to have taken ``age_s`` seconds when completed."""
        self.manager._processed_jobs_this_step = False
        now = time.monotonic()
        for job_id, meta in self.manager._jobs.items():
            tj = meta.transfer_job
            if not tj.is_promotion and tj.submit_time > 0:
                tj.submit_time = now - age_s

    def test_ema_updates_on_store_completion(self, setup):
        bp = self.tier.bp_detector
        fast_latency = 0.01
        # EMA is in s/MiB; with 16-byte blocks, scale = MiB / block_bytes.
        scale = EMABackpressureDetector._MIB / self.tier.block_size_bytes

        # During warmup, EMA stays at 0.
        for i in range(_BP_WARMUP - 1):
            keys = to_keys([300 + i])
            self._store_blocks(keys)
            self._backdate_held_jobs(fast_latency)
            self.tier.release_jobs()
            self._simulate_on_schedule_end()
            assert bp.store_latency_ema == 0.0

        # Final warmup sample seeds EMA with the mean.
        keys = to_keys([400])
        self._store_blocks(keys)
        self._backdate_held_jobs(fast_latency)
        self.tier.release_jobs()
        self._simulate_on_schedule_end()
        fast_s_per_mib = fast_latency * scale
        assert bp.store_latency_ema == pytest.approx(fast_s_per_mib, rel=0.1)

        # After warmup, EMA updates normally.
        new_latency = 0.05
        keys = to_keys([401])
        self._store_blocks(keys)
        self._backdate_held_jobs(new_latency)
        self.tier.release_jobs()
        self._simulate_on_schedule_end()
        new_s_per_mib = new_latency * scale
        expected = _BP_EMA_ALPHA * new_s_per_mib + (1 - _BP_EMA_ALPHA) * fast_s_per_mib
        assert bp.store_latency_ema == pytest.approx(expected, rel=0.1)

    def test_pressure_activates_above_high_water(self, setup):
        bp = self.tier.bp_detector

        # Warm up the detector with fast completions first.
        for i in range(_BP_WARMUP):
            keys = to_keys([200 + i])
            self._store_blocks(keys)
            self._backdate_held_jobs(0.001)
            self.tier.release_jobs()
            self._simulate_on_schedule_end()

        keys = to_keys([100])
        self._store_blocks(keys)
        self._simulate_on_schedule_end()

        # 0.1s for a single 16-byte block → ~6553 s/MiB, well above high water.
        self._backdate_held_jobs(0.1)
        self.tier.release_jobs()
        self._simulate_on_schedule_end()

        assert bp.store_latency_ema > _BP_HIGH_WATER_S
        assert bp.is_under_pressure() is True

    def test_pressure_clears_below_low_water(self, setup):
        bp = self.tier.bp_detector
        bp.store_latency_ema = _BP_HIGH_WATER_S * 2
        bp._under_pressure = True
        bp._completions = _BP_WARMUP

        # Drive several fast completions to bring EMA below low water.
        block_id = 100
        while bp.store_latency_ema >= _BP_LOW_WATER_S:
            keys = to_keys([block_id])
            block_id += 1
            self._store_blocks(keys)
            # Pressure is on, so the cascade was skipped. Force-submit
            # a store job manually to get a completion to feed the EMA.
            job_meta = self.manager.create_store_job(keys, _CTX)
            job_meta.submit_time = time.monotonic() - 0.001
            self.tier.submit_store(job_meta)
            self.tier.release_jobs()
            self._simulate_on_schedule_end()

        assert bp.is_under_pressure() is False

    def test_hysteresis_prevents_oscillation(self, setup):
        bp = self.tier.bp_detector
        bp._completions = _BP_WARMUP
        block_bytes = self.tier.block_size_bytes

        # Set EMA between low and high water marks (in s/MiB).
        mid = (_BP_LOW_WATER_S + _BP_HIGH_WATER_S) / 2
        bp.store_latency_ema = mid
        bp._last_update = time.monotonic()

        # Raw seconds that produce `mid` s/MiB for a given number of blocks.
        def mid_latency_s(num_blocks: int) -> float:
            return mid * (num_blocks * block_bytes / EMABackpressureDetector._MIB)

        # When not under pressure, mid-range EMA should not activate.
        bp._under_pressure = False
        keys = to_keys(range(2))
        self._store_blocks(keys)
        self._backdate_held_jobs(mid_latency_s(len(keys)))
        self.tier.release_jobs()
        self._simulate_on_schedule_end()
        assert bp.is_under_pressure() is False

        # When under pressure, mid-range EMA should not deactivate.
        bp._under_pressure = True
        bp._last_update = time.monotonic()
        keys2 = to_keys(range(10, 12))
        # Force-submit since pressure is on (cascade would skip).
        for k in keys2:
            self.primary.prepare_store([k], _CTX)
            self.primary.complete_store([k], _CTX)
        job_meta = self.manager.create_store_job(keys2, _CTX)
        job_meta.submit_time = time.monotonic() - mid_latency_s(len(keys2))
        self.tier.submit_store(job_meta)
        self.tier.release_jobs()
        self._simulate_on_schedule_end()
        assert bp.is_under_pressure() is True

    def test_stores_skipped_under_pressure(self, setup):
        bp = self.tier.bp_detector
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 4
        initial_blocks = self.tier.get_num_blocks()

        keys = to_keys(range(50, 53))
        self._store_blocks(keys)

        # Blocks should be in primary but NOT in secondary.
        for k in keys:
            assert self.primary.lookup(k, _CTX) is LookupResult.HIT
        assert self.tier.get_num_blocks() == initial_blocks

    def test_stores_resume_after_pressure_clears(self, setup):
        bp = self.tier.bp_detector

        # Start under pressure — stores skipped.
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 4
        keys1 = to_keys(range(60, 62))
        self._store_blocks(keys1)
        assert all(k not in self.tier.blocks for k in keys1)

        # Clear pressure — next store should cascade.
        bp._under_pressure = False
        keys2 = to_keys(range(70, 72))
        self._store_blocks(keys2)
        assert all(k in self.tier.blocks for k in keys2)

    def test_dropped_store_count_tracked(self, setup):
        bp = self.tier.bp_detector
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 4
        policy = bp.policy
        assert policy.pop_stores_dropped() == (0, 0)

        self._store_blocks(to_keys([80]))
        assert policy._stores_dropped == 1
        assert policy._blocks_dropped == 1

        self._store_blocks(to_keys([81, 82]))
        assert policy._stores_dropped == 2
        assert policy._blocks_dropped == 3

    def test_metrics_reported_via_get_stats(self, setup):
        bp = self.tier.bp_detector
        bp._under_pressure = True
        bp.store_latency_ema = 2.5
        policy = bp.policy
        policy._stores_dropped = 5
        policy._blocks_dropped = 12

        stats = self.manager.get_stats()
        assert stats is not None
        reduced = stats.reduce()

        ema_key = TieringOffloadingMetrics.BACKPRESSURE_STORE_LATENCY_EMA
        stores_key = TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED
        blocks_key = TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED
        assert reduced[f"{ema_key}:('1:delayed',)"] == pytest.approx(2.5)
        assert reduced[f"{stores_key}:('1:delayed',)"] == 5
        assert reduced[f"{blocks_key}:('1:delayed',)"] == 12

        # Dropped counts reset after get_stats.
        assert policy._stores_dropped == 0
        assert policy._blocks_dropped == 0

    def test_reset_cache_clears_backpressure(self, setup):
        bp = self.tier.bp_detector
        bp.store_latency_ema = 5.0
        bp._under_pressure = True

        self.manager.reset_cache()

        assert bp.store_latency_ema == 0.0
        assert bp.is_under_pressure() is False

    def test_request_level_tier_respects_backpressure(self, setup):
        """Request-level cascade also skips pressured tiers."""
        # Store blocks in primary first.
        existing = to_keys(range(3))
        self._store_blocks(existing)
        self.tier.release_jobs()
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        # Make tier request-level for a new request.
        self.tier._request_policy = OffloadPolicy.REQUEST_LEVEL
        ctx = ReqContext(req_id="req_rl_bp")
        self.manager.on_new_request(ctx)

        # Activate pressure.
        bp = self.tier.bp_detector
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 4
        initial_blocks = self.tier.get_num_blocks()

        # prepare_store with existing + new blocks.
        new = to_keys(range(10, 12))
        all_keys = existing + new
        result = self.manager.prepare_store(all_keys, ctx)
        assert result is not None

        # Existing blocks would normally cascade to request-level tier,
        # but should be skipped under pressure.
        assert self.tier.get_num_blocks() == initial_blocks

    def test_loads_continue_under_pressure(self, setup):
        """Loads from a pressured tier still work."""
        blocks = to_keys(range(3))
        for b in blocks:
            self.tier.blocks[b] = True

        bp = self.tier.bp_detector
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 4

        # Lookups should still initiate promotion.
        for b in blocks:
            result = self.manager.lookup(b, _CTX)
            assert result is LookupResult.HIT_PENDING


class TestThrottledDropPolicy:
    """Tests for proportional throttling under pressure."""

    def _make_detector(self, **kwargs):
        defaults = dict(
            high_water_s=_BP_HIGH_WATER_S,
            low_water_s=_BP_LOW_WATER_S,
            cooldown_s=0.0,
        )
        defaults.update(kwargs)
        return EMABackpressureDetector(**defaults)

    def test_no_drop_when_not_under_pressure(self):
        bp = self._make_detector()
        for _ in range(20):
            assert bp.should_store(1) is True

    def test_no_drop_at_exactly_high_water(self):
        bp = self._make_detector()
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S
        for _ in range(20):
            assert bp.should_store(1) is True

    def test_partial_drop_at_moderate_pressure(self):
        bp = self._make_detector()
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 2
        results = [bp.should_store(1) for _ in range(20)]
        assert any(results), "some stores should be allowed"
        assert not all(results), "some stores should be dropped"

    def test_full_drop_at_severe_pressure(self):
        bp = self._make_detector()
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 4
        for _ in range(20):
            assert bp.should_store(1) is False

    def test_drop_rate_increases_with_severity(self):
        """Higher EMA → higher drop rate."""
        drop_counts = []
        for multiplier in [1.5, 2.0, 2.5, 3.5]:
            bp = self._make_detector()
            bp._under_pressure = True
            bp.store_latency_ema = _BP_HIGH_WATER_S * multiplier
            drops = sum(1 for _ in range(100) if not bp.should_store(1))
            drop_counts.append(drops)
        for i in range(len(drop_counts) - 1):
            assert drop_counts[i] <= drop_counts[i + 1]

    def test_explicit_drop_store_policy_still_drops_all(self):
        """DropStorePolicy (explicit) still drops everything under pressure."""
        bp = EMABackpressureDetector(
            high_water_s=_BP_HIGH_WATER_S,
            low_water_s=_BP_LOW_WATER_S,
            policy=DropStorePolicy(),
            cooldown_s=0.0,
        )
        bp._under_pressure = True
        bp.store_latency_ema = _BP_HIGH_WATER_S * 1.5
        for _ in range(20):
            assert bp.should_store(1) is False


class TestHealthyBypass:
    """Tests for the fast-path bypass when a tier is consistently healthy."""

    def _make_detector(self, **kwargs):
        defaults = dict(
            high_water_s=_BP_HIGH_WATER_S,
            low_water_s=_BP_LOW_WATER_S,
            cooldown_s=0.0,
        )
        defaults.update(kwargs)
        return EMABackpressureDetector(**defaults)

    def test_not_healthy_initially(self):
        bp = self._make_detector()
        assert bp.is_healthy() is False

    def test_healthy_after_streak(self):
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._last_update = time.monotonic()
        block_bytes = 16
        fast_s_per_mib = _BP_LOW_WATER_S * 0.5
        fast_elapsed = fast_s_per_mib * (block_bytes / EMABackpressureDetector._MIB)
        for _ in range(EMABackpressureDetector._HEALTHY_THRESHOLD + _BP_WARMUP):
            bp.on_store_completed(fast_elapsed, block_bytes)
        assert bp.is_healthy() is True

    def test_healthy_resets_on_high_latency(self):
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._healthy_streak = EMABackpressureDetector._HEALTHY_THRESHOLD
        bp._last_update = time.monotonic()
        block_bytes = 16
        slow_s_per_mib = _BP_HIGH_WATER_S * 2
        slow_elapsed = slow_s_per_mib * (block_bytes / EMABackpressureDetector._MIB)
        bp.on_store_completed(slow_elapsed, block_bytes)
        assert bp.is_healthy() is False

    def test_healthy_bypasses_policy(self):
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._healthy_streak = EMABackpressureDetector._HEALTHY_THRESHOLD
        assert bp.should_store(1) is True

    def test_reset_clears_healthy_streak(self):
        bp = self._make_detector()
        bp._healthy_streak = EMABackpressureDetector._HEALTHY_THRESHOLD
        bp.reset()
        assert bp._healthy_streak == 0
        assert bp.is_healthy() is False


class TestCooldown:
    """Tests for the post-pressure cooldown period."""

    def _make_detector(self, **kwargs):
        defaults = dict(
            high_water_s=_BP_HIGH_WATER_S,
            low_water_s=_BP_LOW_WATER_S,
            cooldown_s=1.0,
        )
        defaults.update(kwargs)
        return EMABackpressureDetector(**defaults)

    def test_cooldown_throttles_stores(self):
        bp = self._make_detector()
        bp._completions = _BP_WARMUP
        bp._under_pressure = True
        bp._ema = _BP_HIGH_WATER_S * 2
        bp._last_update = time.monotonic()

        with patch("time.monotonic") as mock_time:
            t0 = bp._last_update + 10.0
            mock_time.return_value = t0
            assert bp.is_under_pressure() is False
            assert bp._pressure_cleared_at > 0

            mock_time.return_value = t0 + 0.1
            results = [bp.should_store(1) for _ in range(10)]
            assert sum(results) == 5, "cooldown should allow ~50%"

    def test_no_cooldown_when_never_pressured(self):
        bp = self._make_detector()
        for _ in range(20):
            assert bp.should_store(1) is True

    def test_no_cooldown_after_window_expires(self):
        bp = self._make_detector()
        bp._pressure_cleared_at = time.monotonic() - 2.0
        for _ in range(20):
            assert bp.should_store(1) is True

    def test_reset_clears_cooldown(self):
        bp = self._make_detector()
        bp._pressure_cleared_at = time.monotonic()
        bp._cooldown_count = 5
        bp.reset()
        assert bp._pressure_cleared_at == 0.0
        assert bp._cooldown_count == 0
