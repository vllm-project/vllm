# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the tiering admission-policy framework (RFC #53485).

Covers the TieringAdmissionPolicy interface, the AlwaysAdmitPolicy and
BackpressureAdmissionPolicy implementations, the AdmissionPolicyFactory
registry, and the manager wiring that consults the policy.
"""

import time
from collections.abc import Iterable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.admission import (
    AdmissionPolicyFactory,
    AlwaysAdmitPolicy,
    BackpressureAdmissionPolicy,
    TieringAdmissionPolicy,
)
from vllm.v1.kv_offload.tiering.backpressure import EMABackpressureDetector
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

_CTX = ReqContext(req_id="test")
_MOCK_OFFLOADING_SPEC = MagicMock()
_BP_HIGH_WATER_S = 1000.0
_BP_LOW_WATER_S = 500.0


def to_keys(int_ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    mock = MagicMock()
    view = memoryview(torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy())
    mock.create_kv_memoryview.return_value = view
    return mock


def _make_detector(**kwargs) -> EMABackpressureDetector:
    defaults = dict(
        high_water_s=_BP_HIGH_WATER_S,
        low_water_s=_BP_LOW_WATER_S,
    )
    defaults.update(kwargs)
    return EMABackpressureDetector(**defaults)


def _pressured_detector() -> EMABackpressureDetector:
    bp = _make_detector()
    bp._under_pressure = True
    bp.store_latency_ema = _BP_HIGH_WATER_S * 4
    return bp


class _FakeTier:
    """Minimal tier stand-in exposing only what the policy reads."""

    def __init__(self, detector=None, tier_type="fake"):
        self.bp_detector = detector
        self.tier_type = tier_type
        self.block_size_bytes = 16


def _make_job(
    job_id: int = 7,
    keys: list[OffloadKey] | None = None,
    is_promotion: bool = False,
) -> TransferJob:
    keys = keys if keys is not None else to_keys([1, 2])
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.arange(len(keys), dtype=np.int32),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


class TestAlwaysAdmitPolicy:
    def test_admits_cascade(self):
        policy = AlwaysAdmitPolicy()
        assert policy.should_admit(to_keys([1]), 0, False) is True

    def test_admits_promotion(self):
        policy = AlwaysAdmitPolicy()
        assert policy.should_admit(to_keys([1]), 0, True) is True

    def test_admits_any_tier_index(self):
        policy = AlwaysAdmitPolicy()
        for tier_idx in (0, 1, 5):
            assert policy.should_admit(to_keys([1]), tier_idx, False) is True

    def test_lifecycle_hooks_are_noops(self):
        policy = AlwaysAdmitPolicy()
        job = _make_job()
        policy.on_admitted(job)
        policy.on_completed(job, JobResult(job_id=job.job_id, success=True))
        policy.reset()

    def test_get_stats_returns_none(self):
        assert AlwaysAdmitPolicy().get_stats() is None


class TestBackpressureAdmissionPolicy:
    def test_admits_when_no_detector(self):
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=None)])
        assert policy.should_admit(to_keys([1]), 0, False) is True

    def test_admits_healthy_tier(self):
        bp = _make_detector()
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        assert policy.should_admit(to_keys([1]), 0, False) is True

    def test_rejects_pressured_tier(self):
        policy = BackpressureAdmissionPolicy(
            [_FakeTier(detector=_pressured_detector())]
        )
        assert policy.should_admit(to_keys([1]), 0, False) is False

    def test_rejection_records_drop_in_detector_policy(self):
        bp = _pressured_detector()
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        assert bp.policy.pop_stores_dropped() == (0, 0)
        policy.should_admit(to_keys([1, 2, 3]), 0, False)
        assert bp.policy.pop_stores_dropped() == (1, 3)

    def test_promotion_admitted_under_pressure(self):
        policy = BackpressureAdmissionPolicy(
            [_FakeTier(detector=_pressured_detector())]
        )
        assert policy.should_admit(to_keys([1]), 0, True) is True

    def _admitted_meta(self, policy, tier_idx=0):
        from vllm.v1.kv_offload.tiering.manager import JobMetadata

        job = _make_job()
        meta = JobMetadata(job, tier_idx)
        policy.on_admitted(meta)
        return meta

    def test_on_completed_updates_detector_on_success(self):
        bp = _make_detector()
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        meta = self._admitted_meta(policy)
        meta.transfer_job.submit_time = time.monotonic() - 0.01
        policy.on_completed(
            meta, JobResult(job_id=meta.transfer_job.job_id, success=True)
        )
        assert bp.store_latency_ema > 0.0

    def test_on_completed_uses_transfer_bytes_when_reported(self):
        bp = _make_detector()
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        meta = self._admitted_meta(policy)
        meta.transfer_job.submit_time = time.monotonic() - 0.01
        with patch.object(bp, "update") as mock_update:
            policy.on_completed(
                meta,
                JobResult(
                    job_id=meta.transfer_job.job_id,
                    success=True,
                    transfer_bytes=1234,
                ),
            )
        mock_update.assert_called_once()
        assert mock_update.call_args[0][1] == 1234

    def test_on_completed_ignores_failed_job(self):
        bp = _make_detector()
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        meta = self._admitted_meta(policy)
        policy.on_completed(
            meta, JobResult(job_id=meta.transfer_job.job_id, success=False)
        )
        assert bp.store_latency_ema == 0.0

    def test_on_completed_ignores_promotion(self):
        bp = _make_detector()
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        job = _make_job(is_promotion=True)
        from vllm.v1.kv_offload.tiering.manager import JobMetadata

        meta = JobMetadata(job, 0)
        policy.on_admitted(meta)
        policy.on_completed(meta, JobResult(job_id=job.job_id, success=True))
        assert bp.store_latency_ema == 0.0

    def test_on_completed_ignores_unknown_job(self):
        bp = _make_detector()
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        from vllm.v1.kv_offload.tiering.manager import JobMetadata

        meta = JobMetadata(_make_job(job_id=999), 0)
        # Never admitted: must not raise and must not touch the detector.
        policy.on_completed(meta, JobResult(job_id=999, success=True))
        assert bp.store_latency_ema == 0.0

    def test_reset_clears_detectors(self):
        bp = _pressured_detector()
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp)])
        policy.reset()
        assert bp.store_latency_ema == 0.0
        assert bp.is_under_pressure() is False

    def test_get_stats_none_when_empty(self):
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=None)])
        assert policy.get_stats() is None

    def test_get_stats_reports_ema_and_drops(self):
        bp = _pressured_detector()
        bp.store_latency_ema = 2.5
        bp.policy._stores_dropped = 5
        bp.policy._blocks_dropped = 12
        policy = BackpressureAdmissionPolicy([_FakeTier(detector=bp, tier_type="t")])
        stats = policy.get_stats()
        assert stats is not None
        reduced = stats.reduce()
        ema_key = TieringOffloadingMetrics.BACKPRESSURE_STORE_LATENCY_EMA
        stores_key = TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED
        blocks_key = TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED
        assert reduced[f"{ema_key}:('1:t',)"] == pytest.approx(2.5)
        assert reduced[f"{stores_key}:('1:t',)"] == 5
        assert reduced[f"{blocks_key}:('1:t',)"] == 12
        # Drop counters are popped; the EMA gauge reflects live state
        # and persists across reads.
        reduced2 = policy.get_stats().reduce()
        assert f"{ema_key}:('1:t',)" in reduced2
        assert f"{stores_key}:('1:t',)" not in reduced2
        assert f"{blocks_key}:('1:t',)" not in reduced2

    def test_build_metric_definitions_has_backpressure_keys(self):
        defs = BackpressureAdmissionPolicy.build_metric_definitions()
        assert TieringOffloadingMetrics.BACKPRESSURE_STORE_LATENCY_EMA in defs
        assert TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED in defs
        assert TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED in defs


class TestAdmissionPolicyFactory:
    def test_always_resolves(self):
        assert AdmissionPolicyFactory.get_policy_class("always") is AlwaysAdmitPolicy

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown admission policy"):
            AdmissionPolicyFactory.get_policy_class("nope")

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            AdmissionPolicyFactory.register_policy(
                "always",
                "vllm.v1.kv_offload.tiering.admission.always",
                "AlwaysAdmitPolicy",
            )

    def test_register_and_resolve_custom_policy(self):
        AdmissionPolicyFactory.register_policy(
            "test-always-copy",
            "vllm.v1.kv_offload.tiering.admission.always",
            "AlwaysAdmitPolicy",
        )
        try:
            cls = AdmissionPolicyFactory.get_policy_class("test-always-copy")
            assert issubclass(cls, TieringAdmissionPolicy)
        finally:
            del AdmissionPolicyFactory._registry["test-always-copy"]

    def test_backpressure_not_registered_by_default(self):
        # BackpressureAdmissionPolicy needs constructed tier instances, so
        # it is intentionally absent from the name registry.
        with pytest.raises(ValueError, match="Unknown admission policy"):
            AdmissionPolicyFactory.get_policy_class("backpressure")


class _WiringTier(SecondaryTierManager):
    """Secondary tier stub for manager-wiring tests.

    Completions are reported on the next get_finished_jobs() poll with
    configurable success.
    """

    def __init__(self, offloading_spec, primary_kv_view, detector=None):
        super().__init__(
            offloading_spec,
            primary_kv_view,
            "wiring",
            backpressure_detector=detector,
        )
        self.blocks: dict[OffloadKey, bool] = {}
        self._pending: list[JobResult] = []
        self.next_success = True

    def lookup(self, key, req_context):
        return LookupResult.HIT if key in self.blocks else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        for key in job_metadata.keys:
            self.blocks[key] = True
        self._pending.append(
            JobResult(job_id=job_metadata.job_id, success=self.next_success)
        )

    def submit_load(self, job_metadata: TransferJob) -> None:
        self._pending.append(
            JobResult(job_id=job_metadata.job_id, success=self.next_success)
        )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        result = self._pending
        self._pending = []
        return result

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def drain_jobs(self):
        pass

    def has_pending_work(self):
        return bool(self._pending)

    def get_num_blocks(self):
        return len(self.blocks)


class TestAdmissionPolicyWiring:
    """The manager consults its admission policy on every submission."""

    @pytest.fixture
    def setup(self):
        mock_region = _mock_mmap_region(20)
        self.primary = CPUPrimaryTierOffloadingManager(
            num_chunks=20, mmap_region=mock_region
        )
        mock_view = mock_region.create_kv_memoryview()
        self.detector = _make_detector()
        self.tier = _WiringTier(
            _MOCK_OFFLOADING_SPEC, mock_view, detector=self.detector
        )
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary,
            secondary_tiers=[self.tier],
            admission_policy=BackpressureAdmissionPolicy([self.tier]),
        )

    def _start_request(self, ctx=_CTX):
        if ctx.req_id not in self.manager._req_state:
            self.manager.on_new_request(ctx)

    def _store_blocks(self, keys, ctx=_CTX):
        self._start_request(ctx)
        result = self.manager.prepare_store(keys, ctx)
        assert result is not None
        self.manager.complete_store(keys, ctx, success=True)

    def _pressure_on(self):
        # NOTE: _completions stays below warmup so is_under_pressure()
        # returns the flag directly without applying idle decay.
        self.detector._under_pressure = True
        self.detector.store_latency_ema = _BP_HIGH_WATER_S * 4

    def _simulate_on_schedule_end(self):
        # Simulate a fresh scheduler step: the per-step poll gate was
        # reset after the previous step, so this poll actually runs.
        self.manager._processed_jobs_this_step = False
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        self.manager.on_schedule_end(ctx)

    def test_default_policy_is_always_admit(self, setup):
        manager = TieringOffloadingManager(
            primary_tier=self.primary,
            secondary_tiers=[self.tier],
        )
        assert isinstance(manager._admission_policy, AlwaysAdmitPolicy)
        # Even with the detector pressured, the default policy admits.
        self._pressure_on()
        keys = to_keys([1])
        manager.on_new_request(_CTX)
        manager.prepare_store(keys, _CTX)
        manager.complete_store(keys, _CTX, success=True)
        assert all(k in self.tier.blocks for k in keys)

    def test_rejected_cascade_consumes_no_job(self, setup):
        self._pressure_on()
        keys = to_keys([1, 2])
        self._store_blocks(keys)
        assert self.manager._transfer_jobs == {}
        assert all(k not in self.tier.blocks for k in keys)

    def test_rejected_store_leaves_primary_unpinned(self, setup):
        self._pressure_on()
        keys = to_keys([3])
        self._store_blocks(keys)
        block = self.primary._policy.get(keys[0])
        assert block is not None
        assert block.ref_cnt == 0

    def test_promotion_admitted_under_pressure(self, setup):
        self._pressure_on()
        key = to_keys([4])[0]
        self.tier.blocks[key] = True
        result = self.manager.lookup(key, _CTX)
        assert result is LookupResult.HIT_PENDING

    def test_completed_cascade_feeds_detector(self, setup):
        bp = self.detector
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        keys = to_keys([5])
        self._store_blocks(keys)
        for _, meta in self.manager._jobs.items():
            meta.transfer_job.submit_time = time.monotonic() - 0.01
        self._simulate_on_schedule_end()
        assert bp.store_latency_ema > 0.0

    def test_failed_cascade_does_not_feed_detector(self, setup):
        bp = self.detector
        bp._completions = EMABackpressureDetector.DEFAULT_WARMUP_COMPLETIONS
        self.tier.next_success = False
        keys = to_keys([6])
        self._store_blocks(keys)
        self._simulate_on_schedule_end()
        assert bp.store_latency_ema == 0.0

    def test_reset_cache_resets_policy_detectors(self, setup):
        self._pressure_on()
        self.manager.reset_cache()
        assert self.detector.store_latency_ema == 0.0
        assert self.detector.is_under_pressure() is False

    def test_get_stats_includes_backpressure_metrics(self, setup):
        self._pressure_on()
        self.detector.policy._stores_dropped = 2
        self.detector.policy._blocks_dropped = 4
        stats = self.manager.get_stats()
        assert stats is not None
        reduced = stats.reduce()
        stores_key = TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED
        blocks_key = TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED
        assert reduced[f"{stores_key}:('1:wiring',)"] == 2
        assert reduced[f"{blocks_key}:('1:wiring',)"] == 4

    def test_admitted_job_ids_stay_sequential_after_rejection(self, setup):
        self._pressure_on()
        self._store_blocks(to_keys([7]))
        counter_before = self.manager._job_id_counter
        assert self.manager._transfer_jobs == {}
        self.detector._under_pressure = False
        keys = to_keys([8])
        self._start_request()
        assert self.manager.prepare_store(keys, _CTX) is not None
        self.manager.complete_store(keys, _CTX, success=True)
        # The rejected store consumed no job ID: the next admitted
        # job takes the exact counter value from before.
        assert list(self.manager._transfer_jobs.keys()) == [counter_before]

    def test_p2p_rejection_reported_as_miss(self, setup):
        from vllm.v1.kv_offload.tiering.p2p.session.server import (
            ServerRole,
            _ActiveLookup,
        )

        server = ServerRole.__new__(ServerRole)
        server.add_stored_blocks = MagicMock()
        lookup = _ActiveLookup(
            lookup_id=1,
            kv_request_id="req-1",
            ctx=_CTX,
            keys=to_keys([9, 10]),
        )

        class _RejectingParent:
            def create_store_job(self, keys, ctx):
                return None

        ServerRole._pin_and_register_hits(
            server, lookup, lookup.keys, _RejectingParent()
        )
        assert lookup.resolved == {k: False for k in lookup.keys}
        server.add_stored_blocks.assert_not_called()
