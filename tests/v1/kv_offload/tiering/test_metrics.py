# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering import metrics as tiering_metrics
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    TieringOffloadingMetrics,
)
from vllm.v1.kv_offload.tiering.manager import (
    CascadeTransferJob,
    JobMetadata,
    PromotionTransferJob,
)
from vllm.v1.kv_offload.tiering.metrics import TieringMetricsTracker

_CTX = ReqContext(req_id="test")


def to_keys(int_ids: Iterable[int]):
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def test_tiering_metrics_tracker_records_lookup_metrics():
    tracker = TieringMetricsTracker(
        tier_types=["example", "p2p"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )
    tracker.on_new_request(_CTX)

    key = to_keys([1])[0]
    tracker.on_lookup(
        _CTX,
        key,
        tracker.primary_tier_label,
        LookupResult.MISS,
        lookup_duration=0.0,
    )
    tracker.on_lookup(_CTX, key, tracker.tier_label(0), LookupResult.MISS, 0.0)
    tracker.on_lookup(_CTX, key, tracker.tier_label(1), LookupResult.HIT, 0.0)
    tracker.on_lookup(_CTX, key, tracker.tier_label(1), LookupResult.HIT, 0.0)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("0:primary",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("1:example",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("2:p2p",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_HITS][("2:p2p",)] == 1
    assert ("1:example",) not in values[TieringOffloadingMetrics.CHUNK_HITS]


def test_tiering_metrics_tracker_stops_lookup_metrics_after_allocation():
    tracker = TieringMetricsTracker(
        tier_types=["example"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )
    tracker.on_new_request(_CTX)
    key = to_keys([1])[0]

    tracker.on_lookup(
        _CTX,
        key,
        tracker.primary_tier_label,
        LookupResult.MISS,
        lookup_duration=0.0,
    )
    tracker.on_lookup(_CTX, key, tracker.tier_label(0), LookupResult.HIT, 0.0)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("0:primary",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("1:example",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_HITS][("1:example",)] == 1

    tracker.on_request_allocated(_CTX)
    tracker.on_lookup(
        _CTX,
        key,
        tracker.primary_tier_label,
        LookupResult.MISS,
        lookup_duration=0.0,
    )
    tracker.on_lookup(_CTX, key, tracker.tier_label(0), LookupResult.HIT, 0.0)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert TieringOffloadingMetrics.CHUNK_QUERIES not in values
    assert TieringOffloadingMetrics.CHUNK_HITS not in values


def _register(tracker: TieringMetricsTracker, job: JobMetadata) -> None:
    """Register a job the way production does.

    Promotion chunk counts are recorded at allocation (inside
    _promotion_allocation), which hand-built jobs bypass; simulate that here.
    """
    tracker.on_job_registered(job)
    transfer_job = job.transfer_job
    claim = transfer_job.get()
    if transfer_job.is_promotion and claim is not None:
        tracker.safe_counter_update(
            job.tier_idx,
            TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC,
            len(claim[1]),
        )


def test_tiering_metrics_tracker_records_finished_job_metrics():
    tracker = TieringMetricsTracker(
        tier_types=["example"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )
    cascade_key, promotion_key, failed_cascade_key, failed_promotion_key = to_keys(
        range(4)
    )
    jobs = [
        JobMetadata(
            CascadeTransferJob(0, _CTX, [cascade_key], np.array([0])),
            0,
        ),
        JobMetadata(
            PromotionTransferJob(1, _CTX, [promotion_key], np.array([1])),
            0,
        ),
        JobMetadata(
            CascadeTransferJob(2, _CTX, [failed_cascade_key], np.array([2])),
            0,
        ),
        JobMetadata(
            PromotionTransferJob(3, _CTX, [failed_promotion_key], np.array([3])),
            0,
        ),
    ]
    for job in jobs:
        _register(tracker, job)

    tracker.on_job_finished(
        jobs[0], JobResult(job_id=0, success=True, transfer_time=0.5)
    )
    tracker.on_job_finished(
        jobs[1], JobResult(job_id=1, success=True, transfer_time=0.25)
    )
    tracker.on_job_finished(jobs[2], JobResult(job_id=2, success=False))
    tracker.on_job_finished(jobs[3], JobResult(job_id=3, success=False))

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    label = ("1:example",)
    assert values[TieringOffloadingMetrics.WRITE_BYTES][label] == 16
    assert values[TieringOffloadingMetrics.WRITE_TIME][label] == 0.5
    assert values[TieringOffloadingMetrics.READ_BYTES][label] == 16
    assert values[TieringOffloadingMetrics.READ_TIME][label] == 0.25
    assert values[TieringOffloadingMetrics.CASCADE_JOB_FAILURES][label] == 1
    assert values[TieringOffloadingMetrics.PROMOTION_JOB_FAILURES][label] == 1
    tracker.assert_idle()


def test_tiering_metrics_tracker_records_partial_promotion_success_bytes():
    tracker = TieringMetricsTracker(
        tier_types=["example"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )
    keys = to_keys(range(3))
    job = JobMetadata(
        PromotionTransferJob(0, _CTX, keys, np.array([0, 1, 2])),
        0,
    )
    _register(tracker, job)

    tracker.on_job_finished(
        job,
        JobResult(
            job_id=0,
            success=False,
            successful_keys=(keys[0], keys[2]),
            transfer_time=0.5,
        ),
    )

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    label = ("1:example",)
    assert values[TieringOffloadingMetrics.READ_BYTES][label] == 32
    assert values[TieringOffloadingMetrics.READ_TIME][label] == 0.5
    assert values[TieringOffloadingMetrics.PROMOTION_JOB_FAILURES][label] == 1
    tracker.assert_idle()


def test_tiering_metrics_tracker_reports_active_job_and_primary_usage_gauges():
    tracker = TieringMetricsTracker(
        tier_types=["example", "p2p"],
        num_primary_chunks=6,
        primary_chunk_size=16,
    )
    ex_job = JobMetadata(
        CascadeTransferJob(0, _CTX, to_keys([0, 1]), np.array([0, 1])),
        0,
    )
    p2p_job = JobMetadata(
        PromotionTransferJob(1, _CTX, to_keys([2, 3, 4]), np.array([2, 3, 4])),
        1,
    )
    _register(tracker, ex_job)
    _register(tracker, p2p_job)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    ex_label = ("1:example",)
    p2p_label = ("2:p2p",)
    assert values[TieringOffloadingMetrics.PRIMARY_READ_USAGE_PERC][
        ex_label
    ] == pytest.approx(2 / 6)
    assert values[TieringOffloadingMetrics.ACTIVE_CASCADE_JOBS][ex_label] == 1
    assert values[TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC][
        p2p_label
    ] == pytest.approx(3 / 6)
    assert values[TieringOffloadingMetrics.ACTIVE_PROMOTION_JOBS][p2p_label] == 1


def test_tiering_metrics_tracker_records_promotion_latency_histogram(monkeypatch):
    """Tracker measures job-creation-to-completion, not transfer_time.

    Includes failed promotions.
    """
    clock = 100.0
    monkeypatch.setattr(
        tiering_metrics, "time", SimpleNamespace(monotonic=lambda: clock)
    )

    tracker = TieringMetricsTracker(
        tier_types=["fs", "p2p"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )

    promotion_job = JobMetadata(
        PromotionTransferJob(0, _CTX, to_keys([0, 1]), np.array([0, 1])),
        1,
    )
    promotion_job.transfer_job.submit_time = clock
    _register(tracker, promotion_job)
    clock += 1.5
    tracker.on_job_finished(
        promotion_job, JobResult(job_id=0, success=True, transfer_time=0.42)
    )

    stats = tracker.take_stats()
    assert stats is not None
    observations = stats.data["data"][TieringOffloadingMetrics.PROMOTION_LATENCY]
    assert observations == {("2:p2p", "success"): [1.5]}

    cascade_job = JobMetadata(
        CascadeTransferJob(1, _CTX, to_keys([2]), np.array([2])),
        1,
    )
    _register(tracker, cascade_job)
    clock += 0.1
    tracker.on_job_finished(
        cascade_job, JobResult(job_id=1, success=True, transfer_time=0.10)
    )

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert TieringOffloadingMetrics.PROMOTION_LATENCY not in values

    failed_promotion_job = JobMetadata(
        PromotionTransferJob(2, _CTX, to_keys([3]), np.array([3])),
        0,
    )
    failed_promotion_job.transfer_job.submit_time = clock
    _register(tracker, failed_promotion_job)
    clock += 0.2
    tracker.on_job_finished(
        failed_promotion_job,
        JobResult(job_id=2, success=False, transfer_time=0.42),
    )

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    observations = values[TieringOffloadingMetrics.PROMOTION_LATENCY]
    assert observations == {("1:fs", "failure"): pytest.approx([0.2])}
    tracker.assert_idle()


def test_tiering_metrics_tracker_records_promotion_latency_without_transfer_time():
    """A completion carrying no transfer_time still records a latency sample."""
    tracker = TieringMetricsTracker(
        tier_types=["p2p"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )

    promotion_job = JobMetadata(
        PromotionTransferJob(0, _CTX, to_keys([0]), np.array([0])),
        0,
    )
    _register(tracker, promotion_job)
    tracker.on_job_finished(promotion_job, JobResult(job_id=0, success=True))
    tracker.assert_idle()

    stats = tracker.take_stats()
    assert stats is not None
    observations = stats.data["data"][TieringOffloadingMetrics.PROMOTION_LATENCY]
    assert len(observations[("1:p2p", "success")]) == 1


def test_tiering_metrics_tracker_records_promotion_allocation_failures():
    tracker = TieringMetricsTracker(
        tier_types=["ex"],
        num_primary_chunks=1,
        primary_chunk_size=16,
    )

    tracker.on_promotion_allocation_failure()

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES][()] == 1


def test_metrics_promotion_chunk_count_at_acquisition():
    """Promotion chunk count is recorded at acquisition, not registration."""
    keys = to_keys(range(3))
    tracker = TieringMetricsTracker(
        tier_types=["fs"], num_primary_chunks=5, primary_chunk_size=16
    )
    job = JobMetadata(
        PromotionTransferJob(
            1,
            _CTX,
            keys,
            None,
            alloc_fn=lambda k, ctx: (list(k), np.array([0, 1, 2], dtype=np.int32)),
        ),
        0,
    )

    # Registration records only the active-job count.
    tracker.on_job_registered(job)
    state = tracker._tier_states[0]
    assert state.active_promotion_count == 1
    assert state.primary_write_chunk_count == 0

    # Acquisition (inside get(), on whichever thread redeems it) records the
    # chunk count.
    assert job.transfer_job.get() is not None
    tracker.safe_counter_update(
        0, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, len(keys)
    )
    tracker.drain_pending_updates()
    assert state.primary_write_chunk_count == 3

    tracker.on_job_finished(job, JobResult(job_id=1, success=True, transfer_time=0.1))
    tracker.assert_idle()


def test_drain_pending_updates_semantics():
    """Drain applies buffered chunk counts and failures to the right
    counters, resets the buffer, and nets negative deltas to zero."""
    tracker = TieringMetricsTracker(
        tier_types=["fs", "p2p"], num_primary_chunks=5, primary_chunk_size=16
    )
    tracker.safe_counter_update(0, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, 3)
    tracker.safe_counter_update(1, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, 2)
    tracker.safe_counter_update(
        0, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, -3
    )
    tracker.safe_counter_update(
        0, TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES, 1
    )

    tracker.drain_pending_updates()
    assert tracker._tier_states[0].primary_write_chunk_count == 0
    assert tracker._tier_states[1].primary_write_chunk_count == 2
    values = tracker.take_stats().data["data"]
    assert values[TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES][()] == 1

    tracker.safe_counter_update(
        1, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, -2
    )
    tracker.drain_pending_updates()
    tracker.assert_idle()


def test_buffered_increment_drained_before_finished_job_decrement():
    """on_job_finished drains before decrementing, so an increment buffered
    by a tier worker (recorded before the job was reported finished) cannot
    drive the counter negative."""
    keys = to_keys(range(3))
    tracker = TieringMetricsTracker(
        tier_types=["fs"], num_primary_chunks=5, primary_chunk_size=16
    )
    job = JobMetadata(
        PromotionTransferJob(
            1,
            _CTX,
            keys,
            None,
            alloc_fn=lambda k, ctx: (list(k), np.array([0, 1, 2], dtype=np.int32)),
        ),
        0,
    )
    tracker.on_job_registered(job)
    # Worker redeems the claim and buffers the chunk count; no drain yet.
    assert job.transfer_job.get() is not None
    tracker.safe_counter_update(
        0, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, len(keys)
    )
    assert tracker._tier_states[0].primary_write_chunk_count == 0

    tracker.on_job_finished(job, JobResult(job_id=1, success=True))
    tracker.assert_idle()


def test_direct_and_buffered_counter_paths_are_independent():
    """on_primary_chunk_count applies immediately on the scheduler thread;
    safe_counter_update buffers until drain. Both land in the same counter."""
    tracker = TieringMetricsTracker(
        tier_types=["fs"], num_primary_chunks=5, primary_chunk_size=16
    )
    tracker.on_primary_chunk_count(0, 1, is_promotion=True)
    assert tracker._tier_states[0].primary_write_chunk_count == 1

    tracker.safe_counter_update(0, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, 2)
    assert tracker._tier_states[0].primary_write_chunk_count == 1

    tracker.drain_pending_updates()
    assert tracker._tier_states[0].primary_write_chunk_count == 3

    tracker.safe_counter_update(
        0, TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC, -3
    )
    tracker.drain_pending_updates()

    tracker.assert_idle()
