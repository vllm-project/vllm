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
    TransferJob,
)
from vllm.v1.kv_offload.tiering.manager import JobMetadata
from vllm.v1.kv_offload.tiering.metrics import TieringMetricsTracker

_CTX = ReqContext(req_id="test")


def to_keys(int_ids: Iterable[int]):
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def test_tiering_metrics_tracker_records_lookup_metrics():
    tracker = TieringMetricsTracker(
        tier_types=["fs", "p2p"],
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
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("1:fs",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("2:p2p",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_HITS][("2:p2p",)] == 1
    assert ("1:fs",) not in values[TieringOffloadingMetrics.CHUNK_HITS]


def test_tiering_metrics_tracker_stops_lookup_metrics_after_allocation():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
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
    assert values[TieringOffloadingMetrics.CHUNK_QUERIES][("1:fs",)] == 1
    assert values[TieringOffloadingMetrics.CHUNK_HITS][("1:fs",)] == 1

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


def test_tiering_metrics_tracker_records_finished_job_metrics():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )
    cascade_key, promotion_key, failed_cascade_key, failed_promotion_key = to_keys(
        range(4)
    )
    jobs = [
        JobMetadata(
            TransferJob(0, [cascade_key], np.array([0]), False, _CTX),
            0,
        ),
        JobMetadata(
            TransferJob(1, [promotion_key], np.array([1]), True, _CTX),
            0,
        ),
        JobMetadata(
            TransferJob(2, [failed_cascade_key], np.array([2]), False, _CTX),
            0,
        ),
        JobMetadata(
            TransferJob(3, [failed_promotion_key], np.array([3]), True, _CTX),
            0,
        ),
    ]
    for job in jobs:
        tracker.on_job_registered(job)

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
    label = ("1:fs",)
    assert values[TieringOffloadingMetrics.WRITE_BYTES][label] == 16
    assert values[TieringOffloadingMetrics.WRITE_TIME][label] == 0.5
    assert values[TieringOffloadingMetrics.READ_BYTES][label] == 16
    assert values[TieringOffloadingMetrics.READ_TIME][label] == 0.25
    assert values[TieringOffloadingMetrics.CASCADE_JOB_FAILURES][label] == 1
    assert values[TieringOffloadingMetrics.PROMOTION_JOB_FAILURES][label] == 1
    tracker.assert_idle()


def test_tiering_metrics_tracker_records_partial_promotion_success_bytes():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_chunks=5,
        primary_chunk_size=16,
    )
    keys = to_keys(range(3))
    job = JobMetadata(
        TransferJob(0, keys, np.array([0, 1, 2]), True, _CTX),
        0,
    )
    tracker.on_job_registered(job)

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
    label = ("1:fs",)
    assert values[TieringOffloadingMetrics.READ_BYTES][label] == 32
    assert values[TieringOffloadingMetrics.READ_TIME][label] == 0.5
    assert values[TieringOffloadingMetrics.PROMOTION_JOB_FAILURES][label] == 1
    tracker.assert_idle()


def test_tiering_metrics_tracker_reports_active_job_and_primary_usage_gauges():
    tracker = TieringMetricsTracker(
        tier_types=["fs", "p2p"],
        num_primary_chunks=6,
        primary_chunk_size=16,
    )
    fs_job = JobMetadata(
        TransferJob(0, to_keys([0, 1]), np.array([0, 1]), False, _CTX),
        0,
    )
    p2p_job = JobMetadata(
        TransferJob(1, to_keys([2, 3, 4]), np.array([2, 3, 4]), True, _CTX),
        1,
    )
    tracker.on_job_registered(fs_job)
    tracker.on_job_registered(p2p_job)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    fs_label = ("1:fs",)
    p2p_label = ("2:p2p",)
    assert values[TieringOffloadingMetrics.PRIMARY_READ_USAGE_PERC][
        fs_label
    ] == pytest.approx(2 / 6)
    assert values[TieringOffloadingMetrics.ACTIVE_CASCADE_JOBS][fs_label] == 1
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
        TransferJob(0, to_keys([0, 1]), np.array([0, 1]), True, _CTX),
        1,
    )
    promotion_job.transfer_job.submit_time = clock
    tracker.on_job_registered(promotion_job)
    clock += 1.5
    tracker.on_job_finished(
        promotion_job, JobResult(job_id=0, success=True, transfer_time=0.42)
    )

    stats = tracker.take_stats()
    assert stats is not None
    observations = stats.data["data"][TieringOffloadingMetrics.PROMOTION_LATENCY]
    assert observations == {("2:p2p", "success"): [1.5]}

    cascade_job = JobMetadata(
        TransferJob(1, to_keys([2]), np.array([2]), False, _CTX),
        1,
    )
    tracker.on_job_registered(cascade_job)
    clock += 0.1
    tracker.on_job_finished(
        cascade_job, JobResult(job_id=1, success=True, transfer_time=0.10)
    )

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert TieringOffloadingMetrics.PROMOTION_LATENCY not in values

    failed_promotion_job = JobMetadata(
        TransferJob(2, to_keys([3]), np.array([3]), True, _CTX),
        0,
    )
    failed_promotion_job.transfer_job.submit_time = clock
    tracker.on_job_registered(failed_promotion_job)
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
        TransferJob(0, to_keys([0]), np.array([0]), True, _CTX),
        0,
    )
    tracker.on_job_registered(promotion_job)
    tracker.on_job_finished(promotion_job, JobResult(job_id=0, success=True))
    tracker.assert_idle()

    stats = tracker.take_stats()
    assert stats is not None
    observations = stats.data["data"][TieringOffloadingMetrics.PROMOTION_LATENCY]
    assert len(observations[("1:p2p", "success")]) == 1


def test_tiering_metrics_tracker_records_promotion_allocation_failures():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_chunks=1,
        primary_chunk_size=16,
    )

    tracker.on_promotion_allocation_failure()

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES][()] == 1


OUTCOMES = TieringOffloadingMetrics.PROMOTED_CHUNK_OUTCOMES
PENDING = TieringOffloadingMetrics.PROMOTED_CHUNKS_PENDING
REPROMOTED = TieringOffloadingMetrics.CHUNK_REPROMOTIONS


def _promotion_tracker(num_primary_chunks: int = 8) -> TieringMetricsTracker:
    return TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_chunks=num_primary_chunks,
        primary_chunk_size=16,
    )


def _values(tracker: TieringMetricsTracker) -> dict:
    stats = tracker.take_stats()
    assert stats is not None
    return stats.data["data"]


def _outcomes(values: dict) -> dict[str, int]:
    return {labels[1]: count for labels, count in values.get(OUTCOMES, {}).items()}


def test_promoted_chunk_records_exactly_one_outcome():
    tracker = _promotion_tracker()
    key, unpromoted = to_keys(range(2))
    tracker.on_promotion_landed(0, [key])
    # Two loads are in flight; only the first completion is an outcome.
    tracker.on_promoted_reads_started([key, unpromoted])
    tracker.on_promoted_reads_started([key])
    tracker.on_promoted_reads_done([key, unpromoted], success=True, by_gpu_load=True)
    tracker.on_promoted_reads_done([key], success=True, by_gpu_load=True)
    # Events after the outcome, and events for chunks that were never
    # promoted, change nothing.
    tracker.on_primary_evicted([key, unpromoted])
    tracker.on_primary_reset()
    tracker.on_promotion_slots_allocated(0, [key, unpromoted])
    values = _values(tracker)
    assert _outcomes(values) == {"loaded": 1}
    assert REPROMOTED not in values
    assert values[PENDING] == {("1:fs", "unread"): 0, ("1:fs", "reading"): 0}


def test_failed_peer_read_leaves_promoted_chunk_unread():
    tracker = _promotion_tracker()
    (key,) = to_keys([1])
    tracker.on_promotion_landed(0, [key])
    job = JobMetadata(TransferJob(3, [key], np.array([0]), False, _CTX), 0)
    tracker.on_job_registered(job)
    tracker.on_peer_read_job(3, [key])
    tracker.on_job_finished(job, JobResult(job_id=3, success=False))
    values = _values(tracker)
    assert OUTCOMES not in values
    assert values[PENDING][("1:fs", "unread")] == 1

    tracker.on_primary_evicted([key])
    assert _outcomes(_values(tracker)) == {"evicted_unloaded": 1}


def test_repromotion_after_unread_eviction_is_counted_once():
    """Re-promotion memory holds the most recent wasted chunks, up to the
    primary tier's capacity."""
    tracker = _promotion_tracker(num_primary_chunks=2)
    keys = to_keys(range(3))
    tracker.on_promotion_landed(0, keys)
    tracker.on_primary_evicted(keys)
    tracker.on_promotion_slots_allocated(0, keys)
    tracker.on_promotion_slots_allocated(0, keys)
    values = _values(tracker)
    assert _outcomes(values) == {"evicted_unloaded": 3}
    assert values[REPROMOTED][("1:fs",)] == 2  # the oldest one was forgotten


def test_reset_closes_pending_chunks_and_ignores_pre_reset_completions():
    tracker = _promotion_tracker()
    unread, reading, wasted = to_keys(range(3))
    tracker.on_promotion_landed(0, [unread, reading, wasted])
    tracker.on_primary_evicted([wasted])
    tracker.on_promoted_reads_started([reading])
    tracker.on_primary_reset()
    assert _outcomes(_values(tracker)) == {"evicted_unloaded": 1, "reset": 2}

    # A load submitted before the reset completes afterwards, and a promotion
    # after a reset is not a re-promotion.
    tracker.on_promoted_reads_done([reading], success=True, by_gpu_load=True)
    tracker.on_promotion_slots_allocated(0, [wasted])
    values = _values(tracker)
    assert OUTCOMES not in values and REPROMOTED not in values
