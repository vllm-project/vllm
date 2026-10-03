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
    JobMetadata,
    TieringTransferJob,
)
from vllm.v1.kv_offload.tiering.metrics import TieringMetricsTracker

_CTX = ReqContext(req_id="test")


def to_keys(int_ids: Iterable[int]):
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def create_promotion_job(
    job_id: int,
    keys: list,
    chunk_ids: list[int],
    *,
    tracker: TieringMetricsTracker,
    tier_idx: int,
) -> TieringTransferJob:
    """Build a TieringTransferJob that materializes to the given chunk ids."""

    def primary_alloc_fn(keys, ctx):
        tracker.on_promotion_allocated(tier_idx, len(chunk_ids))
        return (list(keys), np.array(chunk_ids, dtype=np.int32))

    return TieringTransferJob(
        job_id, _CTX, keys, is_promotion=True, alloc_fn=primary_alloc_fn
    )


def create_cascade_job(
    job_id: int, keys: list, chunk_ids: list[int]
) -> TieringTransferJob:
    """Build a TieringTransferJob materialized to the given chunk ids."""
    return TieringTransferJob(
        job_id,
        _CTX,
        keys,
        is_promotion=False,
        chunk_ids=np.array(chunk_ids, dtype=np.int32),
    )


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
        JobMetadata(create_cascade_job(0, [cascade_key], [0]), 0),
        JobMetadata(
            create_promotion_job(1, [promotion_key], [1], tracker=tracker, tier_idx=0),
            0,
        ),
        JobMetadata(create_cascade_job(2, [failed_cascade_key], [2]), 0),
        JobMetadata(
            create_promotion_job(
                3, [failed_promotion_key], [3], tracker=tracker, tier_idx=0
            ),
            0,
        ),
    ]
    for job in jobs:
        tracker.on_job_registered(job)
        job.transfer_job.materialize()

    tracker.on_job_finished(
        jobs[0],
        JobResult(job_id=0, success=True, transfer_time=0.5),
    )
    tracker.on_job_finished(
        jobs[1],
        JobResult(job_id=1, success=True, transfer_time=0.25),
    )
    tracker.on_job_finished(
        jobs[2],
        JobResult(job_id=2, success=False),
    )
    tracker.on_job_finished(
        jobs[3],
        JobResult(job_id=3, success=False),
    )

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
        create_promotion_job(0, keys, [0, 1, 2], tracker=tracker, tier_idx=0), 0
    )
    tracker.on_job_registered(job)
    job.transfer_job.materialize()

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
    fs_job = JobMetadata(create_cascade_job(0, to_keys([0, 1]), [0, 1]), 0)
    p2p_job = JobMetadata(
        create_promotion_job(
            1, to_keys([2, 3, 4]), [2, 3, 4], tracker=tracker, tier_idx=1
        ),
        1,
    )
    tracker.on_job_registered(fs_job)
    tracker.on_job_registered(p2p_job)
    fs_job.transfer_job.materialize()
    p2p_job.transfer_job.materialize()

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
        create_promotion_job(0, to_keys([0, 1]), [0, 1], tracker=tracker, tier_idx=1),
        1,
    )
    promotion_job.transfer_job.submit_time = clock
    tracker.on_job_registered(
        promotion_job,
    )
    promotion_job.transfer_job.materialize()
    clock += 1.5
    tracker.on_job_finished(
        promotion_job,
        JobResult(job_id=0, success=True, transfer_time=0.42),
    )

    stats = tracker.take_stats()
    assert stats is not None
    observations = stats.data["data"][TieringOffloadingMetrics.PROMOTION_LATENCY]
    assert observations == {("2:p2p", "success"): [1.5]}

    cascade_job = JobMetadata(create_cascade_job(1, to_keys([2]), [2]), 1)
    tracker.on_job_registered(cascade_job)
    clock += 0.1
    tracker.on_job_finished(
        cascade_job,
        JobResult(job_id=1, success=True, transfer_time=0.10),
    )

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert TieringOffloadingMetrics.PROMOTION_LATENCY not in values

    failed_promotion_job = JobMetadata(
        create_promotion_job(2, to_keys([3]), [3], tracker=tracker, tier_idx=0), 0
    )
    failed_promotion_job.transfer_job.submit_time = clock
    tracker.on_job_registered(
        failed_promotion_job,
    )
    failed_promotion_job.transfer_job.materialize()
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
        create_promotion_job(0, to_keys([0]), [0], tracker=tracker, tier_idx=0), 0
    )
    tracker.on_job_registered(
        promotion_job,
    )
    promotion_job.transfer_job.materialize()
    tracker.on_job_finished(
        promotion_job,
        JobResult(job_id=0, success=True),
    )
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


def test_metrics_promotion_chunk_count_at_acquisition():
    """Promotion chunk count is recorded at acquisition, not registration."""
    keys = to_keys(range(3))
    tracker = TieringMetricsTracker(
        tier_types=["fs"], num_primary_chunks=5, primary_chunk_size=16
    )

    job = JobMetadata(
        create_promotion_job(1, keys, [0, 1, 2], tracker=tracker, tier_idx=0), 0
    )

    # Registration records only the active-job count.
    tracker.on_job_registered(job)
    state = tracker._tier_states[0]
    assert state.active_promotion_count == 1
    assert state.primary_write_chunk_count == 0

    # Acquisition (inside materialize(), on whichever thread redeems it)
    # records the chunk count.
    job.transfer_job.materialize()
    assert job.transfer_job.is_materialized()
    assert state.primary_write_chunk_count == 3

    tracker.on_job_finished(
        job,
        JobResult(job_id=1, success=True, transfer_time=0.1),
    )
    tracker.assert_idle()


def test_promotion_chunk_count_nets_to_zero_after_finish():
    """The chunk count a promotion records at acquisition is removed when
    the job finishes, so assert_idle holds."""
    keys = to_keys(range(3))
    tracker = TieringMetricsTracker(
        tier_types=["fs"], num_primary_chunks=5, primary_chunk_size=16
    )

    job = JobMetadata(
        create_promotion_job(1, keys, [0, 1, 2], tracker=tracker, tier_idx=0), 0
    )
    tracker.on_job_registered(job)
    job.transfer_job.materialize()
    assert job.transfer_job.is_materialized()
    assert tracker._tier_states[0].primary_write_chunk_count == 3

    tracker.on_job_finished(
        job,
        JobResult(job_id=1, success=True),
    )
    tracker.assert_idle()
