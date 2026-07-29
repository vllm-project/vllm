# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import numpy as np
import pytest

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
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
        num_primary_blocks=5,
        primary_block_size=16,
    )
    tracker.on_new_request(_CTX)

    key = to_keys([1])[0]
    tracker.on_lookup(
        _CTX,
        key,
        tracker.primary_tier_label,
        LookupResult.MISS,
        elapsed=0.0,
    )
    tracker.on_lookup(_CTX, key, tracker.tier_label(0), LookupResult.MISS, 0.0)
    tracker.on_lookup(_CTX, key, tracker.tier_label(1), LookupResult.HIT, 0.0)
    tracker.on_lookup(_CTX, key, tracker.tier_label(1), LookupResult.HIT, 0.0)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.BLOCK_QUERIES][("0:primary",)] == 1
    assert values[TieringOffloadingMetrics.BLOCK_QUERIES][("1:fs",)] == 1
    assert values[TieringOffloadingMetrics.BLOCK_QUERIES][("2:p2p",)] == 1
    assert values[TieringOffloadingMetrics.BLOCK_HITS][("2:p2p",)] == 1
    assert ("1:fs",) not in values[TieringOffloadingMetrics.BLOCK_HITS]


def test_tiering_metrics_tracker_stops_lookup_metrics_after_allocation():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_blocks=5,
        primary_block_size=16,
    )
    tracker.on_new_request(_CTX)
    key = to_keys([1])[0]

    tracker.on_lookup(
        _CTX,
        key,
        tracker.primary_tier_label,
        LookupResult.MISS,
        elapsed=0.0,
    )
    tracker.on_lookup(_CTX, key, tracker.tier_label(0), LookupResult.HIT, 0.0)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.BLOCK_QUERIES][("0:primary",)] == 1
    assert values[TieringOffloadingMetrics.BLOCK_QUERIES][("1:fs",)] == 1
    assert values[TieringOffloadingMetrics.BLOCK_HITS][("1:fs",)] == 1

    tracker.on_request_allocated(_CTX)
    tracker.on_lookup(
        _CTX,
        key,
        tracker.primary_tier_label,
        LookupResult.MISS,
        elapsed=0.0,
    )
    tracker.on_lookup(_CTX, key, tracker.tier_label(0), LookupResult.HIT, 0.0)

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert TieringOffloadingMetrics.BLOCK_QUERIES not in values
    assert TieringOffloadingMetrics.BLOCK_HITS not in values


def test_tiering_metrics_tracker_records_finished_job_metrics():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_blocks=5,
        primary_block_size=16,
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


def test_tiering_metrics_tracker_reports_active_job_and_primary_usage_gauges():
    tracker = TieringMetricsTracker(
        tier_types=["fs", "p2p"],
        num_primary_blocks=6,
        primary_block_size=16,
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


def test_tiering_metrics_tracker_records_promotion_allocation_failures():
    tracker = TieringMetricsTracker(
        tier_types=["fs"],
        num_primary_blocks=1,
        primary_block_size=16,
    )

    tracker.on_promotion_allocation_failure()

    stats = tracker.take_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES][()] == 1
