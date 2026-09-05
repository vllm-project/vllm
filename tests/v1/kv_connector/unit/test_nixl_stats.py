# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NixlKVConnectorStats.reduce() metric math.

nixlXferTelemetry is only imported under TYPE_CHECKING in nixl/stats.py, so
these tests bypass record_transfer() and inject data directly — no nixl
hardware or installation required.
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)

MB = 2**20  # bytes per megabyte


def _stats_with_transfers(
    transfers: list[tuple[float, int, int]],
) -> NixlKVConnectorStats:
    """Build a NixlKVConnectorStats pre-populated with (duration_s, bytes, descs)
    rows, bypassing the nixlXferTelemetry dependency."""
    stats = NixlKVConnectorStats()
    for duration_s, total_bytes, descs in transfers:
        stats.data["transfer_duration"].append(duration_s)
        stats.data["post_duration"].append(duration_s * 0.1)  # stub value
        stats.data["bytes_transferred"].append(total_bytes)
        stats.data["num_descriptors"].append(descs)
    return stats


def test_is_empty_on_fresh_stats():
    stats = NixlKVConnectorStats()
    assert stats.is_empty()
    assert stats.num_successful_transfers == 0


def test_reduce_returns_zeros_when_no_successful_transfers():
    stats = NixlKVConnectorStats()
    stats.record_failed_transfer()
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 0
    assert reduced["Avg per-transfer throughput (MB/s)"] == 0


def test_reduce_single_transfer():
    # 2 MB in 2 s → 1 MB/s per-transfer rate; mean of one = 1 MB/s
    stats = _stats_with_transfers([(2.0, 2 * MB, 4)])
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 1
    assert reduced["Avg MB per transfer"] == pytest.approx(2.0, rel=1e-3)
    assert reduced["Avg per-transfer throughput (MB/s)"] == pytest.approx(1.0, rel=1e-3)


def test_reduce_sequential_uniform_transfers():
    # Two transfers with identical per-link rate: mean must equal that rate.
    # 1 MB in 1 s and 2 MB in 2 s → both 1 MB/s; mean = 1 MB/s.
    stats = _stats_with_transfers(
        [
            (1.0, 1 * MB, 2),
            (2.0, 2 * MB, 3),
        ]
    )
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 2
    assert reduced["Avg per-transfer throughput (MB/s)"] == pytest.approx(1.0, rel=1e-3)


def test_reduce_overlapping_transfers_old_formula_would_be_wrong():
    """
    Regression test: the old formula ``total_mb / sum(durations)`` understates
    throughput under concurrency.  With concurrent transfers (C=2) the
    denominator is doubled even though wall-clock time is only half as long.

    Concretely:
      Transfer A: 4 MB in 1 s  → per-link rate = 4 MB/s
      Transfer B: 2 MB in 2 s  → per-link rate = 1 MB/s

    Old formula:  (4+2) MB / (1+2) s = 2.000 MB/s   ← wrong under overlap
    New formula:  mean(4, 1)          = 2.500 MB/s   ← correct per-link avg

    The two values differ, so this test would have failed on the old code.
    """
    stats = _stats_with_transfers(
        [
            (1.0, 4 * MB, 4),
            (2.0, 2 * MB, 4),
        ]
    )
    reduced = stats.reduce()
    throughput = reduced["Avg per-transfer throughput (MB/s)"]

    # New (correct) value.
    assert throughput == pytest.approx(2.5, rel=1e-3)

    # Verify the old formula would have produced a different (wrong) answer.
    old_formula_result = (4 + 2) / (1 + 2)  # = 2.0
    assert throughput != pytest.approx(old_formula_result, rel=1e-3)


def test_reduce_many_concurrent_uniform_transfers():
    """
    C identical concurrent transfers each doing R MB/s must still report R.

    Old formula gives R/C (off by the concurrency factor).
    New formula gives mean(R, R, …, R) = R — correct.
    """
    R = 10.0  # MB/s per link
    C = 8  # concurrency
    duration_s = 0.1
    bytes_per_transfer = int(R * duration_s * MB)
    transfers = [(duration_s, bytes_per_transfer, 1)] * C

    stats = _stats_with_transfers(transfers)
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == C
    assert reduced["Avg per-transfer throughput (MB/s)"] == pytest.approx(R, rel=1e-3)


def test_reduce_key_names():
    """Confirm the metric dict uses the new key name, not the old one."""
    stats = _stats_with_transfers([(1.0, 1 * MB, 1)])
    reduced = stats.reduce()
    assert "Avg per-transfer throughput (MB/s)" in reduced
    assert "Throughput (MB/s)" not in reduced


def test_reduce_ignores_zero_duration_transfer():
    stats = _stats_with_transfers(
        [
            (0.0, 0, 0),
            (1.0, 2 * MB, 1),
        ]
    )
    reduced = stats.reduce()
    assert reduced["Avg per-transfer throughput (MB/s)"] == pytest.approx(2.0)


def test_reduce_returns_zero_when_all_durations_are_zero():
    stats = _stats_with_transfers([(0.0, 0, 0)])
    reduced = stats.reduce()
    assert reduced["Avg per-transfer throughput (MB/s)"] == 0


def test_record_failure_methods():
    stats = NixlKVConnectorStats()
    stats.record_failed_transfer()
    stats.record_failed_notification()
    stats.record_kv_expired_req()
    assert not stats.is_empty()
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 0
    assert reduced["Avg per-transfer throughput (MB/s)"] == 0


def test_aggregate_combines_data():
    a = _stats_with_transfers([(1.0, 1 * MB, 2)])
    b = _stats_with_transfers([(2.0, 2 * MB, 3)])
    a.aggregate(b)
    assert a.num_successful_transfers == 2


def test_clone_and_reset():
    stats = _stats_with_transfers([(0.5, 1 * MB, 1)])
    snap = stats.clone_and_reset()
    assert snap.num_successful_transfers == 1
    assert stats.is_empty()
