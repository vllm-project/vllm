# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
    HiSparseKVConnectorStats,
)


def test_is_empty_on_fresh_stats():
    stats = HiSparseKVConnectorStats()
    assert stats.is_empty()


def test_record_snapshot_and_reduce():
    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(hits=7, misses=3, host_to_device_bytes=48)
    stats.record_snapshot(hits=5, misses=1, host_to_device_bytes=16)
    assert not stats.is_empty()

    reduced = stats.reduce()
    assert reduced["HiSparse hot-buffer hits"] == 12
    assert reduced["HiSparse hot-buffer misses"] == 4
    assert reduced["HiSparse host-to-device bytes"] == 64


def test_aggregate_extends_snapshot_deltas():
    first = HiSparseKVConnectorStats()
    first.record_snapshot(hits=7, misses=3, host_to_device_bytes=48)
    second = HiSparseKVConnectorStats()
    second.record_snapshot(hits=5, misses=1, host_to_device_bytes=16)

    first.aggregate(second)

    assert first.data == {
        "cache_hits": [7, 5],
        "cache_misses": [3, 1],
        "host_to_device_bytes": [48, 16],
    }


def test_aggregate_skips_empty_stats():
    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(hits=2, misses=1, host_to_device_bytes=16)

    stats.aggregate(HiSparseKVConnectorStats())

    assert stats.data["cache_hits"] == [2]


def test_build_kv_connector_stats_round_trip():
    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(hits=12, misses=4, host_to_device_bytes=64)
    payload = stats.to_dict()

    rebuilt = HiSparseConnector.build_kv_connector_stats(data=payload)

    assert rebuilt is not None
    assert isinstance(rebuilt, HiSparseKVConnectorStats)
    assert rebuilt.reduce() == stats.reduce()
