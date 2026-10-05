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


def test_record_host_usage_reports_latest_level():
    stats = HiSparseKVConnectorStats()
    stats.record_host_usage(usage=0.4, pending_page_transfers=2)
    stats.record_host_usage(usage=0.8, pending_page_transfers=0)

    reduced = stats.reduce()
    assert reduced["HiSparse host KV cache usage %"] == 80.0


def test_aggregate_keeps_latest_host_usage_level():
    first = HiSparseKVConnectorStats()
    first.record_host_usage(usage=0.4, pending_page_transfers=2)
    second = HiSparseKVConnectorStats()
    second.record_host_usage(usage=0.8, pending_page_transfers=0)
    second.record_host_usage(usage=0.6, pending_page_transfers=1)

    first.aggregate(second)

    assert first.data["host_cache_usage_perc"] == [0.4, 0.6]
    assert first.data["pending_page_transfers"] == [2, 1]


def test_aggregate_extends_snapshot_deltas():
    first = HiSparseKVConnectorStats()
    first.record_snapshot(hits=7, misses=3, host_to_device_bytes=48)
    second = HiSparseKVConnectorStats()
    second.record_snapshot(hits=5, misses=1, host_to_device_bytes=16)

    first.aggregate(second)

    assert first.data["cache_hits"] == [7, 5]
    assert first.data["cache_misses"] == [3, 1]
    assert first.data["host_to_device_bytes"] == [48, 16]


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


def test_prom_metrics_observe_host_usage_gauges():
    from types import SimpleNamespace
    from typing import Any

    from prometheus_client import Counter, Gauge, Histogram

    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
        HiSparsePromMetrics,
    )

    class _FakeMetric:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.increments: list[int | float] = []
            self.set_values: list[int | float] = []

        def labels(self, *labelvalues: object) -> "_FakeMetric":
            return self

        def inc(self, value: int | float) -> None:
            self.increments.append(value)

        def set(self, value: int | float) -> None:
            self.set_values.append(value)

    created: dict[str, _FakeMetric] = {}

    class _NamedFake(_FakeMetric):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            created[kwargs["name"]] = self

    prom = HiSparsePromMetrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        metric_types={Gauge: _NamedFake, Counter: _NamedFake, Histogram: _NamedFake},
        labelnames=["model_name"],
        per_engine_labelvalues={0: ["model"]},
    )

    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(hits=3, misses=2, host_to_device_bytes=32)
    stats.record_host_usage(usage=0.5, pending_page_transfers=1)
    stats.record_host_usage(usage=0.75, pending_page_transfers=3)
    prom.observe(stats.to_dict())

    assert created["vllm:hisparse_cache_hits"].increments == [3]
    assert created["vllm:hisparse_host_to_device_bytes"].increments == [32]
    assert created["vllm:hisparse_host_cache_usage_perc"].set_values == [0.75]
    assert created["vllm:hisparse_pending_page_transfers"].set_values == [3]
