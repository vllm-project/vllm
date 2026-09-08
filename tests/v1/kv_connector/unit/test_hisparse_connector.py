# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.metrics import (
    HiSparseMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    TypedKVConnectorStats,
)
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def test_no_forward_enqueues_deferred_hisparse_transfers():
    """A zero-token step must still enqueue deferred post-forward transfers."""
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.pre_forward = MagicMock()
    connector.finish_forward = MagicMock()
    connector.post_forward = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(finished_req_ids=set())
    connector.no_forward(scheduler_output)

    connector.pre_forward.assert_called_once_with(scheduler_output)
    connector.finish_forward.assert_called_once_with()


class _FakeCounter:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.increments: list[int | float] = []

    def labels(self, *labelvalues):
        return self

    def inc(self, value):
        self.increments.append(value)


def _hisparse_stats(hits: int, misses: int, nbytes: int) -> TypedKVConnectorStats:
    stats = TypedKVConnectorStats()
    stats.increase_counter(HiSparseMetricName.CACHE_HITS, hits)
    stats.increase_counter(HiSparseMetricName.CACHE_MISSES, misses)
    stats.increase_counter(HiSparseMetricName.HOST_TO_DEVICE_BYTES, nbytes)
    return stats


def test_hisparse_stats_rebuild_and_aggregate_from_payload():
    """Worker payloads rebuild on the logger side and sum across workers."""
    first = HiSparseConnector.build_kv_connector_stats(
        _hisparse_stats(7, 3, 48).to_dict()
    )
    second = HiSparseConnector.build_kv_connector_stats(
        _hisparse_stats(5, 1, 16).to_dict()
    )
    assert first is not None and second is not None

    assert first.aggregate(second).reduce() == {
        HiSparseMetricName.CACHE_HITS: 12,
        HiSparseMetricName.CACHE_MISSES: 4,
        HiSparseMetricName.HOST_TO_DEVICE_BYTES: 64,
    }
    empty = HiSparseConnector.build_kv_connector_stats()
    assert empty is not None and empty.is_empty()


def test_hisparse_prom_metrics_export_counters():
    prom_metrics = HiSparseConnector.build_prom_metrics(
        SimpleNamespace(kv_transfer_config=None),  # type: ignore[arg-type]
        {Gauge: _FakeCounter, Counter: _FakeCounter, Histogram: _FakeCounter},
        ["model_name", "engine"],
        {0: ["model", "0"]},
    )

    prom_metrics.observe(_hisparse_stats(7, 3, 48).to_dict())

    increments = {
        name: metric.increments for (_, name, _), metric in prom_metrics.metrics.items()
    }
    assert increments == {
        HiSparseMetricName.CACHE_HITS: [7],
        HiSparseMetricName.CACHE_MISSES: [3],
        HiSparseMetricName.HOST_TO_DEVICE_BYTES: [48],
    }
