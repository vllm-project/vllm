# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
    _MetricType,
    _StatsKey,
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

LOAD_BYTES = _TransferMetricName.LOAD_BYTES
LOAD_TIME = _TransferMetricName.LOAD_TIME
LOAD_SIZE = _TransferMetricName.LOAD_SIZE
STORE_BYTES = _TransferMetricName.STORE_BYTES
STORE_TIME = _TransferMetricName.STORE_TIME
STORE_SIZE = _TransferMetricName.STORE_SIZE
STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
PENDING_STORES = "vllm:kv_offload_pending_stores"
LOOKUP_LATENCY = "vllm:kv_offload_lookup_latency_seconds"


class _FakeMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.children: list[_FakeMetric] = []
        self.observed: list[int | float] = []
        self.increments: list[int | float] = []
        self.set_values: list[int | float] = []
        self.labelvalues: tuple[object, ...] = ()

    def labels(self, *labelvalues):
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        self.children.append(child)
        return child

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value):
        self.increments.append(value)

    def set(self, value):
        self.set_values.append(value)


class _FakeVllmConfig:
    def __init__(self, store_threshold: int = 2):
        self.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"store_threshold": store_threshold}
        )


def _metric_metadata():
    return {
        LOAD_BYTES: OffloadingCounterMetadata(
            documentation="load bytes",
        ),
        LOAD_TIME: OffloadingCounterMetadata(
            documentation="load time",
        ),
        LOAD_SIZE: OffloadingHistogramMetadata(
            documentation="load size",
        ),
        STORE_BYTES: OffloadingCounterMetadata(
            documentation="store bytes",
        ),
        STORE_TIME: OffloadingCounterMetadata(
            documentation="store time",
        ),
        STORE_SIZE: OffloadingHistogramMetadata(
            documentation="store size",
        ),
        STORES_SKIPPED: OffloadingCounterMetadata(
            documentation="stores skipped",
        ),
        PENDING_STORES: OffloadingGaugeMetadata(
            documentation="pending stores",
        ),
        LOOKUP_LATENCY: OffloadingHistogramMetadata(
            documentation="lookup latency",
        ),
    }


def test_build_kv_connector_stats_with_none():
    """Test that build_kv_connector_stats returns empty stats when given None."""
    stats = OffloadingConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.is_empty()


def test_build_kv_connector_stats_with_empty_dict():
    """Test that build_kv_connector_stats returns empty stats with empty dict."""
    stats = OffloadingConnector.build_kv_connector_stats(data={})

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.is_empty()


def test_build_kv_connector_stats_reconstructs_offload_stats():
    """Test that OffloadingConnector stats are properly reconstructed with
    correct data."""
    serialized_data = {
        _StatsKey.TYPES: {
            LOAD_BYTES: _MetricType.COUNTER,
            LOAD_TIME: _MetricType.COUNTER,
            LOAD_SIZE: _MetricType.HISTOGRAM,
            STORE_BYTES: _MetricType.COUNTER,
            STORE_TIME: _MetricType.COUNTER,
            STORE_SIZE: _MetricType.HISTOGRAM,
            STORES_SKIPPED: _MetricType.COUNTER,
        },
        _StatsKey.DATA: {
            LOAD_BYTES: 24,
            LOAD_TIME: 1.5,
            LOAD_SIZE: [16, 8],
            STORE_BYTES: 3,
            STORE_TIME: 0.3,
            STORE_SIZE: [1, 2],
            STORES_SKIPPED: 5,
        },
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    assert isinstance(stats, OffloadingConnectorStats)
    values = stats.data[_StatsKey.DATA]
    assert values[LOAD_BYTES] == 24
    assert values[LOAD_TIME] == 1.5
    assert values[LOAD_SIZE] == [16, 8]
    assert values[STORE_BYTES] == 3
    assert values[STORE_TIME] == 0.3
    assert values[STORE_SIZE] == [1, 2]
    assert values[STORES_SKIPPED] == 5


def _make_stats_data(
    metric_data: dict[str, Any],
    metric_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured data dict from flat metric data and metadata."""
    metric_types = {}
    for key in metric_data:
        md = metric_metadata[key]
        if isinstance(md, OffloadingCounterMetadata):
            metric_types[key] = _MetricType.COUNTER
        elif isinstance(md, OffloadingGaugeMetadata):
            metric_types[key] = _MetricType.GAUGE
        elif isinstance(md, OffloadingHistogramMetadata):
            metric_types[key] = _MetricType.HISTOGRAM
    return {
        _StatsKey.TYPES: metric_types,
        _StatsKey.DATA: metric_data,
    }


def test_aggregate_same_connector():
    """Test aggregating stats from the same connector type."""
    metadata = _metric_metadata()
    stats1 = OffloadingConnectorStats(
        data=_make_stats_data(
            {
                LOAD_BYTES: 24,
                LOAD_TIME: 1.5,
                LOAD_SIZE: [16, 8],
                STORE_BYTES: 3,
                STORE_TIME: 0.3,
                STORE_SIZE: [1, 2],
                STORES_SKIPPED: 1,
                PENDING_STORES: 3,
                LOOKUP_LATENCY: [0.1],
            },
            metadata,
        ),
    )

    stats2 = OffloadingConnectorStats(
        data=_make_stats_data(
            {
                LOAD_BYTES: 10,
                LOAD_TIME: 1.1,
                LOAD_SIZE: [3, 7],
                STORE_BYTES: 16,
                STORE_TIME: 2,
                STORE_SIZE: [16],
                STORES_SKIPPED: 3,
                PENDING_STORES: 1,
                LOOKUP_LATENCY: [0.2, 0.3],
            },
            metadata,
        ),
    )

    result = stats1.aggregate(stats2)

    assert result is stats1  # Should return self
    values = result.data[_StatsKey.DATA]
    assert values[LOAD_BYTES] == 34
    assert values[LOAD_TIME] == 2.6
    assert values[LOAD_SIZE] == [16, 8, 3, 7]
    assert values[STORE_BYTES] == 19
    assert values[STORE_TIME] == 2.3
    assert values[STORE_SIZE] == [1, 2, 16]
    assert values[STORES_SKIPPED] == 4
    assert values[PENDING_STORES] == 1
    assert values[LOOKUP_LATENCY] == [0.1, 0.2, 0.3]


def test_aggregate_merges_types():
    stats1 = OffloadingConnectorStats(
        data={
            _StatsKey.TYPES: {LOAD_BYTES: _MetricType.COUNTER},
            _StatsKey.DATA: {LOAD_BYTES: 1},
        },
    )
    stats2 = OffloadingConnectorStats(
        data={
            _StatsKey.TYPES: {PENDING_STORES: _MetricType.GAUGE},
            _StatsKey.DATA: {PENDING_STORES: 2},
        },
    )

    result = stats1.aggregate(stats2)

    assert result.data[_StatsKey.DATA][PENDING_STORES] == 2
    assert result.data[_StatsKey.TYPES][PENDING_STORES] == _MetricType.GAUGE


def test_reduce():
    """Test that reduce() correctly reduces connector stats."""
    metadata = _metric_metadata()
    stats = OffloadingConnectorStats(
        data=_make_stats_data(
            {
                LOAD_BYTES: 34,
                LOAD_TIME: 2.6,
                LOAD_SIZE: [16, 8, 3, 7],
                STORE_BYTES: 19,
                STORE_TIME: 2.3,
                STORE_SIZE: [1, 2, 16],
                STORES_SKIPPED: 11,
                PENDING_STORES: 2,
                LOOKUP_LATENCY: [0.1, 0.2, 0.3],
            },
            metadata,
        ),
    )

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    assert reduced[LOAD_BYTES] == 34
    assert reduced[LOAD_TIME] == 2.6
    assert reduced[f"{LOAD_SIZE}_count"] == 4
    assert reduced[f"{LOAD_SIZE}_sum"] == 34
    assert reduced[STORE_BYTES] == 19
    assert reduced[STORE_TIME] == 2.3
    assert reduced[f"{STORE_SIZE}_count"] == 3
    assert reduced[f"{STORE_SIZE}_sum"] == 19
    assert reduced[STORES_SKIPPED] == 11
    assert reduced[PENDING_STORES] == 2
    assert reduced[f"{LOOKUP_LATENCY}_count"] == 3
    assert reduced[f"{LOOKUP_LATENCY}_sum"] == sum([0.1, 0.2, 0.3])


def test_reset():
    """Test that reset() resets all connector stats."""
    metadata = _metric_metadata()
    offload_connector_stats = OffloadingConnectorStats(
        data=_make_stats_data(
            {
                LOAD_BYTES: 10,
                LOAD_TIME: 1.1,
                LOAD_SIZE: [3, 7],
                STORE_BYTES: 16,
                STORE_TIME: 2,
                STORE_SIZE: [16],
                STORES_SKIPPED: 4,
                PENDING_STORES: 2,
                LOOKUP_LATENCY: [0.1],
            },
            metadata,
        ),
    )

    assert not offload_connector_stats.is_empty()

    offload_connector_stats.reset()

    # After reset, stats should be empty
    assert offload_connector_stats.is_empty()


def test_prom_metrics_observes_manager_counter():
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    prom_metrics.observe(
        {
            _StatsKey.TYPES: {STORES_SKIPPED: _MetricType.COUNTER},
            _StatsKey.DATA: {STORES_SKIPPED: 7},
        }
    )

    counter = prom_metrics.offloading_metrics[(0, STORES_SKIPPED)]
    assert counter.increments == [7]
    counter_def = prom_metrics._offloading_metric_defs[STORES_SKIPPED]
    assert counter_def.kwargs["name"] == "vllm:kv_offload_stores_skipped"
    assert counter.labelvalues == ("model", "0")


def test_prom_metrics_observes_flat_transfer_metrics_and_legacy_metrics():
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    prom_metrics.observe(
        {
            _StatsKey.TYPES: {
                LOAD_BYTES: _MetricType.COUNTER,
                LOAD_TIME: _MetricType.COUNTER,
                LOAD_SIZE: _MetricType.HISTOGRAM,
                STORE_BYTES: _MetricType.COUNTER,
                STORE_TIME: _MetricType.COUNTER,
                STORE_SIZE: _MetricType.HISTOGRAM,
            },
            _StatsKey.DATA: {
                LOAD_BYTES: 24,
                LOAD_TIME: 1.5,
                LOAD_SIZE: [16, 8],
                STORE_BYTES: 3,
                STORE_TIME: 0.3,
                STORE_SIZE: [1, 2],
            },
        }
    )

    assert prom_metrics.offloading_metrics[(0, LOAD_BYTES)].increments == [24]
    assert prom_metrics.offloading_metrics[(0, LOAD_TIME)].increments == [1.5]
    assert prom_metrics.offloading_metrics[(0, LOAD_SIZE)].observed == [16, 8]
    assert prom_metrics.offloading_metrics[(0, STORE_BYTES)].increments == [3]
    assert prom_metrics.offloading_metrics[(0, STORE_TIME)].increments == [0.3]
    assert prom_metrics.offloading_metrics[(0, STORE_SIZE)].observed == [1, 2]

    assert prom_metrics.counter_kv_bytes[(0, "CPU_to_GPU")].increments == [24]
    assert prom_metrics.counter_kv_transfer_time[(0, "CPU_to_GPU")].increments == [1.5]
    assert prom_metrics.histogram_transfer_size[(0, "CPU_to_GPU")].observed == [16, 8]
    assert prom_metrics.counter_kv_bytes[(0, "GPU_to_CPU")].increments == [3]
    assert prom_metrics.counter_kv_transfer_time[(0, "GPU_to_CPU")].increments == [0.3]
    assert prom_metrics.histogram_transfer_size[(0, "GPU_to_CPU")].observed == [1, 2]


def test_prom_metrics_observes_manager_gauge_and_histogram():
    metric_definitions = {
        PENDING_STORES: OffloadingGaugeMetadata(
            documentation="Number of currently pending KV offload stores.",
        ),
        LOOKUP_LATENCY: OffloadingHistogramMetadata(
            documentation="KV offload lookup latency.",
            buckets=(0.1, 1.0),
        ),
    }
    with patch.object(
        CPUOffloadingSpec, "build_metric_definitions", return_value=metric_definitions
    ):
        prom_metrics = OffloadPromMetrics(
            vllm_config=_FakeVllmConfig(store_threshold=0),  # type: ignore[arg-type]
            metric_types={
                Gauge: _FakeMetric,
                Counter: _FakeMetric,
                Histogram: _FakeMetric,
            },
            labelnames=["model_name", "engine"],
            per_engine_labelvalues={0: ["model", "0"]},
        )

    prom_metrics.observe(
        {
            _StatsKey.TYPES: {
                PENDING_STORES: _MetricType.GAUGE,
                LOOKUP_LATENCY: _MetricType.HISTOGRAM,
            },
            _StatsKey.DATA: {
                PENDING_STORES: 5,
                LOOKUP_LATENCY: [0.2, 0.4],
            },
        }
    )

    gauge = prom_metrics.offloading_metrics[(0, PENDING_STORES)]
    histogram = prom_metrics.offloading_metrics[(0, LOOKUP_LATENCY)]
    assert gauge.set_values == [5]
    assert histogram.observed == [0.2, 0.4]
    histogram_def = prom_metrics._offloading_metric_defs[LOOKUP_LATENCY]
    assert histogram_def.kwargs["buckets"] == (0.1, 1.0)


def test_prom_metrics_uses_configured_manager_metrics():
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(store_threshold=0),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    assert STORES_SKIPPED not in prom_metrics._offloading_metric_metadata


def test_aggregate_into_empty_stats():
    """Aggregating non-empty stats into a fresh (empty) stats object works."""
    empty = OffloadingConnectorStats()
    assert empty.is_empty()

    non_empty = OffloadingConnectorStats(
        data={
            _StatsKey.TYPES: {
                LOAD_BYTES: _MetricType.COUNTER,
                LOAD_SIZE: _MetricType.HISTOGRAM,
                PENDING_STORES: _MetricType.GAUGE,
            },
            _StatsKey.DATA: {
                LOAD_BYTES: 42,
                LOAD_SIZE: [10, 20],
                PENDING_STORES: 3,
            },
        },
    )

    result = empty.aggregate(non_empty)

    assert result is empty
    values = result.data[_StatsKey.DATA]
    assert values[LOAD_BYTES] == 42
    assert values[LOAD_SIZE] == [10, 20]
    assert values[PENDING_STORES] == 3


def test_prom_metrics_multi_engine_routing():
    """Metrics are routed to the correct engine index."""
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"], 1: ["model", "1"]},
    )

    prom_metrics.observe(
        {
            _StatsKey.TYPES: {LOAD_BYTES: _MetricType.COUNTER},
            _StatsKey.DATA: {LOAD_BYTES: 100},
        },
        engine_idx=1,
    )

    engine0 = prom_metrics.offloading_metrics[(0, LOAD_BYTES)]
    engine1 = prom_metrics.offloading_metrics[(1, LOAD_BYTES)]
    assert engine0.increments == []
    assert engine1.increments == [100]


def test_prom_metrics_rejects_undeclared_metric():
    """observe() asserts if a metric was never declared in metadata."""
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(store_threshold=0),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    with pytest.raises(AssertionError):
        prom_metrics.observe(
            {
                _StatsKey.TYPES: {"unknown:metric": _MetricType.COUNTER},
                _StatsKey.DATA: {"unknown:metric": 1},
            }
        )
