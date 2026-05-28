# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    LOAD_BYTES,
    LOAD_SIZE,
    LOAD_TIME,
    STORE_BYTES,
    STORE_SIZE,
    STORE_TIME,
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

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


def _connector_metric_metadata():
    return OffloadingConnector.get_metric_definitions(_FakeVllmConfig())  # type: ignore[arg-type]


def test_build_kv_connector_stats_with_none():
    """Test that build_kv_connector_stats returns empty stats when given None."""
    stats = OffloadingConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.data == {}
    assert stats.is_empty()


def test_build_kv_connector_stats_with_empty_dict():
    """Test that build_kv_connector_stats returns empty stats with empty dict."""
    stats = OffloadingConnector.build_kv_connector_stats(data={})

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.data == {}
    assert stats.is_empty()


def test_build_kv_connector_stats_reconstructs_offload_stats():
    """Test that OffloadingConnector stats are properly reconstructed with
    correct data."""
    serialized_data = {
        LOAD_BYTES: 24,
        LOAD_TIME: 1.5,
        LOAD_SIZE: [16, 8],
        STORE_BYTES: 3,
        STORE_TIME: 0.3,
        STORE_SIZE: [1, 2],
        STORES_SKIPPED: 5,
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    offload_connector_stats = stats
    assert isinstance(offload_connector_stats, OffloadingConnectorStats)
    assert offload_connector_stats.data[LOAD_BYTES] == 24
    assert offload_connector_stats.data[LOAD_TIME] == 1.5
    assert offload_connector_stats.data[LOAD_SIZE] == [16, 8]
    assert offload_connector_stats.data[STORE_BYTES] == 3
    assert offload_connector_stats.data[STORE_TIME] == 0.3
    assert offload_connector_stats.data[STORE_SIZE] == [1, 2]
    assert offload_connector_stats.data[STORES_SKIPPED] == 5


def test_aggregate_same_connector():
    """Test aggregating stats from the same connector type."""
    stats1 = OffloadingConnectorStats(
        data={
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
        metric_metadata=_metric_metadata(),
    )

    stats2 = OffloadingConnectorStats(
        data={
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
        metric_metadata=_metric_metadata(),
    )

    result = stats1.aggregate(stats2)

    assert result is stats1  # Should return self
    offload_connector_stats = result
    assert offload_connector_stats.data[LOAD_BYTES] == 34
    assert offload_connector_stats.data[LOAD_TIME] == 2.6
    assert offload_connector_stats.data[LOAD_SIZE] == [16, 8, 3, 7]
    assert offload_connector_stats.data[STORE_BYTES] == 19
    assert offload_connector_stats.data[STORE_TIME] == 2.3
    assert offload_connector_stats.data[STORE_SIZE] == [1, 2, 16]
    assert offload_connector_stats.data[STORES_SKIPPED] == 4
    assert offload_connector_stats.data[PENDING_STORES] == 1
    assert offload_connector_stats.data[LOOKUP_LATENCY] == [0.1, 0.2, 0.3]


def test_aggregate_merges_metric_metadata():
    stats1 = OffloadingConnectorStats(
        data={LOAD_BYTES: 1},
        metric_metadata={LOAD_BYTES: OffloadingCounterMetadata("load bytes")},
    )
    stats2 = OffloadingConnectorStats(
        data={PENDING_STORES: 2},
        metric_metadata={PENDING_STORES: OffloadingGaugeMetadata("pending stores")},
    )

    result = stats1.aggregate(stats2)

    assert result.data[PENDING_STORES] == 2
    assert PENDING_STORES in stats1.metric_metadata


def test_reduce():
    """Test that reduce() correctly reduces connector stats."""
    stats = OffloadingConnectorStats(
        data={
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
        metric_metadata=_metric_metadata(),
    )

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    # Check that the stats were reduced (should have aggregated values)
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
    offload_connector_stats = OffloadingConnectorStats(
        data={
            LOAD_BYTES: 10,
            LOAD_TIME: 1.1,
            LOAD_SIZE: [3, 7],
            STORE_BYTES: 16,
            STORE_TIME: 2,
            STORE_SIZE: [16],
            STORES_SKIPPED: 4,
            PENDING_STORES: 2,
            LOOKUP_LATENCY: [0.1],
        }
    )

    assert not offload_connector_stats.is_empty()

    offload_connector_stats.reset()

    # After reset, stats should be empty
    assert offload_connector_stats.is_empty()
    assert offload_connector_stats.data == {}


def test_prom_metrics_observes_manager_counter():
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(),  # type: ignore[arg-type]
        connector_metric_metadata=_connector_metric_metadata(),
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    prom_metrics.observe({STORES_SKIPPED: 7})

    counter = prom_metrics.offloading_metrics[(0, STORES_SKIPPED)]
    assert counter.increments == [7]
    counter_def = prom_metrics._offloading_metric_defs[STORES_SKIPPED]
    assert counter_def.kwargs["name"] == "vllm:kv_offload_stores_skipped"
    assert counter.labelvalues == ("model", "0")


def test_prom_metrics_observes_flat_transfer_metrics_and_legacy_metrics():
    prom_metrics = OffloadPromMetrics(
        vllm_config=_FakeVllmConfig(),  # type: ignore[arg-type]
        connector_metric_metadata=_connector_metric_metadata(),
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
            LOAD_BYTES: 24,
            LOAD_TIME: 1.5,
            LOAD_SIZE: [16, 8],
            STORE_BYTES: 3,
            STORE_TIME: 0.3,
            STORE_SIZE: [1, 2],
        }
    )

    assert prom_metrics.offloading_metrics[(0, LOAD_BYTES)].increments == [24]
    assert prom_metrics.offloading_metrics[(0, LOAD_TIME)].increments == [1.5]
    assert prom_metrics.offloading_metrics[(0, LOAD_SIZE)].observed == [16, 8]
    assert prom_metrics.offloading_metrics[(0, STORE_BYTES)].increments == [3]
    assert prom_metrics.offloading_metrics[(0, STORE_TIME)].increments == [0.3]
    assert prom_metrics.offloading_metrics[(0, STORE_SIZE)].observed == [1, 2]

    assert prom_metrics.counter_kv_bytes[(0, "CPU_to_GPU")].increments == [24]
    assert prom_metrics.counter_kv_transfer_time[(0, "CPU_to_GPU")].increments == [
        1.5
    ]
    assert prom_metrics.histogram_transfer_size[(0, "CPU_to_GPU")].observed == [16, 8]
    assert prom_metrics.counter_kv_bytes[(0, "GPU_to_CPU")].increments == [3]
    assert prom_metrics.counter_kv_transfer_time[(0, "GPU_to_CPU")].increments == [
        0.3
    ]
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
        CPUOffloadingManager, "get_metric_definitions", return_value=metric_definitions
    ):
        prom_metrics = OffloadPromMetrics(
            vllm_config=_FakeVllmConfig(store_threshold=0),  # type: ignore[arg-type]
            connector_metric_metadata=_connector_metric_metadata(),
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
            PENDING_STORES: 5,
            LOOKUP_LATENCY: [0.2, 0.4],
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
        connector_metric_metadata=_connector_metric_metadata(),
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    assert STORES_SKIPPED not in prom_metrics._offloading_metric_metadata
