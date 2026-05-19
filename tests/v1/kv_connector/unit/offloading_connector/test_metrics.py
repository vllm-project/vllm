# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    COUNTER_PREFIX,
    GAUGE_PREFIX,
    HISTOGRAM_PREFIX,
    TRANSFER_PREFIX,
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import (
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)


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
        TRANSFER_PREFIX + "CPU_to_GPU": [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
        ],
        TRANSFER_PREFIX + "GPU_to_CPU": [
            {"op_size": 1, "op_time": 0.1},
            {"op_size": 2, "op_time": 0.2},
        ],
        COUNTER_PREFIX + "stores_skipped": 5,
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    offload_connector_stats = stats
    assert isinstance(offload_connector_stats, OffloadingConnectorStats)
    assert offload_connector_stats.data[TRANSFER_PREFIX + "CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
    ]
    assert offload_connector_stats.data[TRANSFER_PREFIX + "GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
    ]
    assert offload_connector_stats.data[COUNTER_PREFIX + "stores_skipped"] == 5


def test_aggregate_same_connector():
    """Test aggregating stats from the same connector type."""
    stats1 = OffloadingConnectorStats(
        data={
            TRANSFER_PREFIX + "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
            ],
            TRANSFER_PREFIX + "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
            ],
            COUNTER_PREFIX + "stores_skipped": 1,
            GAUGE_PREFIX + "pending_stores": 3,
            HISTOGRAM_PREFIX + "lookup_latency": [0.1],
        }
    )

    stats2 = OffloadingConnectorStats(
        data={
            TRANSFER_PREFIX + "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            TRANSFER_PREFIX + "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
            COUNTER_PREFIX + "stores_skipped": 3,
            GAUGE_PREFIX + "pending_stores": 1,
            HISTOGRAM_PREFIX + "lookup_latency": [0.2, 0.3],
        }
    )

    result = stats1.aggregate(stats2)

    assert result is stats1  # Should return self
    offload_connector_stats = result
    assert offload_connector_stats.data[TRANSFER_PREFIX + "CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
        {"op_size": 3, "op_time": 0.2},
        {"op_size": 7, "op_time": 0.9},
    ]
    assert offload_connector_stats.data[TRANSFER_PREFIX + "GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
        {"op_size": 16, "op_time": 2},
    ]
    assert offload_connector_stats.data[COUNTER_PREFIX + "stores_skipped"] == 4
    assert offload_connector_stats.data[GAUGE_PREFIX + "pending_stores"] == 1
    assert offload_connector_stats.data[HISTOGRAM_PREFIX + "lookup_latency"] == [
        0.1,
        0.2,
        0.3,
    ]


def test_reduce():
    """Test that reduce() correctly reduces connector stats."""
    stats = OffloadingConnectorStats(
        data={
            TRANSFER_PREFIX + "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            TRANSFER_PREFIX + "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
                {"op_size": 16, "op_time": 2},
            ],
            COUNTER_PREFIX + "stores_skipped": 11,
            GAUGE_PREFIX + "pending_stores": 2,
            HISTOGRAM_PREFIX + "lookup_latency": [0.1, 0.2, 0.3],
        }
    )

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    # Check that the stats were reduced (should have aggregated values)
    assert "CPU_to_GPU_total_bytes" in reduced
    assert "CPU_to_GPU_total_time" in reduced
    assert "GPU_to_CPU_total_bytes" in reduced
    assert "GPU_to_CPU_total_time" in reduced
    assert reduced["CPU_to_GPU_total_bytes"] == 34
    assert reduced["CPU_to_GPU_total_time"] == 2.6
    assert reduced["GPU_to_CPU_total_time"] == 2.3
    assert reduced["GPU_to_CPU_total_bytes"] == 19
    assert reduced["stores_skipped"] == 11
    assert reduced["pending_stores"] == 2
    assert reduced["lookup_latency_count"] == 3
    assert reduced["lookup_latency_sum"] == sum([0.1, 0.2, 0.3])


def test_reset():
    """Test that reset() resets all connector stats."""
    offload_connector_stats = OffloadingConnectorStats(
        data={
            TRANSFER_PREFIX + "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            TRANSFER_PREFIX + "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
            COUNTER_PREFIX + "stores_skipped": 4,
            GAUGE_PREFIX + "pending_stores": 2,
            HISTOGRAM_PREFIX + "lookup_latency": [0.1],
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
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    prom_metrics.observe({COUNTER_PREFIX + "stores_skipped": 7})

    counter = prom_metrics.offloading_manager_metrics[(0, "stores_skipped")]
    assert counter.increments == [7]
    counter_def = prom_metrics._offloading_manager_metric_defs["stores_skipped"]
    assert counter_def.kwargs["name"] == "vllm:kv_offload_stores_skipped"
    assert counter.labelvalues == ("model", "0")


def test_prom_metrics_observes_manager_gauge_and_histogram():
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
    prom_metrics._offloading_manager_metric_metadata = {
        "pending_stores": OffloadingGaugeMetadata(
            name="vllm:kv_offload_pending_stores",
            documentation="Number of currently pending KV offload stores.",
        ),
        "lookup_latency": OffloadingHistogramMetadata(
            name="vllm:kv_offload_lookup_latency_seconds",
            documentation="KV offload lookup latency.",
            buckets=(0.1, 1.0),
        ),
    }

    prom_metrics.observe(
        {
            GAUGE_PREFIX + "pending_stores": 5,
            HISTOGRAM_PREFIX + "lookup_latency": [0.2, 0.4],
        }
    )

    gauge = prom_metrics.offloading_manager_metrics[(0, "pending_stores")]
    histogram = prom_metrics.offloading_manager_metrics[(0, "lookup_latency")]
    assert gauge.set_values == [5]
    assert histogram.observed == [0.2, 0.4]
    histogram_def = prom_metrics._offloading_manager_metric_defs["lookup_latency"]
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

    assert prom_metrics._offloading_manager_metric_metadata == {}
