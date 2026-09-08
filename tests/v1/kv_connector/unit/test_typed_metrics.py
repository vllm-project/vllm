# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    CounterMetadata,
    GaugeMetadata,
    HistogramMetadata,
    MetricType,
    StatsKey,
    TypedKVConnectorPromMetrics,
    TypedKVConnectorStats,
)

pytestmark = pytest.mark.cpu_test

HITS = "vllm:test_hits"
DEPTH = "vllm:test_queue_depth"
LATENCY = "vllm:test_latency_seconds"
LABELED = "vllm:test_labeled"

DEFINITIONS = {
    HITS: CounterMetadata(documentation="hits"),
    DEPTH: GaugeMetadata(documentation="depth"),
    LATENCY: HistogramMetadata(documentation="latency", buckets=(0.1, 1.0)),
    LABELED: CounterMetadata(documentation="labeled", labelnames=("kind",)),
}


class _FakeMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.labelvalues: tuple[object, ...] = ()
        self.increments: list[int | float] = []
        self.set_values: list[int | float] = []
        self.observed: list[int | float] = []

    def labels(self, *labelvalues):
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        return child

    def inc(self, value):
        self.increments.append(value)

    def set(self, value):
        self.set_values.append(value)

    def observe(self, value):
        self.observed.append(value)


def _prom_metrics(definitions=DEFINITIONS) -> TypedKVConnectorPromMetrics:
    return TypedKVConnectorPromMetrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),  # type: ignore
        metric_types={Gauge: _FakeMetric, Counter: _FakeMetric, Histogram: _FakeMetric},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"], 1: ["model", "1"]},
        metric_definitions=definitions,
    )


def _record(stats: TypedKVConnectorStats, hits: int, depth: int, latency: float):
    stats.increase_counter(HITS, hits)
    stats.set_gauge(DEPTH, depth)
    stats.observe_histogram(LATENCY, latency)
    stats.increase_counter(LABELED, labelvalues=("a",))


def test_record_aggregate_and_reduce_by_metric_type():
    """Counters sum, gauges keep the latest value, histograms keep samples."""
    first = TypedKVConnectorStats()
    _record(first, hits=3, depth=5, latency=0.2)
    second = TypedKVConnectorStats()
    _record(second, hits=4, depth=1, latency=0.7)

    assert first.aggregate(second) is first
    assert first.reduce() == {
        HITS: 7,
        DEPTH: 1,
        f"{LATENCY}_count": 2,
        f"{LATENCY}_sum": pytest.approx(0.9),
        f"{LABELED}:('a',)": 2,
    }


def test_payload_round_trips_without_metadata():
    """The serialized dict carries the metric types needed to rebuild stats."""
    stats = TypedKVConnectorStats()
    _record(stats, hits=1, depth=2, latency=0.3)

    rebuilt = TypedKVConnectorStats(data=stats.to_dict())

    assert rebuilt.to_dict()[StatsKey.TYPES] == {
        HITS: MetricType.COUNTER,
        DEPTH: MetricType.GAUGE,
        LATENCY: MetricType.HISTOGRAM,
        LABELED: MetricType.COUNTER,
    }
    assert rebuilt.reduce() == stats.reduce()
    assert not rebuilt.is_empty()
    rebuilt.reset()
    assert rebuilt.is_empty()
    assert TypedKVConnectorStats().aggregate(rebuilt).is_empty()


def test_prom_metrics_records_each_metric_type_per_engine():
    prom_metrics = _prom_metrics()
    stats = TypedKVConnectorStats()
    _record(stats, hits=3, depth=5, latency=0.2)

    prom_metrics.observe(stats.to_dict(), engine_idx=1)

    assert prom_metrics.metrics[(1, HITS, ())].increments == [3]
    assert prom_metrics.metrics[(1, DEPTH, ())].set_values == [5]
    assert prom_metrics.metrics[(1, LATENCY, ())].observed == [0.2]
    labeled = prom_metrics.metrics[(1, LABELED, ("a",))]
    assert labeled.increments == [1]
    assert labeled.labelvalues == ("model", "1", "a")
    assert prom_metrics._metric_defs[LATENCY].kwargs == {
        "name": LATENCY,
        "documentation": "latency",
        "labelnames": ["model_name", "engine"],
        "buckets": (0.1, 1.0),
    }
    assert (0, HITS, ()) not in prom_metrics.metrics


@pytest.mark.parametrize(
    "payload",
    [
        {
            StatsKey.TYPES: {"vllm:undeclared": MetricType.COUNTER},
            StatsKey.DATA: {"vllm:undeclared": {(): 1}},
        },
        {
            StatsKey.TYPES: {LABELED: MetricType.COUNTER},
            StatsKey.DATA: {LABELED: {(): 1}},
        },
    ],
    ids=["undeclared_metric", "wrong_label_count"],
)
def test_prom_metrics_rejects_payloads_that_do_not_match_definitions(payload):
    with pytest.raises(AssertionError):
        _prom_metrics().observe(payload)
