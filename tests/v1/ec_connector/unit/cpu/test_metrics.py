# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnectorStats, the CPU EC connector's flat stats
container (mirrors OffloadingConnectorStats's counter/histogram contract,
but rooted in ECConnectorStats)."""

from unittest.mock import Mock

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.ec_transfer.ec_connector.cpu.metrics import (
    ECCPUConnectorProm,
    ECCPUConnectorStats,
    ECCPUMetricName,
    _MetricType,
    _StatsKey,
)

pytestmark = pytest.mark.cpu_test

SAVE_BYTES = ECCPUMetricName.SAVE_BYTES
LOAD_BYTES = ECCPUMetricName.LOAD_BYTES
SAVE_SIZE = ECCPUMetricName.SAVE_SIZE


class _FakeMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observed: list[int | float] = []
        self.increments: list[int | float] = []
        self.labelvalues: tuple[object, ...] = ()

    def labels(self, *labelvalues):
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        return child

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value):
        self.increments.append(value)


def _make_prom(per_engine_labelvalues=None):
    return ECCPUConnectorProm(
        vllm_config=Mock(),
        metric_types={Gauge: _FakeMetric, Counter: _FakeMetric, Histogram: _FakeMetric},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues=per_engine_labelvalues or {0: ["model", "0"]},
    )


def test_fresh_stats_are_empty():
    assert ECCPUConnectorStats().is_empty()


def test_increase_counter_accumulates():
    stats = ECCPUConnectorStats()

    stats.increase_counter(SAVE_BYTES, 24)
    stats.increase_counter(SAVE_BYTES, 6)

    assert not stats.is_empty()
    assert stats.reduce()[SAVE_BYTES] == 30


def test_observe_histogram_accumulates_samples():
    stats = ECCPUConnectorStats()

    stats.observe_histogram(SAVE_SIZE, 16)
    stats.observe_histogram(SAVE_SIZE, 8)

    reduced = stats.reduce()
    assert reduced[f"{SAVE_SIZE}_count"] == 2
    assert reduced[f"{SAVE_SIZE}_sum"] == 24


def test_aggregate_sums_counters_from_both():
    stats1 = ECCPUConnectorStats()
    stats1.increase_counter(SAVE_BYTES, 24)
    stats2 = ECCPUConnectorStats()
    stats2.increase_counter(SAVE_BYTES, 10)
    stats2.increase_counter(LOAD_BYTES, 5)

    result = stats1.aggregate(stats2)

    assert result is stats1
    reduced = result.reduce()
    assert reduced[SAVE_BYTES] == 34
    assert reduced[LOAD_BYTES] == 5


def test_reset_clears_stats():
    stats = ECCPUConnectorStats()
    stats.increase_counter(SAVE_BYTES, 24)
    assert not stats.is_empty()

    stats.reset()

    assert stats.is_empty()


def test_prom_observes_counter():
    prom = _make_prom()

    prom.observe(
        {
            _StatsKey.TYPES: {SAVE_BYTES: _MetricType.COUNTER},
            _StatsKey.DATA: {SAVE_BYTES: 24},
        }
    )

    metric = prom._bound[(0, SAVE_BYTES)]
    assert metric.increments == [24]
    assert metric.labelvalues == ("model", "0")


def test_prom_observes_histogram():
    prom = _make_prom()

    prom.observe(
        {
            _StatsKey.TYPES: {SAVE_SIZE: _MetricType.HISTOGRAM},
            _StatsKey.DATA: {SAVE_SIZE: [16, 8]},
        }
    )

    metric = prom._bound[(0, SAVE_SIZE)]
    assert metric.observed == [16, 8]


def test_prom_routes_to_correct_engine():
    prom = _make_prom({0: ["model", "0"], 1: ["model", "1"]})

    prom.observe(
        {
            _StatsKey.TYPES: {SAVE_BYTES: _MetricType.COUNTER},
            _StatsKey.DATA: {SAVE_BYTES: 100},
        },
        engine_idx=1,
    )

    assert prom._bound[(0, SAVE_BYTES)].increments == []
    assert prom._bound[(1, SAVE_BYTES)].increments == [100]
