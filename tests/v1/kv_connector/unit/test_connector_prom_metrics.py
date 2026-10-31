# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract test for the shared KV connector Prometheus registration."""

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    CounterSpec,
    HistogramSpec,
    KVConnectorPromMetrics,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

TEST_MODEL = "test-model"


class _DeclaredPromMetrics(KVConnectorPromMetrics):
    """Smallest subclass that relies on the shared registration."""

    HISTOGRAM_SPECS = (
        HistogramSpec(
            stats_key="duration",
            name="vllm:test_duration_seconds",
            documentation="Test duration.",
            buckets=(0.5, 1.0),
        ),
    )
    COUNTER_SPECS = (
        CounterSpec(
            stats_keys=("failed_a", "failed_b"),
            name="vllm:test_failed",
            documentation="Test failures, reported through two keys.",
        ),
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """VLLM metrics share the process-wide registry, so drop them per test."""
    unregister_vllm_metrics()
    yield
    unregister_vllm_metrics()


def _labels(engine_idx: int = 0) -> dict[str, str]:
    return {"model_name": TEST_MODEL, "engine": str(engine_idx)}


def _sample(series: str, engine_idx: int = 0) -> float | None:
    return REGISTRY.get_sample_value(series, _labels(engine_idx))


def _build(
    metric_cls: type[KVConnectorPromMetrics],
    per_engine_labelvalues: dict[int, list[object]],
) -> KVConnectorPromMetrics:
    return metric_cls(
        vllm_config=VllmConfig(),
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues=per_engine_labelvalues,
    )


def test_declared_series_are_registered_before_any_observation():
    """A declaration publishes its series at construction, so a scrape taken
    before the first transfer still reports them, at zero."""
    _build(_DeclaredPromMetrics, {0: [TEST_MODEL, "0"]})

    assert _sample("vllm:test_duration_seconds_count") == 0
    assert _sample("vllm:test_failed_total") == 0


def test_a_counter_key_missing_from_the_payload_raises():
    """Only the counter path reaches a declared key that the payload omits, so
    a renamed event counter must surface here rather than record zero."""
    metrics = _build(_DeclaredPromMetrics, {0: [TEST_MODEL, "0"]})

    with pytest.raises(KeyError):
        metrics.observe({"duration": []})
