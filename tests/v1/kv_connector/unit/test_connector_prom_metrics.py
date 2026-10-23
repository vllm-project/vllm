# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the shared KV connector Prometheus declarations."""

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    CounterSpec,
    HistogramSpec,
    KVConnectorPromMetrics,
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
    MooncakePromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

TEST_MODEL = "test-model"


class _DeclaredPromMetrics(KVConnectorPromMetrics):
    """Smallest subclass that relies on the shared registration and recording."""

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


class _UndeclaredPromMetrics(KVConnectorPromMetrics):
    """Declares no series and does not override observe()."""


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
    """A declaration publishes its series at construction, so they exist at zero
    as soon as the connector's metrics are built."""
    _build(_DeclaredPromMetrics, {0: [TEST_MODEL, "0"]})

    assert _sample("vllm:test_duration_seconds_count") == 0
    assert _sample("vllm:test_failed_total") == 0


def test_samples_are_routed_to_the_reported_engine():
    metrics = _build(_DeclaredPromMetrics, {0: [TEST_MODEL, "0"], 1: [TEST_MODEL, "1"]})

    metrics.observe({"duration": [0.4], "failed_a": [1], "failed_b": []}, engine_idx=1)

    assert _sample("vllm:test_duration_seconds_count", engine_idx=0) == 0
    assert _sample("vllm:test_duration_seconds_count", engine_idx=1) == 1
    assert _sample("vllm:test_failed_total", engine_idx=0) == 0
    assert _sample("vllm:test_failed_total", engine_idx=1) == 1


def test_a_counter_sums_every_key_it_declares():
    """Several keys feed one series when different code paths report the same
    kind of event."""
    metrics = _build(_DeclaredPromMetrics, {0: [TEST_MODEL, "0"]})

    metrics.observe({"duration": [], "failed_a": [1, 1], "failed_b": [1]})

    assert _sample("vllm:test_failed_total") == 3


def test_a_payload_missing_a_declared_key_raises():
    """A renamed stats key must surface instead of recording zeros."""
    metrics = _build(_DeclaredPromMetrics, {0: [TEST_MODEL, "0"]})

    with pytest.raises(KeyError):
        metrics.observe({"duration": []})


def test_a_subclass_without_declarations_must_override_observe():
    """Declaring nothing leaves nothing to record into, which is a connector
    bug rather than a silent no-op."""
    metrics = _build(_UndeclaredPromMetrics, {0: [TEST_MODEL, "0"]})

    with pytest.raises(NotImplementedError, match="_UndeclaredPromMetrics"):
        metrics.observe({})


@pytest.mark.parametrize(
    "metric_cls, stats_cls",
    [
        (NixlPromMetrics, NixlKVConnectorStats),
        (MooncakePromMetrics, MooncakeKVConnectorStats),
    ],
)
def test_connector_series_read_keys_the_stats_payload_provides(
    metric_cls: type[KVConnectorPromMetrics],
    stats_cls: type[KVConnectorStats],
):
    """Series keys are plain strings, so pin them against the payload the
    connector actually builds."""
    declared: set[str] = {spec.stats_key for spec in metric_cls.HISTOGRAM_SPECS}
    for counter_spec in metric_cls.COUNTER_SPECS:
        declared.update(counter_spec.stats_keys)

    assert declared <= set(stats_cls().to_dict())
