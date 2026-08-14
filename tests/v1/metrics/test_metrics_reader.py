# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import prometheus_client
import pytest

from vllm.v1.metrics.reader import (
    Counter,
    Gauge,
    Histogram,
    Vector,
    get_metrics_snapshot,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def test_registry(monkeypatch):
    # Use a custom registry for tests
    test_registry = prometheus_client.CollectorRegistry(auto_describe=True)
    monkeypatch.setattr("vllm.v1.metrics.reader.REGISTRY", test_registry)
    return test_registry


@pytest.mark.parametrize("num_engines", [1, 4])
def test_gauge_metric(test_registry, num_engines):
    g = prometheus_client.Gauge(
        "vllm:test_gauge",
        "Test gauge metric",
        labelnames=["model", "engine_index"],
        registry=test_registry,
    )
    for i in range(num_engines):
        g.labels(model="foo", engine_index=str(i)).set(98.5)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Gauge)
        assert m.name == "vllm:test_gauge"
        assert m.value == 98.5
        assert m.labels["model"] == "foo"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])


@pytest.mark.parametrize("num_engines", [1, 4])
def test_counter_metric(test_registry, num_engines):
    c = prometheus_client.Counter(
        "vllm:test_counter",
        "Test counter metric",
        labelnames=["model", "engine_index"],
        registry=test_registry,
    )
    for i in range(num_engines):
        c.labels(model="bar", engine_index=str(i)).inc(19)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Counter)
        assert m.name == "vllm:test_counter"
        assert m.value == 19
        assert m.labels["model"] == "bar"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])


@pytest.mark.parametrize("num_engines", [1, 4])
def test_histogram_metric(test_registry, num_engines):
    h = prometheus_client.Histogram(
        "vllm:test_histogram",
        "Test histogram metric",
        labelnames=["model", "engine_index"],
        buckets=[10, 20, 30, 40, 50],
        registry=test_registry,
    )
    for i in range(num_engines):
        hist = h.labels(model="blaa", engine_index=str(i))
        hist.observe(42)
        hist.observe(21)
        hist.observe(7)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Histogram)
        assert m.name == "vllm:test_histogram"
        assert m.count == 3
        assert m.sum == 70
        assert m.buckets["10.0"] == 1
        assert m.buckets["20.0"] == 1
        assert m.buckets["30.0"] == 2
        assert m.buckets["40.0"] == 2
        assert m.buckets["50.0"] == 3
        assert m.labels["model"] == "blaa"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])


@pytest.mark.parametrize("num_engines", [1, 4])
def test_vector_metric(test_registry, num_engines):
    c = prometheus_client.Counter(
        "vllm:spec_decode_num_accepted_tokens_per_pos",
        "Vector-like counter metric",
        labelnames=["position", "model", "engine_index"],
        registry=test_registry,
    )
    for i in range(num_engines):
        c.labels(position="0", model="llama", engine_index=str(i)).inc(10)
        c.labels(position="1", model="llama", engine_index=str(i)).inc(5)
        c.labels(position="2", model="llama", engine_index=str(i)).inc(1)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Vector)
        assert m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
        assert m.values == [10, 5, 1]
        assert m.labels["model"] == "llama"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])
