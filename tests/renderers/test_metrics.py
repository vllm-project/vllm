# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.renderers.metrics."""

import pytest
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from vllm.renderers import metrics
from vllm.renderers.metrics import observe_request_render_duration
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

METRIC_NAME = "vllm:request_render_duration_seconds"


@pytest.fixture(autouse=True)
def clean_prometheus_registry():
    """Isolate the global registry so tests are order-independent."""
    metrics._request_render_duration.cache_clear()
    unregister_vllm_metrics()
    yield
    metrics._request_render_duration.cache_clear()
    unregister_vllm_metrics()


def _histogram_samples() -> list:
    exposition = generate_latest(REGISTRY).decode()
    for family in text_string_to_metric_families(exposition):
        if family.name == METRIC_NAME:
            return family.samples
    return []


def test_metric_registered_lazily():
    assert METRIC_NAME not in generate_latest(REGISTRY).decode()

    observe_request_render_duration("test-model", "chat", 0.001)

    assert METRIC_NAME in generate_latest(REGISTRY).decode()


def test_observe_records_duration_with_labels():
    observe_request_render_duration("test-model", "chat", 0.5)
    observe_request_render_duration("test-model", "chat", 1.5)
    observe_request_render_duration("test-model", "completion", 0.25)

    samples = {
        (s.labels["model_name"], s.labels["request_type"], s.name): s.value
        for s in _histogram_samples()
    }
    assert samples[("test-model", "chat", f"{METRIC_NAME}_count")] == 2
    assert samples[("test-model", "chat", f"{METRIC_NAME}_sum")] == 2.0
    assert samples[("test-model", "completion", f"{METRIC_NAME}_count")] == 1
    assert samples[("test-model", "completion", f"{METRIC_NAME}_sum")] == 0.25


def test_bucket_range():
    observe_request_render_duration("test-model", "chat", 5.0)

    finite_buckets = sorted(
        float(s.labels["le"])
        for s in _histogram_samples()
        if s.name == f"{METRIC_NAME}_bucket" and s.labels["le"] != "+Inf"
    )
    assert finite_buckets == [
        0.0001,
        0.00025,
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
    ]
