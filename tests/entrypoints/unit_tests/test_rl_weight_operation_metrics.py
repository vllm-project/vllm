# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RPC outcomes and overlapping requests; no model or GPU is required."""

import asyncio

import pytest
from prometheus_client import CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics
from vllm.entrypoints.serve.dev.rlhf.metrics import WeightOperationMetrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
PREFIX = "vllm:rl_weight_update_"


@pytest.mark.parametrize("error", [None, RuntimeError, asyncio.CancelledError])
def test_records_outcomes_and_releases_gauge(error):
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    labels = {"operation": "update"}

    def run():
        with metrics.record("update"):
            assert registry.get_sample_value(PREFIX + "requests_in_flight", labels) == 1
            if error:
                raise error("transfer interrupted")

    if error:
        with pytest.raises(error, match="transfer interrupted"):
            run()
    else:
        run()
    assert registry.get_sample_value(PREFIX + "requests_in_flight", labels) == 0
    status = "error" if error else "success"
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {**labels, "status": status}
        )
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "request_duration_seconds_count", labels)
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "request_duration_seconds_sum", labels) >= 0
    )


def test_overlapping_requests_do_not_clear_each_others_gauge():
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    labels = {"operation": "update"}
    gauge = PREFIX + "requests_in_flight"
    with metrics.record("update"):
        with metrics.record("update"):
            assert registry.get_sample_value(gauge, labels) == 2
        assert registry.get_sample_value(gauge, labels) == 1
    assert registry.get_sample_value(gauge, labels) == 0
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {**labels, "status": "success"}
        )
        == 2
    )


def test_metrics_bind_to_the_registry_returned_by_the_factory(monkeypatch):
    """Import-time creation would bind to whatever registry was default then."""
    registry = CollectorRegistry()
    monkeypatch.setattr(rlhf_metrics, "get_prometheus_registry", lambda: registry)
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)

    metrics = rlhf_metrics.weight_operation_metrics()

    assert isinstance(metrics, WeightOperationMetrics)
    assert PREFIX + "requests_total" in registry._names_to_collectors
    assert PREFIX + "request_duration_seconds" in registry._names_to_collectors
    assert PREFIX + "requests_in_flight" in registry._names_to_collectors


def test_recorder_is_created_once(monkeypatch):
    registries = []
    registry = CollectorRegistry()

    def factory():
        registries.append(registry)
        return registry

    monkeypatch.setattr(rlhf_metrics, "get_prometheus_registry", factory)
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)

    assert (
        rlhf_metrics.weight_operation_metrics()
        is rlhf_metrics.weight_operation_metrics()
    )
    assert len(registries) == 1
