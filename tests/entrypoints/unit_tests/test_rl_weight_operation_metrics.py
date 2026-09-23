# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recorder semantics for logical weight operations; no model or GPU required."""

import asyncio

import pytest
from prometheus_client import CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics
from vllm.entrypoints.serve.dev.rlhf.metrics import WeightOperationMetrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
PREFIX = "vllm:rl_weight_update_"


@pytest.mark.parametrize("error", [None, RuntimeError, asyncio.CancelledError])
def test_records_outcomes_and_releases_gauge(error):
    """A clean exit is success; an exception or cancellation is error."""
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    labels = {"operation": "update"}

    def run():
        with metrics.record("update"):
            assert (
                registry.get_sample_value(PREFIX + "operations_in_flight", labels) == 1
            )
            if error:
                raise error("transfer interrupted")

    if error:
        with pytest.raises(error, match="transfer interrupted"):
            run()
    else:
        run()
    assert registry.get_sample_value(PREFIX + "operations_in_flight", labels) == 0
    status = "error" if error else "success"
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {**labels, "status": status}
        )
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "operation_duration_seconds_count", labels)
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "operation_duration_seconds_sum", labels)
        >= 0
    )


def test_overlapping_operations_do_not_clear_each_others_gauge():
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    labels = {"operation": "update"}
    gauge = PREFIX + "operations_in_flight"
    with metrics.record("update"):
        with metrics.record("update"):
            assert registry.get_sample_value(gauge, labels) == 2
        assert registry.get_sample_value(gauge, labels) == 1
    assert registry.get_sample_value(gauge, labels) == 0
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {**labels, "status": "success"}
        )
        == 2
    )


def test_finish_and_set_version_are_distinct_operations():
    """One request can produce two observations; the failure stays attributed."""
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)

    with metrics.record("finish"):
        pass
    with (
        pytest.raises(RuntimeError, match="version rejected"),
        metrics.record("set_version"),
    ):
        raise RuntimeError("version rejected")

    assert (
        registry.get_sample_value(
            PREFIX + "operations_total",
            {"operation": "finish", "status": "success"},
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total",
            {"operation": "set_version", "status": "error"},
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total",
            {"operation": "finish", "status": "error"},
        )
        is None
    )


def test_metrics_are_created_lazily(monkeypatch):
    """The accessor constructs once, on first call, and caches it.

    Uses a counting subclass on a private registry so this does not depend on the
    process-wide default registry. That importing the module registers no
    collector is asserted in the multiprocess subprocess test, and the real
    default-registry path in test_rl_weight_operation_single_process.py.
    """
    created = []

    class _Counting(WeightOperationMetrics):
        def __init__(self, registry=None):
            created.append(1)
            super().__init__(CollectorRegistry())

    monkeypatch.setattr(rlhf_metrics, "WeightOperationMetrics", _Counting)
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)

    first = rlhf_metrics.weight_operation_metrics()
    second = rlhf_metrics.weight_operation_metrics()

    assert first is second
    assert len(created) == 1
