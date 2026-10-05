# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Duration and in-flight metrics for RL weight-transfer operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from typing import Literal, get_args

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, Histogram

Operation = Literal["init", "start", "start_draft", "update", "finish"]


class WeightOperationMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.duration = Histogram(
            "vllm:rl_weight_update_operation_duration_seconds",
            "Duration of one weight-transfer operation.",
            ["operation"],
            buckets=(0.01, 0.1, 1, 10, 30, 60, 120, 300, 600),
            registry=registry,
        )
        self.in_flight = Gauge(
            "vllm:rl_weight_update_operations_in_flight",
            "Weight-transfer operations currently awaited.",
            ["operation"],
            multiprocess_mode="livesum",
            registry=registry,
        )

        for operation in get_args(Operation):
            self.in_flight.labels(operation).set(0)

    @contextmanager
    def record(self, operation: Operation) -> Iterator[None]:
        in_flight = self.in_flight.labels(operation)
        in_flight.inc()
        started = perf_counter()
        try:
            yield
        finally:
            in_flight.dec()
            self.duration.labels(operation).observe(perf_counter() - started)


_metrics: WeightOperationMetrics | None = None
_metrics_lock = Lock()


def weight_operation_metrics() -> WeightOperationMetrics:
    # Lazy, so the collectors exist only after multiprocess Prometheus setup.
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = WeightOperationMetrics()
    return _metrics
