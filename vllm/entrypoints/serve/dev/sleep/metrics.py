# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics for dispatched sleep-mode frontend operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from typing import Literal, get_args

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, Histogram

Operation = Literal["sleep", "release_kv_cache_memory", "wake"]
BUCKETS = (0.01, 0.1, 1, 10, 30, 60, 120, 300, 600)


class SleepModeOperationMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        registry = REGISTRY if registry is None else registry
        self.duration = Histogram(
            "vllm:sleep_mode_operation_duration_seconds",
            "Duration of one sleep-mode operation.",
            ["operation"],
            buckets=BUCKETS,
            registry=registry,
        )
        self.in_flight = Gauge(
            "vllm:sleep_mode_operations_in_flight",
            "Sleep-mode operations currently awaited.",
            ["operation"],
            multiprocess_mode="livesum",
            registry=registry,
        )

        for operation in get_args(Operation):
            self.in_flight.labels(operation).set(0)

    @contextmanager
    def record(self, operation: Operation) -> Iterator[None]:
        active = self.in_flight.labels(operation)
        started = perf_counter()
        active.inc()
        try:
            yield
        finally:
            active.dec()
            self.duration.labels(operation).observe(perf_counter() - started)


_metrics: SleepModeOperationMetrics | None = None
_metrics_lock = Lock()


def sleep_mode_operation_metrics() -> SleepModeOperationMetrics:
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = SleepModeOperationMetrics()
    return _metrics
