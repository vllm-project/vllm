# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP weight-operation telemetry, not cluster-wide transfer-session state.

Only dispatched operations are counted. Rejected input never enters the recorder.
Durations cover one frontend weight operation, not a transfer-session lifetime.
A cancelled operation (client disconnect) is recorded as ``status="error"``.

Collectors are created on first use on ``get_prometheus_registry()``, so they land
in the same registry the API server exposes on ``/metrics`` - including the
multiprocess registry selected by ``PROMETHEUS_MULTIPROC_DIR``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from typing import Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from vllm.v1.metrics.prometheus import get_prometheus_registry

Operation = Literal["init", "start", "start_draft", "update", "finish", "set_version"]


class WeightOperationMetrics:
    """Bounded-label metrics for successful, failed, and cancelled RPCs."""

    def __init__(self, registry: CollectorRegistry | None = None):
        if registry is None:
            registry = get_prometheus_registry()
        self.requests = Counter(
            "vllm:rl_weight_update_requests_total",
            "Dispatched HTTP weight operations by outcome. 'finish' excludes the "
            "weight-version handshake, which is counted separately as 'set_version'.",
            ["operation", "status"],
            registry=registry,
        )
        self.duration = Histogram(
            "vllm:rl_weight_update_request_duration_seconds",
            "Duration of a dispatched HTTP weight operation.",
            ["operation"],
            registry=registry,
            buckets=(0.01, 0.1, 1, 10, 30, 60, 120, 300, 600),
        )
        self.in_flight = Gauge(
            "vllm:rl_weight_update_requests_in_flight",
            "Currently awaited HTTP weight operations.",
            ["operation"],
            registry=registry,
            multiprocess_mode="livesum",
        )

    @contextmanager
    def record(self, operation: Operation) -> Iterator[None]:
        """Record one dispatched operation; anything but a clean exit is an error."""
        started = perf_counter()
        active = self.in_flight.labels(operation)
        active.inc()
        status = "error"
        try:
            yield
            status = "success"
        finally:
            active.dec()
            self.duration.labels(operation).observe(perf_counter() - started)
            self.requests.labels(operation, status).inc()


_metrics: WeightOperationMetrics | None = None
_metrics_lock = Lock()


def weight_operation_metrics() -> WeightOperationMetrics:
    """Return the process-wide recorder, creating it on first use."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = WeightOperationMetrics()
    return _metrics
