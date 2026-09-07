# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP weight-operation telemetry, not cluster-wide transfer-session state.

Only dispatched RPCs are counted. Rejected input never enters the recorder.
Durations cover individual RPC awaits, not the interval from start to finish.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Literal

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

Operation = Literal["init", "start", "start_draft", "update", "finish"]


class WeightOperationMetrics:
    """Bounded-label metrics for successful, failed, and cancelled RPCs."""

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.requests = Counter(
            "vllm:rl_weight_update_requests_total",
            "Dispatched HTTP weight-operation RPCs by outcome.",
            ["operation", "status"],
            registry=registry,
        )
        self.duration = Histogram(
            "vllm:rl_weight_update_request_duration_seconds",
            "Duration of an individual HTTP weight-operation RPC.",
            ["operation"],
            registry=registry,
            buckets=(0.01, 0.1, 1, 10, 30, 60, 120, 300, 600),
        )
        self.in_flight = Gauge(
            "vllm:rl_weight_update_requests_in_flight",
            "Currently awaited HTTP weight-operation RPCs.",
            ["operation"],
            registry=registry,
            multiprocess_mode="livesum",
        )

    @contextmanager
    def record(self, operation: Operation) -> Iterator[None]:
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


weight_operation_metrics = WeightOperationMetrics()
