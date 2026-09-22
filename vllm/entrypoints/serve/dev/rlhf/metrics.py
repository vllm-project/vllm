# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Telemetry for logical frontend weight operations.

A single HTTP request maps to zero or more *logical frontend weight operations*::

    HTTP request -> input validation -> 0..N logical weight operations
                                     -> outcome / duration / concurrency

Each operation wraps exactly one dispatched engine call, so one
``/finish_weight_update`` with a ``weight_version`` records two operations:
``finish`` for ``engine.finish_weight_update()`` and ``set_version`` for
``engine.update_weight_version()``. Keeping them separate means "finish succeeded,
version handshake failed" is visible instead of being flattened into a single
``finish=error``.

Not modelled here: whole transfer-session duration, RL transaction/lifecycle
state, weight-version labels, or whether the transferred weights were correct.
The metric names say ``operations``, not ``requests``, for that reason: do not
sum them across the ``operation`` label.

Scope of one observation:

- validation failures (malformed JSON, missing/invalid fields) never enter the
  recorder, so they do not touch any of these series; endpoint-level 4xx already
  has the HTTP metrics;
- an exception or a cancellation (client disconnect) inside the block is recorded
  as ``status="error"``;
- durations cover one operation, not a transfer session.

Collectors are instantiated lazily, on first use, so that metric objects are
created after multiprocess Prometheus setup has run (``PROMETHEUS_MULTIPROC_DIR``
must be set before a metric object exists). They are created on the default
registry so that ``prometheus_client`` writes multiprocess mmap data that the
``/metrics`` scrape registry aggregates; the scrape registry is not the creation
registry.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from typing import Literal

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

Operation = Literal["init", "start", "start_draft", "update", "finish", "set_version"]

_OPERATION_HELP = (
    "Logical frontend weight operations by outcome, one observation per "
    "dispatched engine call. 'finish' covers only finish_weight_update(); the "
    "weight-version handshake is counted separately as 'set_version'."
)

_OPERATIONS_NAME = "vllm:rl_weight_update_operations_total"
_DURATION_NAME = "vllm:rl_weight_update_operation_duration_seconds"
_IN_FLIGHT_NAME = "vllm:rl_weight_update_operations_in_flight"

_BUCKETS = (0.01, 0.1, 1, 10, 30, 60, 120, 300, 600)


class WeightOperationMetrics:
    """Bounded-label metrics for successful, failed, and cancelled operations.

    Args:
        registry: Collector registry to create the metrics on. Defaults to the
            process default registry, which is what ``prometheus_client`` expects
            for multiprocess mode: the metric objects become mmap-backed and the
            ``/metrics`` scrape registry aggregates them through
            ``MultiProcessCollector``. Tests pass a private registry for isolation.

    """

    def __init__(self, registry: CollectorRegistry | None = None):
        # Passing ``registry=None`` explicitly means "do not register" to
        # prometheus_client, so the default has to be the registry object itself.
        try:
            if registry is None:
                self.operations = Counter(
                    _OPERATIONS_NAME, _OPERATION_HELP, ["operation", "status"]
                )
                self.duration = Histogram(
                    _DURATION_NAME,
                    "Duration of one logical frontend weight operation.",
                    ["operation"],
                    buckets=_BUCKETS,
                )
                self.in_flight = Gauge(
                    _IN_FLIGHT_NAME,
                    "Logical frontend weight operations currently awaited.",
                    ["operation"],
                    multiprocess_mode="livesum",
                )
            else:
                self.operations = Counter(
                    _OPERATIONS_NAME,
                    _OPERATION_HELP,
                    ["operation", "status"],
                    registry=registry,
                )
                self.duration = Histogram(
                    _DURATION_NAME,
                    "Duration of one logical frontend weight operation.",
                    ["operation"],
                    registry=registry,
                    buckets=_BUCKETS,
                )
                self.in_flight = Gauge(
                    _IN_FLIGHT_NAME,
                    "Logical frontend weight operations currently awaited.",
                    ["operation"],
                    registry=registry,
                    multiprocess_mode="livesum",
                )
        except ValueError:
            # The collectors are already registered (module reload, or a test that
            # resets the singleton). The default registry keeps one collector per
            # sample name for the process lifetime, so adopt the existing ones.
            if registry is not None:
                raise
            self._adopt_registered_collectors()

    def _adopt_registered_collectors(self) -> None:
        self.operations = REGISTRY._names_to_collectors[_OPERATIONS_NAME]
        self.duration = REGISTRY._names_to_collectors[_DURATION_NAME]
        self.in_flight = REGISTRY._names_to_collectors[_IN_FLIGHT_NAME]

    @contextmanager
    def record(self, operation: Operation) -> Iterator[None]:
        """Record one operation; anything but a clean exit is an error."""
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
            self.operations.labels(operation, status).inc()


_metrics: WeightOperationMetrics | None = None
_metrics_lock = Lock()


def weight_operation_metrics() -> WeightOperationMetrics:
    """Return the process-wide recorder, creating it on first use.

    Lazy so the metric objects exist only after multiprocess Prometheus setup.
    """
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = WeightOperationMetrics()
    return _metrics
