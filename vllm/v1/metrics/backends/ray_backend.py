# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray metrics implementation of the metrics backend.

This module provides a Ray-specific implementation of the metrics
abstraction layer, wrapping Ray's util.metrics library.
"""

from typing import Any

try:
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric

    RAY_AVAILABLE = True
except ImportError:
    ray_metrics = None
    Metric = None  # type: ignore[misc,assignment]
    RAY_AVAILABLE = False

import regex as re

try:
    from ray import serve as ray_serve
except ImportError:
    ray_serve = None

from vllm.v1.metrics.backends.abstract import (
    AbstractCounter,
    AbstractGauge,
    AbstractHistogram,
    MetricBackend,
)


def _get_replica_id() -> str | None:
    """Get the current Ray Serve replica ID, or None if not in a Serve context."""
    if ray_serve is None:
        return None
    try:
        return ray_serve.get_replica_context().replica_id.unique_id
    except ray_serve.exceptions.RayServeException:
        return None


def _get_sanitized_opentelemetry_name(name: str) -> str:
    """
    For compatibility with Ray + OpenTelemetry, the metric name must be
    sanitized. In particular, this replaces disallowed character (e.g., ':')
    with '_' in the metric name.
    Allowed characters: a-z, A-Z, 0-9, _

    # ruff: noqa: E501
    Ref: https://github.com/open-telemetry/opentelemetry-cpp/blob/main/sdk/src/metrics/instrument_metadata_validator.cc#L22-L23
    Ref: https://github.com/ray-project/ray/blob/master/src/ray/stats/metric.cc#L107
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _get_tag_keys(labelnames: list[str] | None) -> tuple[str, ...]:
    """Get tag keys for Ray metrics, adding ReplicaId."""
    labels = list(labelnames) if labelnames else []
    labels.append("ReplicaId")
    return tuple(labels)


class RayCounter(AbstractCounter):
    """Ray counter implementation."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Cannot use RayCounter.")

        super().__init__(name, documentation, labelnames, **kwargs)

        tag_keys = _get_tag_keys(labelnames)
        name = _get_sanitized_opentelemetry_name(name)

        self._counter: Metric = ray_metrics.Counter(
            name=name, description=documentation, tag_keys=tag_keys
        )

    def labels(self, *labelvalues: str, **labelkwargs: str) -> "RayCounter":
        """Create a labeled instance of this counter."""
        labeled_counter = RayCounter.__new__(RayCounter)
        labeled_counter.name = self.name
        labeled_counter.documentation = self.documentation
        labeled_counter.labelnames = self.labelnames
        labeled_counter.extra_params = self.extra_params
        labeled_counter._counter = self._counter

        # Apply labels
        if labelvalues:
            # -1 because ReplicaId was added automatically
            expected = len(self._counter._tag_keys) - 1
            if len(labelvalues) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labelvalues)}"
                )
            labelkwargs.update(zip(self._counter._tag_keys, labelvalues))

        labelkwargs["ReplicaId"] = _get_replica_id() or ""

        if labelkwargs:
            for k, v in labelkwargs.items():
                if not isinstance(v, str):
                    labelkwargs[k] = str(v)
            labeled_counter._counter.set_default_tags(labelkwargs)

        return labeled_counter

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter."""
        if amount == 0:
            return
        self._counter.inc(amount)


class RayGauge(AbstractGauge):
    """Ray gauge implementation."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Cannot use RayGauge.")

        # All Ray metrics are keyed by WorkerId, so multiprocess modes like
        # "mostrecent", "all", "sum" do not apply. This logic can be manually
        # implemented at the observability layer (Prometheus/Grafana).
        # Remove multiprocess_mode from kwargs if present
        kwargs.pop("multiprocess_mode", None)

        super().__init__(name, documentation, labelnames, **kwargs)

        tag_keys = _get_tag_keys(labelnames)
        name = _get_sanitized_opentelemetry_name(name)

        self._gauge: Metric = ray_metrics.Gauge(
            name=name, description=documentation, tag_keys=tag_keys
        )

    def labels(self, *labelvalues: str, **labelkwargs: str) -> "RayGauge":
        """Create a labeled instance of this gauge."""
        labeled_gauge = RayGauge.__new__(RayGauge)
        labeled_gauge.name = self.name
        labeled_gauge.documentation = self.documentation
        labeled_gauge.labelnames = self.labelnames
        labeled_gauge.extra_params = self.extra_params
        labeled_gauge._gauge = self._gauge

        # Apply labels
        if labelvalues:
            # -1 because ReplicaId was added automatically
            expected = len(self._gauge._tag_keys) - 1
            if len(labelvalues) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labelvalues)}"
                )
            labelkwargs.update(zip(self._gauge._tag_keys, labelvalues))

        labelkwargs["ReplicaId"] = _get_replica_id() or ""

        if labelkwargs:
            for k, v in labelkwargs.items():
                if not isinstance(v, str):
                    labelkwargs[k] = str(v)
            labeled_gauge._gauge.set_default_tags(labelkwargs)

        return labeled_gauge

    def set(self, value: float) -> None:
        """Set the gauge to a specific value."""
        self._gauge.set(value)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge."""
        self._gauge.inc(amount)

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge."""
        self._gauge.dec(amount)


class RayHistogram(AbstractHistogram):
    """Ray histogram implementation."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
        **kwargs: Any,
    ):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Cannot use RayHistogram.")

        super().__init__(name, documentation, labelnames, **kwargs)

        tag_keys = _get_tag_keys(labelnames)
        name = _get_sanitized_opentelemetry_name(name)
        boundaries = buckets if buckets else []

        self._histogram: Metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
            boundaries=boundaries,
        )

    def labels(self, *labelvalues: str, **labelkwargs: str) -> "RayHistogram":
        """Create a labeled instance of this histogram."""
        labeled_histogram = RayHistogram.__new__(RayHistogram)
        labeled_histogram.name = self.name
        labeled_histogram.documentation = self.documentation
        labeled_histogram.labelnames = self.labelnames
        labeled_histogram.extra_params = self.extra_params
        labeled_histogram._histogram = self._histogram

        # Apply labels
        if labelvalues:
            # -1 because ReplicaId was added automatically
            expected = len(self._histogram._tag_keys) - 1
            if len(labelvalues) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labelvalues)}"
                )
            labelkwargs.update(zip(self._histogram._tag_keys, labelvalues))

        labelkwargs["ReplicaId"] = _get_replica_id() or ""

        if labelkwargs:
            for k, v in labelkwargs.items():
                if not isinstance(v, str):
                    labelkwargs[k] = str(v)
            labeled_histogram._histogram.set_default_tags(labelkwargs)

        return labeled_histogram

    def observe(self, amount: float) -> None:
        """Observe a value."""
        self._histogram.observe(amount)


class RayBackend(MetricBackend):
    """Ray implementation of the metric backend factory."""

    def __init__(self):
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is not installed. Cannot use RayBackend. "
                "Please install Ray with: pip install ray"
            )

    def create_counter(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ) -> RayCounter:
        """Create a Ray counter metric."""
        return RayCounter(name, documentation, labelnames, **kwargs)

    def create_gauge(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ) -> RayGauge:
        """Create a Ray gauge metric."""
        return RayGauge(name, documentation, labelnames, **kwargs)

    def create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
        **kwargs: Any,
    ) -> RayHistogram:
        """Create a Ray histogram metric."""
        return RayHistogram(name, documentation, labelnames, buckets, **kwargs)
