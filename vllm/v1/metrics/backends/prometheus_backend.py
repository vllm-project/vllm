# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prometheus implementation of the metrics backend.

This module provides a Prometheus-specific implementation of the metrics
abstraction layer, wrapping prometheus_client metrics.
"""

from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from vllm.v1.metrics.backends.abstract import (
    AbstractCounter,
    AbstractGauge,
    AbstractHistogram,
    MetricBackend,
)


class PrometheusCounter(AbstractCounter):
    """Prometheus counter implementation."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name, documentation, labelnames, **kwargs)
        self._counter = Counter(
            name=name,
            documentation=documentation,
            labelnames=labelnames or [],
            **kwargs,
        )

    def labels(self, *labelvalues: str, **labelkwargs: str) -> "PrometheusCounter":
        """Create a labeled instance of this counter."""
        labeled_counter = PrometheusCounter.__new__(PrometheusCounter)
        labeled_counter.name = self.name
        labeled_counter.documentation = self.documentation
        labeled_counter.labelnames = self.labelnames
        labeled_counter.extra_params = self.extra_params
        labeled_counter._counter = self._counter.labels(*labelvalues, **labelkwargs)
        return labeled_counter

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter."""
        self._counter.inc(amount)


class PrometheusGauge(AbstractGauge):
    """Prometheus gauge implementation."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name, documentation, labelnames, **kwargs)
        self._gauge = Gauge(
            name=name,
            documentation=documentation,
            labelnames=labelnames or [],
            **kwargs,
        )

    def labels(self, *labelvalues: str, **labelkwargs: str) -> "PrometheusGauge":
        """Create a labeled instance of this gauge."""
        labeled_gauge = PrometheusGauge.__new__(PrometheusGauge)
        labeled_gauge.name = self.name
        labeled_gauge.documentation = self.documentation
        labeled_gauge.labelnames = self.labelnames
        labeled_gauge.extra_params = self.extra_params
        labeled_gauge._gauge = self._gauge.labels(*labelvalues, **labelkwargs)
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


class PrometheusHistogram(AbstractHistogram):
    """Prometheus histogram implementation."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name, documentation, labelnames, **kwargs)
        # If buckets are provided, add them to kwargs
        if buckets is not None:
            kwargs["buckets"] = buckets
        self._histogram = Histogram(
            name=name,
            documentation=documentation,
            labelnames=labelnames or [],
            **kwargs,
        )

    def labels(self, *labelvalues: str, **labelkwargs: str) -> "PrometheusHistogram":
        """Create a labeled instance of this histogram."""
        labeled_histogram = PrometheusHistogram.__new__(PrometheusHistogram)
        labeled_histogram.name = self.name
        labeled_histogram.documentation = self.documentation
        labeled_histogram.labelnames = self.labelnames
        labeled_histogram.extra_params = self.extra_params
        labeled_histogram._histogram = self._histogram.labels(
            *labelvalues, **labelkwargs
        )
        return labeled_histogram

    def observe(self, amount: float) -> None:
        """Observe a value."""
        self._histogram.observe(amount)


class PrometheusBackend(MetricBackend):
    """Prometheus implementation of the metric backend factory."""

    def create_counter(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ) -> PrometheusCounter:
        """Create a Prometheus counter metric."""
        return PrometheusCounter(name, documentation, labelnames, **kwargs)

    def create_gauge(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ) -> PrometheusGauge:
        """Create a Prometheus gauge metric."""
        return PrometheusGauge(name, documentation, labelnames, **kwargs)

    def create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
        **kwargs: Any,
    ) -> PrometheusHistogram:
        """Create a Prometheus histogram metric."""
        return PrometheusHistogram(name, documentation, labelnames, buckets, **kwargs)
