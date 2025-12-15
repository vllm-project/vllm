# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Abstract base classes for metrics backend implementations.

This module defines the interface that all metrics backends must implement.
It provides abstract classes for different metric types (Counter, Gauge, Histogram)
and a MetricBackend factory interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class AbstractMetric(ABC):
    """Base class for all metric types."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize a metric.

        Args:
            name: Metric name (e.g., "vllm:num_requests_running")
            documentation: Human-readable description of the metric
            labelnames: List of label names for dimensional metrics
            **kwargs: Backend-specific parameters (e.g., multiprocess_mode)
        """
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
        self.extra_params = kwargs

    @abstractmethod
    def labels(self, *labelvalues: str, **labelkwargs: str) -> "AbstractMetric":
        """Create a labeled instance of this metric.

        Args:
            *labelvalues: Label values in the same order as labelnames
            **labelkwargs: Label values as keyword arguments

        Returns:
            A new metric instance with the specified labels
        """
        pass


class AbstractCounter(AbstractMetric):
    """Abstract counter metric (monotonically increasing value)."""

    @abstractmethod
    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter.

        Args:
            amount: Amount to increment by (default: 1.0)
        """
        pass


class AbstractGauge(AbstractMetric):
    """Abstract gauge metric (value that can go up or down)."""

    @abstractmethod
    def set(self, value: float) -> None:
        """Set the gauge to a specific value.

        Args:
            value: The value to set
        """
        pass

    @abstractmethod
    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge.

        Args:
            amount: Amount to increment by (default: 1.0)
        """
        pass

    @abstractmethod
    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge.

        Args:
            amount: Amount to decrement by (default: 1.0)
        """
        pass


class AbstractHistogram(AbstractMetric):
    """Abstract histogram metric (bucketed observations)."""

    @abstractmethod
    def observe(self, amount: float) -> None:
        """Observe a value.

        Args:
            amount: The value to observe
        """
        pass


class MetricBackend(ABC):
    """Abstract factory for creating metrics.

    Each backend (Prometheus, OpenTelemetry, etc.) implements this interface
    to provide backend-specific metric instances.
    """

    @abstractmethod
    def create_counter(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ) -> AbstractCounter:
        """Create a counter metric.

        Args:
            name: Metric name
            documentation: Human-readable description
            labelnames: List of label names
            **kwargs: Backend-specific parameters

        Returns:
            A counter metric instance
        """
        pass

    @abstractmethod
    def create_gauge(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        **kwargs: Any,
    ) -> AbstractGauge:
        """Create a gauge metric.

        Args:
            name: Metric name
            documentation: Human-readable description
            labelnames: List of label names
            **kwargs: Backend-specific parameters

        Returns:
            A gauge metric instance
        """
        pass

    @abstractmethod
    def create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
        **kwargs: Any,
    ) -> AbstractHistogram:
        """Create a histogram metric.

        Args:
            name: Metric name
            documentation: Human-readable description
            labelnames: List of label names
            buckets: Histogram buckets (backend-specific default if None)
            **kwargs: Backend-specific parameters

        Returns:
            A histogram metric instance
        """
        pass
