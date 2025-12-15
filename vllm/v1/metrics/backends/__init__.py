# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Metrics backend abstraction layer.

This module provides a unified interface for metrics collection that supports
multiple backend implementations (Prometheus, OpenTelemetry, etc.).

The abstraction allows metrics to be defined once and exported to different
backends without code duplication.
"""

from vllm.v1.metrics.backends.abstract import (
    AbstractCounter,
    AbstractGauge,
    AbstractHistogram,
    MetricBackend,
)
from vllm.v1.metrics.backends.otel_backend import OTELBackend
from vllm.v1.metrics.backends.prometheus_backend import PrometheusBackend

__all__ = [
    "MetricBackend",
    "AbstractCounter",
    "AbstractGauge",
    "AbstractHistogram",
    "PrometheusBackend",
    "OTELBackend",
]
