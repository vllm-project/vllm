# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prometheus metrics for multi-modal preprocessing in APIServer (P0).

These metrics are defined directly in the APIServer process and are
independent of the EngineCore (P1+) metrics managed by
PrometheusStatLogger. They are automatically exposed via the /metrics
endpoint alongside all other vLLM Prometheus metrics.

Latency/size metrics use Histogram (not Gauge) so that every observation
is accumulated losslessly. A Gauge's ``.set()`` overwrites prior values
between scrapes, losing data under high QPS; Histogram accumulates all
observations into buckets and supports cross-instance percentile
aggregation via ``histogram_quantile(sum by (le)(rate(...)))``. For
cheap dashboard queries, precompute the quantile with a Prometheus
recording rule.

Metrics are registered against ``get_prometheus_registry()`` so they
are exposed at /metrics in both single- and multi-API-server modes (the
latter uses a multiprocess CollectorRegistry), matching the existing
APIServer-level Instrumentator setup.

All metrics use lazy initialization to avoid import-time side effects
and gracefully degrade if prometheus_client is not installed.
"""

from __future__ import annotations

import threading

# Lazy-initialized metric instances (populated on first access)
_metrics_lock = threading.Lock()
_metrics_initialized = False
_media_download_latency = None
_media_decode_latency = None
_media_download_bytes = None
_mm_resolve_items_latency = None
_mm_preprocessing_total_latency = None


def _ensure_metrics() -> bool:
    """Initialize Prometheus metrics on first call.

    Returns True if metrics are enabled, False otherwise.
    Gracefully degrades if prometheus_client is not installed.

    Thread-safe: concurrent callers (e.g. the global ThreadPoolExecutor
    used by MediaConnector) will not trigger duplicate registration.
    """
    global _metrics_initialized
    global _media_download_latency, _media_decode_latency
    global _media_download_bytes
    global _mm_resolve_items_latency, _mm_preprocessing_total_latency

    if _metrics_initialized:
        return _media_download_latency is not None

    with _metrics_lock:
        if _metrics_initialized:
            return _media_download_latency is not None

        try:
            from prometheus_client import Histogram
        except ImportError:
            _metrics_initialized = True
            return False

        from vllm.v1.metrics.prometheus import get_prometheus_registry

        registry = get_prometheus_registry()

        _media_download_latency = Histogram(
            name="vllm:mm_media_download_latency_seconds",
            documentation=(
                "Histogram of HTTP media download latency in seconds (per media item)."
            ),
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
            ],
            labelnames=["media_type"],
            registry=registry,
        )

        _media_decode_latency = Histogram(
            name="vllm:mm_media_decode_latency_seconds",
            documentation=(
                "Histogram of media decode/load_bytes latency in seconds "
                "(per media item, CPU-bound)."
            ),
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
            ],
            labelnames=["media_type"],
            registry=registry,
        )

        _media_download_bytes = Histogram(
            name="vllm:mm_media_download_bytes",
            documentation=(
                "Histogram of downloaded media size in bytes (per media item)."
            ),
            buckets=[
                1024,
                10240,
                102400,
                524288,
                1048576,
                5242880,
                10485760,
                52428800,
                104857600,
            ],
            labelnames=["media_type"],
            registry=registry,
        )

        _mm_resolve_items_latency = Histogram(
            name="vllm:mm_resolve_items_latency_seconds",
            documentation=(
                "Histogram of total multi-modal resolve_items latency in "
                "seconds (covers all concurrent downloads + decodes for "
                "one request)."
            ),
            buckets=[
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
            ],
            labelnames=[],
            registry=registry,
        )

        _mm_preprocessing_total_latency = Histogram(
            name="vllm:mm_preprocessing_total_latency_seconds",
            documentation=(
                "Histogram of total chat preprocessing latency in seconds "
                "per request (parse_chat_messages + apply_chat_template + "
                "await mm_data + tokenize)."
            ),
            buckets=[
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
            ],
            labelnames=[],
            registry=registry,
        )

        _metrics_initialized = True

    return True


def observe_media_download(
    media_type: str, elapsed_seconds: float, num_bytes: int
) -> None:
    """Record media download latency and size."""
    if not _ensure_metrics():
        return
    assert _media_download_latency is not None
    assert _media_download_bytes is not None
    _media_download_latency.labels(media_type=media_type).observe(elapsed_seconds)
    _media_download_bytes.labels(media_type=media_type).observe(num_bytes)


def observe_media_decode(media_type: str, elapsed_seconds: float) -> None:
    """Record media decode (load_bytes) latency."""
    if not _ensure_metrics():
        return
    assert _media_decode_latency is not None
    _media_decode_latency.labels(media_type=media_type).observe(elapsed_seconds)


def observe_resolve_items(elapsed_seconds: float) -> None:
    """Record total resolve_items latency for one request."""
    if not _ensure_metrics():
        return
    assert _mm_resolve_items_latency is not None
    _mm_resolve_items_latency.observe(elapsed_seconds)


def observe_preprocessing_total(elapsed_seconds: float) -> None:
    """Record total preprocessing latency for one request batch."""
    if not _ensure_metrics():
        return
    assert _mm_preprocessing_total_latency is not None
    _mm_preprocessing_total_latency.observe(elapsed_seconds)
