# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prometheus metrics for multimodal media loading in the API server.

These metrics live in the frontend process, where media URLs are fetched and
encoded bytes are decoded before the request reaches the engine.
"""

from prometheus_client import Histogram

_MM_MEDIA_LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
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
)

_MM_MEDIA_BYTES_BUCKETS = (
    1 * 1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
)

_media_download_latency = Histogram(
    name="vllm:mm_media_download_latency_seconds",
    documentation="Latency of multimodal media downloads in seconds.",
    labelnames=("media_type",),
    buckets=_MM_MEDIA_LATENCY_BUCKETS,
)

_media_decode_latency = Histogram(
    name="vllm:mm_media_decode_latency_seconds",
    documentation="Latency of multimodal media byte decoding in seconds.",
    labelnames=("media_type",),
    buckets=_MM_MEDIA_LATENCY_BUCKETS,
)

_media_download_bytes = Histogram(
    name="vllm:mm_media_download_bytes",
    documentation="Size of downloaded multimodal media payloads in bytes.",
    labelnames=("media_type",),
    buckets=_MM_MEDIA_BYTES_BUCKETS,
)


def observe_media_download(media_type: str, duration_s: float, num_bytes: int) -> None:
    """Record a successful media download."""
    _media_download_latency.labels(media_type=media_type).observe(duration_s)
    _media_download_bytes.labels(media_type=media_type).observe(num_bytes)


def observe_media_decode(media_type: str, duration_s: float) -> None:
    """Record a successful conversion from encoded bytes/URL to media object."""
    _media_decode_latency.labels(media_type=media_type).observe(duration_s)
