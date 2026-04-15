# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeAlias

from prometheus_client import Counter, Gauge, Histogram

PromMetric: TypeAlias = Gauge | Counter | Histogram


def create_metric_per_engine(
    metric: PromMetric,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, PromMetric]:
    """Create a labeled metric child for each engine index."""
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }


# Default request-phase latency histogram upper bounds (seconds). The first bucket
# is 300ms; see `build_request_latency_buckets` for an optional finer low end.
_REQUEST_LATENCY_BUCKETS_BASE: list[float] = [
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
]

# Optional extra buckets (seconds) below 0.3s for fine-grained Prometheus histograms.
_REQUEST_LATENCY_BUCKETS_FINE_LOW_PREFIX: list[float] = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
]


def build_request_latency_buckets(*, fine_low_end: bool) -> list[float]:
    """Upper bounds (seconds) for request latency Prometheus histograms.

    When ``fine_low_end`` is True, prepends sub-300ms buckets; otherwise matches
    historical vLLM defaults (first finite bucket 0.3s).
    """
    if not fine_low_end:
        return list(_REQUEST_LATENCY_BUCKETS_BASE)
    merged = sorted(
        set(_REQUEST_LATENCY_BUCKETS_FINE_LOW_PREFIX + _REQUEST_LATENCY_BUCKETS_BASE)
    )
    return merged
