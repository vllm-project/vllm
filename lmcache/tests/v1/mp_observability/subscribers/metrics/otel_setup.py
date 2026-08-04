# SPDX-License-Identifier: Apache-2.0

"""Shared OTel MeterProvider + helpers for all metrics subscriber tests.

OTel only allows one MeterProvider per process.  This module sets it up
once so that every test file in this directory reads from the same reader,
and provides a small set of helpers for snapshotting and diffing counter /
histogram values via :data:`reader`.

Import as::

    from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
        reader,
        read_counters,
        histogram_count,
        counter_delta,
    )
"""

# Third Party
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

reader = InMemoryMetricReader()
_provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(_provider)


def read_counters() -> dict[str, int]:
    """Snapshot counter values, summed across all attribute combinations.

    Accumulates into the result dict (rather than overwriting) when the
    same metric name appears in multiple resource/scope buckets, which
    can happen when several test files have populated the shared reader.

    Returns:
        Mapping of metric name -> total value across all data points.
        Histograms are skipped (use :func:`histogram_count` instead).
    """
    data = reader.get_metrics_data()
    result: dict[str, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                total = 0
                any_value = False
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue  # skip histogram data points
                    total += int(dp.value)
                    any_value = True
                if any_value:
                    result[metric.name] = result.get(metric.name, 0) + total
    return result


def histogram_count(name: str) -> int:
    """Return the total observation count for the named histogram.

    Args:
        name: OTel metric name, e.g. ``"lmcache_mp.l1_usage_ratio"``.

    Returns:
        Sum of ``count`` across all data points for that histogram.
        Returns 0 if the histogram has no observations yet.
    """
    data = reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for dp in metric.data.data_points:
                    if hasattr(dp, "count"):
                        total += int(dp.count)
    return total


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Compute ``after - before`` for every metric name in either snapshot."""
    all_keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}
