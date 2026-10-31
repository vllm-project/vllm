# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeAlias, TypeVar

from prometheus_client import Counter, Gauge, Histogram

PromMetric: TypeAlias = Gauge | Counter | Histogram
_KeyT = TypeVar("_KeyT")
_MetricT = TypeVar("_MetricT", bound=PromMetric)


def create_metric_per_engine(
    metric: _MetricT,
    per_engine_labelvalues: dict[_KeyT, list[object]],
) -> dict[_KeyT, _MetricT]:
    """Cache labeled metric children keyed by engine and optional dimensions."""
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
