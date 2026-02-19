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
