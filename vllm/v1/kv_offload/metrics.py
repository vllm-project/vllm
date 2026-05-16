# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Literal

OffloadingMetricType = Literal["counter", "gauge", "histogram"]

@dataclass(frozen=True)
class OffloadingMetricMetadata:
    name: str
    documentation: str
    metric_type: OffloadingMetricType
    buckets: tuple[float, ...] | None = None
