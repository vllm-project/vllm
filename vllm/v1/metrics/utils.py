# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""指标工具函数模块。

本模块提供指标相关的辅助函数，负责：
- 为每个引擎索引创建带标签的 Prometheus 指标子项

主要函数：
- create_metric_per_engine: 为每个引擎创建指标子项
"""

from typing import TypeAlias

from prometheus_client import Counter, Gauge, Histogram

PromMetric: TypeAlias = Gauge | Counter | Histogram


def create_metric_per_engine(
    metric: PromMetric,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, PromMetric]:
    """为每个引擎索引创建带标签的指标子项。

    Prometheus 多进程模式下，需要为每个引擎创建独立的指标实例，
    使用引擎索引作为标签进行区分。

    Args:
        metric: Prometheus 指标（Gauge、Counter 或 Histogram）
        per_engine_labelvalues: 引擎索引到标签值列表的映射

    Returns:
        引擎索引到指标子项的映射字典

    示例:
        >>> labelvalues = {0: ["model_a", "0"], 1: ["model_a", "1"]}
        >>> metrics = create_metric_per_engine(gauge, labelvalues)
        >>> metrics[0].set(100)  # 设置引擎 0 的指标值
        >>> metrics[1].set(200)  # 设置引擎 1 的指标值
    """
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
