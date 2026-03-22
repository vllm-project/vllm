# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus 指标读取器模块。

本模块提供访问 vLLM Prometheus 指标的 API，负责：
- 定义指标数据类（Counter、Gauge、Histogram、Vector）
- 获取 Prometheus 指标的快照
- 解析直方图和向量指标

主要类：
- Metric: 指标基类
- Counter: 单调递增计数器
- Gauge: 可上下波动的数值
- Histogram: 分桶观测值
- Vector: 有序计数器数组（用于 spec decode）

主要函数：
- get_metrics_snapshot: 获取 Prometheus 指标快照
"""

from dataclasses import dataclass

from prometheus_client import REGISTRY
from prometheus_client import Metric as PromMetric
from prometheus_client.samples import Sample


@dataclass
class Metric:
    """Prometheus 指标的基类。

    每个指标都可以与 key=value 标签关联，
    在某些情况下，单个 vLLM 实例可能有多个同名的指标，
    但具有不同的标签集合。

    Attributes:
        name: 指标名称
        labels: 指标标签字典
    """

    name: str
    labels: dict[str, str]


@dataclass
class Counter(Metric):
    """单调递增整数计数器。

    计数器只能增加或重置为零，适用于累计事件的计数，
    如请求总数、生成的 token 总数等。

    Attributes:
        name: 指标名称
        labels: 指标标签字典
        value: 计数器值（整数）
    """

    value: int


@dataclass
class Vector(Metric):
    """有序整数计数器数组。

    这种类型在 Prometheus 中不存在，专门用于建模一个
    特定的指标：vllm:spec_decode_num_accepted_tokens_per_pos。
    它为每个推测解码 token 位置记录接受的 token 数量。

    Attributes:
        name: 指标名称
        labels: 指标标签字典
        values: 计数器值列表
    """

    values: list[int]


@dataclass
class Gauge(Metric):
    """可上下波动的数值。

    Gauge 可以增加或减少，适用于当前状态的度量，
    如当前运行中的请求数、GPU 利用率等。

    Attributes:
        name: 指标名称
        labels: 指标标签字典
        value: 数值（浮点数）
    """

    value: float


@dataclass
class Histogram(Metric):
    """配置分桶中记录的观测值。

    分桶用字典表示，键是分桶的上限，
    值是该分桶中的观测计数。始终存在 '+Inf' 键。

    count 属性是所有分桶的总计数，与 '+Inf' 分桶的计数相同。

    sum 属性是所有观测值的总和。

    Attributes:
        name: 指标名称
        labels: 指标标签字典
        count: 总观测次数
        sum: 所有观测值的总和
        buckets: 分桶字典，键为上限，值为计数
    """

    count: int
    sum: float
    buckets: dict[str, int]


def get_metrics_snapshot() -> list[Metric]:
    """访问内存中 Prometheus 指标的 API。

    该函数收集所有以 "vllm:" 为前缀的指标，并将它们
    转换为相应的数据类（Counter、Gauge、Histogram、Vector）。

    使用示例:
        >>> for metric in llm.get_metrics():
        ...     if isinstance(metric, Counter):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Gauge):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Histogram):
        ...         print(f"{metric}")
        ...         print(f"    sum = {metric.sum}")
        ...         print(f"    count = {metric.count}")
        ...         for bucket_le, value in metrics.buckets.items():
        ...             print(f"    {bucket_le} = {value}")

    Returns:
        Metric 对象列表，包含所有 vLLM 指标
    """
    collected: list[Metric] = []
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        if metric.type == "gauge":
            samples = _get_samples(metric)
            for s in samples:
                collected.append(
                    Gauge(name=metric.name, labels=s.labels, value=s.value)
                )
        elif metric.type == "counter":
            samples = _get_samples(metric, "_total")
            if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                #
                # 特殊的 vllm:num_accepted_tokens_per_pos 用例。
                #
                # 该指标是计数器向量 - 对于每个推测解码 token 位置，
                # 我们使用带 'position' 标签的 Counter 记录接受的 token 数量。
                # 我们将这些转换为整数向量值。
                #
                for labels, values in _digest_num_accepted_by_pos_samples(samples):
                    collected.append(
                        Vector(name=metric.name, labels=labels, values=values)
                    )
            else:
                for s in samples:
                    collected.append(
                        Counter(name=metric.name, labels=s.labels, value=int(s.value))
                    )

        elif metric.type == "histogram":
            #
            # 直方图有多个 '_bucket' 样本，其中
            # 'le' 标签表示分桶的上限。
            # 我们将这些分桶值转换为以 'le' 标签值为索引的字典。
            # 'le=+Inf' 标签是特殊情况，捕获所有观测值。
            #
            bucket_samples = _get_samples(metric, "_bucket")
            count_samples = _get_samples(metric, "_count")
            sum_samples = _get_samples(metric, "_sum")
            for labels, buckets, count_value, sum_value in _digest_histogram(
                bucket_samples, count_samples, sum_samples
            ):
                collected.append(
                    Histogram(
                        name=metric.name,
                        labels=labels,
                        buckets=buckets,
                        count=count_value,
                        sum=sum_value,
                    )
                )
        else:
            raise AssertionError(f"Unknown metric type {metric.type}")

    return collected


def _get_samples(metric: PromMetric, suffix: str | None = None) -> list[Sample]:
    """获取指定后缀的样本列表。

    Args:
        metric: Prometheus 指标对象
        suffix: 样本名称后缀（如 "_total"、"_bucket"）

    Returns:
        匹配的样本列表
    """
    name = (metric.name + suffix) if suffix is not None else metric.name
    return [s for s in metric.samples if s.name == name]


def _strip_label(labels: dict[str, str], key_to_remove: str) -> dict[str, str]:
    """从标签字典中移除指定的键。

    Args:
        labels: 标签字典
        key_to_remove: 要移除的键

    Returns:
        移除指定键后的标签字典副本
    """
    labels_copy = labels.copy()
    labels_copy.pop(key_to_remove)
    return labels_copy


def _digest_histogram(
    bucket_samples: list[Sample], count_samples: list[Sample], sum_samples: list[Sample]
) -> list[tuple[dict[str, str], dict[str, int], int, float]]:
    """解析直方图样本为结构化数据。

    在 DP（数据并行）情况下，我们有每个分桶每个引擎的计数，
    作为带标签的样本列表，以及总计和总和样本。

    输入示例:
        bucket_samples:
          labels = {le: 100, idx: 0}, value = 2
          labels = {le: 200, idx: 0}, value = 4
          labels = {le: Inf, idx: 0}, value = 10
          labels = {le: 100, idx: 1}, value = 1
          labels = {le: 200, idx: 2}, value = 5
          labels = {le: Inf, idx: 3}, value = 7
        count_samples:
          labels = {idx: 0}, value = 10
          labels = {idx: 1}, value = 7
        sum_samples:
          labels = {idx: 0}, value = 2000
          labels = {idx: 1}, value = 1200

    输出:
        [
          {idx: 0}, {"100": 2, "200": 4, "Inf": 10}, 10, 2000
          {idx: 1}, {"100": 1, "200": 5, "Inf": 7},   7, 1200
        ]

    Args:
        bucket_samples: 分桶样本列表
        count_samples: 计数样本列表
        sum_samples: 总和样本列表

    Returns:
        (标签，分桶字典，计数值，总和值) 元组列表
    """
    buckets_by_labels: dict[frozenset[tuple[str, str]], dict[str, int]] = {}
    for s in bucket_samples:
        bucket = s.labels["le"]
        labels_key = frozenset(_strip_label(s.labels, "le").items())
        if labels_key not in buckets_by_labels:
            buckets_by_labels[labels_key] = {}
        buckets_by_labels[labels_key][bucket] = int(s.value)

    counts_by_labels: dict[frozenset[tuple[str, str]], int] = {}
    for s in count_samples:
        labels_key = frozenset(s.labels.items())
        counts_by_labels[labels_key] = int(s.value)

    sums_by_labels: dict[frozenset[tuple[str, str]], float] = {}
    for s in sum_samples:
        labels_key = frozenset(s.labels.items())
        sums_by_labels[labels_key] = s.value

    assert (
        set(buckets_by_labels.keys())
        == set(counts_by_labels.keys())
        == set(sums_by_labels.keys())
    )

    output = []
    label_keys = list(buckets_by_labels.keys())
    for k in label_keys:
        labels = dict(k)
        output.append(
            (labels, buckets_by_labels[k], counts_by_labels[k], sums_by_labels[k])
        )
    return output


def _digest_num_accepted_by_pos_samples(
    samples: list[Sample],
) -> list[tuple[dict[str, str], list[int]]]:
    """解析按位置统计的接受 token 样本。

    在 DP（数据并行）情况下，我们有每个位置每个引擎的计数，
    作为带标签的样本列表。

    输入示例:
        samples:
          labels = {position: 0, idx: 0}, value = 10
          labels = {position: 1, idx: 0}, value = 7
          labels = {position: 2, idx: 0}, value = 2
          labels = {position: 0, idx: 1}, value = 5
          labels = {position: 1, idx: 1}, value = 3
          labels = {position: 2, idx: 1}, value = 1

    输出:
        [
          {idx: 0}, [10, 7, 2]
          {idx: 1}, [5, 3, 1]
        ]

    Args:
        samples: 样本列表

    Returns:
        (标签，值列表) 元组列表
    """
    max_pos = 0
    values_by_labels: dict[frozenset[tuple[str, str]], dict[int, int]] = {}

    for s in samples:
        position = int(s.labels["position"])
        max_pos = max(max_pos, position)

        labels_key = frozenset(_strip_label(s.labels, "position").items())
        if labels_key not in values_by_labels:
            values_by_labels[labels_key] = {}
        values_by_labels[labels_key][position] = int(s.value)

    output = []
    for labels_key, values_by_position in values_by_labels.items():
        labels = dict(labels_key)
        values = [0] * (max_pos + 1)
        for pos, val in values_by_position.items():
            values[pos] = val
        output.append((labels, values))
    return output
