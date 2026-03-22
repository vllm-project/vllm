# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray 指标封装模块。

本模块提供了 Ray Serve 环境下的 Prometheus 指标封装，负责：
- 将 Prometheus 指标 API 适配到 Ray 指标 API
- 自动添加 Ray Serve Replica ID 标签
- 提供 Ray 版本的推测解码、KV 连接器、性能指标类
- 提供 Ray 版本的 Prometheus 统计日志记录器

主要类：
- RayPrometheusMetric: Ray Prometheus 指标基类
- RayGaugeWrapper: Ray Gauge 封装
- RayCounterWrapper: Ray Counter 封装
- RayHistogramWrapper: Ray Histogram 封装
- RaySpecDecodingProm: Ray 推测解码指标
- RayKVConnectorProm: Ray KV 连接器指标
- RayPerfMetricsProm: Ray 性能指标
- RayPrometheusStatLogger: Ray Prometheus 统计日志记录器

主要函数：
- _get_replica_id: 获取 Ray Serve Replica ID
"""

import time

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorProm
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.perf import PerfMetricsProm
from vllm.v1.spec_decode.metrics import SpecDecodingProm

try:
    from ray import serve as ray_serve
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None
    ray_serve = None

import regex as re


def _get_replica_id() -> str | None:
    """获取当前 Ray Serve Replica ID。

    如果不在 Ray Serve 上下文中，则返回 None。

    Returns:
        Replica ID 字符串，如果不在 Serve 上下文中则返回 None
    """
    if ray_serve is None:
        return None
    try:
        return ray_serve.get_replica_context().replica_id.unique_id
    except ray_serve.exceptions.RayServeException:
        return None


class RayPrometheusMetric:
    """Ray Prometheus 指标基类。

    提供 Ray 指标的基础功能和通用的标签处理方法。
    所有 Ray 指标都会自动添加 ReplicaId 标签以区分不同的副本。

    Attributes:
        metric: Ray 指标实例
    """

    def __init__(self):
        """初始化 Ray Prometheus 指标。

        Raises:
            ImportError: 如果 Ray 未安装
        """
        if ray_metrics is None:
            raise ImportError("RayPrometheusMetric requires Ray to be installed.")
        self.metric: Metric = None

    @staticmethod
    def _get_tag_keys(labelnames: list[str] | None) -> tuple[str, ...]:
        """获取标签键列表，自动添加 ReplicaId。

        Args:
            labelnames: 原始标签名称列表

        Returns:
            包含 ReplicaId 的标签键元组
        """
        labels = list(labelnames) if labelnames else []
        labels.append("ReplicaId")
        return tuple(labels)

    def labels(self, *labels, **labelskwargs):
        """设置指标标签。

        自动添加 ReplicaId 标签，并将非字符串值转换为字符串。

        Args:
            *labels: 位置标签参数
            **labelskwargs: 关键字标签参数

        Returns:
            self

        Raises:
            ValueError: 如果位置标签数量不匹配
        """
        if labels:
            # -1 是因为 ReplicaId 是自动添加的
            expected = len(self.metric._tag_keys) - 1
            if len(labels) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labels)}"
                )
            labelskwargs.update(zip(self.metric._tag_keys, labels))

        labelskwargs["ReplicaId"] = _get_replica_id() or ""

        if labelskwargs:
            for k, v in labelskwargs.items():
                if not isinstance(v, str):
                    labelskwargs[k] = str(v)
            self.metric.set_default_tags(labelskwargs)
        return self

    @staticmethod
    def _get_sanitized_opentelemetry_name(name: str) -> str:
        """清理指标名称以符合 OpenTelemetry 规范。

        将不允许的字符（如 ':'）替换为 '_'。
        允许的字符：a-z, A-Z, 0-9, _

        Args:
            name: 原始指标名称

        Returns:
            清理后的指标名称

        参考:
            https://github.com/open-telemetry/opentelemetry-cpp/blob/main/sdk/src/metrics/instrument_metadata_validator.cc#L22-L23
            https://github.com/ray-project/ray/blob/master/src/ray/stats/metric.cc#L107
        """

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class RayGaugeWrapper(RayPrometheusMetric):
    """Ray Gauge 指标封装。

    封装 ray.util.metrics.Gauge 以提供与 prometheus_client.Gauge
    相同的 API 接口。

    Ray Gauge 是一个可升降的数值指标，适用于温度、队列长度等
    可波动的指标。
    """

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        multiprocess_mode: str | None = "",
    ):
        """初始化 Ray Gauge。

        所有 Ray 指标都按 WorkerId 键控，因此多进程模式（如
        "mostrecent"、"all"、"sum"）不适用。此逻辑可以在可观测性层
        （Prometheus/Grafana）手动实现。

        Args:
            name: 指标名称
            documentation: 指标描述
            labelnames: 标签名称列表
            multiprocess_mode: 多进程模式（被忽略）
        """
        # 所有 Ray 指标都按 WorkerId 键控，因此多进程模式如
        # "mostrecent"、"all"、"sum" 不适用。此逻辑可以手动
        # 在可观测性层（Prometheus/Grafana）实现。
        del multiprocess_mode

        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)

        self.metric = ray_metrics.Gauge(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def set(self, value: int | float):
        """设置 Gauge 值。

        Args:
            value: 要设置的值
        """
        return self.metric.set(value)

    def set_to_current_time(self):
        """将 Gauge 设置为当前时间。

        Ray 指标没有 set_to_current_time 方法，
        所以使用 time.time() 手动实现。
        """
        # ray metrics doesn't have set_to_current time
        return self.metric.set(time.time())


class RayCounterWrapper(RayPrometheusMetric):
    """Ray Counter 指标封装。

    封装 ray.util.metrics.Counter 以提供与 prometheus_client.Counter
    相同的 API 接口。

    Counter 是一个只能增加或重置为零的单调递增指标，
    适用于累计统计如请求数、错误数等。
    """

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
    ):
        """初始化 Ray Counter。

        Args:
            name: 指标名称
            documentation: 指标描述
            labelnames: 标签名称列表
        """
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Counter(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def inc(self, value: int | float = 1.0):
        """增加 Counter 值。

        Args:
            value: 增加的值，默认为 1.0
        """
        if value == 0:
            return
        return self.metric.inc(value)


class RayHistogramWrapper(RayPrometheusMetric):
    """Ray Histogram 指标封装。

    封装 ray.util.metrics.Histogram 以提供与 prometheus_client.Histogram
    相同的 API 接口。

    Histogram 用于记录观测值的分布，适用于延迟、请求大小等指标。

    Attributes:
        boundaries: 分桶边界列表
    """

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        """初始化 Ray Histogram。

        Args:
            name: 指标名称
            documentation: 指标描述
            labelnames: 标签名称列表
            buckets: 分桶边界列表
        """
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)

        boundaries = buckets if buckets else []
        self.metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
            boundaries=boundaries,
        )

    def observe(self, value: int | float):
        """记录观测值。

        Args:
            value: 观测值
        """
        return self.metric.observe(value)


class RaySpecDecodingProm(SpecDecodingProm):
    """Ray 推测解码指标。

    RaySpecDecodingProm 用于 RayMetrics 记录 Ray 指标。
    提供与 SpecDecodingProm 相同的指标，但使用 Ray 的
    util.metrics 库。
    """

    _counter_cls = RayCounterWrapper


class RayKVConnectorProm(KVConnectorProm):
    """Ray KV 连接器指标。

    RayKVConnectorProm 用于 RayMetrics 记录 Ray 指标。
    提供与 KV 连接器相同的指标，但使用 Ray 的
    util.metrics 库。
    """

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper


class RayPerfMetricsProm(PerfMetricsProm):
    """Ray 性能指标。

    RayPerfMetricsProm 用于 RayMetrics 记录 Ray 指标。
    提供与 PerfMetricsProm 相同的 MFU 指标，但使用
    Ray 的 util.metrics 库。
    """

    _counter_cls = RayCounterWrapper


class RayPrometheusStatLogger(PrometheusStatLogger):
    """Ray Prometheus 统计日志记录器。

    使用 Ray 指标替代 Prometheus 指标的统计日志记录器。
    适用于在 Ray Serve 部署中运行的 vLLM 实例。

    所有指标类都被替换为对应的 Ray 封装版本：
    - _gauge_cls: RayGaugeWrapper
    - _counter_cls: RayCounterWrapper
    - _histogram_cls: RayHistogramWrapper
    - _spec_decoding_cls: RaySpecDecodingProm
    - _kv_connector_cls: RayKVConnectorProm
    - _perf_metrics_cls: RayPerfMetricsProm
    """

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper
    _spec_decoding_cls = RaySpecDecodingProm
    _kv_connector_cls = RayKVConnectorProm
    _perf_metrics_cls = RayPerfMetricsProm

    @staticmethod
    def _unregister_vllm_metrics():
        """注销 vLLM 指标。

        Ray 模式下无需注销，因为 Ray 自动管理指标生命周期。
        """
        # No-op on purpose
        pass
