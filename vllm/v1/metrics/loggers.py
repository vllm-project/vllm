# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""指标日志记录器模块。

本模块实现了 vLLM 的指标日志记录功能，负责：
- 定义统计日志记录器的基础接口（StatLoggerBase）
- 提供日志记录器实现（LoggingStatLogger）
- 提供聚合日志记录器（AggregatedLoggingStatLogger）
- 提供 Prometheus 指标导出（PrometheusStatLogger）
- 支持自定义统计日志记录器插件
- 管理多引擎的指标记录和聚合

主要类：
- StatLoggerBase: 统计日志记录器基础接口
- AggregateStatLoggerBase: 聚合统计日志记录器基类
- LoggingStatLogger: 日志记录器实现
- AggregatedLoggingStatLogger: 聚合日志记录器
- PerEngineStatLoggerAdapter: 每引擎日志记录器适配器
- PrometheusStatLogger: Prometheus 指标记录器
- StatLoggerManager: 统计日志记录器管理器

主要函数：
- load_stat_logger_plugin_factories: 加载统计日志记录器插件
- build_buckets: 构建指标桶
- build_1_2_5_buckets: 构建 1-2-5 序列桶
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from prometheus_client import Counter, Gauge, Histogram

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphLogging
from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging,
    KVConnectorProm,
)
from vllm.logger import init_logger
from vllm.plugins import STAT_LOGGER_PLUGINS_GROUP, load_plugins_by_group
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.perf import PerfMetricsLogging, PerfMetricsProm
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import (
    CachingMetrics,
    IterationStats,
    MultiModalCacheStats,
    PromptTokenStats,
    SchedulerStats,
)
from vllm.v1.metrics.utils import create_metric_per_engine
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingProm

logger = init_logger(__name__)

PerEngineStatLoggerFactory = Callable[[VllmConfig, int], "StatLoggerBase"]
AggregateStatLoggerFactory = type["AggregateStatLoggerBase"]
StatLoggerFactory = AggregateStatLoggerFactory | PerEngineStatLoggerFactory


class StatLoggerBase(ABC):
    """日志记录器的基础接口。

    API 用户可以定义实现此接口的自定义日志记录器。
    但是请注意，SchedulerStats 和 IterationStats 类
    不被视为稳定的接口，可能会在未来版本中更改。

    Attributes:
        vllm_config: vLLM 配置
        engine_index: 引擎索引
    """

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        """初始化统计日志记录器。

        Args:
            vllm_config: vLLM 配置
            engine_index: 引擎索引，默认为 0
        """
        ...

    @abstractmethod
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """记录统计信息。

        Args:
            scheduler_stats: 调度器统计
            iteration_stats: 迭代统计
            mm_cache_stats: 多模态缓存统计
            engine_idx: 引擎索引
        """
        ...

    @abstractmethod
    def log_engine_initialized(self):
        """记录引擎已初始化。"""
        ...

    def log(self):  # noqa
        """执行日志记录。"""
        pass

    def record_sleep_state(self, is_awake: int, level: int):  # noqa
        """记录睡眠状态。

        Args:
            is_awake: 是否唤醒
            level: 睡眠级别
        """
        pass


def load_stat_logger_plugin_factories() -> list[StatLoggerFactory]:
    """加载统计日志记录器插件工厂。

    从插件组加载所有统计日志记录器插件，
    并验证它们是否是 StatLoggerBase 的子类。

    Returns:
        统计日志记录器工厂列表

    Raises:
        TypeError: 如果插件不是 StatLoggerBase 的子类
    """
    factories: list[StatLoggerFactory] = []

    for name, plugin_class in load_plugins_by_group(STAT_LOGGER_PLUGINS_GROUP).items():
        if not isinstance(plugin_class, type) or not issubclass(
            plugin_class, StatLoggerBase
        ):
            raise TypeError(
                f"统计日志记录器插件 {name!r} 必须是 "
                f"StatLoggerBase 的子类（得到 {plugin_class!r}）。"
            )

        factories.append(plugin_class)

    return factories


class AggregateStatLoggerBase(StatLoggerBase):
    """聚合统计日志记录器基类。

    用于在多个 DP 引擎之间聚合指标。
    """

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]):
        """初始化聚合统计日志记录器。

        Args:
            vllm_config: vLLM 配置
            engine_indexes: 引擎索引列表
        """
        ...


class LoggingStatLogger(StatLoggerBase):
    """日志统计记录器。

    将统计信息记录到标准输出的日志记录器。

    Attributes:
        engine_index: 引擎索引
        vllm_config: vLLM 配置
        last_scheduler_stats: 上次调度器统计
        prefix_caching_metrics: 前缀缓存指标
        connector_prefix_caching_metrics: 连接器前缀缓存指标
        mm_caching_metrics: 多模态缓存指标
        spec_decoding_logging: 推测解码日志
        kv_connector_logging: KV 连接器日志
        cudagraph_logging: CUDA Graph 日志
        last_prompt_throughput: 上次提示吞吐量
        last_generation_throughput: 上次生成吞吐量
        engine_is_idle: 引擎是否空闲
        aggregated: 是否聚合
    """

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        """初始化日志统计记录器。

        Args:
            vllm_config: vLLM 配置
            engine_index: 引擎索引，默认为 0
        """
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self._reset(time.monotonic())

        self.last_scheduler_stats = SchedulerStats()

        # 缓存指标，不能重置
        # TODO: 使间隔可配置
        self.prefix_caching_metrics = CachingMetrics()
        self.connector_prefix_caching_metrics = CachingMetrics()
        self.mm_caching_metrics = CachingMetrics()

        self.spec_decoding_logging = SpecDecodingLogging()
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self.kv_connector_logging = KVConnectorLogging(kv_transfer_config)
        self.cudagraph_logging = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            self.cudagraph_logging = CUDAGraphLogging(
                self.vllm_config.compilation_config.cudagraph_mode,
                self.vllm_config.compilation_config.cudagraph_capture_sizes,
            )
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0
        self.engine_is_idle = False
        self.aggregated = False

        if self._enable_perf_stats():
            self.perf_metrics_logging = PerfMetricsLogging(vllm_config)

    def _reset(self, now):
        """重置统计。

        Args:
            now: 当前时间（单调时间）
        """
        self.last_log_time = now

        # 当前本地日志间隔内跟踪的统计
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0
        self.num_corrupted_reqs: int = 0
        self.num_preemptions: int = 0

    def _enable_perf_stats(self) -> bool:
        """检查是否启用性能统计。

        Returns:
            是否启用 MFU 指标
        """
        return self.vllm_config.observability_config.enable_mfu_metrics

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        """跟踪迭代统计。

        保存用于 token 计数器的跟踪统计。
        对提示吞吐量使用计算的 token（不包括缓存/传输的）。

        Args:
            iteration_stats: 迭代统计
        """
        # 保存用于 token 计数器的跟踪统计
        # 对提示吞吐量使用计算的 token（不包括缓存/传输的）
        self.num_prompt_tokens += iteration_stats.prompt_token_stats.computed
        self.num_generation_tokens += iteration_stats.num_generation_tokens
        self.num_corrupted_reqs += iteration_stats.num_corrupted_reqs
        self.num_preemptions += iteration_stats.num_preempted_reqs

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        """计算跟踪统计的吞吐量指标。

        Args:
            tracked_stats: 跟踪的统计
            now: 当前时间

        Returns:
            吞吐量（tokens/s）
        """
        # 计算跟踪统计的摘要指标
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)

    @property
    def log_prefix(self):
        """返回日志前缀。

        Returns:
            引擎日志前缀
        """
        return "Engine {:03d}: ".format(self.engine_index)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """记录统计信息到标准输出。

        记录迭代统计、调度器统计、缓存统计等信息，
        并更新相应的指标。

        Args:
            scheduler_stats: 调度器统计
            iteration_stats: 迭代统计
            mm_cache_stats: 多模态缓存统计
            engine_idx: 引擎索引
        """
        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            self.prefix_caching_metrics.observe(scheduler_stats.prefix_cache_stats)

            if scheduler_stats.connector_prefix_cache_stats is not None:
                self.connector_prefix_caching_metrics.observe(
                    scheduler_stats.connector_prefix_cache_stats
                )

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_logging.observe(scheduler_stats.spec_decoding_stats)
            if kv_connector_stats := scheduler_stats.kv_connector_stats:
                self.kv_connector_logging.observe(kv_connector_stats)
            if (
                self.cudagraph_logging is not None
                and scheduler_stats.cudagraph_stats is not None
            ):
                self.cudagraph_logging.observe(scheduler_stats.cudagraph_stats)
            if not self.aggregated:
                self.last_scheduler_stats = scheduler_stats
            if (perf_stats := scheduler_stats.perf_stats) and self._enable_perf_stats():
                self.perf_metrics_logging.observe(perf_stats)
        if mm_cache_stats:
            self.mm_caching_metrics.observe(mm_cache_stats)

    def _update_stats(self):
        """更新统计信息。

        计算当前时间间隔内的吞吐量，并重置跟踪统计。
        同时检测引擎是否处于空闲状态。
        """
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)

        self._reset(now)
        self.engine_is_idle = not any(
            (
                prompt_throughput,
                generation_throughput,
                self.last_prompt_throughput,
                self.last_generation_throughput,
            )
        )
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput

    def aggregate_scheduler_stats(self):
        """聚合调度器统计。

        对于每引擎日志记录器，此方法为空操作。
        """
        # noop for per engine loggers
        return

    def log(self):
        """执行日志记录。

        更新统计信息，聚合并输出日志。
        空闲引擎使用 debug 级别，否则使用 info 级别。
        """
        self._update_stats()
        self.aggregate_scheduler_stats()
        # Avoid log noise on an idle production system
        log_fn = logger.debug if self.engine_is_idle else logger.info
        # Format and print output.
        log_parts = [
            "Avg prompt throughput: %.1f tokens/s",
            "Avg generation throughput: %.1f tokens/s",
            "Running: %d reqs",
            "Waiting: %d reqs",
        ]
        log_args: list[int | float | str] = [
            self.last_prompt_throughput,
            self.last_generation_throughput,
            self.last_scheduler_stats.num_running_reqs,
            self.last_scheduler_stats.num_waiting_reqs,
        ]

        if self.num_preemptions > 0:
            log_parts.append("Preemptions: %d")
            log_args.append(self.num_preemptions)

        log_parts.extend(
            [
                "GPU KV cache usage: %.1f%%",
                "Prefix cache hit rate: %.1f%%",
            ]
        )
        log_args.extend(
            [
                self.last_scheduler_stats.kv_cache_usage * 100,
                self.prefix_caching_metrics.hit_rate * 100,
            ]
        )

        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            log_parts.append("Corrupted: %d reqs")
            log_args.append(self.num_corrupted_reqs)
        if not self.connector_prefix_caching_metrics.empty:
            log_parts.append("External prefix cache hit rate: %.1f%%")
            log_args.append(self.connector_prefix_caching_metrics.hit_rate * 100)
        if not self.mm_caching_metrics.empty:
            log_parts.append("MM cache hit rate: %.1f%%")
            log_args.append(self.mm_caching_metrics.hit_rate * 100)

        log_fn(
            self.log_prefix + ", ".join(log_parts),
            *log_args,
        )

        self.spec_decoding_logging.log(log_fn=log_fn)
        self.kv_connector_logging.log(log_fn=log_fn)
        if self.cudagraph_logging is not None:
            self.cudagraph_logging.log(log_fn=log_fn)
        if self._enable_perf_stats():
            self.perf_metrics_logging.log(log_fn=log_fn, log_prefix=self.log_prefix)

    def log_engine_initialized(self):
        """记录引擎已初始化。

        当缓存块数量确定后，记录引擎初始化信息。
        """
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.debug(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d",
                self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks,
            )


class AggregatedLoggingStatLogger(LoggingStatLogger, AggregateStatLoggerBase):
    """聚合日志统计记录器。

    在多个数据并行引擎之间聚合指标。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int],
    ):
        """初始化聚合日志统计记录器。

        Args:
            vllm_config: vLLM 配置
            engine_indexes: 引擎索引列表
        """
        self.engine_indexes = engine_indexes
        self.last_scheduler_stats_dict: dict[int, SchedulerStats] = {
            idx: SchedulerStats() for idx in self.engine_indexes
        }
        LoggingStatLogger.__init__(self, vllm_config, engine_index=-1)
        self.aggregated = True

    @property
    def log_prefix(self):
        """返回日志前缀。

        Returns:
            聚合引擎日志前缀
        """
        return "{} Engines Aggregated: ".format(len(self.engine_indexes))

    def _enable_perf_stats(self) -> bool:
        """检查是否启用性能统计。

        对于聚合日志记录器，禁用每 GPU 性能统计，
        因为跨引擎聚合可能导致误导性数字。

        Returns:
            False（总是禁用）
        """
        # Adding per_gpu perf stats across engines can lead to misleading numbers.
        return False

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """记录统计信息。

        记录各引擎的统计信息并保存用于聚合。

        Args:
            scheduler_stats: 调度器统计
            iteration_stats: 迭代统计
            mm_cache_stats: 多模态缓存统计
            engine_idx: 引擎索引
        """
        if engine_idx not in self.engine_indexes:
            logger.warning("Unexpected engine_idx: %d", engine_idx)
            return
        LoggingStatLogger.record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=mm_cache_stats,
            engine_idx=engine_idx,
        )
        if scheduler_stats is not None:
            self.last_scheduler_stats_dict[engine_idx] = scheduler_stats

    def aggregate_scheduler_stats(self):
        """聚合所有引擎的调度器统计。

        对所有引擎的请求数量和 KV 缓存使用率进行聚合。
        """
        self.last_scheduler_stats = SchedulerStats()
        for last_scheduler_stats in self.last_scheduler_stats_dict.values():
            self.last_scheduler_stats.num_waiting_reqs += (
                last_scheduler_stats.num_waiting_reqs
            )
            self.last_scheduler_stats.num_running_reqs += (
                last_scheduler_stats.num_running_reqs
            )
            self.last_scheduler_stats.kv_cache_usage += (
                last_scheduler_stats.kv_cache_usage
            )
        self.last_scheduler_stats.kv_cache_usage /= len(self.last_scheduler_stats_dict)

    def log(self):
        """执行日志记录。

        调用父类的日志记录方法。
        """
        LoggingStatLogger.log(self)

    def log_engine_initialized(self):
        """记录引擎已初始化。

        记录所有引擎的初始化信息。
        """
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "%d Engines: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d",
                len(self.engine_indexes),
                self.vllm_config.cache_config.num_gpu_blocks,
            )


class PerEngineStatLoggerAdapter(AggregateStatLoggerBase):
    """每引擎日志记录器适配器。

    为每个引擎创建独立的日志记录器实例，
    并代理所有日志记录调用到相应的引擎日志记录器。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int],
        per_engine_stat_logger_factory: PerEngineStatLoggerFactory,
    ) -> None:
        """初始化每引擎日志记录器适配器。

        Args:
            vllm_config: vLLM 配置
            engine_indexes: 引擎索引列表
            per_engine_stat_logger_factory: 每引擎日志记录器工厂
        """
        self.per_engine_stat_loggers = {}
        self.engine_indexes = engine_indexes
        for engine_index in engine_indexes:
            self.per_engine_stat_loggers[engine_index] = per_engine_stat_logger_factory(
                vllm_config, engine_index
            )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """记录统计信息到对应的引擎日志记录器。

        Args:
            scheduler_stats: 调度器统计
            iteration_stats: 迭代统计
            mm_cache_stats: 多模态缓存统计
            engine_idx: 引擎索引
        """
        if engine_idx not in self.per_engine_stat_loggers:
            logger.warning("Unexpected engine_idx: %d", engine_idx)
            return
        self.per_engine_stat_loggers[engine_idx].record(
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=mm_cache_stats,
            engine_idx=engine_idx,
        )

    def log(self):
        """执行所有引擎的日志记录。"""
        for per_engine_stat_logger in self.per_engine_stat_loggers.values():
            per_engine_stat_logger.log()

    def log_engine_initialized(self):
        """记录所有引擎已初始化。"""
        for per_engine_stat_logger in self.per_engine_stat_loggers.values():
            per_engine_stat_logger.log_engine_initialized()


class PrometheusStatLogger(AggregateStatLoggerBase):
    """Prometheus 指标记录器。

    将 vLLM 指标导出到 Prometheus 监控系统。
    """

    _gauge_cls = Gauge
    _counter_cls = Counter
    _histogram_cls = Histogram
    _spec_decoding_cls = SpecDecodingProm
    _kv_connector_cls = KVConnectorProm
    _perf_metrics_cls = PerfMetricsProm

    def __init__(
        self, vllm_config: VllmConfig, engine_indexes: list[int] | None = None
    ):
        """初始化 Prometheus 指标记录器。

        注册所有 Prometheus 指标。

        Args:
            vllm_config: vLLM 配置
            engine_indexes: 引擎索引列表，默认为 [0]
        """
        if engine_indexes is None:
            engine_indexes = [0]

        self.engine_indexes = engine_indexes

        unregister_vllm_metrics()
        self.vllm_config = vllm_config
        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = vllm_config.observability_config.show_hidden_metrics
        self.kv_cache_metrics_enabled = (
            vllm_config.observability_config.kv_cache_metrics
        )

        labelnames = ["model_name", "engine"]
        model_name = vllm_config.model_config.served_model_name
        max_model_len = vllm_config.model_config.max_model_len

        self.per_engine_labelvalues: dict[int, list[object]] = {
            idx: [model_name, str(idx)] for idx in engine_indexes
        }
        per_engine_labelvalues = self.per_engine_labelvalues

        self.spec_decoding_prom = self._spec_decoding_cls(
            vllm_config.speculative_config, labelnames, per_engine_labelvalues
        )
        self.kv_connector_prom = self._kv_connector_cls(
            vllm_config, labelnames, per_engine_labelvalues
        )
        self.perf_metrics_prom = self._perf_metrics_cls(
            vllm_config, labelnames, per_engine_labelvalues
        )

        #
        # Scheduler state
        #
        gauge_scheduler_running = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_scheduler_running = create_metric_per_engine(
            gauge_scheduler_running, per_engine_labelvalues
        )

        gauge_scheduler_waiting = self._gauge_cls(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_scheduler_waiting = create_metric_per_engine(
            gauge_scheduler_waiting, per_engine_labelvalues
        )

        gauge_engine_sleep_state = self._gauge_cls(
            name="vllm:engine_sleep_state",
            documentation=(
                "Engine sleep state; awake = 0 means engine is sleeping; "
                "awake = 1 means engine is awake; "
                "weights_offloaded = 1 means sleep level 1; "
                "discard_all = 1 means sleep level 2."
            ),
            labelnames=labelnames + ["sleep_state"],
            multiprocess_mode="mostrecent",
        )

        self.gauge_engine_sleep_state = {}
        sleep_state = ["awake", "weights_offloaded", "discard_all"]

        for s in sleep_state:
            self.gauge_engine_sleep_state[s] = {
                idx: gauge_engine_sleep_state.labels(
                    engine=idx, model_name=model_name, sleep_state=s
                )
                for idx in engine_indexes
            }

        # Setting default values
        self.record_sleep_state()

        gauge_kv_cache_usage = self._gauge_cls(
            name="vllm:kv_cache_usage_perc",
            documentation="KV-cache usage. 1 means 100 percent usage.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_kv_cache_usage = create_metric_per_engine(
            gauge_kv_cache_usage, per_engine_labelvalues
        )

        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            counter_corrupted_requests = self._counter_cls(
                name="vllm:corrupted_requests",
                documentation=(
                    "Corrupted requests, in terms of total number of requests "
                    "with NaNs in logits."
                ),
                labelnames=labelnames,
            )
            self.counter_corrupted_requests = create_metric_per_engine(
                counter_corrupted_requests, per_engine_labelvalues
            )

        counter_prefix_cache_queries = self._counter_cls(
            name="vllm:prefix_cache_queries",
            documentation=(
                "Prefix cache queries, in terms of number of queried tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_queries = create_metric_per_engine(
            counter_prefix_cache_queries, per_engine_labelvalues
        )

        counter_prefix_cache_hits = self._counter_cls(
            name="vllm:prefix_cache_hits",
            documentation=("Prefix cache hits, in terms of number of cached tokens."),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_hits = create_metric_per_engine(
            counter_prefix_cache_hits, per_engine_labelvalues
        )

        #
        # External - KV connector prefix cache
        #

        counter_connector_prefix_cache_queries = self._counter_cls(
            name="vllm:external_prefix_cache_queries",
            documentation=(
                "External prefix cache queries from KV connector "
                "cross-instance cache sharing, in terms of number of queried tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_connector_prefix_cache_queries = create_metric_per_engine(
            counter_connector_prefix_cache_queries, per_engine_labelvalues
        )

        counter_connector_prefix_cache_hits = self._counter_cls(
            name="vllm:external_prefix_cache_hits",
            documentation=(
                "External prefix cache hits from KV connector "
                "cross-instance cache sharing, in terms of number of cached tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_connector_prefix_cache_hits = create_metric_per_engine(
            counter_connector_prefix_cache_hits, per_engine_labelvalues
        )

        #
        # Multi-modal cache
        #

        counter_mm_cache_queries = self._counter_cls(
            name="vllm:mm_cache_queries",
            documentation=(
                "Multi-modal cache queries, in terms of number of queried items."
            ),
            labelnames=labelnames,
        )
        self.counter_mm_cache_queries = create_metric_per_engine(
            counter_mm_cache_queries, per_engine_labelvalues
        )

        counter_mm_cache_hits = self._counter_cls(
            name="vllm:mm_cache_hits",
            documentation=(
                "Multi-modal cache hits, in terms of number of cached items."
            ),
            labelnames=labelnames,
        )
        self.counter_mm_cache_hits = create_metric_per_engine(
            counter_mm_cache_hits, per_engine_labelvalues
        )

        #
        # Counters
        #
        counter_num_preempted_reqs = self._counter_cls(
            name="vllm:num_preemptions",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames,
        )
        self.counter_num_preempted_reqs = create_metric_per_engine(
            counter_num_preempted_reqs, per_engine_labelvalues
        )

        counter_prompt_tokens = self._counter_cls(
            name="vllm:prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
        )
        self.counter_prompt_tokens = create_metric_per_engine(
            counter_prompt_tokens, per_engine_labelvalues
        )

        # Labeled prompt token counters by source
        counter_prompt_tokens_by_source = self._counter_cls(
            name="vllm:prompt_tokens_by_source",
            documentation="Number of prompt tokens by source.",
            labelnames=labelnames + ["source"],
        )
        self.counter_prompt_tokens_by_source: dict[str, dict[int, Counter]] = {}
        for source in PromptTokenStats.ALL_SOURCES:
            self.counter_prompt_tokens_by_source[source] = {
                idx: counter_prompt_tokens_by_source.labels(
                    model_name, str(idx), source
                )
                for idx in engine_indexes
            }

        # Cached prompt tokens counter
        counter_prompt_tokens_cached = self._counter_cls(
            name="vllm:prompt_tokens_cached",
            documentation="Number of cached prompt tokens (local + external).",
            labelnames=labelnames,
        )
        self.counter_prompt_tokens_cached = create_metric_per_engine(
            counter_prompt_tokens_cached, per_engine_labelvalues
        )

        # Recomputed tokens (last token recomputed when entire prompt is cached)
        counter_prompt_tokens_recomputed = self._counter_cls(
            name="vllm:prompt_tokens_recomputed",
            documentation="Number of cached tokens recomputed for forward pass.",
            labelnames=labelnames,
        )
        self.counter_prompt_tokens_recomputed = create_metric_per_engine(
            counter_prompt_tokens_recomputed, per_engine_labelvalues
        )

        counter_generation_tokens = self._counter_cls(
            name="vllm:generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
        )
        self.counter_generation_tokens = create_metric_per_engine(
            counter_generation_tokens, per_engine_labelvalues
        )

        self.counter_request_success: dict[FinishReason, dict[int, Counter]] = {}
        counter_request_success_base = self._counter_cls(
            name="vllm:request_success",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"],
        )
        for reason in FinishReason:
            self.counter_request_success[reason] = {
                idx: counter_request_success_base.labels(
                    model_name, str(idx), str(reason)
                )
                for idx in engine_indexes
            }

        #
        # Histograms of counts
        #
        histogram_num_prompt_tokens_request = self._histogram_cls(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_num_prompt_tokens_request = create_metric_per_engine(
            histogram_num_prompt_tokens_request, per_engine_labelvalues
        )

        histogram_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_num_generation_tokens_request = create_metric_per_engine(
            histogram_num_generation_tokens_request, per_engine_labelvalues
        )

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        # See: https://github.com/vllm-project/vllm/pull/18053
        histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
            documentation="Histogram of number of tokens per engine_step.",
            buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            labelnames=labelnames,
        )
        self.histogram_iteration_tokens = create_metric_per_engine(
            histogram_iteration_tokens, per_engine_labelvalues
        )

        histogram_max_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_max_num_generation_tokens",
            documentation="Histogram of maximum number of requested generation tokens.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_max_num_generation_tokens_request = create_metric_per_engine(
            histogram_max_num_generation_tokens_request, per_engine_labelvalues
        )

        histogram_n_request = self._histogram_cls(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            buckets=[1, 2, 5, 10, 20],
            labelnames=labelnames,
        )
        self.histogram_n_request = create_metric_per_engine(
            histogram_n_request, per_engine_labelvalues
        )

        histogram_max_tokens_request = self._histogram_cls(
            name="vllm:request_params_max_tokens",
            documentation="Histogram of the max_tokens request parameter.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_max_tokens_request = create_metric_per_engine(
            histogram_max_tokens_request, per_engine_labelvalues
        )

        #
        # Histogram of timing intervals
        #
        histogram_time_to_first_token = self._histogram_cls(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001,
                0.005,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                20.0,
                40.0,
                80.0,
                160.0,
                640.0,
                2560.0,
            ],
            labelnames=labelnames,
        )
        self.histogram_time_to_first_token = create_metric_per_engine(
            histogram_time_to_first_token, per_engine_labelvalues
        )

        histogram_inter_token_latency = self._histogram_cls(
            name="vllm:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            buckets=[
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                20.0,
                40.0,
                80.0,
            ],
            labelnames=labelnames,
        )
        self.histogram_inter_token_latency = create_metric_per_engine(
            histogram_inter_token_latency, per_engine_labelvalues
        )

        histogram_request_time_per_output_token = self._histogram_cls(
            name="vllm:request_time_per_output_token_seconds",
            documentation="Histogram of time_per_output_token_seconds per request.",
            buckets=[
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                20.0,
                40.0,
                80.0,
            ],
            labelnames=labelnames,
        )
        self.histogram_request_time_per_output_token = create_metric_per_engine(
            histogram_request_time_per_output_token, per_engine_labelvalues
        )

        request_latency_buckets = [
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
        histogram_e2e_time_request = self._histogram_cls(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of e2e request latency in seconds.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_e2e_time_request = create_metric_per_engine(
            histogram_e2e_time_request, per_engine_labelvalues
        )

        histogram_queue_time_request = self._histogram_cls(
            name="vllm:request_queue_time_seconds",
            documentation="Histogram of time spent in WAITING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_queue_time_request = create_metric_per_engine(
            histogram_queue_time_request, per_engine_labelvalues
        )

        histogram_inference_time_request = self._histogram_cls(
            name="vllm:request_inference_time_seconds",
            documentation="Histogram of time spent in RUNNING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_inference_time_request = create_metric_per_engine(
            histogram_inference_time_request, per_engine_labelvalues
        )

        histogram_prefill_time_request = self._histogram_cls(
            name="vllm:request_prefill_time_seconds",
            documentation="Histogram of time spent in PREFILL phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_prefill_time_request = create_metric_per_engine(
            histogram_prefill_time_request, per_engine_labelvalues
        )

        histogram_decode_time_request = self._histogram_cls(
            name="vllm:request_decode_time_seconds",
            documentation="Histogram of time spent in DECODE phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_decode_time_request = create_metric_per_engine(
            histogram_decode_time_request, per_engine_labelvalues
        )

        histogram_prefill_kv_computed_request = self._histogram_cls(
            name="vllm:request_prefill_kv_computed_tokens",
            documentation=(
                "Histogram of new KV tokens computed during prefill "
                "(excluding cached tokens)."
            ),
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_prefill_kv_computed_request = create_metric_per_engine(
            histogram_prefill_kv_computed_request, per_engine_labelvalues
        )

        #
        # KV Cache residency metrics
        #
        if self.kv_cache_metrics_enabled:
            kv_cache_residency_buckets = [
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                120,
                300,
                600,
                1200,
                1800,
            ]

            histogram_kv_block_lifetime = self._histogram_cls(
                name="vllm:kv_block_lifetime_seconds",
                documentation=(
                    "Histogram of KV cache block lifetime from allocation to eviction. "
                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
                ),
                buckets=kv_cache_residency_buckets,
                labelnames=labelnames,
            )
            self.histogram_kv_block_lifetime = create_metric_per_engine(
                histogram_kv_block_lifetime, per_engine_labelvalues
            )

            histogram_kv_block_idle_before_evict = self._histogram_cls(
                name="vllm:kv_block_idle_before_evict_seconds",
                documentation=(
                    "Histogram of idle time before KV cache block eviction. "
                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
                ),
                buckets=kv_cache_residency_buckets,
                labelnames=labelnames,
            )
            self.histogram_kv_block_idle_before_evict = create_metric_per_engine(
                histogram_kv_block_idle_before_evict, per_engine_labelvalues
            )

            histogram_kv_block_reuse_gap = self._histogram_cls(
                name="vllm:kv_block_reuse_gap_seconds",
                documentation=(
                    "Histogram of time gaps between consecutive KV cache block "
                    "accesses. Only the most recent accesses are recorded "
                    "(ring buffer). Sampled metrics (controlled by "
                    "--kv-cache-metrics-sample)."
                ),
                buckets=kv_cache_residency_buckets,
                labelnames=labelnames,
            )
            self.histogram_kv_block_reuse_gap = create_metric_per_engine(
                histogram_kv_block_reuse_gap, per_engine_labelvalues
            )
        else:
            self.histogram_kv_block_lifetime = {}
            self.histogram_kv_block_idle_before_evict = {}
            self.histogram_kv_block_reuse_gap = {}

        #
        # LoRA metrics
        #

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        # See: https://github.com/vllm-project/vllm/pull/18053
        self.gauge_lora_info: Gauge | None = None
        if vllm_config.lora_config is not None:
            if len(self.engine_indexes) > 1:
                logger.warning(
                    "vllm:lora_requests_info prometheus metrics may be "
                    "incorrect/misleading with data parallel deployments."
                )
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = vllm_config.lora_config.max_loras
            self.gauge_lora_info = self._gauge_cls(
                name="vllm:lora_requests_info",
                documentation="Running stats on lora requests.",
                multiprocess_mode="sum",
                labelnames=[
                    self.labelname_max_lora,
                    self.labelname_waiting_lora_adapters,
                    self.labelname_running_lora_adapters,
                ],
            )

    def log_metrics_info(self, type: str, config_obj: SupportsMetricsInfo):
        """记录配置指标信息。

        将配置信息作为 Prometheus 指标记录。
        Info 类型指标在 Prometheus 多进程模式下使用 Gauge 模拟。

        Args:
            type: 指标类型（如 "cache_config"）
            config_obj: 支持指标信息的配置对象
        """
        metrics_info = config_obj.metrics_info()
        metrics_info["engine"] = ""

        name, documentation = None, None
        if type == "cache_config":
            name = "vllm:cache_config_info"
            documentation = "Information of the LLMEngine CacheConfig"
        assert name is not None, f"Unknown metrics info type {type}"

        # Info type metrics are syntactic sugar for a gauge permanently set to 1
        # Since prometheus multiprocessing mode does not support Info, emulate
        # info here with a gauge.
        info_gauge = self._gauge_cls(
            name=name,
            documentation=documentation,
            multiprocess_mode="mostrecent",
            labelnames=metrics_info.keys(),
        )
        for engine_index in self.engine_indexes:
            metrics_info = config_obj.metrics_info()
            metrics_info["engine"] = str(engine_index)
            info_gauge.labels(**metrics_info).set(1)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """记录统计信息到 Prometheus。

        更新所有已注册的 Prometheus 指标。

        Args:
            scheduler_stats: 调度器统计
            iteration_stats: 迭代统计
            mm_cache_stats: 多模态缓存统计
            engine_idx: 引擎索引
        """
        """Log to prometheus."""
        if scheduler_stats is not None:
            self.gauge_scheduler_running[engine_idx].set(
                scheduler_stats.num_running_reqs
            )
            self.gauge_scheduler_waiting[engine_idx].set(
                scheduler_stats.num_waiting_reqs
            )
            self.gauge_kv_cache_usage[engine_idx].set(scheduler_stats.kv_cache_usage)

            self.counter_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.queries
            )
            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits
            )

            if scheduler_stats.connector_prefix_cache_stats is not None:
                self.counter_connector_prefix_cache_queries[engine_idx].inc(
                    scheduler_stats.connector_prefix_cache_stats.queries
                )
                self.counter_connector_prefix_cache_hits[engine_idx].inc(
                    scheduler_stats.connector_prefix_cache_stats.hits
                )

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_prom.observe(
                    scheduler_stats.spec_decoding_stats, engine_idx
                )

            if scheduler_stats.kv_connector_stats is not None:
                self.kv_connector_prom.observe(
                    scheduler_stats.kv_connector_stats, engine_idx
                )

            if scheduler_stats.perf_stats is not None:
                self.perf_metrics_prom.observe(scheduler_stats.perf_stats, engine_idx)

            if (
                self.kv_cache_metrics_enabled
                and scheduler_stats.kv_cache_eviction_events
            ):
                lifetime_hist = self.histogram_kv_block_lifetime[engine_idx]
                idle_hist = self.histogram_kv_block_idle_before_evict[engine_idx]
                reuse_hist = self.histogram_kv_block_reuse_gap[engine_idx]

                for event in scheduler_stats.kv_cache_eviction_events:
                    lifetime_hist.observe(event.lifetime_seconds)
                    idle_hist.observe(event.idle_seconds)
                    for gap in event.reuse_gaps_seconds:
                        reuse_hist.observe(gap)

            if self.gauge_lora_info is not None:
                running_lora_adapters = ",".join(
                    scheduler_stats.running_lora_adapters.keys()
                )
                waiting_lora_adapters = ",".join(
                    scheduler_stats.waiting_lora_adapters.keys()
                )
                lora_info_labels = {
                    self.labelname_running_lora_adapters: running_lora_adapters,
                    self.labelname_waiting_lora_adapters: waiting_lora_adapters,
                    self.labelname_max_lora: self.max_lora,
                }
                self.gauge_lora_info.labels(**lora_info_labels).set_to_current_time()

        if mm_cache_stats is not None:
            self.counter_mm_cache_queries[engine_idx].inc(mm_cache_stats.queries)
            self.counter_mm_cache_hits[engine_idx].inc(mm_cache_stats.hits)

        if iteration_stats is None:
            return
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            self.counter_corrupted_requests[engine_idx].inc(
                iteration_stats.num_corrupted_reqs
            )
        self.counter_num_preempted_reqs[engine_idx].inc(
            iteration_stats.num_preempted_reqs
        )
        self.counter_prompt_tokens[engine_idx].inc(iteration_stats.num_prompt_tokens)
        # Labeled prompt token counters by source
        pts = iteration_stats.prompt_token_stats
        for source in PromptTokenStats.ALL_SOURCES:
            self.counter_prompt_tokens_by_source[source][engine_idx].inc(
                pts.get_by_source(source)
            )
        self.counter_prompt_tokens_cached[engine_idx].inc(pts.cached_tokens)
        self.counter_prompt_tokens_recomputed[engine_idx].inc(pts.recomputed_tokens)
        self.counter_generation_tokens[engine_idx].inc(
            iteration_stats.num_generation_tokens
        )
        self.histogram_iteration_tokens[engine_idx].observe(
            iteration_stats.num_prompt_tokens + iteration_stats.num_generation_tokens
        )

        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request[engine_idx].observe(
                max_gen_tokens
            )
        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request[engine_idx].observe(n_param)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token[engine_idx].observe(ttft)
        for itl in iteration_stats.inter_token_latencies_iter:
            self.histogram_inter_token_latency[engine_idx].observe(itl)

        for finished_request in iteration_stats.finished_requests:
            self.counter_request_success[finished_request.finish_reason][
                engine_idx
            ].inc()
            self.histogram_e2e_time_request[engine_idx].observe(
                finished_request.e2e_latency
            )
            self.histogram_queue_time_request[engine_idx].observe(
                finished_request.queued_time
            )
            self.histogram_prefill_time_request[engine_idx].observe(
                finished_request.prefill_time
            )
            self.histogram_inference_time_request[engine_idx].observe(
                finished_request.inference_time
            )
            self.histogram_decode_time_request[engine_idx].observe(
                finished_request.decode_time
            )
            # Calculate prefill KV compute (excludes cached tokens)
            prefill_kv_computed = finished_request.num_prompt_tokens - max(
                finished_request.num_cached_tokens, 0
            )
            self.histogram_prefill_kv_computed_request[engine_idx].observe(
                prefill_kv_computed
            )
            self.histogram_num_prompt_tokens_request[engine_idx].observe(
                finished_request.num_prompt_tokens
            )
            self.histogram_num_generation_tokens_request[engine_idx].observe(
                finished_request.num_generation_tokens
            )
            self.histogram_request_time_per_output_token[engine_idx].observe(
                finished_request.mean_time_per_output_token
            )
            if finished_request.max_tokens_param:
                self.histogram_max_tokens_request[engine_idx].observe(
                    finished_request.max_tokens_param
                )

    def record_sleep_state(self, sleep: int = 0, level: int = 0):
        """记录引擎睡眠状态。

        更新 Prometheus 指标以反映引擎的睡眠状态。

        Args:
            sleep: 睡眠标志，0=唤醒，1=睡眠
            level: 睡眠级别，1=卸载权重，2=丢弃所有
        """
        awake = 1
        discard_all = 0
        weights_offloaded = 0

        if sleep == 1:
            awake = 0
            if level == 1:
                weights_offloaded = 1
            elif level == 2:
                discard_all = 1

        for engine_idx in self.engine_indexes:
            self.gauge_engine_sleep_state["discard_all"][engine_idx].set(discard_all)
            self.gauge_engine_sleep_state["weights_offloaded"][engine_idx].set(
                weights_offloaded
            )
            self.gauge_engine_sleep_state["awake"][engine_idx].set(awake)

    def log_engine_initialized(self):
        """记录引擎已初始化。

        使用缓存配置信息记录引擎初始化指标。
        """
        self.log_metrics_info("cache_config", self.vllm_config.cache_config)


def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]:
    """构建指标桶。

    通过递增的 10 的幂次乘以尾数值来构建桶，
    直到值超过指定的最大值。

    Args:
        mantissa_lst: 尾数列表
        max_value: 最大值

    Returns:
        桶列表
    """
    exponent = 0
    buckets: list[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> list[int]:
    """构建 1-2-5 序列桶。

    示例：
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]

    Args:
        max_value: 最大值

    Returns:
        1-2-5 序列桶列表
    """
    return build_buckets([1, 2, 5], max_value)


class StatLoggerManager:
    """统计日志记录器管理器。

    日志记录发生在 EngineCore 级别（每个调度器）。

    - DP（数据并行）：每个 AsyncLLM 有多个 EngineCore，每个 EngineCore 有自己的日志记录器
    - 使用 Local Logger 时，为 N 个 EngineCore 创建 N 份副本
    - 使用 Prometheus 时，需要单个日志记录器处理 N 个"labels"

    该类从 AsyncLLM 中抽象出这些实现细节，
    允许 AsyncLLM 简单地调用 .record() 和 .log() 方法。

    Attributes:
        engine_indexes: 引擎索引列表
        stat_loggers: 统计日志记录器列表
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_idxs: list[int] | None = None,
        custom_stat_loggers: list[StatLoggerFactory] | None = None,
        enable_default_loggers: bool = True,
        aggregate_engine_logging: bool = False,
        client_count: int = 1,
    ):
        """初始化统计日志记录器管理器。

        Args:
            vllm_config: vLLM 配置
            engine_idxs: 引擎索引列表，默认为 [0]
            custom_stat_loggers: 自定义统计日志记录器工厂列表
            enable_default_loggers: 是否启用默认日志记录器
            aggregate_engine_logging: 是否聚合引擎日志记录
            client_count: 客户端数量
        """
        self.engine_indexes = engine_idxs if engine_idxs else [0]
        self.stat_loggers: list[AggregateStatLoggerBase] = []
        stat_logger_factories: list[StatLoggerFactory] = []
        if custom_stat_loggers is not None:
            stat_logger_factories.extend(custom_stat_loggers)
        if enable_default_loggers and logger.isEnabledFor(logging.INFO):
            if client_count > 1:
                logger.warning(
                    "AsyncLLM created with api_server_count more than 1; "
                    "disabling stats logging to avoid incomplete stats."
                )
            else:
                default_logger_factory = (
                    AggregatedLoggingStatLogger
                    if aggregate_engine_logging
                    else LoggingStatLogger
                )
                stat_logger_factories.append(default_logger_factory)
        custom_prometheus_logger: bool = False
        for stat_logger_factory in stat_logger_factories:
            if isinstance(stat_logger_factory, type) and issubclass(
                stat_logger_factory, AggregateStatLoggerBase
            ):
                global_stat_logger = stat_logger_factory(
                    vllm_config=vllm_config,
                    engine_indexes=self.engine_indexes,
                )
                if isinstance(global_stat_logger, PrometheusStatLogger):
                    custom_prometheus_logger = True
            else:
                # per engine logger
                global_stat_logger = PerEngineStatLoggerAdapter(
                    vllm_config=vllm_config,
                    engine_indexes=self.engine_indexes,
                    per_engine_stat_logger_factory=stat_logger_factory,  # type: ignore[arg-type]
                )
            self.stat_loggers.append(global_stat_logger)
        if not custom_prometheus_logger:
            self.stat_loggers.append(
                PrometheusStatLogger(vllm_config, self.engine_indexes)
            )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int | None = None,
    ):
        """记录统计信息到所有日志记录器。

        Args:
            scheduler_stats: 调度器统计
            iteration_stats: 迭代统计
            mm_cache_stats: 多模态缓存统计
            engine_idx: 引擎索引，默认为 0
        """
        if engine_idx is None:
            engine_idx = 0
        for stat_logger in self.stat_loggers:
            stat_logger.record(
                scheduler_stats,
                iteration_stats,
                mm_cache_stats=mm_cache_stats,
                engine_idx=engine_idx,
            )

    def record_sleep_state(self, sleep: int = 0, level: int = 0):
        """记录睡眠状态到所有日志记录器。

        Args:
            sleep: 睡眠标志
            level: 睡眠级别
        """
        for logger in self.stat_loggers:
            logger.record_sleep_state(sleep, level)

    def log(self):
        """执行所有日志记录器的日志记录。"""
        for logger in self.stat_loggers:
            logger.log()

    def log_engine_initialized(self):
        """记录所有引擎已初始化。"""
        for agg_logger in self.stat_loggers:
            agg_logger.log_engine_initialized()
