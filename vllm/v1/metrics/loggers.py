# SPDX-License-Identifier: Apache-2.0

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, cast

import numpy as np
import prometheus_client

from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.executor.ray_utils import ray
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingProm

if ray is not None:
    from ray.util import metrics as ray_metrics
else:
    ray_metrics = None

logger = init_logger(__name__)

_LOCAL_LOGGING_INTERVAL_SEC = 5.0

StatLoggerFactory = Callable[[VllmConfig, int], "StatLoggerBase"]


class StatLoggerBase(ABC):
    """Interface for logging metrics.

    API users may define custom loggers that implement this interface.
    However, note that the `SchedulerStats` and `IterationStats` classes
    are not considered stable interfaces and may change in future versions.
    """

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        ...

    @abstractmethod
    def record(self, scheduler_stats: SchedulerStats,
               iteration_stats: Optional[IterationStats]):
        ...

    @abstractmethod
    def log_engine_initialized(self):
        ...

    def log(self):  # noqa
        pass


class LoggingStatLogger(StatLoggerBase):

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self._reset(time.monotonic())
        self.last_scheduler_stats = SchedulerStats()
        # Prefix cache metrics. This cannot be reset.
        # TODO: Make the interval configurable.
        self.prefix_caching_metrics = PrefixCachingMetrics()
        self.spec_decoding_logging = SpecDecodingLogging()
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: list[int] = []
        self.num_generation_tokens: list[int] = []

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(iteration_stats.num_prompt_tokens)
        self.num_generation_tokens.append(
            iteration_stats.num_generation_tokens)

    def _get_throughput(self, tracked_stats: list[int], now: float) -> float:
        # Compute summary metrics for tracked stats
        return float(np.sum(tracked_stats) / (now - self.last_log_time))

    def record(self, scheduler_stats: SchedulerStats,
               iteration_stats: Optional[IterationStats]):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        self.prefix_caching_metrics.observe(scheduler_stats.prefix_cache_stats)

        if scheduler_stats.spec_decoding_stats is not None:
            self.spec_decoding_logging.observe(
                scheduler_stats.spec_decoding_stats)

        self.last_scheduler_stats = scheduler_stats

    def log(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(
            self.num_generation_tokens, now)

        self._reset(now)

        scheduler_stats = self.last_scheduler_stats

        log_fn = logger.info
        if not any(
            (prompt_throughput, generation_throughput,
             self.last_prompt_throughput, self.last_generation_throughput)):
            # Avoid log noise on an idle production system
            log_fn = logger.debug
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput

        # Format and print output.
        log_fn(
            "Engine %03d: "
            "Avg prompt throughput: %.1f tokens/s, "
            "Avg generation throughput: %.1f tokens/s, "
            "Running: %d reqs, Waiting: %d reqs, "
            "GPU KV cache usage: %.1f%%, "
            "Prefix cache hit rate: %.1f%%",
            self.engine_index,
            prompt_throughput,
            generation_throughput,
            scheduler_stats.num_running_reqs,
            scheduler_stats.num_waiting_reqs,
            scheduler_stats.gpu_cache_usage * 100,
            self.prefix_caching_metrics.hit_rate * 100,
        )
        self.spec_decoding_logging.log(log_fn=log_fn)

    def log_engine_initialized(self):
        logger.info(
            "vllm cache_config_info with initialization " \
            "after num_gpu_blocks is: %d",
            self.vllm_config.cache_config.num_gpu_blocks)


class Metrics:
    """
    vLLM uses a multiprocessing-based frontend for the OpenAI server.
    This means that we need to run prometheus_client in multiprocessing mode
    See https://prometheus.github.io/client_python/multiprocess/ for more
    details on limitations.
    """

    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram
    _spec_decoding_cls = SpecDecodingProm

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self._unregister_vllm_metrics()

        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = (
            vllm_config.observability_config.show_hidden_metrics)

        labels = {
            "model_name": vllm_config.model_config.served_model_name,
            "engine": str(engine_index),
        }
        labelnames = list(labels.keys())

        max_model_len = vllm_config.model_config.max_model_len

        self.spec_decoding_prom = self._spec_decoding_cls(
            vllm_config.speculative_config, labelnames, list(labels.values()))

        #
        # Scheduler state
        #
        self.gauge_scheduler_running = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            labelnames=labelnames,
        ).labels(**labels)

        self.gauge_scheduler_waiting = self._gauge_cls(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
        ).labels(**labels)

        #
        # GPU cache
        #
        self.gauge_gpu_cache_usage = self._gauge_cls(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
        ).labels(**labels)

        self.counter_gpu_prefix_cache_queries = self._counter_cls(
            name="vllm:gpu_prefix_cache_queries",
            documentation=
            "GPU prefix cache queries, in terms of number of queried tokens.",
            labelnames=labelnames,
        ).labels(**labels)

        self.counter_gpu_prefix_cache_hits = self._counter_cls(
            name="vllm:gpu_prefix_cache_hits",
            documentation=
            "GPU prefix cache hits, in terms of number of cached tokens.",
            labelnames=labelnames,
        ).labels(**labels)

        #
        # Counters
        #
        self.counter_num_preempted_reqs = self._counter_cls(
            name="vllm:num_preemptions_total",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames,
        ).labels(**labels)

        self.counter_prompt_tokens = self._counter_cls(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
        ).labels(**labels)

        self.counter_generation_tokens = self._counter_cls(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
        ).labels(**labels)

        self.counter_request_success: dict[FinishReason,
                                           prometheus_client.Counter] = {}
        counter_request_success_base = self._counter_cls(
            name="vllm:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"],
        )

        for reason in FinishReason:
            request_success_labels = {"finished_reason": str(reason), **labels}
            self.counter_request_success[
                reason] = counter_request_success_base.labels(
                    **request_success_labels)

        #
        # Histograms of counts
        #
        self.histogram_num_prompt_tokens_request = self._histogram_cls(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        ).labels(**labels)

        self.histogram_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        ).labels(**labels)

        self.histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
            documentation="Histogram of number of tokens per engine_step.",
            buckets=[
                1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
            ],
            labelnames=labelnames,
        ).labels(**labels)

        self.histogram_max_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_max_num_generation_tokens",
            documentation=
            "Histogram of maximum number of requested generation tokens.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        ).labels(**labels)

        self.histogram_n_request = self._histogram_cls(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            buckets=[1, 2, 5, 10, 20],
            labelnames=labelnames,
        ).labels(**labels)

        self.histogram_max_tokens_request = self._histogram_cls(
            name="vllm:request_params_max_tokens",
            documentation="Histogram of the max_tokens request parameter.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        ).labels(**labels)

        #
        # Histogram of timing intervals
        #
        self.histogram_time_to_first_token = self._histogram_cls(
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
        ).labels(**labels)

        self.histogram_time_per_output_token = self._histogram_cls(
            name="vllm:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
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
        ).labels(**labels)

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
        self.histogram_e2e_time_request = self._histogram_cls(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of e2e request latency in seconds.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        ).labels(**labels)
        self.histogram_queue_time_request = self._histogram_cls(
            name="vllm:request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        ).labels(**labels)
        self.histogram_inference_time_request = self._histogram_cls(
            name="vllm:request_inference_time_seconds",
            documentation=
            "Histogram of time spent in RUNNING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        ).labels(**labels)
        self.histogram_prefill_time_request = self._histogram_cls(
            name="vllm:request_prefill_time_seconds",
            documentation=
            "Histogram of time spent in PREFILL phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        ).labels(**labels)
        self.histogram_decode_time_request = self._histogram_cls(
            name="vllm:request_decode_time_seconds",
            documentation=
            "Histogram of time spent in DECODE phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        ).labels(**labels)

        #
        # LoRA metrics
        #
        self.gauge_lora_info: Optional[prometheus_client.Gauge] = None
        if vllm_config.lora_config is not None:
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = vllm_config.lora_config.max_loras
            self.gauge_lora_info = self._gauge_cls(
                name="vllm:lora_requests_info",
                documentation="Running stats on lora requests.",
                labelnames=[
                    self.labelname_max_lora,
                    self.labelname_waiting_lora_adapters,
                    self.labelname_running_lora_adapters,
                ],
            )

    @staticmethod
    def _unregister_vllm_metrics():
        # Unregister any existing vLLM collectors (for CI/CD
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                prometheus_client.REGISTRY.unregister(collector)


class _RayGaugeWrapper:
    """Wraps around ray.util.metrics.Gauge to provide same API as
    prometheus_client.Gauge"""

    def __init__(self,
                 name: str,
                 documentation: str = "",
                 labelnames: Optional[list[str]] = None,
                 multiprocess_mode: str = ""):
        del multiprocess_mode
        labelnames_tuple = tuple(labelnames) if labelnames else None
        self._gauge = ray_metrics.Gauge(name=name,
                                        description=documentation,
                                        tag_keys=labelnames_tuple)

    def labels(self, **labels):
        self._gauge.set_default_tags(labels)
        return self

    def set(self, value: Union[int, float]):
        return self._gauge.set(value)

    def set_to_current_time(self):
        # ray metrics doesn't have set_to_current time, https://docs.ray.io/en/latest/_modules/ray/util/metrics.html
        return self._gauge.set(time.time())


class _RayCounterWrapper:
    """Wraps around ray.util.metrics.Counter to provide same API as
    prometheus_client.Counter"""

    def __init__(self,
                 name: str,
                 documentation: str = "",
                 labelnames: Optional[list[str]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        self._counter = ray_metrics.Counter(name=name,
                                            description=documentation,
                                            tag_keys=labelnames_tuple)

    def labels(self, **labels):
        self._counter.set_default_tags(labels)
        return self

    def inc(self, value: Union[int, float] = 1.0):
        if value == 0:
            return
        return self._counter.inc(value)


class _RayHistogramWrapper:
    """Wraps around ray.util.metrics.Histogram to provide same API as
    prometheus_client.Histogram"""

    def __init__(self,
                 name: str,
                 documentation: str = "",
                 labelnames: Optional[list[str]] = None,
                 buckets: Optional[list[float]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        boundaries = buckets if buckets else []
        self._histogram = ray_metrics.Histogram(name=name,
                                                description=documentation,
                                                tag_keys=labelnames_tuple,
                                                boundaries=boundaries)

    def labels(self, **labels):
        self._histogram.set_default_tags(labels)
        return self

    def observe(self, value: Union[int, float]):
        return self._histogram.observe(value)


class RaySpecDecodingProm(SpecDecodingProm):
    """
    RaySpecDecodingProm is used by RayMetrics to log to Ray metrics.
    Provides the same metrics as SpecDecodingProm but uses Ray's
    util.metrics library.
    """

    _counter_cls: type[prometheus_client.Counter] = cast(
        type[prometheus_client.Counter], _RayCounterWrapper)


class RayMetrics(Metrics):
    """
    RayMetrics is used by RayPrometheusStatLogger to log to Ray metrics.
    Provides the same metrics as Metrics but uses Ray's util.metrics library.
    """

    _gauge_cls: type[prometheus_client.Gauge] = cast(
        type[prometheus_client.Gauge], _RayGaugeWrapper)
    _counter_cls: type[prometheus_client.Counter] = cast(
        type[prometheus_client.Counter], _RayCounterWrapper)
    _histogram_cls: type[prometheus_client.Histogram] = cast(
        type[prometheus_client.Histogram], _RayHistogramWrapper)
    _spec_decoding_cls: type[SpecDecodingProm] = cast(type[SpecDecodingProm],
                                                      RaySpecDecodingProm)

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        if ray_metrics is None:
            raise ImportError("RayMetrics requires Ray to be installed.")
        super().__init__(vllm_config, engine_index)

    @staticmethod
    def _unregister_vllm_metrics():
        # No-op on purpose
        pass


class PrometheusStatLogger(StatLoggerBase):
    _metrics_cls = Metrics

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.metrics = self._metrics_cls(vllm_config=vllm_config,
                                         engine_index=engine_index)

        #
        # Cache config info metric
        #
        self.log_metrics_info("cache_config", vllm_config.cache_config)

    def log_metrics_info(self, type: str, config_obj: SupportsMetricsInfo):
        metrics_info = config_obj.metrics_info()
        metrics_info["engine"] = self.engine_index

        name, documentation = None, None
        if type == "cache_config":
            name = "vllm:cache_config_info"
            documentation = "Information of the LLMEngine CacheConfig"
        assert name is not None, f"Unknown metrics info type {type}"

        # Info type metrics are syntactic sugar for a gauge permanently set to 1
        # Since prometheus multiprocessing mode does not support Info, emulate
        # info here with a gauge.
        info_gauge = prometheus_client.Gauge(
            name=name,
            documentation=documentation,
            labelnames=metrics_info.keys()).labels(**metrics_info)
        info_gauge.set(1)

    def record(self, scheduler_stats: SchedulerStats,
               iteration_stats: Optional[IterationStats]):
        """Log to prometheus."""
        self.metrics.gauge_scheduler_running.set(
            scheduler_stats.num_running_reqs)
        self.metrics.gauge_scheduler_waiting.set(
            scheduler_stats.num_waiting_reqs)

        self.metrics.gauge_gpu_cache_usage.set(scheduler_stats.gpu_cache_usage)

        self.metrics.counter_gpu_prefix_cache_queries.inc(
            scheduler_stats.prefix_cache_stats.queries)
        self.metrics.counter_gpu_prefix_cache_hits.inc(
            scheduler_stats.prefix_cache_stats.hits)

        if scheduler_stats.spec_decoding_stats is not None:
            self.metrics.spec_decoding_prom.observe(
                scheduler_stats.spec_decoding_stats)

        if iteration_stats is None:
            return

        self.metrics.counter_num_preempted_reqs.inc(
            iteration_stats.num_preempted_reqs)
        self.metrics.counter_prompt_tokens.inc(
            iteration_stats.num_prompt_tokens)
        self.metrics.counter_generation_tokens.inc(
            iteration_stats.num_generation_tokens)
        self.metrics.histogram_iteration_tokens.observe(
            iteration_stats.num_prompt_tokens +
            iteration_stats.num_generation_tokens)

        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.metrics.histogram_max_num_generation_tokens_request.observe(
                max_gen_tokens)
        for n_param in iteration_stats.n_params_iter:
            self.metrics.histogram_n_request.observe(n_param)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.metrics.histogram_time_to_first_token.observe(ttft)
        for tpot in iteration_stats.time_per_output_tokens_iter:
            self.metrics.histogram_time_per_output_token.observe(tpot)

        for finished_request in iteration_stats.finished_requests:
            self.metrics.counter_request_success[
                finished_request.finish_reason].inc()
            self.metrics.histogram_e2e_time_request.observe(
                finished_request.e2e_latency)
            self.metrics.histogram_queue_time_request.observe(
                finished_request.queued_time)
            self.metrics.histogram_prefill_time_request.observe(
                finished_request.prefill_time)
            self.metrics.histogram_inference_time_request.observe(
                finished_request.inference_time)
            self.metrics.histogram_decode_time_request.observe(
                finished_request.decode_time)
            self.metrics.histogram_num_prompt_tokens_request.observe(
                finished_request.num_prompt_tokens)
            self.metrics.histogram_num_generation_tokens_request.observe(
                finished_request.num_generation_tokens)
            self.metrics.histogram_max_tokens_request.observe(
                finished_request.max_tokens_param)

        if self.metrics.gauge_lora_info is not None:
            running_lora_adapters = ",".join(
                iteration_stats.running_lora_adapters.keys())
            waiting_lora_adapters = ",".join(
                iteration_stats.waiting_lora_adapters.keys())
            lora_info_labels = {
                self.metrics.labelname_running_lora_adapters:
                running_lora_adapters,
                self.metrics.labelname_waiting_lora_adapters:
                waiting_lora_adapters,
                self.metrics.labelname_max_lora: self.metrics.max_lora,
            }
            self.metrics.gauge_lora_info.labels(
                **lora_info_labels).set_to_current_time()


class RayPrometheusStatLogger(PrometheusStatLogger):
    """RayPrometheusStatLogger uses Ray metrics instead."""

    _metrics_cls = RayMetrics

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        return None


def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum.

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
    """
    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)


def setup_default_loggers(
    vllm_config: VllmConfig,
    log_stats: bool,
    engine_num: int,
    custom_stat_loggers: Optional[list[StatLoggerFactory]] = None,
) -> list[list[StatLoggerBase]]:
    """Setup logging and prometheus metrics."""
    if not log_stats:
        return []

    factories: list[StatLoggerFactory]
    if custom_stat_loggers is not None:
        factories = custom_stat_loggers
    else:
        factories = [PrometheusStatLogger]
        if logger.isEnabledFor(logging.INFO):
            factories.append(LoggingStatLogger)

    stat_loggers: list[list[StatLoggerBase]] = []
    for i in range(engine_num):
        per_engine_stat_loggers: list[StatLoggerBase] = []
        for logger_factory in factories:
            per_engine_stat_loggers.append(logger_factory(vllm_config, i))
        stat_loggers.append(per_engine_stat_loggers)

    return stat_loggers
