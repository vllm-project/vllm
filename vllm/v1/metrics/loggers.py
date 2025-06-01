# SPDX-License-Identifier: Apache-2.0

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import prometheus_client

from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingProm

logger = init_logger(__name__)

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
    def record(self, scheduler_stats: Optional[SchedulerStats],
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

    def record(self, scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats]):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            self.prefix_caching_metrics.observe(
                scheduler_stats.prefix_cache_stats)

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
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d", self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks)


class PrometheusStatLogger(StatLoggerBase):
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram
    _spec_decoding_cls = SpecDecodingProm

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):

        unregister_vllm_metrics()
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = \
            vllm_config.observability_config.show_hidden_metrics

        labelnames = ["model_name", "engine"]
        labelvalues = [
            vllm_config.model_config.served_model_name,
            str(engine_index)
        ]

        max_model_len = vllm_config.model_config.max_model_len

        self.spec_decoding_prom = self._spec_decoding_cls(
            vllm_config.speculative_config, labelnames, labelvalues)

        #
        # Scheduler state
        #
        self.gauge_scheduler_running = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames).labels(*labelvalues)

        self.gauge_scheduler_waiting = self._gauge_cls(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames).labels(*labelvalues)

        #
        # GPU cache
        #
        self.gauge_gpu_cache_usage = self._gauge_cls(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_gpu_prefix_cache_queries = self._counter_cls(
            name="vllm:gpu_prefix_cache_queries",
            documentation=
            "GPU prefix cache queries, in terms of number of queried tokens.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_gpu_prefix_cache_hits = self._counter_cls(
            name="vllm:gpu_prefix_cache_hits",
            documentation=
            "GPU prefix cache hits, in terms of number of cached tokens.",
            labelnames=labelnames).labels(*labelvalues)

        #
        # Counters
        #
        self.counter_num_preempted_reqs = self._counter_cls(
            name="vllm:num_preemptions",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_prompt_tokens = self._counter_cls(
            name="vllm:prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_generation_tokens = self._counter_cls(
            name="vllm:generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_request_success: dict[FinishReason,
                                           prometheus_client.Counter] = {}
        counter_request_success_base = self._counter_cls(
            name="vllm:request_success",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"])
        for reason in FinishReason:
            self.counter_request_success[
                reason] = counter_request_success_base.labels(*(labelvalues +
                                                                [str(reason)]))

        #
        # Histograms of counts
        #
        self.histogram_num_prompt_tokens_request = \
            self._histogram_cls(
                name="vllm:request_prompt_tokens",
                documentation="Number of prefill tokens processed.",
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_num_generation_tokens_request = \
            self._histogram_cls(
                name="vllm:request_generation_tokens",
                documentation="Number of generation tokens processed.",
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        # See: https://github.com/vllm-project/vllm/pull/18053
        self.histogram_iteration_tokens = \
            self._histogram_cls(
                name="vllm:iteration_tokens_total",
                documentation="Histogram of number of tokens per engine_step.",
                buckets=[
                    1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                    16384
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_max_num_generation_tokens_request = \
            self._histogram_cls(
                name="vllm:request_max_num_generation_tokens",
                documentation=
                "Histogram of maximum number of requested generation tokens.",
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_n_request = \
            self._histogram_cls(
                name="vllm:request_params_n",
                documentation="Histogram of the n request parameter.",
                buckets=[1, 2, 5, 10, 20],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_max_tokens_request = \
            self._histogram_cls(
                name="vllm:request_params_max_tokens",
                documentation="Histogram of the max_tokens request parameter.",
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        #
        # Histogram of timing intervals
        #
        self.histogram_time_to_first_token = \
            self._histogram_cls(
                name="vllm:time_to_first_token_seconds",
                documentation="Histogram of time to first token in seconds.",
                buckets=[
                    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0,
                    640.0, 2560.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_time_per_output_token = \
            self._histogram_cls(
                name="vllm:time_per_output_token_seconds",
                documentation="Histogram of time per output token in seconds.",
                buckets=[
                    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
            40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
        ]
        self.histogram_e2e_time_request = \
            self._histogram_cls(
                name="vllm:e2e_request_latency_seconds",
                documentation="Histogram of e2e request latency in seconds.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_queue_time_request = \
            self._histogram_cls(
                name="vllm:request_queue_time_seconds",
                documentation=
                "Histogram of time spent in WAITING phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_inference_time_request = \
            self._histogram_cls(
                name="vllm:request_inference_time_seconds",
                documentation=
                "Histogram of time spent in RUNNING phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_prefill_time_request = \
            self._histogram_cls(
                name="vllm:request_prefill_time_seconds",
                documentation=
                "Histogram of time spent in PREFILL phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_decode_time_request = \
            self._histogram_cls(
                name="vllm:request_decode_time_seconds",
                documentation=
                "Histogram of time spent in DECODE phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)

        #
        # LoRA metrics
        #

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        self.gauge_lora_info: Optional[prometheus_client.Gauge] = None
        if vllm_config.lora_config is not None:
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = vllm_config.lora_config.max_loras
            self.gauge_lora_info = \
                self._gauge_cls(
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
        info_gauge = self._gauge_cls(
            name=name,
            documentation=documentation,
            multiprocess_mode="mostrecent",
            labelnames=metrics_info.keys(),
        ).labels(**metrics_info)
        info_gauge.set(1)

    def record(self, scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats]):
        """Log to prometheus."""
        if scheduler_stats is not None:
            self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
            self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)

            self.gauge_gpu_cache_usage.set(scheduler_stats.gpu_cache_usage)

            self.counter_gpu_prefix_cache_queries.inc(
                scheduler_stats.prefix_cache_stats.queries)
            self.counter_gpu_prefix_cache_hits.inc(
                scheduler_stats.prefix_cache_stats.hits)

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_prom.observe(
                    scheduler_stats.spec_decoding_stats)

        if iteration_stats is None:
            return

        self.counter_num_preempted_reqs.inc(iteration_stats.num_preempted_reqs)
        self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens.inc(
            iteration_stats.num_generation_tokens)
        self.histogram_iteration_tokens.observe(
            iteration_stats.num_prompt_tokens + \
            iteration_stats.num_generation_tokens)

        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request.observe(
                max_gen_tokens)
        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request.observe(n_param)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token.observe(ttft)
        for tpot in iteration_stats.time_per_output_tokens_iter:
            self.histogram_time_per_output_token.observe(tpot)

        for finished_request in iteration_stats.finished_requests:
            self.counter_request_success[finished_request.finish_reason].inc()
            self.histogram_e2e_time_request.observe(
                finished_request.e2e_latency)
            self.histogram_queue_time_request.observe(
                finished_request.queued_time)
            self.histogram_prefill_time_request.observe(
                finished_request.prefill_time)
            self.histogram_inference_time_request.observe(
                finished_request.inference_time)
            self.histogram_decode_time_request.observe(
                finished_request.decode_time)
            self.histogram_num_prompt_tokens_request.observe(
                finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request.observe(
                finished_request.num_generation_tokens)
            self.histogram_max_tokens_request.observe(
                finished_request.max_tokens_param)

        if self.gauge_lora_info is not None:
            running_lora_adapters = \
                ",".join(iteration_stats.running_lora_adapters.keys())
            waiting_lora_adapters = \
                ",".join(iteration_stats.waiting_lora_adapters.keys())
            lora_info_labels = {
                self.labelname_running_lora_adapters: running_lora_adapters,
                self.labelname_waiting_lora_adapters: waiting_lora_adapters,
                self.labelname_max_lora: self.max_lora,
            }
            self.gauge_lora_info.labels(**lora_info_labels)\
                                .set_to_current_time()

    def log_engine_initialized(self):
        self.log_metrics_info("cache_config", self.vllm_config.cache_config)


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
