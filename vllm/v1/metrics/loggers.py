# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import prometheus_client

from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging)
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
    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
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
        kv_tranfer_config = self.vllm_config.kv_transfer_config
        self.kv_transfer_logging = KVConnectorLogging(kv_tranfer_config)
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens += iteration_stats.num_prompt_tokens
        self.num_generation_tokens += iteration_stats.num_generation_tokens

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        # Compute summary metrics for tracked stats
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            self.prefix_caching_metrics.observe(
                scheduler_stats.prefix_cache_stats)

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_logging.observe(
                    scheduler_stats.spec_decoding_stats)
            if kv_connector_stats := scheduler_stats.kv_connector_stats:
                self.kv_transfer_logging.observe(kv_connector_stats)
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
            scheduler_stats.kv_cache_usage * 100,
            self.prefix_caching_metrics.hit_rate * 100,
        )
        self.spec_decoding_logging.log(log_fn=log_fn)
        self.kv_transfer_logging.log(log_fn=log_fn)

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

    def __init__(self,
                 vllm_config: VllmConfig,
                 engine_indexes: Optional[list[int]] = None):
        if engine_indexes is None:
            engine_indexes = [0]
        self.engine_indexes = engine_indexes

        unregister_vllm_metrics()
        self.vllm_config = vllm_config
        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = \
            vllm_config.observability_config.show_hidden_metrics

        labelnames = ["model_name", "engine"]
        model_name = vllm_config.model_config.served_model_name
        max_model_len = vllm_config.model_config.max_model_len

        spec_decode_labelvalues: dict[int, list[str]] = {
            idx: [model_name, str(idx)]
            for idx in engine_indexes
        }

        self.spec_decoding_prom = self._spec_decoding_cls(
            vllm_config.speculative_config, labelnames,
            spec_decode_labelvalues)

        #
        # Scheduler state
        #
        gauge_scheduler_running = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames)
        self.gauge_scheduler_running = make_per_engine(gauge_scheduler_running,
                                                       engine_indexes,
                                                       model_name)

        gauge_scheduler_waiting = self._gauge_cls(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames)
        self.gauge_scheduler_waiting = make_per_engine(gauge_scheduler_waiting,
                                                       engine_indexes,
                                                       model_name)

        #
        # GPU cache
        #
        # Deprecated in 0.9.2 - Renamed as vllm:kv_cache_usage_perc
        # With 0.11.x you can enable with --show-hidden-metrics-for-version=0.10
        # TODO: remove in 0.12.0
        if self.show_hidden_metrics:
            gauge_gpu_cache_usage = self._gauge_cls(
                name="vllm:gpu_cache_usage_perc",
                documentation=(
                    "GPU KV-cache usage. 1 means 100 percent usage."
                    "DEPRECATED: Use vllm:kv_cache_usage_perc instead."),
                multiprocess_mode="mostrecent",
                labelnames=labelnames)
            self.gauge_gpu_cache_usage = make_per_engine(
                gauge_gpu_cache_usage, engine_indexes, model_name)

        # Deprecated in 0.9.2 - Renamed as vllm:prefix_cache_queries
        # With 0.11.x you can enable with --show-hidden-metrics-for-version=0.10
        # TODO: remove in 0.12.0
        if self.show_hidden_metrics:
            counter_gpu_prefix_cache_queries = self._counter_cls(
                name="vllm:gpu_prefix_cache_queries",
                documentation=(
                    "GPU prefix cache queries, in terms of number of queried"
                    "tokens. DEPRECATED: Use vllm:prefix_cache_queries instead."
                ),
                labelnames=labelnames)
            self.counter_gpu_prefix_cache_queries = make_per_engine(
                counter_gpu_prefix_cache_queries, engine_indexes, model_name)

        # Deprecated in 0.9.2 - Renamed as vllm:prefix_cache_hits
        # With 0.11.x you can enable with --show-hidden-metrics-for-version=0.10
        # TODO: remove in 0.12.0
        if self.show_hidden_metrics:
            counter_gpu_prefix_cache_hits = self._counter_cls(
                name="vllm:gpu_prefix_cache_hits",
                documentation=(
                    "GPU prefix cache hits, in terms of number of cached "
                    "tokens. DEPRECATED: Use vllm:prefix_cache_hits instead."),
                labelnames=labelnames)
            self.counter_gpu_prefix_cache_hits = make_per_engine(
                counter_gpu_prefix_cache_hits, engine_indexes, model_name)

        gauge_kv_cache_usage = self._gauge_cls(
            name="vllm:kv_cache_usage_perc",
            documentation="KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        self.gauge_kv_cache_usage = make_per_engine(gauge_kv_cache_usage,
                                                    engine_indexes, model_name)

        counter_prefix_cache_queries = self._counter_cls(
            name="vllm:prefix_cache_queries",
            documentation=(
                "Prefix cache queries, in terms of number of queried tokens."),
            labelnames=labelnames)
        self.counter_prefix_cache_queries = make_per_engine(
            counter_prefix_cache_queries, engine_indexes, model_name)

        counter_prefix_cache_hits = self._counter_cls(
            name="vllm:prefix_cache_hits",
            documentation=(
                "Prefix cache hits, in terms of number of cached tokens."),
            labelnames=labelnames)
        self.counter_prefix_cache_hits = make_per_engine(
            counter_prefix_cache_hits, engine_indexes, model_name)

        #
        # Counters
        #
        counter_num_preempted_reqs = self._counter_cls(
            name="vllm:num_preemptions",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames)
        self.counter_num_preempted_reqs = make_per_engine(
            counter_num_preempted_reqs, engine_indexes, model_name)

        counter_prompt_tokens = self._counter_cls(
            name="vllm:prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        self.counter_prompt_tokens = make_per_engine(counter_prompt_tokens,
                                                     engine_indexes,
                                                     model_name)

        counter_generation_tokens = self._counter_cls(
            name="vllm:generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        self.counter_generation_tokens = make_per_engine(
            counter_generation_tokens, engine_indexes, model_name)

        self.counter_request_success: dict[FinishReason, dict[
            int, prometheus_client.Counter]] = {}
        counter_request_success_base = self._counter_cls(
            name="vllm:request_success",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"])
        for reason in FinishReason:
            self.counter_request_success[reason] = {
                idx:
                counter_request_success_base.labels(model_name, str(idx),
                                                    str(reason))
                for idx in engine_indexes
            }

        #
        # Histograms of counts
        #
        histogram_num_prompt_tokens_request = self._histogram_cls(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames)
        self.histogram_num_prompt_tokens_request = make_per_engine(
            histogram_num_prompt_tokens_request, engine_indexes, model_name)

        histogram_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames)
        self.histogram_num_generation_tokens_request = make_per_engine(
            histogram_num_generation_tokens_request, engine_indexes,
            model_name)

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        # See: https://github.com/vllm-project/vllm/pull/18053
        histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
            documentation="Histogram of number of tokens per engine_step.",
            buckets=[
                1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
            ],
            labelnames=labelnames)
        self.histogram_iteration_tokens = make_per_engine(
            histogram_iteration_tokens, engine_indexes, model_name)

        histogram_max_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_max_num_generation_tokens",
            documentation=
            "Histogram of maximum number of requested generation tokens.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames)
        self.histogram_max_num_generation_tokens_request = make_per_engine(
            histogram_max_num_generation_tokens_request, engine_indexes,
            model_name)

        histogram_n_request = self._histogram_cls(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            buckets=[1, 2, 5, 10, 20],
            labelnames=labelnames)
        self.histogram_n_request = make_per_engine(histogram_n_request,
                                                   engine_indexes, model_name)

        histogram_max_tokens_request = self._histogram_cls(
            name="vllm:request_params_max_tokens",
            documentation="Histogram of the max_tokens request parameter.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames)
        self.histogram_max_tokens_request = make_per_engine(
            histogram_max_tokens_request, engine_indexes, model_name)

        #
        # Histogram of timing intervals
        #
        histogram_time_to_first_token = self._histogram_cls(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=labelnames)
        self.histogram_time_to_first_token = make_per_engine(
            histogram_time_to_first_token, engine_indexes, model_name)

        # Deprecated in 0.11 - Renamed as vllm:inter_token_latency_seconds
        # TODO: in 0.12, only enable if show_hidden_metrics=True
        histogram_time_per_output_token = self._histogram_cls(
            name="vllm:time_per_output_token_seconds",
            documentation=(
                "Histogram of time per output token in seconds."
                "DEPRECATED: Use vllm:inter_token_latency_seconds instead."),
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=labelnames)
        self.histogram_time_per_output_token = make_per_engine(
            histogram_time_per_output_token, engine_indexes, model_name)

        histogram_inter_token_latency = self._histogram_cls(
            name="vllm:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=labelnames)
        self.histogram_inter_token_latency = make_per_engine(
            histogram_inter_token_latency, engine_indexes, model_name)

        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
            40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
        ]
        histogram_e2e_time_request = self._histogram_cls(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of e2e request latency in seconds.",
            buckets=request_latency_buckets,
            labelnames=labelnames)
        self.histogram_e2e_time_request = make_per_engine(
            histogram_e2e_time_request, engine_indexes, model_name)

        histogram_queue_time_request = self._histogram_cls(
            name="vllm:request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames)
        self.histogram_queue_time_request = make_per_engine(
            histogram_queue_time_request, engine_indexes, model_name)

        histogram_inference_time_request = self._histogram_cls(
            name="vllm:request_inference_time_seconds",
            documentation=
            "Histogram of time spent in RUNNING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames)
        self.histogram_inference_time_request = make_per_engine(
            histogram_inference_time_request, engine_indexes, model_name)

        histogram_prefill_time_request = self._histogram_cls(
            name="vllm:request_prefill_time_seconds",
            documentation=
            "Histogram of time spent in PREFILL phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames)
        self.histogram_prefill_time_request = make_per_engine(
            histogram_prefill_time_request, engine_indexes, model_name)

        histogram_decode_time_request = self._histogram_cls(
            name="vllm:request_decode_time_seconds",
            documentation=
            "Histogram of time spent in DECODE phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames)
        self.histogram_decode_time_request = make_per_engine(
            histogram_decode_time_request, engine_indexes, model_name)

        #
        # LoRA metrics
        #

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        self.gauge_lora_info: Optional[prometheus_client.Gauge] = None
        if vllm_config.lora_config is not None:
            if len(self.engine_indexes) > 1:
                raise NotImplementedError(
                    "LoRA in DP mode is not supported yet.")
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

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        """Log to prometheus."""
        if scheduler_stats is not None:
            self.gauge_scheduler_running[engine_idx].set(
                scheduler_stats.num_running_reqs)
            self.gauge_scheduler_waiting[engine_idx].set(
                scheduler_stats.num_waiting_reqs)

            if self.show_hidden_metrics:
                self.gauge_gpu_cache_usage[engine_idx].set(
                    scheduler_stats.kv_cache_usage)
            self.gauge_kv_cache_usage[engine_idx].set(
                scheduler_stats.kv_cache_usage)

            if self.show_hidden_metrics:
                self.counter_gpu_prefix_cache_queries[engine_idx].inc(
                    scheduler_stats.prefix_cache_stats.queries)
                self.counter_gpu_prefix_cache_hits[engine_idx].inc(
                    scheduler_stats.prefix_cache_stats.hits)

            self.counter_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.queries)
            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits)

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_prom.observe(
                    scheduler_stats.spec_decoding_stats, engine_idx)

        if iteration_stats is None:
            return

        self.counter_num_preempted_reqs[engine_idx].inc(
            iteration_stats.num_preempted_reqs)
        self.counter_prompt_tokens[engine_idx].inc(
            iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens[engine_idx].inc(
            iteration_stats.num_generation_tokens)
        self.histogram_iteration_tokens[engine_idx].observe(
            iteration_stats.num_prompt_tokens + \
            iteration_stats.num_generation_tokens)

        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request[
                engine_idx].observe(max_gen_tokens)
        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request[engine_idx].observe(n_param)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token[engine_idx].observe(ttft)
        for itl in iteration_stats.inter_token_latencies_iter:
            self.histogram_inter_token_latency[engine_idx].observe(itl)
            self.histogram_time_per_output_token[engine_idx].observe(itl)

        for finished_request in iteration_stats.finished_requests:
            self.counter_request_success[
                finished_request.finish_reason][engine_idx].inc()
            self.histogram_e2e_time_request[engine_idx].observe(
                finished_request.e2e_latency)
            self.histogram_queue_time_request[engine_idx].observe(
                finished_request.queued_time)
            self.histogram_prefill_time_request[engine_idx].observe(
                finished_request.prefill_time)
            self.histogram_inference_time_request[engine_idx].observe(
                finished_request.inference_time)
            self.histogram_decode_time_request[engine_idx].observe(
                finished_request.decode_time)
            self.histogram_num_prompt_tokens_request[engine_idx].observe(
                finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request[engine_idx].observe(
                finished_request.num_generation_tokens)
            if finished_request.max_tokens_param:
                self.histogram_max_tokens_request[engine_idx].observe(
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


PromMetric = Union[
    prometheus_client.Gauge,
    prometheus_client.Counter,
    prometheus_client.Histogram,
]


def make_per_engine(metric: PromMetric, engine_idxs: list[int],
                    model_name: str) -> dict[int, PromMetric]:
    return {idx: metric.labels(model_name, str(idx)) for idx in engine_idxs}


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


class StatLoggerManager:
    """
    StatLoggerManager:
        Logging happens at the level of the EngineCore (per scheduler).
         * DP: >1 EngineCore per AsyncLLM - loggers for each EngineCore.
         * With Local Logger, just make N copies for N EngineCores.
         * With Prometheus, we need a single logger with N "labels"

        This class abstracts away this implementation detail from
        the AsyncLLM, allowing the AsyncLLM to just call .record()
        and .log() to a simple interface.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_idxs: Optional[list[int]] = None,
        custom_stat_loggers: Optional[list[StatLoggerFactory]] = None,
        enable_default_loggers: bool = True,
        client_count: int = 1,
    ):
        self.engine_idxs = engine_idxs if engine_idxs else [0]

        factories: list[StatLoggerFactory] = []
        if custom_stat_loggers is not None:
            factories.extend(custom_stat_loggers)

        if enable_default_loggers and logger.isEnabledFor(logging.INFO):
            if client_count > 1:
                logger.warning(
                    "AsyncLLM created with api_server_count more than 1; "
                    "disabling stats logging to avoid incomplete stats.")
            else:
                factories.append(LoggingStatLogger)

        # engine_idx: StatLogger
        self.per_engine_logger_dict: dict[int, list[StatLoggerBase]] = {}
        prometheus_factory = PrometheusStatLogger
        for engine_idx in self.engine_idxs:
            loggers: list[StatLoggerBase] = []
            for logger_factory in factories:
                # If we get a custom prometheus logger, use that
                # instead. This is typically used for the ray case.
                if (isinstance(logger_factory, type)
                        and issubclass(logger_factory, PrometheusStatLogger)):
                    prometheus_factory = logger_factory
                    continue
                loggers.append(logger_factory(vllm_config,
                                              engine_idx))  # type: ignore
            self.per_engine_logger_dict[engine_idx] = loggers

        # For Prometheus, need to share the metrics between EngineCores.
        # Each EngineCore's metrics are expressed as a unique label.
        self.prometheus_logger = prometheus_factory(vllm_config, engine_idxs)

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: Optional[int] = None,
    ):
        if engine_idx is None:
            engine_idx = 0

        per_engine_loggers = self.per_engine_logger_dict[engine_idx]
        for logger in per_engine_loggers:
            logger.record(scheduler_stats, iteration_stats, engine_idx)

        self.prometheus_logger.record(scheduler_stats, iteration_stats,
                                      engine_idx)

    def log(self):
        for per_engine_loggers in self.per_engine_logger_dict.values():
            for logger in per_engine_loggers:
                logger.log()

    def log_engine_initialized(self):
        self.prometheus_logger.log_engine_initialized()

        for per_engine_loggers in self.per_engine_logger_dict.values():
            for logger in per_engine_loggers:
                logger.log_engine_initialized()
