# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import prometheus_client

from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = init_logger(__name__)

_LOCAL_LOGGING_INTERVAL_SEC = 5.0


class StatLoggerBase(ABC):

    @abstractmethod
    def log(self, scheduler_stats: SchedulerStats,
            iteration_stats: IterationStats):
        ...


class LoggingStatLogger(StatLoggerBase):

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

        # Prefix cache metrics. TODO: Make the interval configurable.
        self.prefix_caching_metrics = PrefixCachingMetrics()

    def _local_interval_elapsed(self, now: float) -> bool:
        # Log every _LOCAL_LOGGING_INTERVAL_SEC.
        elapsed_time = now - self.last_log_time
        return elapsed_time > _LOCAL_LOGGING_INTERVAL_SEC

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(iteration_stats.num_prompt_tokens)
        self.num_generation_tokens.append(
            iteration_stats.num_generation_tokens)

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        # Compute summary metrics for tracked stats
        return float(np.sum(tracked_stats) / (now - self.last_log_time))

    def log(self, scheduler_stats: SchedulerStats,
            iteration_stats: IterationStats):
        """Log Stats to standard output."""

        self._track_iteration_stats(iteration_stats)

        self.prefix_caching_metrics.observe(scheduler_stats.prefix_cache_stats)

        now = time.monotonic()
        if not self._local_interval_elapsed(now):
            return

        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(
            self.num_generation_tokens, now)

        self._reset(now)

        # Format and print output.
        logger.info(
            "Avg prompt throughput: %.1f tokens/s, "
            "Avg generation throughput: %.1f tokens/s, "
            "Running: %d reqs, Waiting: %d reqs, "
            "GPU KV cache usage: %.1f%%, "
            "Prefix cache hit rate: %.1f%%",
            prompt_throughput,
            generation_throughput,
            scheduler_stats.num_running_reqs,
            scheduler_stats.num_waiting_reqs,
            scheduler_stats.gpu_cache_usage * 100,
            self.prefix_caching_metrics.hit_rate * 100,
        )


class PrometheusStatLogger(StatLoggerBase):

    def __init__(self, vllm_config: VllmConfig):
        self._unregister_vllm_metrics()

        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = \
            vllm_config.observability_config.show_hidden_metrics

        labelnames = ["model_name"]
        labelvalues = [vllm_config.model_config.served_model_name]

        max_model_len = vllm_config.model_config.max_model_len

        self.gauge_scheduler_running = prometheus_client.Gauge(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            labelnames=labelnames).labels(*labelvalues)

        self.gauge_scheduler_waiting = prometheus_client.Gauge(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.gauge_gpu_cache_usage = prometheus_client.Gauge(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames).labels(*labelvalues)

        self.gauge_total_tokens_in_queue = prometheus_client.Gauge(
            name="vllm:total_tokens_in_queue",
            documentation=
            "Total number of tokens currently waiting in the queue",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_gpu_prefix_cache_queries = prometheus_client.Counter(
            name="vllm:gpu_prefix_cache_queries",
            documentation=
            "GPU prefix cache queries, in terms of number of queried blocks.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_gpu_prefix_cache_hits = prometheus_client.Counter(
            name="vllm:gpu_prefix_cache_hits",
            documentation=
            "GPU prefix cache hits, in terms of number of cached blocks.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_num_preempted_reqs = prometheus_client.Counter(
            name="vllm:num_preemptions_total",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_prompt_tokens = prometheus_client.Counter(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_generation_tokens = prometheus_client.Counter(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_total_evicted_tokens = prometheus_client.Counter(
            name="vllm:total_evicted_tokens_total",
            documentation="Total number of tokens evicted from KV cache.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_request_success: Dict[FinishReason,
                                           prometheus_client.Counter] = {}
        counter_request_success_base = prometheus_client.Counter(
            name="vllm:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"])
        for reason in FinishReason:
            self.counter_request_success[
                reason] = counter_request_success_base.labels(*(labelvalues +
                                                                [str(reason)]))

        self.histogram_num_prompt_tokens_request = \
            prometheus_client.Histogram(
                name="vllm:request_prompt_tokens",
                documentation="Number of prefill tokens processed.",
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_num_generation_tokens_request = \
            prometheus_client.Histogram(
                name="vllm:request_generation_tokens",
                documentation="Number of generation tokens processed.",
                buckets=build_1_2_5_buckets(max_model_len),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_iteration_tokens = \
            prometheus_client.Histogram(
                name="vllm:iteration_tokens_total",
                documentation="Histogram of number of tokens per engine_step.",
                buckets=build_cudagraph_buckets(vllm_config),
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_time_to_first_token = \
            prometheus_client.Histogram(
                name="vllm:time_to_first_token_seconds",
                documentation="Histogram of time to first token in seconds.",
                buckets=[
                    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_time_per_output_token = \
            prometheus_client.Histogram(
                name="vllm:time_per_output_token_seconds",
                documentation="Histogram of time per output token in seconds.",
                buckets=[
                    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                    0.75, 1.0, 2.5
                ],
                labelnames=labelnames).labels(*labelvalues)

        self.histogram_time_per_prefill_token_request = \
            prometheus_client.Histogram(
                name="vllm:time_per_prefill_token_request_seconds",
                documentation="Time spent per token during prefill  "
                "phase in seconds",
                buckets=[
                    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0
                ],
                labelnames=labelnames).labels(*labelvalues)

        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
            40.0, 50.0, 60.0
        ]
        self.histogram_e2e_time_request = \
            prometheus_client.Histogram(
                name="vllm:e2e_request_latency_seconds",
                documentation="Histogram of e2e request latency in seconds.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_queue_time_request = \
            prometheus_client.Histogram(
                name="vllm:request_queue_time_seconds",
                documentation=
                "Histogram of time spent in WAITING phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_inference_time_request = \
            prometheus_client.Histogram(
                name="vllm:request_inference_time_seconds",
                documentation=
                "Histogram of time spent in RUNNING phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_prefill_time_request = \
            prometheus_client.Histogram(
                name="vllm:request_prefill_time_seconds",
                documentation=
                "Histogram of time spent in PREFILL phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        self.histogram_decode_time_request = \
            prometheus_client.Histogram(
                name="vllm:request_decode_time_seconds",
                documentation=
                "Histogram of time spent in DECODE phase for request.",
                buckets=request_latency_buckets,
                labelnames=labelnames).labels(*labelvalues)
        max_token_capacity = min(
            vllm_config.model_config.max_model_len *
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.gauge_max_token_capacity_per_batch = prometheus_client.Gauge(
            name="vllm:max_token_capacity_per_batch",
            documentation=
            "Maximum tokens processed by the model server at max batch size",
            labelnames=labelnames).labels(*labelvalues)
        self.gauge_max_token_capacity_per_batch.set(max_token_capacity)

        self.gauge_lora_info: Optional[prometheus_client.Gauge] = None
        if vllm_config.lora_config is not None:
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = vllm_config.lora_config.max_loras
            self.gauge_lora_info = \
                prometheus_client.Gauge(
                    name="vllm:lora_requests_info",
                    documentation="Running stats on lora requests.",
                    labelnames=[
                        self.labelname_max_lora,
                        self.labelname_waiting_lora_adapters,
                        self.labelname_running_lora_adapters,
                    ])

        self.log_metrics_info("cache_config", vllm_config.cache_config)

    def log_metrics_info(self, type: str, config_obj: SupportsMetricsInfo):
        metrics_info = config_obj.metrics_info()

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

    def log(self, scheduler_stats: SchedulerStats,
            iteration_stats: IterationStats):
        """Log to prometheus."""
        self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)
        self.gauge_total_tokens_in_queue.set(
            scheduler_stats.total_tokens_in_queue)
        self.gauge_gpu_cache_usage.set(scheduler_stats.gpu_cache_usage)

        self.counter_gpu_prefix_cache_queries.inc(
            scheduler_stats.prefix_cache_stats.queries)
        self.counter_gpu_prefix_cache_hits.inc(
            scheduler_stats.prefix_cache_stats.hits)

        self.counter_num_preempted_reqs.inc(iteration_stats.num_preempted_reqs)
        self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens.inc(
            iteration_stats.num_generation_tokens)
        self.counter_total_evicted_tokens.inc(
            scheduler_stats.num_evicted_tokens)
        self.histogram_iteration_tokens.observe(
            iteration_stats.num_prompt_tokens + \
            iteration_stats.num_generation_tokens)

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
            if finished_request.num_prompt_tokens > 0:
                time_per_prefill_token = (finished_request.prefill_time /
                                          finished_request.num_prompt_tokens)
                self.histogram_time_per_prefill_token_request.observe(
                    time_per_prefill_token)
            self.histogram_inference_time_request.observe(
                finished_request.inference_time)
            self.histogram_decode_time_request.observe(
                finished_request.decode_time)
            self.histogram_num_prompt_tokens_request.observe(
                finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request.observe(
                finished_request.num_generation_tokens)

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

    @staticmethod
    def _unregister_vllm_metrics():
        # Unregister any existing vLLM collectors (for CI/CD
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                prometheus_client.REGISTRY.unregister(collector)


def build_buckets(mantissa_lst: List[int], max_value: int) -> List[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum.

    """
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)


def build_cudagraph_buckets(vllm_config: VllmConfig) -> List[int]:
    if not vllm_config.model_config.enforce_eager:
        buckets = vllm_config.compilation_config.\
            cudagraph_capture_sizes.copy()
        buckets.sort()
        return buckets
    else:
        return [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8096]
