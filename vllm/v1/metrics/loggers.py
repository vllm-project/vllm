import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import prometheus_client

from vllm.config import ModelConfig
from vllm.logger import init_logger
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
            "Running: %d reqs, Waiting: %d reqs ",
            prompt_throughput,
            generation_throughput,
            scheduler_stats.num_running_reqs,
            scheduler_stats.num_waiting_reqs,
        )


class PrometheusStatLogger(StatLoggerBase):

    def __init__(self, model_config: ModelConfig):
        self._unregister_vllm_metrics()

        labelnames = ["model_name"]
        labelvalues = [model_config.served_model_name]

        max_model_len = model_config.max_model_len

        self.gauge_scheduler_running = prometheus_client.Gauge(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            labelnames=labelnames).labels(*labelvalues)

        self.gauge_scheduler_waiting = prometheus_client.Gauge(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_prompt_tokens = prometheus_client.Counter(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames).labels(*labelvalues)

        self.counter_generation_tokens = prometheus_client.Counter(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames).labels(*labelvalues)

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

    def log(self, scheduler_stats: SchedulerStats,
            iteration_stats: IterationStats):
        """Log to prometheus."""
        self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)

        self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens.inc(
            iteration_stats.num_generation_tokens)

        for finished_request in iteration_stats.finished_requests:
            self.histogram_num_prompt_tokens_request.observe(
                finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request.observe(
                finished_request.num_generation_tokens)

        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token.observe(ttft)
        for tpot in iteration_stats.time_per_output_tokens_iter:
            self.histogram_time_per_output_token.observe(tpot)

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
