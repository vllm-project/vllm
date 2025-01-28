import time
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import prometheus_client

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

    def __init__(self, labels: Dict[str, str]):
        self.labels = labels

        labelnames = self.labels.keys()
        labelvalues = self.labels.values()

        self._unregister_vllm_metrics()

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

    def log(self, scheduler_stats: SchedulerStats,
            iteration_stats: IterationStats):
        """Log to prometheus."""
        self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)

        self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens.inc(
            iteration_stats.num_generation_tokens)

    @staticmethod
    def _unregister_vllm_metrics():
        # Unregister any existing vLLM collectors (for CI/CD
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                prometheus_client.REGISTRY.unregister(collector)
