import time
from abc import ABC, abstractmethod
from typing import Dict

import prometheus_client

from vllm.logger import init_logger
from vllm.v1.metrics.stats import SchedulerStats

logger = init_logger(__name__)

_LOCAL_LOGGING_INTERVAL_SEC = 5.0


class StatLoggerBase(ABC):

    @abstractmethod
    def log(self, scheduler_stats: SchedulerStats):
        ...


class LoggingStatLogger(StatLoggerBase):

    def __init__(self):
        self.last_log_time = time.monotonic()

    def log(self, scheduler_stats: SchedulerStats):
        """Log Stats to standard output."""

        # Log every _LOCAL_LOGGING_INTERVAL_SEC.
        now = time.monotonic()
        if now - self.last_log_time < _LOCAL_LOGGING_INTERVAL_SEC:
            return
        self.last_log_time = now

        # Format and print output.
        logger.info(
            "Running: %d reqs, Waiting: %d reqs ",
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

    def log(self, scheduler_stats: SchedulerStats):
        """Log to prometheus."""
        self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
        self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)

    @staticmethod
    def _unregister_vllm_metrics():
        # Unregister any existing vLLM collectors (for CI/CD
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                prometheus_client.REGISTRY.unregister(collector)
