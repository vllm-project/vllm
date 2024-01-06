import time
import numpy as np
from aioprometheus import Counter, Gauge, Histogram
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Dict, Callable

labels = {}


def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
counter_prompt_tokens = Counter("vllm_prompt_tokens_total",
                                "Number of prefill tokens processed.")
counter_generation_tokens = Counter("vllm_generation_tokens_total",
                                    "Number of generation tokens processed.")

gauge_scheduler_running = Gauge(
    "vllm_requests_running_total",
    "Number of requests currently running on GPU.")
gauge_scheduler_swapped = Gauge("vllm_requests_swapped_total",
                                "Number of requests swapped to CPU.")
gauge_scheduler_waiting = Gauge("vllm_requests_waiting_total",
                                "Number of requests waiting to be processed.")

gauge_gpu_cache_usage = Gauge(
    "vllm_gpu_cache_usage_perc",
    "GPU KV-cache usage. 1 means 100 percent usage.")
gauge_cpu_cache_usage = Gauge(
    "vllm_cpu_cache_usage_perc",
    "CPU KV-cache usage. 1 means 100 percent usage.")

histogram_time_to_first_token = Histogram(
    "vllm_time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])
histogram_time_per_output_tokens = Histogram(
    "vllm_time_per_output_tokens_seconds",
    "Histogram of time per output token in seconds.",
    buckets=[0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 5.0])
histogram_e2e_request_latency = Histogram(
    "vllm_e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=[0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0])

# end-metrics-definitions

# Stats defintions.
# These are the interface between the LLMEngine and the Logger/Metrics.
@dataclass
class Stats:
    # System stats representing snapshot of state.
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float
    # Raw stats from most recent model iteration.
    num_prompt_tokens: int
    num_generation_tokens: int
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    time_e2e_requests: List[float]

class TrackedStats:
    self.stats: Union[List[int], List[float]] = []
    self.times: List[float] = []

    def append(self, new_stat: Union[int, float], now: float) -> None:
        self.stats.append(new_item)
        self.times.append(now)

    def reset(self) -> None:
        self.stats = []
        self.times = []

    def get_throughput(self) -> float:
        if self.stats.len() > 1:
            return np.sum(self.stats) / self.times[-1]
        return 0.0
        
class PrometheusMetric(ABC):
    """Log Stats to a Prometheus Metric and Tracks Metrics For Local Logging"""
    def __init__(self, stats_attr: str, template: Template) -> None:
        # Attribute to extract correct data from Stats in self.log.
        self.stats_attr = stats_attr
        # String template with $metric for local logging.
        self.template = template

    @abstractmethod
    # Extract data from stats, log to prometheus, and track data.
    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    # Compute metric from tracked data log.
    def compute_metric(self) -> str:
        raise NotImplementedError

class CounterMetric(PrometheusMetric):
    """Implementation of PrometheusMetric for Counters"""
    def __init__(self, counter: Counter, stats_attr: str, template: Template) -> None:
        self.counter = counter
        self.tracked_stats = TrackedStats()
        super().__init__(stats_attr, template)

    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        stat = getattr(stats, self.stats_attr, 0)
        self.counter.add(labels, stat)
        self.tracked_stats.append(stat, now)

    def compute_metric(self):
        metric = self.tracked_data.get_throughput()
        self.tracked_stats.reset()
        return self.template.safe_substitute(metric=metric)

class GaugeMetric(PrometheusMetric):
    """Implementation of PrometheusMetric for Gauges"""
    def __init__(self, gauge: Gauge, stats_attr: str, template: Template) -> None:
        self.gauge = gauge
        self.metric = 0
        super().__init__(stats_attr, template)

    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        stat = getattr(stats, self.stats_attr, 0)
        self.gauge.set(labels, stat)
        self.metric = stat

    def compute_metric(self):
        return self.template.safe_substitute(metric=self.metric)

class HistogramMetric(PrometheusMetric):
    """Implementation of PrometheusMetric for Histograms"""
    def __init__(self, histogram: Histogram, stats_attr: str, template: Template) -> None:
        self.histogram = histogram
        self.tracked_stats = TrackedStats()
        super().__init__(stats_attr, template)

    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        stat = getattr(stats, self.stats_attr, 0)
        self.gauge.set(labels, stat)
        self.last_stat = stat

    def compute_metric(self):
        metric = self.tracked_data.get_throughput()
        return self.template.safe_substitute(metric=self.last_stat)



    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        self.counter.add(labels, getattr(stats, self.attr, 0))
    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        self.gauge.set(labels, getattr(stats, self.attr, 0))
    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        for metric in getattr(stats, self.attr, []):
            self.histogram.observe(labels, metric)


# Functions to extract Metric from Stats.
def _cache_usage(num_total: int, num_free: int) -> float:
    return 1.0 - num_free / num_total if num_total > 0 else 0.

class PrometheusLogger:
    """PrometheusLogger used by LLMEngine to log stats to Prom."""

    def __init__(self) -> None:
        self.metrics: List[PrometheusMetric] = [
            GaugeMetric(gauge_scheduler_running, "num_running"),
            GaugeMetric(gauge_scheduler_swapped, "num_swapped"),
            GaugeMetric(gauge_scheduler_waiting, "num_waiting"),
            GaugeMetric(gauge_gpu_cache_usage, "gpu_cache_usage"),
            GaugeMetric(gauge_cpu_cache_usage, "cpu_cache_usage"),
            CounterMetric(counter_prompt_tokens, "num_prompt_tokens"),
            CounterMetric(counter_generation_tokens, "num_generation_tokens"),
            HistogramMetric(histogram_time_to_first_token,
                            "time_to_first_tokens"),
            HistogramMetric(histogram_inter_token_latency,
                            "time_per_output_tokens"),
            HistogramMetric(histogram_e2e_request_latency,
                            "time_e2e_requests"),
        ]

    def log(self, stats: Stats) -> None:
        for metric in self.metrics:
            metric.log(stats=stats, labels=labels)