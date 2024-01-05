from aioprometheus import Counter, Gauge, Histogram
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Dict, Callable

labels = {}


def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)


# Stats defintions.
# These are the interface between the LLMEngine and the Logger/Metrics.
@dataclass
class Stats:
    # System stats representing snapshot of state.
    total_gpu_blocks: int
    total_cpu_blocks: int
    free_gpu_blocks: int
    free_cpu_blocks: int
    num_running: int
    num_waiting: int
    num_swapped: int

    # Raw stats from most recent model iteration.
    prompt_run: bool
    num_batched_tokens: int
    iter_timings: List[float]
    e2e_timings: List[float]


class PrometheusMetric(ABC):
    """Log Stats to a Prometheus Metric."""

    @abstractmethod
    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        raise NotImplementedError


@dataclass
class CounterMetric(PrometheusMetric):
    """Compute and log Counter"""
    counter: Counter
    fn: Callable[[Stats], Union[int, float]]

    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        self.counter.add(labels, self.fn(stats))


@dataclass
class GaugeMetric(PrometheusMetric):
    """Compute and log Gauge"""
    gauge: Gauge
    fn: Callable[[Stats], Union[int, float]]

    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        self.gauge.set(labels, self.fn(stats))


@dataclass
class HistogramMetric(PrometheusMetric):
    """Compute and log Histogram"""
    histogram: Histogram
    fn: Callable[[Stats], Union[List[int], List[float]]]

    def log(self, stats: Stats, labels: Dict[str, str]) -> None:
        for metric in self.fn(stats):
            self.histogram.observe(labels, metric)


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
histogram_inter_token_latency = Histogram(
    "vllm_inter_token_latency_seconds",
    "Histogram of inter token latency in seconds.",
    buckets=[0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 5.0])
histogram_e2e_request_latency = Histogram(
    "vllm_e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=[0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0])

# end-metrics-definitions


# Functions to extract Metric from Stats.
def _cache_usage(num_total: int, num_free: int) -> float:
    return 1.0 - num_free / num_total if num_total > 0 else 0.


prompt_tokens_fn = lambda stats: stats.num_batched_tokens if stats.prompt_run else 0
generation_tokens_fn = lambda stats: stats.num_batched_tokens if not stats.prompt_run else 0
scheduler_running_fn = lambda stats: stats.num_running
scheduler_swapped_fn = lambda stats: stats.num_swapped
scheduler_waiting_fn = lambda stats: stats.num_waiting
gpu_cache_usage_fn = lambda stats: _cache_usage(stats.total_gpu_blocks, stats.
                                                free_gpu_blocks)
cpu_cache_usage_fn = lambda stats: _cache_usage(stats.total_cpu_blocks, stats.
                                                free_cpu_blocks)
time_to_first_token_fn = lambda stats: stats.iter_timings if stats.prompt_run else [
]
inter_token_latency_fn = lambda stats: stats.iter_timings if not stats.prompt_run else [
]
e2e_request_latency_fn = lambda stats: stats.e2e_timings


class PrometheusLogger:
    """PrometheusLogger used by LLMEngine to log stats to Prom."""

    def __init__(self) -> None:
        self.metrics: List[PrometheusMetric] = [
            GaugeMetric(gauge_scheduler_running, scheduler_running_fn),
            GaugeMetric(gauge_scheduler_swapped, scheduler_swapped_fn),
            GaugeMetric(gauge_scheduler_waiting, scheduler_waiting_fn),
            GaugeMetric(gauge_gpu_cache_usage, gpu_cache_usage_fn),
            GaugeMetric(gauge_cpu_cache_usage, cpu_cache_usage_fn),
            CounterMetric(counter_prompt_tokens, prompt_tokens_fn),
            CounterMetric(counter_generation_tokens, generation_tokens_fn),
            HistogramMetric(histogram_time_to_first_token,
                            time_to_first_token_fn),
            HistogramMetric(histogram_inter_token_latency,
                            inter_token_latency_fn),
            HistogramMetric(histogram_e2e_request_latency,
                            e2e_request_latency_fn),
        ]

    def log(self, stats: Stats) -> None:
        for metric in self.metrics:
            metric.log(stats=stats, labels=labels)
