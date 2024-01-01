from aioprometheus import Counter, Gauge, Histogram
from abc import ABC
from dataclasses import dataclass
from typing import Union, List, Dict, Callable, Optional

labels = {}

def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)

# Stats defintions. 
# These are the interface between the LLMEngine and the Logger/Metrics.
@dataclass
class SystemStats:
    """Raw snapshot of the system state at a given time."""
    total_gpu_blocks: int
    total_cpu_blocks: int
    free_gpu_blocks: int
    free_cpu_blocks: int
    num_running: int
    num_waiting: int
    num_swapped: int

@dataclass
class IterationStats:
    """Raw stats from most recent model iteration."""
    prompt_run: bool
    num_batched_tokens: int
    latency_timings: List[float]

Stats = Union[SystemStats, IterationStats]

@dataclass
class PrometheusMetric(ABC):
    """Metric and Function from Stats -> Metric."""
    metric: Union[Counter, Gauge, Histogram]
    fn: Callable[[Stats], Union[List[int], List[float], Union[int,float]]]

    def log(self, stats: Stats, labels: Dict[str,str]) -> None:
        raise NotImplementedError

class CounterMetric(PrometheusMetric):
    def log(self, stats: Stats, labels: Dict[str,str]) -> None:
        self.metric.add(labels, self.fn(stats))

class GaugeMetric(PrometheusMetric):
    def log(self, stats: Stats, labels: Dict[str,str]) -> None:
        self.metric.set(labels, self.fn(stats))

class HistogramMetric(PrometheusMetric):    
    def log(self, stats: Stats, labels: Dict[str,str]) -> None:
        for metric in self.fn(stats):
            self.metric.observe(labels, metric)

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
counter_prompt_tokens = Counter(
    "vllm_prompt_tokens_total",
    "Number of prefill tokens processed.")
counter_generation_tokens = Counter(
    "vllm_generation_tokens_total",
    "Number of generation tokens processed.")

gauge_scheduler_running = Gauge(
    "vllm_requests_running_total",
    "Number of requests that is currently running for inference.")
gauge_scheduler_swapped = Gauge(
    "vllm_requests_stopped_total",
    "Number requests swapped to CPU.")
gauge_scheduler_waiting = Gauge(
    "vllm_requests_waiting_total",
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
    buckets = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)
histogram_inter_token_latency = Histogram(
    "vllm_inter_token_latency_seconds",
    "Histogram of inter token latency in seconds.",
    buckets = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 5.0]
)
# end-metrics-definitions

# Functions to convert Stats --> Metrics.
def _cache_usage(num_total: int, num_free: int) -> float:
    return 1.0 - num_free / num_total if num_total > 0 else 0.

_prompt_tokens_fn = lambda stats: stats.num_batched_tokens if stats.prompt_run else 0
_generation_tokens_fn = lambda stats: stats.num_batched_tokens if not stats.prompt_run else 0
_scheduler_running_fn = lambda stats: stats.num_running
_scheduler_swapped_fn = lambda stats: stats.num_swapped
_scheduler_waiting_fn = lambda stats: stats.num_waiting
_gpu_cache_usage_fn = lambda stats: _cache_usage(stats.total_gpu_blocks, stats.free_gpu_blocks)
_cpu_cache_usage_fn = lambda stats: _cache_usage(stats.total_cpu_blocks, stats.free_cpu_blocks)
_time_to_first_token_fn = lambda stats: stats.latency_timings if stats.prompt_run else []
_inter_token_latency_fn = lambda stats: stats.latency_timings if not stats.prompt_run else []

class PrometheusLogger:
    """PrometheusLogger is used by LLMEngine to log statistics to Prometheus.
    
    There are two types of PrometheusMetrics:
        - system_metrics are snapshots of the system state. Logged every _LOGGING_INTERVAL.
        - iteration_metrics are info about the most recent model step. Logged every iteration.
    """
    def __init__(self):
        self._LOGGING_INTERVAL_SEC = 5
        self.last_system_logging_time = 0.0

        self.system_metrics: List[PrometheusMetric] = [
            GaugeMetric(metric=gauge_scheduler_running, fn=_scheduler_running_fn),
            GaugeMetric(metric=gauge_scheduler_swapped, fn=_scheduler_swapped_fn),
            GaugeMetric(metric=gauge_scheduler_waiting, fn=_scheduler_waiting_fn),
            GaugeMetric(metric=gauge_gpu_cache_usage, fn=_gpu_cache_usage_fn),
            GaugeMetric(metric=gauge_cpu_cache_usage, fn=_cpu_cache_usage_fn),
        ]
        self.iteration_metrics: List[PrometheusMetric] = [
            CounterMetric(metric=counter_prompt_tokens, fn=_prompt_tokens_fn),
            CounterMetric(metric=counter_generation_tokens, fn=_generation_tokens_fn),
            HistogramMetric(metric=histogram_time_to_first_token, fn=_time_to_first_token_fn),
            HistogramMetric(metric=histogram_inter_token_latency, fn=_inter_token_latency_fn),
        ]

    def should_log_system(self, now: float) -> bool:
        return now - self.last_system_logging_time >= self._LOGGING_INTERVAL_SEC

    def log(
        self, now: float,
        system_stats: Optional[SystemStats], 
        iteration_stats: IterationStats,
    ) -> None:
        # Log iteration_stats every iteration.
        for metric in self.iteration_metrics:
            metric.log(stats=iteration_stats, labels=labels)
        
        # Log system_stats every _LOGGING_INTERVAL_SEC.
        if system_stats is not None:
            self.last_system_logging_time = now
            for metric in self.system_metrics:
                metric.log(stats=system_stats, labels=labels)
        