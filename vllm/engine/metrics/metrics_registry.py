from aioprometheus import Counter, Gauge, Histogram

from vllm.engine.metrics.metrics_utils import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    Stats,
)

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

counter_time_to_first_token = Counter(
    "vllm_time_to_first_token_total_seconds",
    "Time to first token in seconds."
)
counter_inter_token_latency = Counter(
    "vllm_inter_token_latency_total_seconds",
    "Inter token latency in seconds."
)

histogram_time_to_first_token = Histogram(
    "vllm_time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets = [0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0]
)
histogram_inter_token_latency = Histogram(
    "vllm_inter_token_latency_seconds",
    "Histogram of inter token latency in seconds.",
    buckets = [0.0005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 1.0, 5.0]
)

# end-metrics-definitions

################################################################################################################################
# Number of Tokens Processed
class CounterPromptTokens(CounterMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = sum(stats.iteration_stats.num_prompt_tokens)
    
class CounterGenerationTokens(CounterMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = sum(stats.iteration_stats.num_generation_tokens)

################################################################################################################################
# Scheduler State
class GaugeSchedulerRunning(GaugeMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = stats.system_stats.num_running
    def to_str(self) -> str:
        return f"Running: {self.metric} reqs"

class GaugeSchedulerSwapped(GaugeMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = stats.system_stats.num_swapped
    def to_str(self) -> str:
        return f"Swapped: {self.metric} reqs"

class GaugeSchedulerWaiting(GaugeMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = stats.system_stats.num_waiting
    def to_str(self) -> str:
        return f"Waiting: {self.metric} reqs"

################################################################################################################################
# Cache Usage
def compute_cache_usage(num_total_blocks: int, num_free_blocks: int):
    if num_total_blocks <= 0:
        return 0.0

    return 1.0 - num_free_blocks / num_total_blocks

class GaugeGPUCacheUsage(GaugeMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = compute_cache_usage(
            num_total_blocks=stats.system_stats.num_total_gpu_blocks,
            num_free_blocks=stats.system_stats.num_free_gpu_blocks
        )
    def to_str(self) -> str:
        return f"GPU KV cache usage: {self.metric * 100:.1f}%"
    
class GaugeCPUCacheUsage(GaugeMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = compute_cache_usage(
            num_total_blocks=stats.system_stats.num_total_cpu_blocks,
            num_free_blocks=stats.system_stats.num_free_cpu_blocks
        )
    def to_str(self) -> str:
        return f"CPU KV cache usage: {self.metric * 100:.1f}%"

################################################################################################################################
# Request Level Timings
class CounterTimeToFirstToken(CounterMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = sum(stats.iteration_stats.time_to_first_token)

class CounterInterTokenLatency(CounterMetric):
    def compute(self, stats: Stats) -> None:
        self.metric = sum(stats.iteration_stats.inter_token_latency)
        
class HistogramTimeToFirstToken(HistogramMetric):
    def compute(self, stats: Stats) -> None:
        self.metrics = stats.iteration_stats.time_to_first_token
    
class HistogramInterTokenLatency(HistogramMetric):
    def compute(self, stats: Stats) -> None:
        self.metrics = stats.iteration_stats.inter_token_latency

################################################################################################################################
# Metric Registry
METRIC_REGISTRY = [
    (counter_prompt_tokens, CounterPromptTokens),
    (counter_generation_tokens, CounterGenerationTokens),
    (gauge_scheduler_running, GaugeSchedulerRunning),
    (gauge_scheduler_swapped, GaugeSchedulerSwapped),
    (gauge_scheduler_waiting, GaugeSchedulerWaiting),
    (gauge_gpu_cache_usage, GaugeGPUCacheUsage),
    (gauge_cpu_cache_usage, GaugeCPUCacheUsage),
    (counter_time_to_first_token, CounterTimeToFirstToken),
    (counter_inter_token_latency, CounterInterTokenLatency),
    (histogram_time_to_first_token, HistogramTimeToFirstToken),
    (histogram_inter_token_latency, HistogramInterTokenLatency),
]
