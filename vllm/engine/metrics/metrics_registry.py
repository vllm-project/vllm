from aioprometheus import Counter, Gauge, Histogram
import numpy as np
from typing import List, Tuple, Optional, Tuple, Union

from vllm.engine.metrics.metrics_utils import (
    PrometheusMetric, 
    GaugeMetric,
    HistogramMetric,
    Stats,
)

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
gauge_avg_prompt_throughput = Counter("vllm:avg_prompt_throughput_toks_per_s",
                                    "Average prefill throughput in tokens/s.")
gauge_avg_generation_throughput = Counter(
    "vllm:avg_generation_throughput_toks_per_s",
    "Average generation throughput in tokens/s.")

gauge_scheduler_running = Gauge(
    "vllm:num_requests_running",
    "Number of requests that is currently running for inference.")
gauge_scheduler_swapped = Gauge("vllm:num_requests_swapped",
                                "Number requests swapped to CPU.")
gauge_scheduler_waiting = Gauge("vllm:num_requests_waiting",
                                "Number of requests waiting to be processed.")

gauge_gpu_cache_usage = Gauge(
    "vllm:gpu_cache_usage_perc",
    "GPU KV-cache usage. 1 means 100 percent usage.")
gauge_cpu_cache_usage = Gauge(
    "vllm:cpu_cache_usage_perc",
    "CPU KV-cache usage. 1 means 100 percent usage.")

gauge_avg_time_to_first_token = Gauge(
    "vllm:avg_time_to_first_token",
    "Avg time to first token in ms."
)
gauge_avg_inter_token_latency = Gauge(
    "vllm:avg_inter_token_latency",
    "Avg inter token latency in ms."
)

histogram_time_to_first_token = Histogram(
    "vllm:time_to_first_token",
    "Histogram of time to first token in ms."
)
histogram_inter_token_latency = Histogram(
    "vllm:inter_token_latency",
    "Histogram of inter token latency in ms."
)

# end-metrics-definitions

################################################################################################################################
# Average Throughput
def compute_avg_tput(now: float, num_tokens_list: List[Tuple[float, int]]) -> float:
    if len(num_tokens_list) <= 1:
        return None
    
    total_num_tokens = sum(n for _, n in num_tokens_list[:-1])
    window = now - num_tokens_list[0][0]
    return total_num_tokens / window

class AvgPromptThroughputMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = compute_avg_tput(now, stats.iteration_stats.num_prompt_tokens)
    def to_str(self) -> str:
        if self.metric is None:
            return f"Avg prompt throughput: (no prefills in logging window)"
        return f"Avg prompt throughput: {self.metric:.1f} tokens/sec"
    
class AvgGenerationThroughputMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = compute_avg_tput(now, stats.iteration_stats.num_generation_tokens)
    def to_str(self) -> str:
        if self.metric is None:
            return "Avg generation throughput: (no generations in logging window)"
        return f"Avg generation throughput: {self.metric:.1f} tokens/sec"

################################################################################################################################
# Scheduler State
class SchedulerRunningMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = stats.system_stats.num_running
    def to_str(self) -> str:
        return f"Running: {self.metric} reqs"

class SchedulerSwappedMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = stats.system_stats.num_swapped
    def to_str(self) -> str:
        return f"Swapped: {self.metric} reqs"

class SchedulerWaitingMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = stats.system_stats.num_waiting
    def to_str(self) -> str:
        return f"Waiting: {self.metric} reqs"

################################################################################################################################
# Cache Usage
def compute_cache_usage(num_total_blocks: int, num_free_blocks: int):
    if num_total_blocks <= 0:
        return 0.0

    return 1.0 - num_free_blocks / num_total_blocks

class GPUCacheUsageMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = compute_cache_usage(
            num_total_blocks=stats.system_stats.num_total_gpu_blocks,
            num_free_blocks=stats.system_stats.num_free_gpu_blocks
        )
    def to_str(self) -> str:
        return f"GPU KV cache usage: {self.metric * 100:.1f}%"
    
class CPUCacheUsageMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = compute_cache_usage(
            num_total_blocks=stats.system_stats.num_total_cpu_blocks,
            num_free_blocks=stats.system_stats.num_free_cpu_blocks
        )
    def to_str(self) -> str:
        return f"CPU KV cache usage: {self.metric * 100:.1f}%"

################################################################################################################################
# Request Level Timings
def compute_avg_latency(request_latencies: List[Tuple[float, float]]) -> Optional[float]:
    if len(request_latencies) <= 1:
        return None
    # avg latency in ms
    return np.mean(latency for _, latency in request_latencies[:-1])

def extract_latencies(request_latencies: List[Tuple[float, float]]) -> List[float]:
    if len(request_latencies) <= 1:
        return []
    return [latency for _, latency in request_latencies[:-1]]

class AvgTimeToFirstTokenMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = compute_avg_latency(stats.iteration_stats.time_to_first_token)

    def to_str(self) -> str:
        if self.metric is None:
            return f"Avg TTFT: (no prefills in logging window)"
        return f"Avg TTFT: {self.metric:.1f} ms"

class TimetoFirstTokenMetric(HistogramMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metrics = extract_latencies(stats.iteration_stats.time_to_first_token)    

class AvgInterTokenLatencyMetric(GaugeMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metric = compute_avg_latency(stats.iteration_stats.inter_token_latency)

    def to_str(self) -> str:
        if self.metric is None:
            return f"Avg Inter Token Latency: (no decodes in logging window)"
        return f"Avg Inter Token Latency: {self.metric:.1f} ms"
    
class InterTokenLatencyMetric(HistogramMetric):
    def update_metric(self, now: float, stats: Stats) -> None:
        self.metrics = extract_latencies(stats.iteration_stats.inter_token_latency)

################################################################################################################################
# Metric Registry
METRIC_REGISTRY: List[Tuple[Union[Gauge, Histogram], PrometheusMetric]] = {
    "avg_prompt_throughput": (gauge_avg_prompt_throughput, AvgPromptThroughputMetric),
    "avg_generation_throughput": (gauge_avg_generation_throughput, AvgGenerationThroughputMetric),
    "scheduler_running": (gauge_scheduler_running, SchedulerRunningMetric),
    "scheduler_swapped": (gauge_scheduler_swapped, SchedulerSwappedMetric),
    "scheduler_waiting": (gauge_scheduler_waiting, SchedulerWaitingMetric),
    "gpu_cache_usage": (gauge_gpu_cache_usage, GPUCacheUsageMetric),
    "cpu_cache_usage": (gauge_cpu_cache_usage, CPUCacheUsageMetric),
    "avg_time_to_first_token": (gauge_avg_time_to_first_token, AvgTimeToFirstTokenMetric),
    "avg_inter_token_latency": (gauge_avg_inter_token_latency, AvgInterTokenLatencyMetric),
    "avg_time_to_first_token": (histogram_time_to_first_token, TimetoFirstTokenMetric),
    "avg_inter_token_latency": (histogram_inter_token_latency, InterTokenLatencyMetric),
}