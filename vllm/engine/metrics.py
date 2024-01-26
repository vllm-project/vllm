from vllm.logger import init_logger
from aioprometheus import Counter, Gauge, Histogram

import time
import numpy as np
from typing import List
from dataclasses import dataclass

logger = init_logger(__name__)

labels = {}


def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)


# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
gauge_avg_prompt_throughput = Gauge("vllm:avg_prompt_throughput_toks_per_s",
                                    "Average prefill throughput in tokens/s.")
gauge_avg_generation_throughput = Gauge(
    "vllm:avg_generation_throughput_toks_per_s",
    "Average generation throughput in tokens/s.")
counter_prompt_tokens = Counter("vllm:prompt_tokens_total",
                                "Number of prefill tokens processed.")
counter_generation_tokens = Counter("vllm:generation_tokens_total",
                                    "Number of generation tokens processed.")

gauge_scheduler_running = Gauge(
    "vllm:num_requests_running",
    "Number of requests currently running on GPU.")
gauge_scheduler_swapped = Gauge("vllm:num_requests_swapped",
                                "Number of requests swapped to CPU.")
gauge_scheduler_waiting = Gauge("vllm:num_requests_waiting",
                                "Number of requests waiting to be processed.")

gauge_gpu_cache_usage = Gauge(
    "vllm:gpu_cache_usage_perc",
    "GPU KV-cache usage. 1 means 100 percent usage.")
gauge_cpu_cache_usage = Gauge(
    "vllm:cpu_cache_usage_perc",
    "CPU KV-cache usage. 1 means 100 percent usage.")

histogram_time_to_first_token = Histogram(
    "vllm:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets=[
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0
    ])
histogram_time_per_output_tokens = Histogram(
    "vllm:time_per_output_token_seconds",
    "Histogram of time per output token in seconds.",
    buckets=[
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5
    ])
histogram_e2e_request_latency = Histogram(
    "vllm:e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
# end-metrics-definitions


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float

    # System stats.
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float

    # Raw stats from last model iteration.
    num_prompt_tokens: int
    num_generation_tokens: int
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    time_e2e_requests: List[float]


class StatLogger:
    """StatLogger is used LLMEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float) -> None:
        # Metadata for logging locally.
        self.last_local_log = time.monotonic()
        self.local_interval = local_interval

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def _log_prometheus(self, stats: Stats) -> None:
        # Set system stat gauges.
        gauge_scheduler_running.set(labels, stats.num_running)
        gauge_scheduler_swapped.set(labels, stats.num_swapped)
        gauge_scheduler_waiting.set(labels, stats.num_waiting)
        gauge_gpu_cache_usage.set(labels, stats.gpu_cache_usage)
        gauge_cpu_cache_usage.set(labels, stats.cpu_cache_usage)

        # Add to token counters.
        counter_prompt_tokens.add(labels, stats.num_prompt_tokens)
        counter_generation_tokens.add(labels, stats.num_generation_tokens)

        # Observe request level latencies in histograms.
        for ttft in stats.time_to_first_tokens:
            histogram_time_to_first_token.observe(labels, ttft)
        for tpot in stats.time_per_output_tokens:
            histogram_time_per_output_tokens.observe(labels, tpot)
        for e2e in stats.time_e2e_requests:
            histogram_e2e_request_latency.observe(labels, e2e)

    def _log_prometheus_interval(self, prompt_throughput: float,
                                 generation_throughput: float) -> None:
        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on the vLLM side.
        # Moving forward, we should use counters like counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the grafana/prometheus side.
        # See https://github.com/vllm-project/vllm/pull/2316#discussion_r1464204666
        gauge_avg_prompt_throughput.set(labels, prompt_throughput)
        gauge_avg_generation_throughput.set(labels, generation_throughput)

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to prometheus and tracked stats every iteration. 
           Logs to Stdout every self.local_interval seconds."""

        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens)
        self.num_generation_tokens.append(stats.num_generation_tokens)

        # Log locally every local_interval seconds.
        if self._local_interval_elapsed(stats.now):

            # Compute summary metrics for tracked stats (and log them to promethus if applicable).
            prompt_throughput = self._get_throughput(self.num_prompt_tokens,
                                                     now=stats.now)
            generation_throughput = self._get_throughput(
                self.num_generation_tokens, now=stats.now)
            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput)

            # Log to stdout.
            logger.info(
                f"Avg prompt throughput: {prompt_throughput:.1f} tokens/s, "
                f"Avg generation throughput: {generation_throughput:.1f} tokens/s, "
                f"Running: {stats.num_running} reqs, "
                f"Swapped: {stats.num_swapped} reqs, "
                f"Pending: {stats.num_waiting} reqs, "
                f"GPU KV cache usage: {stats.gpu_cache_usage * 100:.1f}%, "
                f"CPU KV cache usage: {stats.cpu_cache_usage * 100:.1f}%")

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now
