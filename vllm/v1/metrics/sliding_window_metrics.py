# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from prometheus_client import Gauge

from vllm.v1.metrics.prometheus import get_prometheus_registry

# Sliding window metrics
sliding_window_latency = Gauge(
    "vllm:sliding_window_latency_ms",
    "Average latency over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_throughput = Gauge(
    "vllm:sliding_window_throughput_tokens_per_sec",
    "Average throughput over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_ttft = Gauge(
    "vllm:sliding_window_time_to_first_token_ms",
    "Average time to first token over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_prompt_tokens = Gauge(
    "vllm:sliding_window_prompt_tokens",
    "Average number of prompt tokens over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_generation_tokens = Gauge(
    "vllm:sliding_window_generation_tokens",
    "Average number of generation tokens over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_queued_time = Gauge(
    "vllm:sliding_window_queued_time_ms",
    "Average queued time over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_prefill_time = Gauge(
    "vllm:sliding_window_prefill_time_ms",
    "Average prefill time over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

sliding_window_decode_time = Gauge(
    "vllm:sliding_window_decode_time_ms",
    "Average decode time over last N requests",
    ["window_size"],
    registry=get_prometheus_registry()
)

def update_sliding_window_metrics(stats):
    """Update all sliding window metrics.
    
    Args:
        stats: SlidingWindowStats instance containing the metrics.
    """
    window_size = str(stats.window_size)
    
    sliding_window_latency.labels(window_size=window_size).set(
        stats.get_metric("latency_ms"))
    sliding_window_throughput.labels(window_size=window_size).set(
        stats.get_metric("throughput_tokens_per_sec"))
    sliding_window_ttft.labels(window_size=window_size).set(
        stats.get_metric("time_to_first_token_ms"))
    sliding_window_prompt_tokens.labels(window_size=window_size).set(
        stats.get_metric("prompt_tokens"))
    sliding_window_generation_tokens.labels(window_size=window_size).set(
        stats.get_metric("generation_tokens"))
    sliding_window_queued_time.labels(window_size=window_size).set(
        stats.get_metric("queued_time_ms"))
    sliding_window_prefill_time.labels(window_size=window_size).set(
        stats.get_metric("prefill_time_ms"))
    sliding_window_decode_time.labels(window_size=window_size).set(
        stats.get_metric("decode_time_ms"))