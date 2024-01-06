from aioprometheus import Counter, Gauge, Histogram
from vllm.engine.metrics.metrics import (CounterMetric, GaugeMetric,
                                         HistogramMetric)

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
    buckets=[
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5
    ])
histogram_e2e_request_latency = Histogram(
    "vllm_e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
# end-metrics-definitions

METRICS_REGISTRY = [
    CounterMetric(
        counter=counter_prompt_tokens,
        attr="num_prompt_tokens",
        template="Prompt throughput: {:0.2f} tok/sec",
    ),
    CounterMetric(
        counter=counter_generation_tokens,
        attr="num_generation_tokens",
        template="Generation throughput: {:0.2f} tok/s",
    ),
    GaugeMetric(gauge=gauge_scheduler_running,
                attr="num_running",
                template="Running: {} reqs"),
    GaugeMetric(
        gauge=gauge_scheduler_swapped,
        attr="num_swapped",
        template="Swapped: {} reqs",
    ),
    GaugeMetric(
        gauge=gauge_scheduler_waiting,
        attr="num_waiting",
        template="Waiting: {} reqs",
    ),
    GaugeMetric(
        gauge=gauge_gpu_cache_usage,
        attr="gpu_cache_usage",
        template="GPU KV cache usage: {:0.1f}%",
    ),
    GaugeMetric(
        gauge=gauge_cpu_cache_usage,
        attr="cpu_cache_usage",
        template="CPU KV cache usage: {:0.1f}%",
    ),
    HistogramMetric(
        histogram=histogram_time_to_first_token,
        attr="time_to_first_tokens",
        template="Avg TTFT: {:0.2f}s",
        log_local=False,
    ),
    HistogramMetric(
        histogram=histogram_time_per_output_tokens,
        attr="time_per_output_tokens",
        template="Avg TPOT: {:0.2f}s",
        log_local=False,
    ),
    HistogramMetric(
        histogram=histogram_e2e_request_latency,
        attr="time_e2e_requests",
        template="Avg E2E Latency: {:0.2f}s",
        log_local=False,
    ),
]
