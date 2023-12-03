from aioprometheus import Gauge

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
gauge_avg_prompt_throughput = Gauge("vllm:avg_prompt_throughput_toks_per_s",
                                    "Average prefill throughput in tokens/s.")
gauge_avg_generation_throughput = Gauge(
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
# end-metrics-definitions

labels = {}


def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)


def record_metrics(
    avg_prompt_throughput: float,
    avg_generation_throughput: float,
    scheduler_running: int,
    scheduler_swapped: int,
    scheduler_waiting: int,
    gpu_cache_usage: float,
    cpu_cache_usage: float,
):
    gauge_avg_prompt_throughput.set(labels, avg_prompt_throughput)
    gauge_avg_generation_throughput.set(labels, avg_generation_throughput)
    gauge_scheduler_running.set(labels, scheduler_running)
    gauge_scheduler_swapped.set(labels, scheduler_swapped)
    gauge_scheduler_waiting.set(labels, scheduler_waiting)
    gauge_gpu_cache_usage.set(labels, gpu_cache_usage)
    gauge_cpu_cache_usage.set(labels, cpu_cache_usage)
