import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Counter as CollectionsCounter
from typing import Dict, List, Optional, Protocol, Union

import numpy as np
import prometheus_client

from vllm.executor.ray_utils import ray
from vllm.logger import init_logger

if ray is not None:
    from ray.util import metrics as ray_metrics
else:
    ray_metrics = None

if TYPE_CHECKING:
    from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics

logger = init_logger(__name__)

prometheus_client.disable_created_metrics()

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.


# begin-metrics-definitions
class Metrics:
    labelname_finish_reason = "finished_reason"
    _base_library = prometheus_client

    def __init__(self, labelnames: List[str], max_model_len: int):
        # Unregister any existing vLLM collectors
        self._unregister_vllm_metrics()

        # Config Information
        self.info_cache_config = prometheus_client.Info(
            name='vllm:cache_config',
            documentation='information of cache_config')

        # System stats
        #   Scheduler State
        self.gauge_scheduler_running = self._base_library.Gauge(
            name="vllm:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames)
        self.gauge_scheduler_waiting = self._base_library.Gauge(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames)
        self.gauge_scheduler_swapped = self._base_library.Gauge(
            name="vllm:num_requests_swapped",
            documentation="Number of requests swapped to CPU.",
            labelnames=labelnames)
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = self._base_library.Gauge(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        self.gauge_cpu_cache_usage = self._base_library.Gauge(
            name="vllm:cpu_cache_usage_perc",
            documentation="CPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)

        # Iteration stats
        self.counter_num_preemption = self._base_library.Counter(
            name="vllm:num_preemptions_total",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames)
        self.counter_prompt_tokens = self._base_library.Counter(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        self.counter_generation_tokens = self._base_library.Counter(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        self.histogram_time_to_first_token = self._base_library.Histogram(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0
            ])
        self.histogram_time_per_output_token = self._base_library.Histogram(
            name="vllm:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5
            ])

        # Request stats
        #   Latency
        self.histogram_e2e_time_request = self._base_library.Histogram(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        #   Metadata
        self.histogram_num_prompt_tokens_request = self._base_library.Histogram(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_generation_tokens_request = \
            self._base_library.Histogram(
                name="vllm:request_generation_tokens",
                documentation="Number of generation tokens processed.",
                labelnames=labelnames,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        self.histogram_best_of_request = self._base_library.Histogram(
            name="vllm:request_params_best_of",
            documentation="Histogram of the best_of request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_n_request = self._base_library.Histogram(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.counter_request_success = self._base_library.Counter(
            name="vllm:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [Metrics.labelname_finish_reason])

        # Deprecated in favor of vllm:prompt_tokens_total
        self.gauge_avg_prompt_throughput = self._base_library.Gauge(
            name="vllm:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
        )
        # Deprecated in favor of vllm:generation_tokens_total
        self.gauge_avg_generation_throughput = self._base_library.Gauge(
            name="vllm:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
        )

    def _unregister_vllm_metrics(self) -> None:
        for collector in list(self._base_library.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                self._base_library.REGISTRY.unregister(collector)


class RayMetrics(Metrics):
    """
    RayMetrics is used by RayPrometheusStatLogger to log to Ray metrics.
    Provides the same metrics as Metrics but uses Ray's util.metrics library.
    """
    _base_library = ray_metrics

    def __init__(self, labelnames: List[str], max_model_len: int):
        if ray_metrics is None:
            raise ImportError("RayMetrics requires Ray to be installed.")
        super().__init__(labelnames, max_model_len)

    def _unregister_vllm_metrics(self) -> None:
        # No-op on purpose
        pass


# end-metrics-definitions


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by 
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float

    # System stats (should have _sys suffix)
    #   Scheduler State
    num_running_sys: int
    num_waiting_sys: int
    num_swapped_sys: int
    #   KV Cache Usage in %
    gpu_cache_usage_sys: float
    cpu_cache_usage_sys: float

    # Iteration stats (should have _iter suffix)
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]
    num_preemption_iter: int

    # Request stats (should have _requests suffix)
    #   Latency
    time_e2e_requests: List[float]
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    best_of_requests: List[int]
    n_requests: List[int]
    finished_reason_requests: List[str]

    spec_decode_metrics: Optional["SpecDecodeWorkerMetrics"] = None


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> Dict[str, str]:
        ...


def local_interval_elapsed(now: float, last_log: float,
                           local_interval: float) -> bool:
    elapsed_time = now - last_log
    return elapsed_time > local_interval


def get_throughput(tracked_stats: List[int], now: float,
                   last_log: float) -> float:
    return float(np.sum(tracked_stats) / (now - last_log))


class StatLoggerBase(ABC):
    """Base class for StatLogger."""

    def __init__(self, local_interval: float) -> None:
        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.last_local_log = time.time()
        self.local_interval = local_interval

    @abstractmethod
    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError

    @abstractmethod
    def log(self, stats: Stats) -> None:
        raise NotImplementedError


class LoggingStatLogger(StatLoggerBase):
    """LoggingStatLogger is used in LLMEngine to log to Stdout."""

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to Stdout every self.local_interval seconds."""

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens.append(stats.num_generation_tokens_iter)

        # Log locally every local_interval seconds.
        if local_interval_elapsed(stats.now, self.last_local_log,
                                  self.local_interval):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = get_throughput(self.num_prompt_tokens,
                                               now=stats.now,
                                               last_log=self.last_local_log)
            generation_throughput = get_throughput(
                self.num_generation_tokens,
                now=stats.now,
                last_log=self.last_local_log)

            # Log to stdout.
            logger.info(
                "Avg prompt throughput: %.1f tokens/s, "
                "Avg generation throughput: %.1f tokens/s, "
                "Running: %d reqs, Swapped: %d reqs, "
                "Pending: %d reqs, GPU KV cache usage: %.1f%%, "
                "CPU KV cache usage: %.1f%%.",
                prompt_throughput,
                generation_throughput,
                stats.num_running_sys,
                stats.num_swapped_sys,
                stats.num_waiting_sys,
                stats.gpu_cache_usage_sys * 100,
                stats.cpu_cache_usage_sys * 100,
            )

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now

            if stats.spec_decode_metrics is not None:
                logger.info(
                    self._format_spec_decode_metrics_str(
                        stats.spec_decode_metrics))

    def _format_spec_decode_metrics_str(
            self, metrics: "SpecDecodeWorkerMetrics") -> str:

        return ("Speculative metrics: "
                f"Draft acceptance rate: {metrics.draft_acceptance_rate:.3f}, "
                f"System efficiency: {metrics.system_efficiency:.3f}, "
                f"Number of speculative tokens: {metrics.num_spec_tokens}, "
                f"Number of accepted tokens: {metrics.accepted_tokens}, "
                f"Number of draft tokens tokens: {metrics.draft_tokens}, "
                f"Number of emitted tokens tokens: {metrics.emitted_tokens}.")


class PrometheusStatLogger(StatLoggerBase):
    """PrometheusStatLogger is used LLMEngine to log to Promethus."""
    _metrics_cls = Metrics

    def __init__(self, local_interval: float, labels: Dict[str, str],
                 max_model_len: int) -> None:
        super().__init__(local_interval)
        # Prometheus metrics
        self.labels = labels
        self.metrics = self._metrics_cls(labelnames=list(labels.keys()),
                                         max_model_len=max_model_len)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        if type == "cache_config":
            self.metrics.info_cache_config.info(obj.metrics_info())

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(self, counter, data: CollectionsCounter,
                            label_key: str) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram, data: Union[List[int],
                                                    List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def _log_prometheus(self, stats: Stats) -> None:
        # System state data
        self._log_gauge(self.metrics.gauge_scheduler_running,
                        stats.num_running_sys)
        self._log_gauge(self.metrics.gauge_scheduler_swapped,
                        stats.num_swapped_sys)
        self._log_gauge(self.metrics.gauge_scheduler_waiting,
                        stats.num_waiting_sys)
        self._log_gauge(self.metrics.gauge_gpu_cache_usage,
                        stats.gpu_cache_usage_sys)
        self._log_gauge(self.metrics.gauge_cpu_cache_usage,
                        stats.cpu_cache_usage_sys)

        # Iteration level data
        self._log_counter(self.metrics.counter_num_preemption,
                          stats.num_preemption_iter)
        self._log_counter(self.metrics.counter_prompt_tokens,
                          stats.num_prompt_tokens_iter)
        self._log_counter(self.metrics.counter_generation_tokens,
                          stats.num_generation_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_to_first_token,
                            stats.time_to_first_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_per_output_token,
                            stats.time_per_output_tokens_iter)

        # Request level data
        # Latency
        self._log_histogram(self.metrics.histogram_e2e_time_request,
                            stats.time_e2e_requests)
        # Metadata
        finished_reason_counter = CollectionsCounter(
            stats.finished_reason_requests)
        self._log_counter_labels(self.metrics.counter_request_success,
                                 finished_reason_counter,
                                 Metrics.labelname_finish_reason)
        self._log_histogram(self.metrics.histogram_num_prompt_tokens_request,
                            stats.num_prompt_tokens_requests)
        self._log_histogram(
            self.metrics.histogram_num_generation_tokens_request,
            stats.num_generation_tokens_requests)
        self._log_histogram(self.metrics.histogram_n_request, stats.n_requests)
        self._log_histogram(self.metrics.histogram_best_of_request,
                            stats.best_of_requests)

    def _log_prometheus_interval(self, prompt_throughput: float,
                                 generation_throughput: float) -> None:
        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on
        # the vLLM side. Moving forward, we should use counters like
        # counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the
        # grafana/prometheus side. See
        # https://github.com/vllm-project/vllm/pull/2316#discussion_r1464204666
        self.metrics.gauge_avg_prompt_throughput.labels(
            **self.labels).set(prompt_throughput)
        self.metrics.gauge_avg_generation_throughput.labels(
            **self.labels).set(generation_throughput)

    def log(self, stats: Stats):
        """Logs to prometheus and tracked stats every iteration."""
        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens.append(stats.num_generation_tokens_iter)

        # Log locally every local_interval seconds.
        if local_interval_elapsed(stats.now, self.last_local_log,
                                  self.local_interval):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = get_throughput(self.num_prompt_tokens,
                                               now=stats.now,
                                               last_log=self.last_local_log)
            generation_throughput = get_throughput(
                self.num_generation_tokens,
                now=stats.now,
                last_log=self.last_local_log)

            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput)

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now


class RayPrometheusStatLogger(PrometheusStatLogger):
    """RayPrometheusStatLogger uses Ray metrics instead."""
    _metrics_cls = RayMetrics