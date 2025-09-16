"""Metric collection utilities for the custom vLLM metrics proxy."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest)

from vllm.v1.metrics import reader as metrics_reader
from vllm.v1.metrics.reader import Counter as CounterMetric
from vllm.v1.metrics.reader import Gauge as GaugeMetric
from vllm.v1.metrics.reader import Histogram as HistogramMetric
from vllm.v1.metrics.reader import Metric as BaseMetric
from vllm.v1.metrics.reader import Vector as VectorMetric

logger = logging.getLogger(__name__)


@dataclass
class RequestOutcome:
    """Information gathered about a single proxied request."""

    queue_time: float
    inference_time: float
    ttft: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    finish_reasons: List[str]
    max_tokens: Optional[int]
    n: int
    success: bool
    status_code: int
    stream: bool
    error: Optional[str] = None


@dataclass
class RequestSummary:
    """A human readable view of :class:`RequestOutcome`."""

    queue_time: float
    inference_time: float
    e2e_time: float
    ttft: Optional[float]
    prefill_time: float
    decode_time: float
    inter_token_latency: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    finish_reasons: List[str] = field(default_factory=list)
    success: bool = True
    status_code: int = 200
    n: int = 1
    max_tokens: Optional[int] = None


def build_buckets(mantissa_lst: Iterable[int], max_value: int) -> List[int]:
    """Build a list of monotonically increasing histogram buckets."""

    exponent = 0
    buckets: List[int] = []
    while True:
        for mantissa in mantissa_lst:
            value = mantissa * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """Convenience wrapper that produces 1/2/5 style histogram buckets."""

    return build_buckets([1, 2, 5], max_value)


class ProxyMetricsRecorder:
    """Collects vLLM style Prometheus metrics for proxied requests."""

    def __init__(self,
                 model_name: str,
                 engine_label: str = "proxy",
                 *,
                 max_model_len: int = 4096,
                 registry: Optional[CollectorRegistry] = None) -> None:
        self.model_name = model_name
        self.engine_label = engine_label
        self.registry = registry or CollectorRegistry()
        self.max_model_len = max_model_len

        labelnames = ["model_name", "engine"]
        label_values = (self.model_name, self.engine_label)

        # Scheduler gauges.
        self._running_metric = Gauge(
            "vllm:num_requests_running",
            "Number of requests currently forwarded to the upstream server.",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._waiting_metric = Gauge(
            "vllm:num_requests_waiting",
            "Number of requests waiting for a free upstream slot.",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)

        # Cache related gauges/counters (recorded as zero – proxy cannot infer).
        self._kv_cache_usage = Gauge(
            "vllm:kv_cache_usage_perc",
            "Proxy placeholder for KV cache usage (always 0).",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._kv_cache_usage.set(0.0)

        self._prefix_cache_queries = Counter(
            "vllm:prefix_cache_queries",
            "Proxy placeholder for prefix cache queries (always 0).",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._prefix_cache_hits = Counter(
            "vllm:prefix_cache_hits",
            "Proxy placeholder for prefix cache hits (always 0).",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._num_preemptions = Counter(
            "vllm:num_preemptions",
            "Proxy placeholder for engine preemptions (always 0).",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)

        # Token accounting.
        self._prompt_tokens = Counter(
            "vllm:prompt_tokens",
            "Number of prompt tokens processed by the proxy.",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._generation_tokens = Counter(
            "vllm:generation_tokens",
            "Number of generated tokens processed by the proxy.",
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)

        self._request_success_metric = Counter(
            "vllm:request_success",
            "Count of successfully processed requests forwarded by the proxy.",
            labelnames=labelnames + ["finished_reason"],
            registry=self.registry,
        )

        # Token histograms.
        self._prompt_tokens_hist = Histogram(
            "vllm:request_prompt_tokens",
            "Histogram of prompt tokens per request.",
            buckets=build_1_2_5_buckets(self.max_model_len),
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._generation_tokens_hist = Histogram(
            "vllm:request_generation_tokens",
            "Histogram of generated tokens per request.",
            buckets=build_1_2_5_buckets(self.max_model_len),
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._max_generation_tokens_hist = Histogram(
            "vllm:request_max_num_generation_tokens",
            "Histogram of requested max generation tokens.",
            buckets=build_1_2_5_buckets(self.max_model_len),
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._max_tokens_param_hist = Histogram(
            "vllm:request_params_max_tokens",
            "Histogram of the max_tokens parameter provided to the proxy.",
            buckets=build_1_2_5_buckets(self.max_model_len),
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._n_param_hist = Histogram(
            "vllm:request_params_n",
            "Histogram of the n parameter provided to the proxy.",
            buckets=[1, 2, 5, 10, 20],
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)

        # Timing histograms.
        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
            40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
        ]
        self._ttft_hist = Histogram(
            "vllm:time_to_first_token_seconds",
            "Histogram of observed time to first token.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._time_per_token_hist = Histogram(
            "vllm:time_per_output_token_seconds",
            "Histogram of average time per generated token (proxy estimate).",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._inter_token_latency_hist = Histogram(
            "vllm:inter_token_latency_seconds",
            "Histogram of inter-token latency (proxy estimate).",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._e2e_latency_hist = Histogram(
            "vllm:e2e_request_latency_seconds",
            "Histogram of end-to-end latency observed by the proxy.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._queue_time_hist = Histogram(
            "vllm:request_queue_time_seconds",
            "Histogram of queue wait time before contacting the upstream server.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._inference_time_hist = Histogram(
            "vllm:request_inference_time_seconds",
            "Histogram of total upstream processing time.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._prefill_time_hist = Histogram(
            "vllm:request_prefill_time_seconds",
            "Histogram of estimated prefill time (proxied).",
            buckets=request_latency_buckets,
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)
        self._decode_time_hist = Histogram(
            "vllm:request_decode_time_seconds",
            "Histogram of estimated decode time (proxied).",
            buckets=request_latency_buckets,
            labelnames=labelnames,
            registry=self.registry,
        ).labels(*label_values)

        self._running_count = 0
        self._waiting_count = 0

    # ------------------------------------------------------------------
    # Gauge helpers
    # ------------------------------------------------------------------
    def increment_waiting(self) -> None:
        self._waiting_count += 1
        self._waiting_metric.set(self._waiting_count)

    def decrement_waiting(self) -> None:
        self._waiting_count = max(self._waiting_count - 1, 0)
        self._waiting_metric.set(self._waiting_count)

    def increment_running(self) -> None:
        self._running_count += 1
        self._running_metric.set(self._running_count)

    def decrement_running(self) -> None:
        self._running_count = max(self._running_count - 1, 0)
        self._running_metric.set(self._running_count)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def observe_queue_time(self, value: float) -> None:
        self._queue_time_hist.observe(max(value, 0.0))

    def finalize_request(self, outcome: RequestOutcome) -> RequestSummary:
        """Update Prometheus metrics using the provided outcome."""

        e2e_time = outcome.queue_time + outcome.inference_time
        prefill_time = 0.0
        if outcome.ttft is not None:
            prefill_time = max(min(outcome.ttft, outcome.inference_time), 0.0)
            self._ttft_hist.observe(outcome.ttft)
        decode_time = max(outcome.inference_time - prefill_time, 0.0)

        self._e2e_latency_hist.observe(e2e_time)
        self._inference_time_hist.observe(max(outcome.inference_time, 0.0))
        self._prefill_time_hist.observe(prefill_time)
        self._decode_time_hist.observe(decode_time)

        inter_token_latency: Optional[float] = None
        if outcome.completion_tokens and outcome.completion_tokens > 0 and decode_time > 0:
            inter_token_latency = decode_time / outcome.completion_tokens
            self._inter_token_latency_hist.observe(inter_token_latency)
            self._time_per_token_hist.observe(inter_token_latency)

        if outcome.prompt_tokens is not None:
            self._prompt_tokens.inc(max(outcome.prompt_tokens, 0))
            self._prompt_tokens_hist.observe(max(outcome.prompt_tokens, 0))
        if outcome.completion_tokens is not None:
            self._generation_tokens.inc(max(outcome.completion_tokens, 0))
            self._generation_tokens_hist.observe(max(outcome.completion_tokens, 0))

        if outcome.max_tokens is not None:
            self._max_generation_tokens_hist.observe(max(outcome.max_tokens, 0))
            self._max_tokens_param_hist.observe(max(outcome.max_tokens, 0))

        self._n_param_hist.observe(max(outcome.n, 0))

        if outcome.success:
            finish_reasons = outcome.finish_reasons or ["stop"]
            for reason in finish_reasons:
                normalized = (reason or "unknown").lower()
                self._request_success_metric.labels(self.model_name,
                                                    self.engine_label,
                                                    normalized).inc()
        else:
            finish_reasons = outcome.finish_reasons

        summary = RequestSummary(queue_time=outcome.queue_time,
                                 inference_time=outcome.inference_time,
                                 e2e_time=e2e_time,
                                 ttft=outcome.ttft,
                                 prefill_time=prefill_time,
                                 decode_time=decode_time,
                                 inter_token_latency=inter_token_latency,
                                 prompt_tokens=outcome.prompt_tokens,
                                 completion_tokens=outcome.completion_tokens,
                                 total_tokens=outcome.total_tokens,
                                 finish_reasons=finish_reasons,
                                 success=outcome.success,
                                 status_code=outcome.status_code,
                                 n=outcome.n,
                                 max_tokens=outcome.max_tokens)

        return summary

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return a JSON serialisable snapshot of all vLLM metrics."""

        grouped: Dict[str, List[Dict[str, Any]]] = {
            "gauges": [],
            "counters": [],
            "histograms": [],
            "vectors": [],
        }
        for metric in collect_metrics_from_registry(self.registry):
            payload = asdict(metric)
            if isinstance(metric, GaugeMetric):
                grouped["gauges"].append(payload)
            elif isinstance(metric, CounterMetric):
                grouped["counters"].append(payload)
            elif isinstance(metric, HistogramMetric):
                grouped["histograms"].append(payload)
            elif isinstance(metric, VectorMetric):
                grouped["vectors"].append(payload)
        return grouped

    def render_prometheus(self) -> str:
        """Render the stored metrics using Prometheus exposition format."""

        return generate_latest(self.registry).decode("utf-8")


def collect_metrics_from_registry(
        registry: CollectorRegistry) -> List[BaseMetric]:
    """Collect metrics from an arbitrary Prometheus registry."""

    collected: List[BaseMetric] = []
    for metric in registry.collect():
        if not metric.name.startswith("vllm:"):
            continue
        if metric.type == "gauge":
            samples = metrics_reader._get_samples(metric)
            for sample in samples:
                collected.append(
                    GaugeMetric(name=metric.name,
                                labels=dict(sample.labels),
                                value=sample.value))
        elif metric.type == "counter":
            samples = metrics_reader._get_samples(metric, "_total")
            if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                for labels, values in metrics_reader._digest_num_accepted_by_pos_samples(
                        samples):
                    collected.append(
                        VectorMetric(name=metric.name,
                                     labels=labels,
                                     values=values))
            else:
                for sample in samples:
                    collected.append(
                        CounterMetric(name=metric.name,
                                      labels=dict(sample.labels),
                                      value=int(sample.value)))
        elif metric.type == "histogram":
            bucket_samples = metrics_reader._get_samples(metric, "_bucket")
            count_samples = metrics_reader._get_samples(metric, "_count")
            sum_samples = metrics_reader._get_samples(metric, "_sum")
            for labels, buckets, count_value, sum_value in metrics_reader._digest_histogram(
                    bucket_samples, count_samples, sum_samples):
                collected.append(
                    HistogramMetric(name=metric.name,
                                    labels=labels,
                                    buckets=buckets,
                                    count=count_value,
                                    sum=sum_value))
        else:
            logger.debug("Unsupported metric type encountered: %s", metric.type)
    return collected
