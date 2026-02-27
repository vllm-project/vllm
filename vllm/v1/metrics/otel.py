# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenTelemetry Metrics StatLogger for vLLM.

Exports the same metrics as PrometheusStatLogger via OTLP protocol,
enabling integration with any OpenTelemetry-compatible metrics backend
(e.g., Prometheus, Datadog, Grafana Cloud, etc.).
"""

import atexit
import logging
import os
import traceback

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import (
    IterationStats,
    MultiModalCacheStats,
    PromptTokenStats,
    SchedulerStats,
)

logger = init_logger(__name__)

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as OTLPGrpcMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as OTLPHttpMetricExporter,
    )
    from opentelemetry.metrics import (
        Meter,
        set_meter_provider,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    _IS_OTEL_METRICS_AVAILABLE = True
    _otel_metrics_import_error = None
except ImportError:
    _IS_OTEL_METRICS_AVAILABLE = False
    _otel_metrics_import_error = traceback.format_exc()
    Meter = None  # type: ignore[assignment, misc]
    MeterProvider = None  # type: ignore[assignment, misc]


def is_otel_metrics_available() -> bool:
    return _IS_OTEL_METRICS_AVAILABLE


def _get_metric_exporter(endpoint: str):
    """Create an OTLP metric exporter based on the configured protocol."""
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc")
    if protocol == "grpc":
        return OTLPGrpcMetricExporter(endpoint=endpoint, insecure=True)
    elif protocol == "http/protobuf":
        return OTLPHttpMetricExporter(endpoint=endpoint)
    else:
        raise ValueError(
            f"Unsupported OTLP metrics protocol '{protocol}' is configured. "
            "Supported protocols: 'grpc', 'http/protobuf'."
        )


def _init_otel_meter(
    endpoint: str,
    export_interval_millis: int = 10000,
) -> Meter:
    """Initialize the OpenTelemetry MeterProvider and return a Meter."""
    if not _IS_OTEL_METRICS_AVAILABLE:
        raise ValueError(
            "OpenTelemetry metrics packages are not available. "
            "Ensure opentelemetry-sdk and opentelemetry-exporter-otlp "
            f"are installed.\nOriginal error:\n{_otel_metrics_import_error}"
        )

    resource = Resource.create({
        "service.name": "vllm",
        "vllm.process_id": str(os.getpid()),
    })

    exporter = _get_metric_exporter(endpoint)
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=export_interval_millis,
    )

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    set_meter_provider(provider)
    atexit.register(provider.shutdown)

    return provider.get_meter("vllm")


# Import the base class here to avoid circular imports at module level
from vllm.v1.metrics.loggers import AggregateStatLoggerBase  # noqa: E402


class OTelMetricsStatLogger(AggregateStatLoggerBase):
    """StatLogger that exports metrics via OpenTelemetry OTLP protocol.

    Mirrors the metrics exported by PrometheusStatLogger, using OTel SDK
    instruments (Counter, Histogram, UpDownCounter as gauge equivalent).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int] | None = None,
    ):
        if engine_indexes is None:
            engine_indexes = [0]

        self.engine_indexes = engine_indexes
        self.vllm_config = vllm_config

        endpoint = vllm_config.observability_config.otlp_metrics_endpoint
        if endpoint is None:
            raise ValueError(
                "otlp_metrics_endpoint must be set to use OTelMetricsStatLogger"
            )

        self.meter = _init_otel_meter(endpoint)
        model_name = vllm_config.model_config.served_model_name

        # Common attributes for all metrics
        self._base_attributes = {"model_name": model_name}

        self._create_instruments()

    def _engine_attrs(self, engine_idx: int) -> dict[str, str]:
        """Return attributes dict including engine index."""
        return {**self._base_attributes, "engine": str(engine_idx)}

    def _create_instruments(self):
        """Create all OTel metric instruments mirroring Prometheus metrics."""

        # --- Gauges (using UpDownCounter as OTel gauge equivalent) ---
        self.gauge_scheduler_running = self.meter.create_up_down_counter(
            name="vllm.num_requests_running",
            description="Number of requests in model execution batches.",
            unit="requests",
        )
        self.gauge_scheduler_waiting = self.meter.create_up_down_counter(
            name="vllm.num_requests_waiting",
            description="Number of requests waiting to be processed.",
            unit="requests",
        )
        self.gauge_kv_cache_usage = self.meter.create_gauge(
            name="vllm.kv_cache_usage_perc",
            description="KV-cache usage. 1 means 100 percent usage.",
        )

        # --- Counters ---
        self.counter_num_preempted_reqs = self.meter.create_counter(
            name="vllm.num_preemptions",
            description="Cumulative number of preemptions from the engine.",
            unit="requests",
        )
        self.counter_prompt_tokens = self.meter.create_counter(
            name="vllm.prompt_tokens",
            description="Number of prefill tokens processed.",
            unit="tokens",
        )
        self.counter_generation_tokens = self.meter.create_counter(
            name="vllm.generation_tokens",
            description="Number of generation tokens processed.",
            unit="tokens",
        )
        self.counter_request_success = self.meter.create_counter(
            name="vllm.request_success",
            description="Count of successfully processed requests.",
            unit="requests",
        )
        self.counter_prefix_cache_queries = self.meter.create_counter(
            name="vllm.prefix_cache_queries",
            description=(
                "Prefix cache queries, in terms of number of queried tokens."
            ),
            unit="tokens",
        )
        self.counter_prefix_cache_hits = self.meter.create_counter(
            name="vllm.prefix_cache_hits",
            description="Prefix cache hits, in terms of number of cached tokens.",
            unit="tokens",
        )
        self.counter_prompt_tokens_cached = self.meter.create_counter(
            name="vllm.prompt_tokens_cached",
            description="Number of cached prompt tokens (local + external).",
            unit="tokens",
        )
        self.counter_prompt_tokens_recomputed = self.meter.create_counter(
            name="vllm.prompt_tokens_recomputed",
            description="Number of cached tokens recomputed for forward pass.",
            unit="tokens",
        )

        # Per-source prompt token counters
        self.counter_prompt_tokens_by_source = self.meter.create_counter(
            name="vllm.prompt_tokens_by_source",
            description="Number of prompt tokens by source.",
            unit="tokens",
        )

        # External prefix cache (KV connector)
        self.counter_connector_prefix_cache_queries = self.meter.create_counter(
            name="vllm.external_prefix_cache_queries",
            description=(
                "External prefix cache queries from KV connector "
                "cross-instance cache sharing."
            ),
            unit="tokens",
        )
        self.counter_connector_prefix_cache_hits = self.meter.create_counter(
            name="vllm.external_prefix_cache_hits",
            description=(
                "External prefix cache hits from KV connector "
                "cross-instance cache sharing."
            ),
            unit="tokens",
        )

        # Multi-modal cache
        self.counter_mm_cache_queries = self.meter.create_counter(
            name="vllm.mm_cache_queries",
            description=(
                "Multi-modal cache queries, in terms of number of queried items."
            ),
            unit="items",
        )
        self.counter_mm_cache_hits = self.meter.create_counter(
            name="vllm.mm_cache_hits",
            description=(
                "Multi-modal cache hits, in terms of number of cached items."
            ),
            unit="items",
        )

        # --- Histograms ---
        self.histogram_time_to_first_token = self.meter.create_histogram(
            name="vllm.time_to_first_token_seconds",
            description="Histogram of time to first token in seconds.",
            unit="s",
        )
        self.histogram_inter_token_latency = self.meter.create_histogram(
            name="vllm.inter_token_latency_seconds",
            description="Histogram of inter-token latency in seconds.",
            unit="s",
        )
        self.histogram_e2e_time_request = self.meter.create_histogram(
            name="vllm.e2e_request_latency_seconds",
            description="Histogram of end-to-end request latency in seconds.",
            unit="s",
        )
        self.histogram_queue_time_request = self.meter.create_histogram(
            name="vllm.request_queue_time_seconds",
            description=(
                "Histogram of time spent in WAITING phase for request."
            ),
            unit="s",
        )
        self.histogram_inference_time_request = self.meter.create_histogram(
            name="vllm.request_inference_time_seconds",
            description=(
                "Histogram of time spent in RUNNING phase for request."
            ),
            unit="s",
        )
        self.histogram_prefill_time_request = self.meter.create_histogram(
            name="vllm.request_prefill_time_seconds",
            description=(
                "Histogram of time spent in PREFILL phase for request."
            ),
            unit="s",
        )
        self.histogram_decode_time_request = self.meter.create_histogram(
            name="vllm.request_decode_time_seconds",
            description=(
                "Histogram of time spent in DECODE phase for request."
            ),
            unit="s",
        )
        self.histogram_num_prompt_tokens_request = self.meter.create_histogram(
            name="vllm.request_prompt_tokens",
            description="Number of prefill tokens processed per request.",
            unit="tokens",
        )
        self.histogram_num_generation_tokens_request = self.meter.create_histogram(
            name="vllm.request_generation_tokens",
            description="Number of generation tokens processed per request.",
            unit="tokens",
        )
        self.histogram_iteration_tokens = self.meter.create_histogram(
            name="vllm.iteration_tokens_total",
            description="Histogram of number of tokens per engine_step.",
            unit="tokens",
        )
        self.histogram_request_time_per_output_token = self.meter.create_histogram(
            name="vllm.request_time_per_output_token_seconds",
            description=(
                "Histogram of time_per_output_token_seconds per request."
            ),
            unit="s",
        )
        self.histogram_max_num_generation_tokens_request = self.meter.create_histogram(
            name="vllm.request_max_num_generation_tokens",
            description=(
                "Histogram of maximum number of requested generation tokens."
            ),
            unit="tokens",
        )
        self.histogram_n_request = self.meter.create_histogram(
            name="vllm.request_params_n",
            description="Histogram of the n request parameter.",
        )
        self.histogram_max_tokens_request = self.meter.create_histogram(
            name="vllm.request_params_max_tokens",
            description="Histogram of the max_tokens request parameter.",
            unit="tokens",
        )
        self.histogram_prefill_kv_computed_request = self.meter.create_histogram(
            name="vllm.request_prefill_kv_computed_tokens",
            description=(
                "Histogram of new KV tokens computed during prefill "
                "(excluding cached tokens)."
            ),
            unit="tokens",
        )

        # Track last gauge values per engine for delta computation
        self._last_running: dict[int, int] = {
            idx: 0 for idx in self.engine_indexes
        }
        self._last_waiting: dict[int, int] = {
            idx: 0 for idx in self.engine_indexes
        }

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """Record metrics to OpenTelemetry."""
        attrs = self._engine_attrs(engine_idx)

        if scheduler_stats is not None:
            # Gauge updates via UpDownCounter delta
            running_delta = (
                scheduler_stats.num_running_reqs
                - self._last_running.get(engine_idx, 0)
            )
            self.gauge_scheduler_running.add(running_delta, attrs)
            self._last_running[engine_idx] = scheduler_stats.num_running_reqs

            waiting_delta = (
                scheduler_stats.num_waiting_reqs
                - self._last_waiting.get(engine_idx, 0)
            )
            self.gauge_scheduler_waiting.add(waiting_delta, attrs)
            self._last_waiting[engine_idx] = scheduler_stats.num_waiting_reqs

            # KV cache usage (true gauge)
            self.gauge_kv_cache_usage.set(
                scheduler_stats.kv_cache_usage, attrs
            )

            # Prefix cache counters
            self.counter_prefix_cache_queries.add(
                scheduler_stats.prefix_cache_stats.queries, attrs
            )
            self.counter_prefix_cache_hits.add(
                scheduler_stats.prefix_cache_stats.hits, attrs
            )

            # External prefix cache (KV connector)
            if scheduler_stats.connector_prefix_cache_stats is not None:
                self.counter_connector_prefix_cache_queries.add(
                    scheduler_stats.connector_prefix_cache_stats.queries,
                    attrs,
                )
                self.counter_connector_prefix_cache_hits.add(
                    scheduler_stats.connector_prefix_cache_stats.hits, attrs
                )

        if mm_cache_stats is not None:
            self.counter_mm_cache_queries.add(
                mm_cache_stats.queries, attrs
            )
            self.counter_mm_cache_hits.add(mm_cache_stats.hits, attrs)

        if iteration_stats is None:
            return

        # Token counters
        self.counter_num_preempted_reqs.add(
            iteration_stats.num_preempted_reqs, attrs
        )
        self.counter_prompt_tokens.add(
            iteration_stats.num_prompt_tokens, attrs
        )
        self.counter_generation_tokens.add(
            iteration_stats.num_generation_tokens, attrs
        )

        # Per-source prompt token counters
        prompt_token_stats = iteration_stats.prompt_token_stats
        for source in PromptTokenStats.ALL_SOURCES:
            source_attrs = {**attrs, "source": source}
            self.counter_prompt_tokens_by_source.add(
                prompt_token_stats.get_by_source(source), source_attrs
            )
        self.counter_prompt_tokens_cached.add(
            prompt_token_stats.cached_tokens, attrs
        )
        self.counter_prompt_tokens_recomputed.add(
            prompt_token_stats.recomputed_tokens, attrs
        )

        # Iteration-level histograms
        self.histogram_iteration_tokens.record(
            iteration_stats.num_prompt_tokens
            + iteration_stats.num_generation_tokens,
            attrs,
        )

        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request.record(
                max_gen_tokens, attrs
            )
        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request.record(n_param, attrs)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token.record(ttft, attrs)
        for itl in iteration_stats.inter_token_latencies_iter:
            self.histogram_inter_token_latency.record(itl, attrs)

        # Per-finished-request metrics
        for finished_request in iteration_stats.finished_requests:
            reason_attrs = {
                **attrs,
                "finished_reason": str(finished_request.finish_reason),
            }
            self.counter_request_success.add(1, reason_attrs)

            self.histogram_e2e_time_request.record(
                finished_request.e2e_latency, attrs
            )
            self.histogram_queue_time_request.record(
                finished_request.queued_time, attrs
            )
            self.histogram_prefill_time_request.record(
                finished_request.prefill_time, attrs
            )
            self.histogram_inference_time_request.record(
                finished_request.inference_time, attrs
            )
            self.histogram_decode_time_request.record(
                finished_request.decode_time, attrs
            )

            prefill_kv_computed = (
                finished_request.num_prompt_tokens
                - max(finished_request.num_cached_tokens, 0)
            )
            self.histogram_prefill_kv_computed_request.record(
                prefill_kv_computed, attrs
            )
            self.histogram_num_prompt_tokens_request.record(
                finished_request.num_prompt_tokens, attrs
            )
            self.histogram_num_generation_tokens_request.record(
                finished_request.num_generation_tokens, attrs
            )
            self.histogram_request_time_per_output_token.record(
                finished_request.mean_time_per_output_token, attrs
            )
            if finished_request.max_tokens_param:
                self.histogram_max_tokens_request.record(
                    finished_request.max_tokens_param, attrs
                )

    def log_engine_initialized(self):
        logger.info(
            "OTel metrics logger initialized, exporting to %s",
            self.vllm_config.observability_config.otlp_metrics_endpoint,
        )

    def log(self):
        # OTel SDK handles periodic export via PeriodicExportingMetricReader.
        # No manual flush needed on each log interval.
        pass

    def record_sleep_state(self, sleep: int = 0, level: int = 0):
        # Sleep state tracking is Prometheus-specific (gauge with labels).
        # For OTel, we skip this as it requires ObservableGauge callbacks
        # which don't fit the push model well.
        pass
