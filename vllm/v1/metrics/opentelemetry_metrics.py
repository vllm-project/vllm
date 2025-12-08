# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenTelemetry Metrics Logger for vLLM.

This module provides push-based metrics export to OpenTelemetry collectors,
as an alternative to the pull-based Prometheus metrics.
"""

import os

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.metrics.loggers import AggregateStatLoggerBase
from vllm.v1.metrics.stats import (
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
)

logger = init_logger(__name__)

_is_otel_available = False
otel_import_error_traceback: str | None = None

try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    _is_otel_available = True
except ImportError:
    import traceback

    otel_import_error_traceback = traceback.format_exc()


def is_otel_metrics_available() -> bool:
    """Check if OpenTelemetry metrics SDK is available."""
    return _is_otel_available


class OpenTelemetryMetricsLogger(AggregateStatLoggerBase):
    """
    OpenTelemetry-based metrics logger that pushes metrics to OTEL collectors.

    This logger uses the OpenTelemetry Metrics SDK to push metrics directly
    to an OTLP endpoint, eliminating the need for external scrapers.

    Configuration via environment variables:
        - OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: OTLP endpoint for metrics
          (default: http://localhost:4317)
        - OTEL_EXPORTER_OTLP_METRICS_PROTOCOL: Protocol (grpc or http/protobuf)
          (default: grpc)
        - OTEL_METRIC_EXPORT_INTERVAL: Export interval in milliseconds
          (default: 60000)
        - OTEL_SERVICE_NAME: Service name for metrics
          (default: vllm)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int] | None = None,
    ):
        if not is_otel_metrics_available():
            raise ValueError(
                "OpenTelemetry Metrics SDK is not available. "
                "Install with: pip install opentelemetry-exporter-otlp-proto-grpc\n"
                f"Original error:\n{otel_import_error_traceback}"
            )

        if engine_indexes is None:
            engine_indexes = [0]

        self.engine_indexes = engine_indexes
        self.vllm_config = vllm_config

        # Get configuration from environment
        endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "http://localhost:4317",
        )
        protocol = os.getenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc")
        export_interval_millis = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000"))
        service_name = os.getenv("OTEL_SERVICE_NAME", "vllm")

        logger.info(
            "Initializing OpenTelemetry metrics exporter: "
            "endpoint=%s, protocol=%s, export_interval=%dms, service=%s",
            endpoint,
            protocol,
            export_interval_millis,
            service_name,
        )

        # Create OTLP exporter
        if protocol == "grpc":
            exporter = OTLPMetricExporter(endpoint=endpoint)
        elif protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter as HTTPMetricExporter,
            )

            exporter = HTTPMetricExporter(endpoint=endpoint)
        else:
            raise ValueError(f"Unsupported OTLP protocol: {protocol}")

        # Create periodic reader with configured export interval
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=export_interval_millis,
        )

        # Create resource with service name and model info
        resource = Resource.create(
            {
                "service.name": service_name,
                "vllm.model": vllm_config.model_config.served_model_name,
                "vllm.version": "1.0.0",  # TODO: Import from vllm.version
            }
        )

        # Set up meter provider
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

        # Get meter for creating instruments
        self.meter = metrics.get_meter("vllm.metrics")

        # Common labels/attributes
        model_name = vllm_config.model_config.served_model_name
        self.common_attributes = {"model_name": model_name}

        # Create metric instruments (similar to Prometheus metrics)
        self._create_instruments()

        logger.info("OpenTelemetry metrics logger initialized successfully")

    def _create_instruments(self):
        """Create OpenTelemetry metric instruments."""

        # Scheduler state gauges (ObservableGauge for async observation)
        self.gauge_scheduler_running = self.meter.create_observable_gauge(
            name="vllm.num_requests_running",
            description="Number of requests in model execution batches",
            callbacks=[self._observe_running_requests],
        )

        self.gauge_scheduler_waiting = self.meter.create_observable_gauge(
            name="vllm.num_requests_waiting",
            description="Number of requests waiting to be processed",
            callbacks=[self._observe_waiting_requests],
        )

        # KV cache usage
        self.gauge_kv_cache_usage = self.meter.create_observable_gauge(
            name="vllm.kv_cache_usage_perc",
            description="KV-cache usage (1 = 100%)",
            callbacks=[self._observe_kv_cache_usage],
        )

        # Counters for tokens and requests
        self.counter_prompt_tokens = self.meter.create_counter(
            name="vllm.prompt_tokens",
            description="Number of prefill tokens processed",
            unit="tokens",
        )

        self.counter_generation_tokens = self.meter.create_counter(
            name="vllm.generation_tokens",
            description="Number of generation tokens processed",
            unit="tokens",
        )

        self.counter_request_success = self.meter.create_counter(
            name="vllm.request_success",
            description="Count of successfully processed requests",
            unit="requests",
        )

        self.counter_num_preempted_reqs = self.meter.create_counter(
            name="vllm.num_preemptions",
            description="Cumulative number of preemptions",
            unit="preemptions",
        )

        # Prefix cache metrics
        self.counter_prefix_cache_queries = self.meter.create_counter(
            name="vllm.prefix_cache_queries",
            description="Prefix cache queries (number of queried tokens)",
            unit="tokens",
        )

        self.counter_prefix_cache_hits = self.meter.create_counter(
            name="vllm.prefix_cache_hits",
            description="Prefix cache hits (number of cached tokens)",
            unit="tokens",
        )

        # Histograms for latencies
        self.histogram_time_to_first_token = self.meter.create_histogram(
            name="vllm.time_to_first_token_seconds",
            description="Time to first token in seconds",
            unit="s",
        )

        self.histogram_inter_token_latency = self.meter.create_histogram(
            name="vllm.inter_token_latency_seconds",
            description="Inter-token latency in seconds",
            unit="s",
        )

        self.histogram_e2e_time_request = self.meter.create_histogram(
            name="vllm.e2e_request_latency_seconds",
            description="End-to-end request latency in seconds",
            unit="s",
        )

        self.histogram_queue_time_request = self.meter.create_histogram(
            name="vllm.request_queue_time_seconds",
            description="Time spent in WAITING phase",
            unit="s",
        )

        self.histogram_inference_time_request = self.meter.create_histogram(
            name="vllm.request_inference_time_seconds",
            description="Time spent in RUNNING phase",
            unit="s",
        )

        # Store latest scheduler stats for observable gauges
        self._latest_scheduler_stats: dict[int, SchedulerStats] = {}

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """Record metrics from scheduler and iteration stats."""

        # Store scheduler stats for gauge observations
        if scheduler_stats is not None:
            self._latest_scheduler_stats[engine_idx] = scheduler_stats

            # Record prefix cache metrics
            self.counter_prefix_cache_queries.add(
                scheduler_stats.prefix_cache_stats.queries,
                attributes={**self.common_attributes, "engine": str(engine_idx)},
            )
            self.counter_prefix_cache_hits.add(
                scheduler_stats.prefix_cache_stats.hits,
                attributes={**self.common_attributes, "engine": str(engine_idx)},
            )

        if iteration_stats is None:
            return

        attrs = {**self.common_attributes, "engine": str(engine_idx)}

        # Record token counters
        self.counter_prompt_tokens.add(
            iteration_stats.num_prompt_tokens, attributes=attrs
        )
        self.counter_generation_tokens.add(
            iteration_stats.num_generation_tokens, attributes=attrs
        )

        # Record preemptions
        self.counter_num_preempted_reqs.add(
            iteration_stats.num_preempted_reqs, attributes=attrs
        )

        # Record latency histograms
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token.record(ttft, attributes=attrs)

        for itl in iteration_stats.inter_token_latencies_iter:
            self.histogram_inter_token_latency.record(itl, attributes=attrs)

        # Record finished request metrics
        for finished_request in iteration_stats.finished_requests:
            finish_attrs = {
                **attrs,
                "finished_reason": str(finished_request.finish_reason),
            }

            self.counter_request_success.add(1, attributes=finish_attrs)
            self.histogram_e2e_time_request.record(
                finished_request.e2e_latency, attributes=attrs
            )
            self.histogram_queue_time_request.record(
                finished_request.queued_time, attributes=attrs
            )
            self.histogram_inference_time_request.record(
                finished_request.inference_time, attributes=attrs
            )

    def _observe_running_requests(
        self, options: "metrics.CallbackOptions"
    ) -> list["metrics.Observation"]:
        """Callback for running requests gauge."""
        observations = []
        for engine_idx, stats in self._latest_scheduler_stats.items():
            attrs = {**self.common_attributes, "engine": str(engine_idx)}
            observations.append(
                metrics.Observation(stats.num_running_reqs, attributes=attrs)
            )
        return observations

    def _observe_waiting_requests(
        self, options: "metrics.CallbackOptions"
    ) -> list["metrics.Observation"]:
        """Callback for waiting requests gauge."""
        observations = []
        for engine_idx, stats in self._latest_scheduler_stats.items():
            attrs = {**self.common_attributes, "engine": str(engine_idx)}
            observations.append(
                metrics.Observation(stats.num_waiting_reqs, attributes=attrs)
            )
        return observations

    def _observe_kv_cache_usage(
        self, options: "metrics.CallbackOptions"
    ) -> list["metrics.Observation"]:
        """Callback for KV cache usage gauge."""
        observations = []
        for engine_idx, stats in self._latest_scheduler_stats.items():
            attrs = {**self.common_attributes, "engine": str(engine_idx)}
            observations.append(
                metrics.Observation(stats.kv_cache_usage, attributes=attrs)
            )
        return observations

    def log_engine_initialized(self):
        """Log that the engine has been initialized."""
        logger.info(
            "OpenTelemetry metrics logger: Engine initialized with %d GPU blocks",
            self.vllm_config.cache_config.num_gpu_blocks or 0,
        )
