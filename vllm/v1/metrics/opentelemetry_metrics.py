# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenTelemetry Metrics Logger for vLLM.

This module provides push-based metrics export to OpenTelemetry collectors,
as an alternative to the pull-based Prometheus metrics.
"""

import os
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.metrics.loggers import AggregateStatLoggerBase
from vllm.v1.metrics.stats import (
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
)

if TYPE_CHECKING:
    from vllm.config import SupportsMetricsInfo

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

        # Engine sleep state
        self.gauge_engine_sleep_state = self.meter.create_observable_gauge(
            name="vllm.engine_sleep_state",
            description=(
                "Engine sleep state (awake=1, weights_offloaded=1, discard_all=1)"
            ),
            callbacks=[self._observe_sleep_state],
        )

        # LoRA metrics
        self.gauge_lora_info = None
        self.max_lora = None
        if self.vllm_config.lora_config is not None:
            if len(self.engine_indexes) > 1:
                raise NotImplementedError("LoRA in DP mode is not supported yet.")
            self.max_lora = self.vllm_config.lora_config.max_loras
            self.gauge_lora_info = self.meter.create_observable_gauge(
                name="vllm.lora_requests_info",
                description="Running stats on LoRA requests",
                callbacks=[self._observe_lora_info],
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

        # External - KV connector prefix cache
        self.counter_connector_prefix_cache_queries = self.meter.create_counter(
            name="vllm.external_prefix_cache_queries",
            description=(
                "External prefix cache queries from KV connector (queried tokens)"
            ),
            unit="tokens",
        )

        self.counter_connector_prefix_cache_hits = self.meter.create_counter(
            name="vllm.external_prefix_cache_hits",
            description=(
                "External prefix cache hits from KV connector (cached tokens)"
            ),
            unit="tokens",
        )

        # Multi-modal cache
        self.counter_mm_cache_queries = self.meter.create_counter(
            name="vllm.mm_cache_queries",
            description="Multi-modal cache queries (number of queried items)",
            unit="items",
        )

        self.counter_mm_cache_hits = self.meter.create_counter(
            name="vllm.mm_cache_hits",
            description="Multi-modal cache hits (number of cached items)",
            unit="items",
        )

        # Corrupted requests counter (conditional on env var)
        import vllm.envs as envs

        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            self.counter_corrupted_requests = self.meter.create_counter(
                name="vllm.corrupted_requests",
                description="Corrupted requests with NaNs in logits",
                unit="requests",
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

        self.histogram_prefill_time_request = self.meter.create_histogram(
            name="vllm.request_prefill_time_seconds",
            description="Time spent in PREFILL phase",
            unit="s",
        )

        self.histogram_decode_time_request = self.meter.create_histogram(
            name="vllm.request_decode_time_seconds",
            description="Time spent in DECODE phase",
            unit="s",
        )

        # Histograms of counts
        self.histogram_num_prompt_tokens_request = self.meter.create_histogram(
            name="vllm.request_prompt_tokens",
            description="Number of prefill tokens per request",
            unit="tokens",
        )

        self.histogram_num_generation_tokens_request = self.meter.create_histogram(
            name="vllm.request_generation_tokens",
            description="Number of generation tokens per request",
            unit="tokens",
        )

        self.histogram_iteration_tokens = self.meter.create_histogram(
            name="vllm.iteration_tokens_total",
            description="Number of tokens per engine_step",
            unit="tokens",
        )

        self.histogram_max_num_generation_tokens_request = self.meter.create_histogram(
            name="vllm.request_max_num_generation_tokens",
            description="Maximum number of requested generation tokens",
            unit="tokens",
        )

        self.histogram_n_request = self.meter.create_histogram(
            name="vllm.request_params_n",
            description="The n request parameter",
            unit="1",
        )

        self.histogram_max_tokens_request = self.meter.create_histogram(
            name="vllm.request_params_max_tokens",
            description="The max_tokens request parameter",
            unit="tokens",
        )

        self.histogram_request_time_per_output_token = self.meter.create_histogram(
            name="vllm.request_time_per_output_token_seconds",
            description="Time per output token per request",
            unit="s",
        )

        # Store latest scheduler stats for observable gauges
        self._latest_scheduler_stats: dict[int, SchedulerStats] = {}

        # Store latest sleep state for observable gauge
        self._sleep_state: dict[int, dict[str, int]] = {}
        for engine_idx in self.engine_indexes:
            self._sleep_state[engine_idx] = {
                "awake": 1,
                "weights_offloaded": 0,
                "discard_all": 0,
            }

        # Store LoRA adapter state for observable gauge
        self._lora_state: dict[int, dict[str, str]] = {}
        if self.gauge_lora_info is not None:
            for engine_idx in self.engine_indexes:
                self._lora_state[engine_idx] = {
                    "waiting_lora_adapters": "",
                    "running_lora_adapters": "",
                }

        # Store cache config info for observable gauge
        self._cache_config_info: dict[int, dict[str, str]] = {}
        self.gauge_cache_config_info: metrics.ObservableGauge | None = None

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

            attrs = {**self.common_attributes, "engine": str(engine_idx)}

            # Record prefix cache metrics
            self.counter_prefix_cache_queries.add(
                scheduler_stats.prefix_cache_stats.queries,
                attributes=attrs,
            )
            self.counter_prefix_cache_hits.add(
                scheduler_stats.prefix_cache_stats.hits,
                attributes=attrs,
            )

            # Record connector prefix cache metrics
            if scheduler_stats.connector_prefix_cache_stats is not None:
                self.counter_connector_prefix_cache_queries.add(
                    scheduler_stats.connector_prefix_cache_stats.queries,
                    attributes=attrs,
                )
                self.counter_connector_prefix_cache_hits.add(
                    scheduler_stats.connector_prefix_cache_stats.hits,
                    attributes=attrs,
                )

            # Update LoRA adapter state
            if self.gauge_lora_info is not None:
                running_lora_adapters = ",".join(
                    scheduler_stats.running_lora_adapters.keys()
                )
                waiting_lora_adapters = ",".join(
                    scheduler_stats.waiting_lora_adapters.keys()
                )
                self._lora_state[engine_idx] = {
                    "running_lora_adapters": running_lora_adapters,
                    "waiting_lora_adapters": waiting_lora_adapters,
                }

        # Record multimodal cache metrics
        if mm_cache_stats is not None:
            attrs = {**self.common_attributes, "engine": str(engine_idx)}
            self.counter_mm_cache_queries.add(mm_cache_stats.queries, attributes=attrs)
            self.counter_mm_cache_hits.add(mm_cache_stats.hits, attributes=attrs)

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

        # Record corrupted requests counter
        import vllm.envs as envs

        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            self.counter_corrupted_requests.add(
                iteration_stats.num_corrupted_reqs, attributes=attrs
            )

        # Record iteration tokens histogram
        self.histogram_iteration_tokens.record(
            iteration_stats.num_prompt_tokens + iteration_stats.num_generation_tokens,
            attributes=attrs,
        )

        # Record per-iteration histograms
        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request.record(
                max_gen_tokens, attributes=attrs
            )

        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request.record(n_param, attributes=attrs)

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

            # Latency histograms
            self.histogram_e2e_time_request.record(
                finished_request.e2e_latency, attributes=attrs
            )
            self.histogram_queue_time_request.record(
                finished_request.queued_time, attributes=attrs
            )
            self.histogram_inference_time_request.record(
                finished_request.inference_time, attributes=attrs
            )
            self.histogram_prefill_time_request.record(
                finished_request.prefill_time, attributes=attrs
            )
            self.histogram_decode_time_request.record(
                finished_request.decode_time, attributes=attrs
            )

            # Token count histograms
            self.histogram_num_prompt_tokens_request.record(
                finished_request.num_prompt_tokens, attributes=attrs
            )
            self.histogram_num_generation_tokens_request.record(
                finished_request.num_generation_tokens, attributes=attrs
            )

            # Time per output token histogram
            self.histogram_request_time_per_output_token.record(
                finished_request.mean_time_per_output_token, attributes=attrs
            )

            # Max tokens param histogram (if provided)
            if finished_request.max_tokens_param:
                self.histogram_max_tokens_request.record(
                    finished_request.max_tokens_param, attributes=attrs
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

    def _observe_sleep_state(
        self, options: "metrics.CallbackOptions"
    ) -> list["metrics.Observation"]:
        """Callback for engine sleep state gauge."""
        observations = []
        for engine_idx, states in self._sleep_state.items():
            for state_name, state_value in states.items():
                attrs = {
                    **self.common_attributes,
                    "engine": str(engine_idx),
                    "sleep_state": state_name,
                }
                observations.append(metrics.Observation(state_value, attributes=attrs))
        return observations

    def _observe_lora_info(
        self, options: "metrics.CallbackOptions"
    ) -> list["metrics.Observation"]:
        """Callback for LoRA requests info gauge."""
        observations = []
        if self.gauge_lora_info is not None:
            for engine_idx, lora_info in self._lora_state.items():
                attrs = {
                    **self.common_attributes,
                    "engine": str(engine_idx),
                    "max_lora": str(self.max_lora),
                    "waiting_lora_adapters": (lora_info["waiting_lora_adapters"]),
                    "running_lora_adapters": (lora_info["running_lora_adapters"]),
                }
                # Use current timestamp as value
                # (similar to Prometheus set_to_current_time)
                import time

                observations.append(metrics.Observation(time.time(), attributes=attrs))
        return observations

    def record_sleep_state(self, is_awake: int = 1, level: int = 0):
        """Record engine sleep state.

        Args:
            is_awake: 0 if sleeping, 1 if awake
            level: Sleep level (0=awake, 1=weights_offloaded, 2=discard_all)
        """
        for engine_idx in self.engine_indexes:
            if is_awake == 1:
                self._sleep_state[engine_idx]["awake"] = 1
                self._sleep_state[engine_idx]["weights_offloaded"] = 0
                self._sleep_state[engine_idx]["discard_all"] = 0
            else:
                self._sleep_state[engine_idx]["awake"] = 0
                if level == 1:
                    self._sleep_state[engine_idx]["weights_offloaded"] = 1
                    self._sleep_state[engine_idx]["discard_all"] = 0
                elif level == 2:
                    self._sleep_state[engine_idx]["weights_offloaded"] = 0
                    self._sleep_state[engine_idx]["discard_all"] = 1
                else:
                    self._sleep_state[engine_idx]["weights_offloaded"] = 0
                    self._sleep_state[engine_idx]["discard_all"] = 0

    def log_metrics_info(self, type: str, config_obj: "SupportsMetricsInfo"):
        """Log metrics info from config objects.

        Args:
            type: Type of config (e.g., "cache_config")
            config_obj: Config object that supports metrics_info()
        """
        from vllm.config import SupportsMetricsInfo

        if not isinstance(config_obj, SupportsMetricsInfo):
            return

        # Create the observable gauge if not already created
        if self.gauge_cache_config_info is None:
            if type == "cache_config":
                self.gauge_cache_config_info = self.meter.create_observable_gauge(
                    name="vllm.cache_config_info",
                    description="Information of the LLMEngine CacheConfig",
                    callbacks=[self._observe_cache_config_info],
                )
            else:
                logger.warning("Unknown metrics info type: %s", type)
                return

        # Store the cache config info for each engine
        for engine_index in self.engine_indexes:
            metrics_info = config_obj.metrics_info()
            # Convert all values to strings for OTEL attributes
            metrics_info_str = {k: str(v) for k, v in metrics_info.items()}
            metrics_info_str["engine"] = str(engine_index)
            self._cache_config_info[engine_index] = metrics_info_str

    def _observe_cache_config_info(
        self, options: "metrics.CallbackOptions"
    ) -> list["metrics.Observation"]:
        """Callback for cache config info gauge."""
        observations = []
        for engine_idx, config_info in self._cache_config_info.items():
            attrs = {**self.common_attributes, **config_info}
            # Info gauges are always set to 1
            observations.append(metrics.Observation(1, attributes=attrs))
        return observations

    def log_engine_initialized(self):
        """Log that the engine has been initialized."""
        logger.info(
            "OpenTelemetry metrics logger: Engine initialized with %d GPU blocks",
            self.vllm_config.cache_config.num_gpu_blocks or 0,
        )
        # Log cache config info
        self.log_metrics_info("cache_config", self.vllm_config.cache_config)
