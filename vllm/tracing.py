# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate"]

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: str | None = None

# Global TracerProvider singleton to prevent overwriting
_global_tracer_provider = None
_global_tracer_endpoint: str | None = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(
    instrumenting_module_name: str, otlp_traces_endpoint: str
) -> Tracer | None:
    """Initialize tracer with a global singleton TracerProvider.

    CRITICAL: Uses a singleton pattern to avoid overwriting the global provider.
    Multiple calls with the same endpoint will reuse the same provider.

    IMPORTANT: First call sets the endpoint. Subsequent calls with different
    endpoints will log a warning and use the original endpoint (first call wins).

    Args:
        instrumenting_module_name: Scope name (e.g., "vllm.api", "vllm.scheduler")
        otlp_traces_endpoint: OTLP endpoint URL

    Returns:
        Tracer instance for the specified scope
    """
    global _global_tracer_provider, _global_tracer_endpoint

    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )

    # Use singleton pattern: create provider once, reuse for all tracers
    if _global_tracer_provider is None:
        logger.info(
            "Initializing global TracerProvider with endpoint: %s",
            otlp_traces_endpoint
        )
        _global_tracer_provider = TracerProvider()
        _global_tracer_endpoint = otlp_traces_endpoint

        span_exporter = get_span_exporter(otlp_traces_endpoint)
        _global_tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        set_tracer_provider(_global_tracer_provider)

        logger.info("Global TracerProvider initialized successfully")
    else:
        # Warn if endpoint mismatch (first call wins)
        if otlp_traces_endpoint != _global_tracer_endpoint:
            logger.warning(
                "TracerProvider already initialized with endpoint '%s'. "
                "Ignoring different endpoint '%s' for scope '%s'. "
                "First init_tracer call sets the endpoint for all scopes.",
                _global_tracer_endpoint,
                otlp_traces_endpoint,
                instrumenting_module_name
            )
        logger.debug(
            "Reusing existing global TracerProvider for scope: %s",
            instrumenting_module_name
        )

    # Get tracer for this specific scope from the singleton provider
    tracer = _global_tracer_provider.get_tracer(instrumenting_module_name)
    logger.info(
        "Created tracer for scope '%s' from global provider",
        instrumenting_module_name
    )

    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    """Extract trace context from headers using W3C Trace Context propagation.

    Args:
        headers: HTTP headers dict that may contain traceparent/tracestate

    Returns:
        Context object if trace headers found, None otherwise
    """
    if not is_otel_available():
        logger.debug("OTEL not available, cannot extract trace context")
        return None

    headers = headers or {}

    if "traceparent" in headers:
        logger.debug("Extracting trace context from headers (traceparent present)")
    else:
        logger.debug("No traceparent header found in request")

    try:
        context = TraceContextTextMapPropagator().extract(headers)
        logger.debug("Successfully extracted trace context: %s", context is not None)
        return context
    except Exception as e:
        logger.debug(
            "Failed to extract trace context: %s",
            e
        )
        return None


def inject_trace_context(span, carrier: dict[str, str] | None = None) -> dict[str, str] | None:
    """Inject span context into carrier dict for W3C Trace Context propagation.

    Args:
        span: The span whose context should be injected
        carrier: Optional dict to inject into (modified in place). If None, creates new dict.

    Returns:
        The carrier dict with injected context. Returns carrier unchanged if:
        - OTEL is unavailable (returns input carrier, which may be None)
        - span is None (returns input carrier, which may be None)
        - Injection fails due to exception (returns carrier, guaranteed dict after line 117)
    """
    if not is_otel_available():
        logger.debug("OTEL not available, skipping trace context injection")
        return carrier

    if span is None:
        logger.debug("Span is None, skipping trace context injection")
        return carrier

    try:
        # Create or use existing carrier
        if carrier is None:
            carrier = {}

        # Inject span context using W3C Trace Context propagator
        from opentelemetry import trace
        context = trace.set_span_in_context(span)
        TraceContextTextMapPropagator().inject(carrier, context=context)

        logger.debug("Injected trace context into carrier (traceparent set)")

        return carrier
    except Exception as e:
        # Injection failure should not break request processing
        logger.debug(
            "Failed to inject trace context: %s",
            e
        )
        return carrier


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"
    # Journey event attributes (for request lifecycle span events)
    JOURNEY_EVENT_TYPE = "event.type"
    JOURNEY_TS_MONOTONIC = "ts.monotonic"
    # API event attributes (for API layer span events)
    EVENT_TS_MONOTONIC = "event.ts.monotonic"
    JOURNEY_PHASE = "phase"
    JOURNEY_PREFILL_DONE_TOKENS = "prefill.done_tokens"
    JOURNEY_PREFILL_TOTAL_TOKENS = "prefill.total_tokens"
    JOURNEY_DECODE_DONE_TOKENS = "decode.done_tokens"
    JOURNEY_DECODE_MAX_TOKENS = "decode.max_tokens"
    JOURNEY_NUM_PREEMPTIONS = "num_preemptions"
    JOURNEY_SCHEDULER_STEP = "scheduler.step"
    JOURNEY_SCHEDULE_KIND = "schedule.kind"
    JOURNEY_FINISH_STATUS = "finish.status"
    # Step batch summary attributes (for step-level tracing)
    STEP_ID = "step.id"
    STEP_TS_START_NS = "step.ts_start_ns"
    STEP_TS_END_NS = "step.ts_end_ns"
    STEP_DURATION_US = "step.duration_us"
    QUEUE_RUNNING_DEPTH = "queue.running_depth"
    QUEUE_WAITING_DEPTH = "queue.waiting_depth"
    BATCH_NUM_PREFILL_REQS = "batch.num_prefill_reqs"
    BATCH_NUM_DECODE_REQS = "batch.num_decode_reqs"
    BATCH_SCHEDULED_TOKENS = "batch.scheduled_tokens"
    BATCH_PREFILL_TOKENS = "batch.prefill_tokens"
    BATCH_DECODE_TOKENS = "batch.decode_tokens"
    BATCH_NUM_FINISHED = "batch.num_finished"
    BATCH_NUM_PREEMPTED = "batch.num_preempted"
    KV_USAGE_GPU_RATIO = "kv.usage_gpu_ratio"
    KV_BLOCKS_TOTAL_GPU = "kv.blocks_total_gpu"
    KV_BLOCKS_FREE_GPU = "kv.blocks_free_gpu"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
