# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate"]

# Environment variable for selecting the traces exporter type.
# Supported values: "otlp" (default), "console", "none"
OTEL_TRACES_EXPORTER = "OTEL_TRACES_EXPORTER"

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: str | None = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        SpanExporter,
    )
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

    class SpanExporter:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(
    instrumenting_module_name: str, otlp_traces_endpoint: str
) -> Tracer | None:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    trace_provider = TracerProvider()

    span_exporters = get_span_exporters(otlp_traces_endpoint)
    for span_exporter in span_exporters:
        # Use SimpleSpanProcessor for ConsoleSpanExporter to ensure
        # immediate output, use BatchSpanProcessor for others for
        # better performance.
        if isinstance(span_exporter, ConsoleSpanExporter):
            trace_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        else:
            trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporters(endpoint: str) -> list[SpanExporter]:
    """Get span exporters based on OTEL_TRACES_EXPORTER environment variable.

    Args:
        endpoint: The OTLP endpoint URL (used when exporter is "otlp").

    Returns:
        A list of SpanExporter instances. Empty list if exporter is "none".

    Supported OTEL_TRACES_EXPORTER values (comma-separated):
        - "otlp" (default): Export traces via OTLP protocol to the specified
          endpoint. The protocol (grpc/http) is controlled by
          OTEL_EXPORTER_OTLP_TRACES_PROTOCOL.
        - "console": Print traces to stdout. Useful for local debugging.
        - "none": Disable trace export (must be used alone).

    Examples:
        OTEL_TRACES_EXPORTER=otlp          # Export to OTLP collector
        OTEL_TRACES_EXPORTER=console       # Print to stdout
        OTEL_TRACES_EXPORTER=console,otlp  # Both console and OTLP
        OTEL_TRACES_EXPORTER=none          # Disable tracing
    """
    exporter_env = os.environ.get(OTEL_TRACES_EXPORTER, "otlp")
    exporter_types_raw = [e.strip().lower() for e in exporter_env.split(",")]
    # Deduplicate exporter types while preserving order
    exporter_types = list(dict.fromkeys(exporter_types_raw))

    # Handle "none" - must be used alone
    if "none" in exporter_types:
        if len(exporter_types) > 1:
            logger.warning(
                "OTEL_TRACES_EXPORTER contains 'none' with other exporters. "
                "'none' takes precedence, tracing is disabled."
            )
        logger.info("Trace exporter is disabled (OTEL_TRACES_EXPORTER=none)")
        return []

    exporters: list[SpanExporter] = []
    for exporter_type in exporter_types:
        if exporter_type == "console":
            logger.info("Using ConsoleSpanExporter for tracing")
            exporters.append(ConsoleSpanExporter())
        elif exporter_type == "otlp":
            exporters.append(_get_otlp_span_exporter(endpoint))
        else:
            raise ValueError(
                f"Unsupported OTEL_TRACES_EXPORTER value: '{exporter_type}'. "
                "Supported values are: 'otlp', 'console', 'none'."
            )

    return exporters


def _get_otlp_span_exporter(endpoint: str):
    """Get OTLP span exporter based on protocol configuration."""
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
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' is configured. "
            "Supported values are: 'grpc', 'http/protobuf'."
        )

    logger.info(
        "Using OTLPSpanExporter for tracing (endpoint=%s, protocol=%s)",
        endpoint,
        protocol,
    )
    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


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


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
