# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Optional

from vllm import IMPORT_START_TIME
from vllm.logger import init_logger
from vllm.utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate"]

logger = init_logger(__name__)

_VLLM_OTEL_DOCS = "https://docs.vllm.ai/en/stable/examples/online_serving/opentelemetry.html"

_is_otel_imported = False
_is_tracer_provider_initialized = False
otel_import_error_traceback: Optional[str] = None
try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.context.context import Context
    from opentelemetry.sdk import resources
    from opentelemetry.sdk import trace as sdktrace
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Span, SpanKind, Tracer
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator)
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

    class Span:  # type: ignore

        def end(*args, **kwargs):
            pass

    class Tracer:  # type: ignore
        """No-op tracer used when otel is unavailable."""

        def start_span(*args, **kwargs):
            return Span()

        @contextmanager
        def start_as_current_span(*args, **kwargs):
            yield Span()


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer_provider(otlp_traces_endpoint: Optional[str] = None):
    """Initialize the process global otel trace provider.

    Args:
        otlp_traces_endpoint: optional endpoint provided to vLLM as command
        line argument, kept for compatability with v0 request tracing. Prefer
        using OTEL_EXPORTER_OTLP_TRACES_ENDPOINT env var.
    """
    # Avoid re-initializing the global trace provider. This could happen if
    # using VLLM_USE_V1=0 with openai.api_server entrypoint as that initializes
    # a tracer both in api_server and in LLMEngine. However, the v0 tracing
    # might be used without api_server and removing the init in LLMEngine
    # would break that use case.
    global _is_tracer_provider_initialized
    if _is_tracer_provider_initialized:
        return
    else:
        _is_tracer_provider_initialized = True

    if not is_otel_available():
        logger.info(
            "OpenTelemetry packages are not available. Falling back to "
            "no-op trace exporter. See %s for "
            "opentelemetry package installation instructions.",
            _VLLM_OTEL_DOCS)
        return

    if otlp_traces_endpoint is not None:
        # vLLM allows providing the trace exporter endpoint as a CLI flag
        # --otlp-traces-endpoint, as opposed to the standard of relying on
        # environment variables defined by
        # https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables
        # Preserve current behaviour of using the flag as endpoint value if set.
        logger.info("Overriding env var %s with %s",
                    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, otlp_traces_endpoint)
        os.environ[OTEL_EXPORTER_OTLP_TRACES_ENDPOINT] = otlp_traces_endpoint

    resource = resources.get_aggregated_resources([
        resources.OTELResourceDetector(),
        resources.ProcessResourceDetector(),
    ])

    provider = sdktrace.TracerProvider(resource=resource)
    exporter, protocol = get_span_exporter()
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    logger.info(
        "Initialized OTLP opentelemetry tracer provider exporting to "
        "%s over %s.", exporter._endpoint, protocol)


def get_tracer(instrumenting_module_name: str) -> Tracer:
    """Create a module scoped tracer, similar to logger.

    Usage:

    from vllm.tracing import get_tracer
    tracer = get_tracer(__name__)

    @tracer.start_as_current_span("vllm.asyncllm")
    def init_asyncllm(...):

        with tracer.start_as_current_span("vllm.asyncllm.tokenizer") as span:
            init_tokenizer(...)
            span.set_attributes({
                "tokenizer.mode": "auto",
            })

    Args:
        instrumenting_module_name: commonly just __name__. Sets the scope name
        in the trace which helps understands where the spans come from.

    Returns:
        an opentelemetry tracer object used to create spans
    """
    if not is_otel_available():
        return Tracer()
    return trace.get_tracer(instrumenting_module_name)


def init_tracer(instrumenting_module_name: str,
                otlp_traces_endpoint: Optional[str] = None) -> Tracer:
    init_tracer_provider(otlp_traces_endpoint)
    return get_tracer(instrumenting_module_name)


def get_span_exporter():
    """Create either gRPC or HTTP otel trace exporter.

    Python OpenTelemetry libraries could, but don't yet, infer this from
    environment variables and create the appropriate exporter by default.
    """
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter)
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter)  # type: ignore
    else:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(), protocol


def get_traceparent() -> dict[str, str]:
    """Get the current otel trace context.

    Used for propagating trace context between processes so that a single start
    up trace can cover spans in both API server and the engine.

    Returns:
        dict with traceparent if otel is available
    """
    if not is_otel_available():
        return {}
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def get_root_span(tracer: Tracer,
                  span_name: str,
                  traceparent: Optional[dict] = None) -> Span:
    """Create the top/root span in this process.

    For convenience, add a span under the root span capturing time spent
    importing python modules.

    Args:
        tracer: the Tracer to attach the root span to.
        span_name: the name for the root span, e.g. "vllm.startup",
        "vllm.engine_core".
        traceparent: add spans to an ongoing trace capture by attaching to the
        given traceparent.

    Returns:
        root span under which all other spans for the process can be nested.
    """
    if not is_otel_available():
        return Span()

    if traceparent is not None:
        extracted_context = propagate.extract(traceparent)
        context.attach(extracted_context)
        logger.info("Attached opentelemetry span to provided trace parent %s.",
                    extracted_context)

    root_span = tracer.start_span(span_name, start_time=IMPORT_START_TIME)
    ctx = trace.set_span_in_context(root_span)
    context.attach(ctx)

    with tracer.start_as_current_span("vllm.python_imports",
                                      start_time=IMPORT_START_TIME):
        pass

    return root_span


def set_span_attributes(attributes: dict):
    """Add span attributes to the current span."""
    if not is_otel_available():
        return
    trace.get_current_span().set_attributes(attributes)


def extract_trace_context(
        headers: Optional[Mapping[str, str]]) -> Optional[Context]:
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
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = (
        "gen_ai.latency.time_in_model_forward")
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = (
        "gen_ai.latency.time_in_model_execute")


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning(
        "Received a request with trace context but tracing is disabled")
