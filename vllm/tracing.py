import os
from typing import Mapping, Optional

from vllm.logger import init_logger
from vllm.utils import run_once

logger = init_logger(__name__)

_is_otel_installed = False
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.ai import SpanAttributes as BaseSpanAttributes
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator)
    _is_otel_installed = True
except ImportError:

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass


def is_otel_installed() -> bool:
    return _is_otel_installed


def init_tracer(instrumenting_module_name: str,
                otlp_traces_endpoint: str) -> Optional[Tracer]:
    trace_provider = TracerProvider()

    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter)
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter)
    else:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(headers: Mapping[str, str]) -> Optional[Context]:
    if is_otel_installed():
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


class SpanAttributes(BaseSpanAttributes):
    # The following span attribute names are added here because they are missing
    # from the Semantic Conventions for LLM.
    LLM_REQUEST_ID = "gen_ai.request.id"
    LLM_REQUEST_BEST_OF = "gen_ai.request.best_of"
    LLM_REQUEST_N = "gen_ai.request.n"
    LLM_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    LLM_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    LLM_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    LLM_LATENCY_E2E = "gen_ai.latency.e2e"


def contains_trace_context(headers: Mapping[str, str]) -> bool:
    return "traceparent" in headers or "tracestate" in headers


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning(
        "Received a request with trace context but tracing is disabled")
