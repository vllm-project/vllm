from typing import Mapping, Optional

otel_installed = False
try:
    from opentelemetry.context.context import Context
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter)
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.ai import SpanAttributes as BaseSpanAttributes
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator)
    otel_installed = True
except ImportError:

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass


def init_tracer(instrumenting_module_name: str) -> Optional[Tracer]:
    if not otel_installed:
        return None

    trace_provider = TracerProvider()

    # The endpoint of OTLPSpanExporter is set from envvars:
    #  OTEL_EXPORTER_OTLP_ENDPOINT
    #  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def extract_trace_context(headers: Mapping[str, str]) -> Optional[Context]:
    if otel_installed:
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
