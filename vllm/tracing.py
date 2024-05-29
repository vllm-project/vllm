import opentelemetry.semconv.ai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider


def init_tracer(instrumenting_module_name):
    trace_provider = TracerProvider()

    # The endpoint of OTLPSpanExporter is set from envvars:
    #  OTEL_EXPORTER_OTLP_ENDPOINT
    #  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


class SpanAttributes(opentelemetry.semconv.ai.SpanAttributes):
    # The following span attribute names are added here because they are missing
    # from the Semantic Conventions for LLM.
    LLM_REQUEST_ID = "gen_ai.request.id"
    LLM_REQUEST_BEST_OF = "gen_ai.request.best_of"
    LLM_REQUEST_N = "gen_ai.request.n"
    LLM_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    LLM_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    LLM_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    LLM_LATENCY_E2E = "gen_ai.latency.e2e"
