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
