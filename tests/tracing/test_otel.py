# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from vllm.tracing import SpanKind, extract_trace_context, extract_trace_headers
from vllm.tracing.otel import (
    get_trace_headers_from_context,
    manual_instrument_otel,
    shutdown_otel_tracer,
)


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    shutdown_otel_tracer()

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    for env_var in ("traceparent", "tracestate", "TRACEPARENT", "TRACESTATE"):
        monkeypatch.delenv(env_var, raising=False)

    yield exporter

    shutdown_otel_tracer()


def test_extract_trace_headers_is_case_insensitive():
    headers = {
        "Traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
        "TraceState": "rojo=1",
    }

    assert extract_trace_headers(headers) == {
        "traceparent": headers["Traceparent"],
        "tracestate": headers["TraceState"],
    }


def test_get_trace_headers_from_context_preserves_parent(
    span_exporter: InMemorySpanExporter,
):
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("parent") as parent:
        parent_context = parent.get_span_context()
        trace_headers = get_trace_headers_from_context()

    assert trace_headers is not None

    manual_instrument_otel(
        "llm_request",
        start_time=1,
        context=extract_trace_context(trace_headers),
        kind=SpanKind.SERVER,
        use_environment_context=False,
    )

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    llm_request_span = spans["llm_request"]

    assert llm_request_span.context.trace_id == parent_context.trace_id
    assert llm_request_span.parent is not None
    assert llm_request_span.parent.span_id == parent_context.span_id
    assert llm_request_span.context.span_id != parent_context.span_id


def test_manual_instrument_otel_uses_environment_context_by_default(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_id = "11111111111111111111111111111111"
    parent_span_id = "2222222222222222"
    monkeypatch.setenv("traceparent", f"00-{trace_id}-{parent_span_id}-01")

    manual_instrument_otel("llm_request", start_time=1, kind=SpanKind.SERVER)

    (span,) = span_exporter.get_finished_spans()
    assert span.parent is not None
    assert span.parent.trace_id == int(trace_id, 16)
    assert span.parent.span_id == int(parent_span_id, 16)


def test_manual_instrument_otel_can_disable_environment_fallback(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_id = "11111111111111111111111111111111"
    parent_span_id = "2222222222222222"
    monkeypatch.setenv("traceparent", f"00-{trace_id}-{parent_span_id}-01")

    manual_instrument_otel(
        "llm_request",
        start_time=1,
        kind=SpanKind.SERVER,
        use_environment_context=False,
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.parent is None
    assert span.context.trace_id != int(trace_id, 16)
    assert span.context.span_id != int(parent_span_id, 16)
