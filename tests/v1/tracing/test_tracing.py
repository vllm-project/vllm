# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# type: ignore
import threading
from collections.abc import Iterable
from concurrent import futures
from typing import Callable, Generator, Literal

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from unittest.mock import patch, MagicMock

from vllm import LLM, SamplingParams
from vllm.tracing import SpanAttributes, get_span_exporters, OTEL_TRACES_EXPORTER

FAKE_TRACE_SERVER_ADDRESS = "localhost:4317"

FieldName = Literal[
    "bool_value", "string_value", "int_value", "double_value", "array_value"
]


def decode_value(value: AnyValue):
    field_decoders: dict[FieldName, Callable] = {
        "bool_value": (lambda v: v.bool_value),
        "string_value": (lambda v: v.string_value),
        "int_value": (lambda v: v.int_value),
        "double_value": (lambda v: v.double_value),
        "array_value": (
            lambda v: [decode_value(item) for item in v.array_value.values]
        ),
    }
    for field, decoder in field_decoders.items():
        if value.HasField(field):
            return decoder(value)
    raise ValueError(f"Couldn't decode value: {value}")


def decode_attributes(attributes: Iterable[KeyValue]):
    return {kv.key: decode_value(kv.value) for kv in attributes}


class FakeTraceService(TraceServiceServicer):
    def __init__(self):
        self.request = None
        self.evt = threading.Event()

    def Export(self, request, context):
        self.request = request
        self.evt.set()
        return ExportTraceServiceResponse()


@pytest.fixture
def trace_service() -> Generator[FakeTraceService, None, None]:
    """Fixture to set up a fake gRPC trace service"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    service = FakeTraceService()
    add_TraceServiceServicer_to_server(service, server)
    server.add_insecure_port(FAKE_TRACE_SERVER_ADDRESS)
    server.start()

    yield service

    server.stop(None)


def test_traces(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,
):
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")

        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.1,
            max_tokens=256,
        )
        model = "facebook/opt-125m"
        llm = LLM(
            model=model,
            otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
            gpu_memory_utilization=0.3,
            disable_log_stats=False,
        )
        prompts = ["This is a short prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        print(f"test_traces outputs is : {outputs}")

        timeout = 10
        if not trace_service.evt.wait(timeout):
            raise TimeoutError(
                f"The fake trace service didn't receive a trace within "
                f"the {timeout} seconds timeout"
            )

        request = trace_service.request
        assert len(request.resource_spans) == 1, (
            f"Expected 1 resource span, but got {len(request.resource_spans)}"
        )
        assert len(request.resource_spans[0].scope_spans) == 1, (
            f"Expected 1 scope span, "
            f"but got {len(request.resource_spans[0].scope_spans)}"
        )
        assert len(request.resource_spans[0].scope_spans[0].spans) == 1, (
            f"Expected 1 span, "
            f"but got {len(request.resource_spans[0].scope_spans[0].spans)}"
        )

        attributes = decode_attributes(
            request.resource_spans[0].scope_spans[0].spans[0].attributes
        )
        # assert attributes.get(SpanAttributes.GEN_AI_RESPONSE_MODEL) == model
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_ID) == outputs[0].request_id
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE)
            == sampling_params.temperature
        )
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_TOP_P) == sampling_params.top_p
        )
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS)
            == sampling_params.max_tokens
        )
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_N) == sampling_params.n
        assert attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == len(
            outputs[0].prompt_token_ids
        )
        completion_tokens = sum(len(o.token_ids) for o in outputs[0].outputs)
        assert (
            attributes.get(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS)
            == completion_tokens
        )

        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE) > 0
        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN) > 0
        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_E2E) > 0


# Unit tests for get_span_exporters
class TestGetSpanExporters:
    """Unit tests for get_span_exporters function."""

    def test_default_otlp_exporter(self, monkeypatch: pytest.MonkeyPatch):
        """Default should be OTLP exporter when env var not set."""
        monkeypatch.delenv(OTEL_TRACES_EXPORTER, raising=False)
        with patch("vllm.tracing._get_otlp_span_exporter") as mock_otlp:
            mock_otlp.return_value = MagicMock(name="OTLPSpanExporter")
            exporters = get_span_exporters("http://localhost:4317")
            assert len(exporters) == 1
            mock_otlp.assert_called_once_with("http://localhost:4317")

    def test_console_exporter(self, monkeypatch: pytest.MonkeyPatch):
        """OTEL_TRACES_EXPORTER=console should return ConsoleSpanExporter."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "console")
        exporters = get_span_exporters("http://localhost:4317")
        assert len(exporters) == 1
        assert isinstance(exporters[0], ConsoleSpanExporter)

    def test_none_exporter(self, monkeypatch: pytest.MonkeyPatch):
        """OTEL_TRACES_EXPORTER=none should return empty list."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "none")
        exporters = get_span_exporters("http://localhost:4317")
        assert exporters == []

    def test_multiple_exporters(self, monkeypatch: pytest.MonkeyPatch):
        """OTEL_TRACES_EXPORTER=console,otlp should return both exporters."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "console,otlp")
        with patch("vllm.tracing._get_otlp_span_exporter") as mock_otlp:
            mock_otlp.return_value = MagicMock(name="OTLPSpanExporter")
            exporters = get_span_exporters("http://localhost:4317")
            assert len(exporters) == 2
            assert isinstance(exporters[0], ConsoleSpanExporter)
            mock_otlp.assert_called_once()

    def test_multiple_exporters_with_spaces(self, monkeypatch: pytest.MonkeyPatch):
        """Should handle spaces in comma-separated values."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, " console , otlp ")
        with patch("vllm.tracing._get_otlp_span_exporter") as mock_otlp:
            mock_otlp.return_value = MagicMock(name="OTLPSpanExporter")
            exporters = get_span_exporters("http://localhost:4317")
            assert len(exporters) == 2

    def test_none_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        """When 'none' is combined with others, none should take precedence."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "none,otlp")
        exporters = get_span_exporters("http://localhost:4317")
        assert exporters == []

    def test_invalid_exporter_raises_error(self, monkeypatch: pytest.MonkeyPatch):
        """Invalid exporter type should raise ValueError."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "invalid")
        with pytest.raises(ValueError, match="Unsupported OTEL_TRACES_EXPORTER"):
            get_span_exporters("http://localhost:4317")

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch):
        """Exporter types should be case-insensitive."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "CONSOLE")
        exporters = get_span_exporters("http://localhost:4317")
        assert len(exporters) == 1
        assert isinstance(exporters[0], ConsoleSpanExporter)

    def test_duplicate_exporters_deduplicated(self, monkeypatch: pytest.MonkeyPatch):
        """Duplicate exporter types should be deduplicated."""
        monkeypatch.setenv(OTEL_TRACES_EXPORTER, "console,console")
        exporters = get_span_exporters("http://localhost:4317")
        assert len(exporters) == 1
        assert isinstance(exporters[0], ConsoleSpanExporter)
