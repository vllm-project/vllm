# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# type: ignore
from __future__ import annotations

import threading
from collections.abc import Iterable
from concurrent import futures
from typing import Callable, Generator, Literal

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer, add_TraceServiceServicer_to_server)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_INSECURE)

from vllm import LLM, SamplingParams
from vllm.tracing import SpanAttributes

FAKE_TRACE_SERVER_ADDRESS = "localhost:4317"

FieldName = Literal['bool_value', 'string_value', 'int_value', 'double_value',
                    'array_value']


def decode_value(value: AnyValue):
    field_decoders: dict[FieldName, Callable] = {
        "bool_value": (lambda v: v.bool_value),
        "string_value": (lambda v: v.string_value),
        "int_value": (lambda v: v.int_value),
        "double_value": (lambda v: v.double_value),
        "array_value":
        (lambda v: [decode_value(item) for item in v.array_value.values]),
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
        m.setenv("VLLM_USE_V1", "1")
        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.1,
            max_tokens=256,
        )
        model = "facebook/opt-125m"
        llm = LLM(model=model,
                  otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
                  gpu_memory_utilization=0.3,
                  disable_log_stats=False)
        prompts = ["This is a short prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        print(f"test_traces outputs is : {outputs}")

        timeout = 10
        if not trace_service.evt.wait(timeout):
            raise TimeoutError(
                f"The fake trace service didn't receive a trace within "
                f"the {timeout} seconds timeout")

        request = trace_service.request
        assert len(request.resource_spans) == 1, (
            f"Expected 1 resource span, "
            f"but got {len(request.resource_spans)}")
        assert len(request.resource_spans[0].scope_spans) == 1, (
            f"Expected 1 scope span, "
            f"but got {len(request.resource_spans[0].scope_spans)}")
        assert len(request.resource_spans[0].scope_spans[0].spans) == 1, (
            f"Expected 1 span, "
            f"but got {len(request.resource_spans[0].scope_spans[0].spans)}")

        attributes = decode_attributes(
            request.resource_spans[0].scope_spans[0].spans[0].attributes)
        # assert attributes.get(SpanAttributes.GEN_AI_RESPONSE_MODEL) == model
        assert attributes.get(
            SpanAttributes.GEN_AI_REQUEST_ID) == outputs[0].request_id
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE
                              ) == sampling_params.temperature
        assert attributes.get(
            SpanAttributes.GEN_AI_REQUEST_TOP_P) == sampling_params.top_p
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS
                              ) == sampling_params.max_tokens
        assert attributes.get(
            SpanAttributes.GEN_AI_REQUEST_N) == sampling_params.n
        assert attributes.get(
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == len(
                outputs[0].prompt_token_ids)
        completion_tokens = sum(len(o.token_ids) for o in outputs[0].outputs)
        assert attributes.get(
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS) == completion_tokens

        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE) > 0
        assert attributes.get(
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN) > 0
        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_E2E) > 0
