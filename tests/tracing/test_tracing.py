import os
import threading
from concurrent import futures
from typing import Callable, Dict, Iterable, Literal

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
    field_decoders: Dict[FieldName, Callable] = {
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
def trace_service():
    """Fixture to set up a fake gRPC trace service"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    service = FakeTraceService()
    add_TraceServiceServicer_to_server(service, server)
    server.add_insecure_port(FAKE_TRACE_SERVER_ADDRESS)
    server.start()

    yield service

    server.stop(None)


def test_traces(trace_service):
    os.environ[OTEL_EXPORTER_OTLP_TRACES_INSECURE] = "true"

    sampling_params = SamplingParams(temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=256)
    model = "facebook/opt-125m"
    llm = LLM(
        model=model,
        otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
    )
    prompts = ["This is a short prompt"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    timeout = 5
    if not trace_service.evt.wait(timeout):
        raise TimeoutError(
            f"The fake trace service didn't receive a trace within "
            f"the {timeout} seconds timeout")

    attributes = decode_attributes(trace_service.request.resource_spans[0].
                                   scope_spans[0].spans[0].attributes)
    assert attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) == model
    assert attributes.get(
        SpanAttributes.LLM_REQUEST_ID) == outputs[0].request_id
    assert attributes.get(
        SpanAttributes.LLM_REQUEST_TEMPERATURE) == sampling_params.temperature
    assert attributes.get(
        SpanAttributes.LLM_REQUEST_TOP_P) == sampling_params.top_p
    assert attributes.get(
        SpanAttributes.LLM_REQUEST_MAX_TOKENS) == sampling_params.max_tokens
    assert attributes.get(
        SpanAttributes.LLM_REQUEST_BEST_OF) == sampling_params.best_of
    assert attributes.get(SpanAttributes.LLM_REQUEST_N) == sampling_params.n
    assert attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == len(
        outputs[0].prompt_token_ids)
    completion_tokens = sum(len(o.token_ids) for o in outputs[0].outputs)
    assert attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) == completion_tokens
    metrics = outputs[0].metrics
    assert attributes.get(
        SpanAttributes.LLM_LATENCY_TIME_IN_QUEUE) == metrics.time_in_queue
    ttft = metrics.first_token_time - metrics.arrival_time
    assert attributes.get(
        SpanAttributes.LLM_LATENCY_TIME_TO_FIRST_TOKEN) == ttft
    e2e_time = metrics.finished_time - metrics.arrival_time
    assert attributes.get(SpanAttributes.LLM_LATENCY_E2E) == e2e_time
