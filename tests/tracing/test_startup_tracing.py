# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Generator, Iterable
from concurrent import futures
from typing import Callable, Literal

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer, add_TraceServiceServicer_to_server)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_EXPORTER_OTLP_TRACES_INSECURE,
    OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)

from ..utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"

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
        self.requests = []
        self.evt = threading.Event()

    def Export(self, request, context):
        self.requests.append(request)
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


def test_traces(trace_service: FakeTraceService, ):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]

    envs = {
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: FAKE_TRACE_SERVER_ADDRESS,
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL: "grpc",
        OTEL_EXPORTER_OTLP_TRACES_INSECURE: "true",
        'VLLM_USE_V1': '1',
    }

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs):
        pass

    timeout = 15
    if not trace_service.evt.wait(timeout):
        raise TimeoutError(
            f"The fake trace service didn't receive a trace within "
            f"the {timeout} seconds timeout")

    spans = {}
    for request in trace_service.requests:
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    spans[span.name] = span

    assert len(spans) == 12, (f"Expected 12 spans but got {len(spans)}.")

    found_spans = set(spans.keys())
    expected_spans = set([
        'vllm.startup',
        'vllm.python_imports',
        'vllm.asyncllm',
        'vllm.asyncllm.tokenizer',
        'vllm.model_registry.inspect_model',
        'vllm.engine_core',
        'vllm.engine_core_client',
        'vllm.engine_core.kv_cache',
        'vllm.engine_core.model_executor',
        'vllm.engine_core.model_runner.load_model',
        'vllm.engine_core.model_runner.profile_run',
        'vllm.api_server.init_app_state',
    ])
    print(spans)
    print(found_spans)
    assert expected_spans <= found_spans, (
        f"Missing expected span names: {expected_spans - found_spans}")

    load_model_span = spans["vllm.engine_core.model_runner.load_model"]
    attributes = decode_attributes(load_model_span.attributes)

    assert attributes["model.name"] == "facebook/opt-125m", (
        f"Unexpected model name {attributes['model.name']}.")

    assert attributes["model.dtype"] == "torch.bfloat16", (
        f"Unexpected model name {attributes['model.dtype']}.")

    assert attributes["model.config_format"] == "auto", (
        f"Unexpected model name {attributes['model.config_format']}.")

    assert attributes["model.load_format"] == "auto", (
        f"Unexpected model name {attributes['model.load_format']}.")
