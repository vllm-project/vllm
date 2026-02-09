# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable, Generator, Iterable
from concurrent import futures
from typing import Any, Literal

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

FAKE_TRACE_SERVER_ADDRESS = "localhost:4317"

FieldName = Literal[
    "bool_value", "string_value", "int_value", "double_value", "array_value"
]


def decode_value(value: AnyValue):
    """Decode an OpenTelemetry AnyValue protobuf message to a Python value."""
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


def decode_attributes(attributes: Iterable[KeyValue]) -> dict[str, Any]:
    """Decode OpenTelemetry KeyValue attributes to a Python dictionary."""
    return {kv.key: decode_value(kv.value) for kv in attributes}


class FakeTraceService(TraceServiceServicer):
    """A fake gRPC trace service for testing OpenTelemetry trace exports."""

    def __init__(self):
        self.requests: list[ExportTraceServiceRequest] = []
        self.evt = threading.Event()
        self._lock = threading.Lock()

    def Export(self, request, context):
        with self._lock:
            self.requests.append(request)
        self.evt.set()
        return ExportTraceServiceResponse()

    @property
    def request(self) -> ExportTraceServiceRequest | None:
        """Returns the first request received (for backward compatibility)."""
        with self._lock:
            return self.requests[0] if self.requests else None

    def get_all_spans(self) -> list[dict]:
        """Returns all spans from all received requests as decoded dicts."""
        spans = []
        with self._lock:
            for request in self.requests:
                for resource_span in request.resource_spans:
                    for scope_span in resource_span.scope_spans:
                        for span in scope_span.spans:
                            spans.append(
                                {
                                    "name": span.name,
                                    "attributes": decode_attributes(span.attributes),
                                    "trace_id": span.trace_id.hex(),
                                    "span_id": span.span_id.hex(),
                                    "parent_span_id": span.parent_span_id.hex()
                                    if span.parent_span_id
                                    else None,
                                    "start_time_unix_nano": span.start_time_unix_nano,
                                    "end_time_unix_nano": span.end_time_unix_nano,
                                }
                            )
        return spans

    def wait_for_spans(self, count: int = 1, timeout: float = 10) -> bool:
        """Wait until at least `count` spans have been received."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.get_all_spans()) >= count:
                return True
            time.sleep(0.1)
        return False

    def clear(self):
        """Clear all received requests."""
        with self._lock:
            self.requests.clear()
        self.evt.clear()


@pytest.fixture
def trace_service() -> Generator[FakeTraceService, None, None]:
    """Fixture to set up a fake gRPC trace service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    service = FakeTraceService()
    add_TraceServiceServicer_to_server(service, server)
    server.add_insecure_port(FAKE_TRACE_SERVER_ADDRESS)
    server.start()

    yield service

    server.stop(grace=None)


@pytest.fixture
def trace_server_address() -> str:
    """Returns the address of the fake trace server."""
    return FAKE_TRACE_SERVER_ADDRESS
