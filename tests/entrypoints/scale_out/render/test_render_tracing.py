# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tracing tests for the GPU-less render server (`vllm launch render`)."""

import socket
import time
from collections.abc import Generator
from concurrent import futures

import grpc
import httpx
import pytest
import pytest_asyncio
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    add_TraceServiceServicer_to_server,
)

from tests.tracing.conftest import (
    FAKE_TRACE_SERVER_ADDRESS,
    FakeTraceService,
)
from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"

# W3C traceparent header with a fixed trace id and parent span id.
TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.fixture(scope="module")
def trace_service() -> Generator[FakeTraceService, None, None]:
    """Module-scoped fake OTLP trace service for the render server process."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    service = FakeTraceService()
    add_TraceServiceServicer_to_server(service, server)
    server.add_insecure_port(FAKE_TRACE_SERVER_ADDRESS)
    server.start()

    host, port = FAKE_TRACE_SERVER_ADDRESS.rsplit(":", 1)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    yield service

    server.stop(grace=None)


@pytest.fixture(scope="module")
def server(trace_service):
    args = ["--otlp-traces-endpoint", FAKE_TRACE_SERVER_ADDRESS]
    with RemoteLaunchRenderServer(MODEL_NAME, args, max_wait_seconds=120) as srv:
        yield srv


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


def _wait_for_span(
    trace_service: FakeTraceService,
    span_name: str,
    timeout: float = 30.0,
) -> dict:
    """Wait until a span with the given name has been exported."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for span in trace_service.get_all_spans():
            if span["name"] == span_name:
                return span
        time.sleep(0.25)
    raise AssertionError(f"No span named '{span_name}' was exported within {timeout}s")


def _assert_span_linked_to_trace(span: dict, traceparent: str) -> None:
    """Assert the span belongs to the trace carried by `traceparent`."""
    _, trace_id, parent_span_id, _ = traceparent.split("-")
    assert span["trace_id"] == trace_id
    assert span["parent_span_id"] == parent_span_id


@pytest.mark.asyncio
async def test_render_chat_completion_span(client, trace_service):
    """A span is exported for the chat render endpoint, linked to the caller's
    trace context."""
    response = await client.post(
        "/v1/chat/completions/render",
        headers={"traceparent": TRACEPARENT},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )

    assert response.status_code == 200

    span = _wait_for_span(trace_service, "render_chat_completion")
    _assert_span_linked_to_trace(span, TRACEPARENT)


@pytest.mark.asyncio
async def test_render_completion_span(client, trace_service):
    """A span is exported for the completion render endpoint."""
    response = await client.post(
        "/v1/completions/render",
        headers={"traceparent": TRACEPARENT},
        json={
            "model": MODEL_NAME,
            "prompt": "Once upon a time",
        },
    )

    assert response.status_code == 200

    span = _wait_for_span(trace_service, "render_completion")
    _assert_span_linked_to_trace(span, TRACEPARENT)


@pytest.mark.asyncio
async def test_tokenize_span(client, trace_service):
    """A span is exported for the tokenize endpoint."""
    response = await client.post(
        "/tokenize",
        headers={"traceparent": TRACEPARENT},
        json={
            "model": MODEL_NAME,
            "prompt": "Hello world",
        },
    )

    assert response.status_code == 200

    span = _wait_for_span(trace_service, "tokenize")
    _assert_span_linked_to_trace(span, TRACEPARENT)
