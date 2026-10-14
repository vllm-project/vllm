# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tracing tests for the GPU-less render server (`vllm launch render`)."""

import json
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
from prometheus_client.parser import text_string_to_metric_families

from tests.tracing.conftest import (
    FAKE_TRACE_SERVER_ADDRESS,
    FakeTraceService,
)
from tests.utils import RemoteLaunchRenderServer
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
TOOLS_TEMPLATE = "TOOLS={{ (tools or []) | tojson }}|END|{{ messages[0].content }}"

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
    args = [
        "--otlp-traces-endpoint",
        FAKE_TRACE_SERVER_ADDRESS,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--exclude-tools-when-tool-choice-none",
        "--chat-template",
        TOOLS_TEMPLATE,
    ]
    with RemoteLaunchRenderServer(
        MODEL_NAME,
        args,
        max_wait_seconds=120,
        env_dict={"OTEL_BSP_SCHEDULE_DELAY": "100"},
    ) as srv:
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


def _render_count(metrics: str, api_surface: str, status: str) -> float:
    return sum(
        sample.value
        for family in text_string_to_metric_families(metrics)
        for sample in family.samples
        if sample.name == "vllm:tool_call_render_requests_total"
        and sample.labels["api_surface"] == api_surface
        and sample.labels["status"] == status
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
@pytest.mark.parametrize("choice", ["auto", "none", "required", "named"])
async def test_render_tool_intent_and_telemetry(
    client, trace_service, api_surface, choice
):
    """Real rendering preserves tool metadata/policy without exporting its values."""
    schema = {
        "type": "object",
        "properties": {"private_schema_field": {"type": "string"}},
    }
    chat_tool = {
        "type": "function",
        "function": {"name": "private-render-tool", "parameters": schema},
    }
    payload: dict[str, object] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "private-render-prompt"}],
        "max_tokens": 1,
        "stream": choice == "auto",
    }
    if api_surface == "messages":
        payload["tools"] = [{"name": "private-render-tool", "input_schema": schema}]
        payload["tool_choice"] = (
            {"type": "tool", "name": "private-render-tool"}
            if choice == "named"
            else {"type": "any" if choice == "required" else choice}
        )
        route, span_name = "/v1/messages/render", "render_messages"
    else:
        payload["tools"] = [chat_tool]
        payload["tool_choice"] = (
            {"type": "function", "function": {"name": "private-render-tool"}}
            if choice == "named"
            else choice
        )
        route, span_name = "/v1/chat/completions/render", "render_chat_completion"
    before = _render_count((await client.get("/metrics")).text, api_surface, "rendered")
    trace_service.clear()
    response = await client.post(
        route, json=payload, headers={"traceparent": TRACEPARENT}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    rendered = get_tokenizer(MODEL_NAME).decode(data["token_ids"])
    tools = json.loads(rendered.split("TOOLS=", 1)[1].split("|END|", 1)[0])
    assert len(tools) == (0 if choice == "none" else 1)
    if tools:
        assert tools[0]["function"]["name"] == "private-render-tool"
        assert tools[0]["function"]["parameters"] == schema
    assert data["stream"] == payload["stream"]
    assert data["sampling_params"]["max_tokens"] == 1
    constraints = data["sampling_params"].get("structured_outputs")
    if choice in ("required", "named"):
        assert constraints and any(
            constraints.get(field) for field in ("json", "grammar", "structural_tag")
        ), constraints
    span = _wait_for_span(trace_service, span_name)
    _assert_span_linked_to_trace(span, TRACEPARENT)
    assert span["attributes"] == {
        "tool_calling.present": True,
        "tool_calling.tool_count": 1,
        "tool_calling.tool_choice": choice,
        "tool_calling.api_surface": api_surface,
        "tool_calling.render_status": "rendered",
    }
    metrics = (await client.get("/metrics")).text
    assert _render_count(metrics, api_surface, "rendered") - before == 1
    for value in (
        "private-render-tool",
        "private_schema_field",
        "private-render-prompt",
    ):
        assert value not in metrics
        assert value not in str(span)
    for request in trace_service.requests:
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for exported_span in scope.spans:
                    assert not exported_span.events
                    assert not exported_span.status.message


@pytest.mark.asyncio
@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
async def test_invalid_tool_choice_is_not_counted_as_a_render(client, api_surface):
    """Pre-handler validation must not invent a render or field-rejection outcome."""
    payload: dict[str, object] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1,
    }
    if api_surface == "messages":
        payload["tool_choice"] = {"type": "tool"}
        route = "/v1/messages/render"
    else:
        payload["tool_choice"] = {"type": "function", "function": {}}
        route = "/v1/chat/completions/render"
    before = (await client.get("/metrics")).text
    response = await client.post(route, json=payload)
    assert response.status_code in (400, 422)
    after = (await client.get("/metrics")).text
    for status in ("rendered", "error"):
        assert _render_count(after, api_surface, status) == _render_count(
            before, api_surface, status
        )
