# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for the Responses API WebSocket mode.

These tests use Starlette's TestClient to exercise the full WebSocket flow
(FastAPI route -> WebSocketResponsesConnection -> mock serving layer)
WITHOUT a real model engine.
"""

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Stub out heavy dependencies BEFORE any vllm imports.
#
# The real protocol / engine / serving modules pull in torch, openai, regex,
# CUDA, etc.  We replace them with lightweight fakes that expose only the
# classes the WebSocket code path actually touches at runtime.
# ---------------------------------------------------------------------------

# 1. Prevent vllm/__init__.py from importing torch via env_override
if "vllm.env_override" not in sys.modules:
    sys.modules["vllm.env_override"] = types.ModuleType("vllm.env_override")


# 2. Build fake Pydantic models that mirror the real protocol surface used
#    by WebSocketResponsesConnection._process_response_create.


class _FakeResponsesRequest(BaseModel):
    """Minimal stand-in for ResponsesRequest."""

    model: str = ""
    input: str | list = ""
    stream: bool = True
    request_id: str = "resp_test"

    model_config = ConfigDict(extra="allow")


class _FakeResponsesResponse(BaseModel):
    """Minimal stand-in for ResponsesResponse."""

    id: str = "resp_test"
    created_at: int = 0
    model: str = ""
    output: list = []
    status: str = "completed"


class _FakeResponseCreatedEvent(BaseModel):
    type: str = "response.created"
    response: _FakeResponsesResponse = _FakeResponsesResponse()


class _FakeResponseCompletedEvent(BaseModel):
    type: str = "response.completed"
    response: _FakeResponsesResponse = _FakeResponsesResponse()


class _FakeStreamingResponsesResponse(BaseModel):
    type: str = "unknown"


class _FakeErrorResponse(BaseModel):
    """Minimal stand-in for ErrorResponse."""

    class _Error(BaseModel):
        message: str = ""
        code: int = 400

    error: _Error = _Error()


class _FakeOpenAIServingResponses:
    """Minimal stand-in for OpenAIServingResponses type."""

    pass


class _FakeValidateJsonRequest:
    """Dependency stub."""

    pass


# 3. Register stub modules in sys.modules so that both api_router.py
#    (top-level imports) and websocket.py (lazy imports inside
#    _process_response_create) resolve without touching the real packages.

_STUB_MODULES: dict[str, dict[str, object]] = {
    "vllm.entrypoints.openai.engine.protocol": {
        "ErrorResponse": _FakeErrorResponse,
    },
    "vllm.entrypoints.openai.responses.protocol": {
        "ResponsesRequest": _FakeResponsesRequest,
        "ResponsesResponse": _FakeResponsesResponse,
        "ResponseCreatedEvent": _FakeResponseCreatedEvent,
        "ResponseCompletedEvent": _FakeResponseCompletedEvent,
        "StreamingResponsesResponse": _FakeStreamingResponsesResponse,
    },
    "vllm.entrypoints.openai.responses.serving": {
        "OpenAIServingResponses": _FakeOpenAIServingResponses,
    },
    "vllm.entrypoints.openai.utils": {
        "validate_json_request": lambda: None,
    },
    "vllm.entrypoints.utils": {
        "load_aware_call": lambda f: f,
        "with_cancellation": lambda f: f,
    },
}

for _modname, _attrs in _STUB_MODULES.items():
    if _modname not in sys.modules:
        _mod = types.ModuleType(_modname)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_modname] = _mod

# ---------------------------------------------------------------------------
# Now safe to import vllm modules
# ---------------------------------------------------------------------------

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from vllm.entrypoints.openai.responses.api_router import router  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(
    *,
    serving: object | None = None,
    max_connections: int = 100,
    active_connections: int = 0,
) -> FastAPI:
    """Create a minimal FastAPI app wired up like the real api_server.py."""
    app = FastAPI()
    app.include_router(router)

    # Mirror the state attributes set by api_server.py
    app.state.openai_serving_responses = serving
    app.state.ws_responses_active_connections = active_connections
    app.state.ws_responses_max_connections = max_connections
    app.state.ws_responses_lock = asyncio.Lock()
    return app


class _FakeTextDelta(BaseModel):
    """A fake streaming delta event."""

    type: str = "response.output_text.delta"
    delta: str = "Hello"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketConnectAndReceiveEvents:
    """Connect, send response.create, receive streaming events."""

    def test_websocket_connect_and_receive_events(self):
        """Full round-trip: connect -> send response.create -> receive
        streamed events (created, delta, completed)."""
        serving = AsyncMock()

        created_event = _FakeResponseCreatedEvent()
        delta_event = _FakeTextDelta()
        completed_resp = _FakeResponsesResponse(id="resp_abc", status="completed")
        completed_event = _FakeResponseCompletedEvent(response=completed_resp)

        async def fake_create_responses(request, raw_request=None):
            async def gen():
                yield created_event
                yield delta_event
                yield completed_event

            return gen()

        serving.create_responses = fake_create_responses

        app = _make_app(serving=serving)
        client = TestClient(app)

        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "test-model",
                    "input": "Say hello",
                }
            )

            received = []
            # We expect 3 events: created, delta, completed
            for _ in range(3):
                data = ws.receive_text()
                received.append(json.loads(data))

            assert received[0]["type"] == "response.created"
            assert received[1]["type"] == "response.output_text.delta"
            assert received[1]["delta"] == "Hello"
            assert received[2]["type"] == "response.completed"
            assert received[2]["response"]["id"] == "resp_abc"


class TestWebSocketConnectionLimit:
    """Verify connection limit enforcement."""

    def test_websocket_connection_limit(self):
        """When active connections == max, the server returns
        websocket_connection_limit_reached and closes."""
        serving = AsyncMock()
        app = _make_app(serving=serving, max_connections=1, active_connections=1)
        client = TestClient(app)

        with client.websocket_connect("/v1/responses") as ws:
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "error"
            assert payload["status"] == 429
            assert payload["error"]["code"] == "websocket_connection_limit_reached"


class TestWebSocketInvalidJson:
    """Verify invalid JSON handling."""

    def test_websocket_invalid_json(self):
        """Sending invalid JSON returns an invalid_json error and the
        connection stays open for further messages."""
        serving = AsyncMock()
        app = _make_app(serving=serving)
        client = TestClient(app)

        with client.websocket_connect("/v1/responses") as ws:
            # Send garbage
            ws.send_text("this is not json{{{")
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "error"
            assert payload["error"]["code"] == "invalid_json"

            # Connection should still be open — send another (valid) message
            # that triggers a different error to prove the conn is alive.
            ws.send_text('{"type": "bogus.event"}')
            data2 = ws.receive_text()
            payload2 = json.loads(data2)
            assert payload2["error"]["code"] == "unknown_event_type"


class TestWebSocketUnknownEventType:
    """Verify unknown event type handling."""

    def test_websocket_unknown_event_type(self):
        """Sending an event with type 'session.update' returns
        unknown_event_type error."""
        serving = AsyncMock()
        app = _make_app(serving=serving)
        client = TestClient(app)

        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json({"type": "session.update"})
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "error"
            assert payload["error"]["code"] == "unknown_event_type"
            assert "session.update" in payload["error"]["message"]
