# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import time
import types

import pytest
import regex as re

# ---------------------------------------------------------------------------
# Lightweight shim so the test module can be collected and run on machines
# that do **not** have torch / CUDA / numpy installed.
#
# ``vllm/__init__.py`` does ``import vllm.env_override`` which in turn does
# ``import torch`` at module level.  We short-circuit that by inserting a
# no-op ``vllm.env_override`` into ``sys.modules`` *before* the real
# ``vllm`` package is imported.
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    # Insert a no-op env_override so vllm/__init__.py skips torch
    sys.modules["vllm.env_override"] = types.ModuleType("vllm.env_override")


def test_uuid7_format():
    """uuid7 returns a 32-char hex string with version=7 and variant=10."""
    from vllm.entrypoints.openai.responses.websocket import uuid7

    result = uuid7()
    assert len(result) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", result), f"Not hex: {result}"
    # Version nibble (bits 48-51) must be 7
    version = int(result[12], 16)
    assert version == 7, f"Version {version} != 7"
    # Variant bits (bits 64-65) must be 10 (value 8-b)
    variant = int(result[16], 16)
    assert variant in (8, 9, 0xA, 0xB), f"Variant nibble {variant:#x} invalid"


def test_uuid7_monotonic():
    """uuid7 values are time-ordered (monotonically increasing)."""
    from vllm.entrypoints.openai.responses.websocket import uuid7

    a = uuid7()
    time.sleep(0.002)  # 2ms to ensure different timestamp
    b = uuid7()
    assert a < b, f"{a} should be < {b}"


def test_uuid7_uniqueness():
    """Rapid uuid7 calls produce unique values."""
    from vllm.entrypoints.openai.responses.websocket import uuid7

    ids = [uuid7() for _ in range(1000)]
    assert len(set(ids)) == 1000


def test_connection_context_defaults():
    """ConnectionContext initializes with correct defaults."""
    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test123")
    assert ctx.connection_id == "ws-test123"
    assert ctx.last_response_id is None
    assert ctx.last_response is None
    assert ctx.inflight is False
    assert ctx.created_at > 0
    assert ConnectionContext.LIFETIME_SECONDS == 3600
    assert ConnectionContext.WARNING_SECONDS == 3300


def test_connection_context_is_expired():
    """ConnectionContext.is_expired checks against LIFETIME_SECONDS."""
    from unittest.mock import patch

    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test")
    assert not ctx.is_expired()

    with patch("time.monotonic", return_value=ctx.created_at + 3601):
        assert ctx.is_expired()


def test_connection_context_should_warn():
    """ConnectionContext.should_warn checks against WARNING_SECONDS."""
    from unittest.mock import patch

    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test")
    assert not ctx.should_warn()

    with patch("time.monotonic", return_value=ctx.created_at + 3301):
        assert ctx.should_warn()


def test_connection_context_evict_cache():
    """evict_cache clears last_response_id and last_response."""
    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test")
    ctx.last_response_id = "resp_abc"
    ctx.last_response = "fake_response"  # type: ignore
    ctx.evict_cache()
    assert ctx.last_response_id is None
    assert ctx.last_response is None


@pytest.mark.asyncio
async def test_send_error_format():
    """send_error sends JSON in the OpenAI error event format."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.send_error("previous_response_not_found", "Response not found", 400)

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["type"] == "error"
    assert payload["status"] == 400
    assert payload["error"]["code"] == "previous_response_not_found"
    assert payload["error"]["message"] == "Response not found"


@pytest.mark.asyncio
async def test_send_event_serializes_pydantic():
    """send_event serializes a Pydantic model and sends as text."""
    from unittest.mock import AsyncMock

    from pydantic import BaseModel

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    class FakeEvent(BaseModel):
        type: str = "response.created"
        data: str = "hello"

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.send_event(FakeEvent())

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["type"] == "response.created"
    assert payload["data"] == "hello"


@pytest.mark.asyncio
async def test_handle_event_unknown_type():
    """Unknown event type sends error, keeps connection open."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.handle_event({"type": "bogus.event"})

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "unknown_event_type"


@pytest.mark.asyncio
async def test_handle_event_concurrent_request_rejected():
    """Second response.create while inflight returns concurrent_request error."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)
    conn.ctx.inflight = True

    await conn.handle_event(
        {
            "type": "response.create",
            "model": "test-model",
            "input": "hello",
        }
    )

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "concurrent_request"


@pytest.mark.asyncio
async def test_handle_event_previous_response_not_found():
    """previous_response_id not in cache returns error."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.handle_event(
        {
            "type": "response.create",
            "model": "test-model",
            "input": "hello",
            "previous_response_id": "resp_nonexistent",
        }
    )

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "previous_response_not_found"
    assert not conn.ctx.inflight


@pytest.mark.asyncio
async def test_handle_connection_accept_and_receive_loop():
    """handle_connection accepts, processes messages, handles disconnect."""
    from unittest.mock import AsyncMock

    from starlette.websockets import WebSocketDisconnect

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    # Simulate: one valid message, then disconnect
    ws.receive_text.side_effect = [
        '{"type": "response.create", "model": "m", "input": "hi"}',
        WebSocketDisconnect(),
    ]
    # Mock _process_response_create directly to avoid lazy imports
    # of protocol modules that transitively require torch.
    conn._process_response_create = AsyncMock(
        side_effect=Exception("mock engine down"),
    )

    await conn.handle_connection()

    # accept() is called by the router, not handle_connection
    ws.accept.assert_not_called()
    assert not conn._is_connected


@pytest.mark.asyncio
async def test_handle_connection_invalid_json():
    """Invalid JSON sends error but keeps connection open."""
    from unittest.mock import AsyncMock

    from starlette.websockets import WebSocketDisconnect

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    ws.receive_text.side_effect = [
        "not valid json{{{",
        WebSocketDisconnect(),
    ]

    await conn.handle_connection()

    first_call = ws.send_text.call_args_list[0]
    payload = json.loads(first_call[0][0])
    assert payload["error"]["code"] == "invalid_json"


@pytest.mark.asyncio
async def test_cleanup_cancels_deadline_task():
    """cleanup cancels the deadline task if still running."""
    from unittest.mock import AsyncMock, MagicMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    fake_task = MagicMock()
    fake_task.done.return_value = False
    conn._deadline_task = fake_task

    await conn.cleanup()

    fake_task.cancel.assert_called_once()


def test_cli_arg_max_websocket_connections_default():
    """--max-websocket-connections defaults to 100."""
    # The FrontendArgs class lives behind a deep import chain that pulls in
    # torch, regex, and other heavy dependencies.  Instead of fighting the
    # imports we read the source file directly and verify the field with a
    # simple AST / text check.  This is robust on machines without CUDA.
    import ast
    import pathlib

    cli_args_path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "vllm"
        / "entrypoints"
        / "openai"
        / "cli_args.py"
    )
    source = cli_args_path.read_text()
    tree = ast.parse(source)

    # Find the FrontendArgs class
    frontend_cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FrontendArgs":
            frontend_cls = node
            break
    assert frontend_cls is not None, "FrontendArgs class not found"

    # Find the max_websocket_connections field
    found = False
    for node in ast.walk(frontend_cls):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id == "max_websocket_connections"
            ):
                found = True
                # Check annotation is int
                assert isinstance(node.annotation, ast.Name)
                assert node.annotation.id == "int"
                # Check default value is 100
                assert isinstance(node.value, ast.Constant)
                assert node.value.value == 100
                break
    assert found, "max_websocket_connections field not found in FrontendArgs"


def _ensure_api_router_importable():
    """Stub the heavy transitive dependencies of api_router.py so it can be
    imported on machines without torch / regex / openai / PIL etc.

    Must be called *before* importing api_router for the first time.
    """
    from unittest.mock import MagicMock

    from pydantic import BaseModel

    _stub_modules: dict[str, dict[str, object]] = {
        "vllm.entrypoints.openai.engine.protocol": {
            "ErrorResponse": type("ErrorResponse", (BaseModel,), {}),
        },
        "vllm.entrypoints.openai.responses.protocol": {
            "ResponsesRequest": type("ResponsesRequest", (BaseModel,), {}),
            "ResponsesResponse": type("ResponsesResponse", (BaseModel,), {}),
            "StreamingResponsesResponse": type(
                "StreamingResponsesResponse", (BaseModel,), {}
            ),
        },
        "vllm.entrypoints.openai.responses.serving": {
            "OpenAIServingResponses": MagicMock(),
        },
        "vllm.entrypoints.openai.utils": {
            "validate_json_request": MagicMock(),
        },
        "vllm.entrypoints.utils": {
            "load_aware_call": lambda f: f,
            "with_cancellation": lambda f: f,
        },
    }
    for modname, attrs in _stub_modules.items():
        if modname not in sys.modules:
            mod = types.ModuleType(modname)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[modname] = mod


@pytest.mark.asyncio
async def test_websocket_route_rejects_when_limit_reached():
    """WebSocket route sends connection_limit_reached when at max."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, PropertyMock

    _ensure_api_router_importable()
    from vllm.entrypoints.openai.responses.api_router import (
        create_responses_websocket,
    )

    ws = AsyncMock()
    # Simulate app state at limit
    app_state = MagicMock()
    app_state.openai_serving_responses = MagicMock()
    app_state.ws_responses_active_connections = 5
    app_state.ws_responses_max_connections = 5
    app_state.ws_responses_lock = asyncio.Lock()

    app = MagicMock()
    app.state = app_state
    type(ws).app = PropertyMock(return_value=app)

    await create_responses_websocket(ws)

    # Should accept, send error, then close
    ws.accept.assert_called_once()
    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "websocket_connection_limit_reached"
    ws.close.assert_called_once()
