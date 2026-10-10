# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the realtime WebSocket connection loop."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.websockets import WebSocketDisconnect

from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection

pytestmark = pytest.mark.skip_global_cleanup


class _FakeWebSocket:
    """Minimal WebSocket that mirrors Starlette's frame semantics.

    ``receive`` returns raw ASGI messages; ``receive_text`` reproduces
    Starlette's behavior of indexing ``message["text"]`` (which raises
    ``KeyError`` on a binary frame). This lets the same test exercise both the
    old ``receive_text`` path and the new ``receive`` path faithfully.
    """

    def __init__(self, frames: list[dict]):
        self._frames = iter([*frames, {"type": "websocket.disconnect", "code": 1000}])
        self.accept = AsyncMock()
        self.send_text = AsyncMock()

    async def receive(self) -> dict:
        return next(self._frames)

    async def receive_text(self) -> str:
        message = await self.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        return message["text"]

    def sent_error_codes(self) -> list[str]:
        codes = []
        for call in self.send_text.await_args_list:
            payload = json.loads(call.args[0])
            if payload.get("type") == "error":
                codes.append(payload.get("code"))
        return codes


def _make_connection(frames: list[dict]):
    websocket = _FakeWebSocket(frames)
    conn = RealtimeConnection(websocket=websocket, serving=MagicMock())
    return conn, websocket


@pytest.mark.asyncio
async def test_binary_frame_reports_error_and_keeps_session_open():
    """A binary frame must not crash the session; the client gets a clean
    error event and the loop continues to the next frame.

    Previously ``receive_text()`` raised ``KeyError`` on the missing ``text``
    key, which propagated past the per-event handler and killed the session
    without ever sending an error to the client.
    """
    conn, websocket = _make_connection(
        [{"type": "websocket.receive", "bytes": b"\x00\x01\x02\x03"}]
    )

    await conn.handle_connection()

    assert "invalid_frame" in websocket.sent_error_codes()


@pytest.mark.asyncio
async def test_text_frame_is_routed_to_handle_event():
    """A normal text frame is decoded and dispatched to handle_event."""
    conn, websocket = _make_connection(
        [{"type": "websocket.receive", "text": '{"type": "input_audio_buffer.append"}'}]
    )
    conn.handle_event = AsyncMock()

    await conn.handle_connection()

    conn.handle_event.assert_awaited_once_with({"type": "input_audio_buffer.append"})


@pytest.mark.asyncio
async def test_disconnect_frame_ends_loop_cleanly():
    """A disconnect frame ends the loop without emitting an error event."""
    conn, websocket = _make_connection([])

    await conn.handle_connection()

    assert conn._is_connected is False
    assert websocket.sent_error_codes() == []
