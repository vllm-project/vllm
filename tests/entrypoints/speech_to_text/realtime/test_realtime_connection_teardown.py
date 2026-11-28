# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teardown guarantees for the realtime WebSocket connection.

Regression tests for https://github.com/vllm-project/vllm/issues/57724.
``handle_connection`` sent ``session.created`` before entering its
``try``, so a client that vanished during the handshake skipped
``finally: await self.cleanup()`` entirely, leaving the audio queue and
generation task uncollected. ``_is_connected`` was also cleared only in
the ``WebSocketDisconnect`` arm, so any other error, and cancellation,
left an aborted connection advertising itself as live.

These drive ``RealtimeConnection`` directly with a fake websocket: the
defect is in connection lifecycle handling and needs no model or engine.
"""

import asyncio

import pytest
from starlette.websockets import WebSocketDisconnect

from vllm.entrypoints.speech_to_text.realtime.connection import (
    RealtimeConnection,
)


class FakeWebSocket:
    """Accepts, then fails the nth send and disconnects on receive."""

    def __init__(self, fail_send_with: BaseException | None = None):
        self.fail_send_with = fail_send_with
        self.accepted = False
        self.sent = 0

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        self.sent += 1
        if self.fail_send_with is not None and self.sent == 1:
            raise self.fail_send_with

    async def receive_text(self):
        raise WebSocketDisconnect()


class FakeServing:
    def _is_model_supported(self, model):
        return True


def make_connection(fail_send_with: BaseException | None = None):
    connection = RealtimeConnection(FakeWebSocket(fail_send_with), FakeServing())
    cleaned = {"ran": False}
    original = connection.cleanup

    async def spy():
        cleaned["ran"] = True
        await original()

    connection.cleanup = spy
    return connection, cleaned


@pytest.mark.asyncio
async def test_handshake_failure_still_cleans_up():
    """A send failure during the handshake must not skip cleanup."""
    connection, cleaned = make_connection(RuntimeError("client went away"))

    await connection.handle_connection()

    assert cleaned["ran"]
    assert connection._is_connected is False
    assert connection.audio_queue.get_nowait() is None


@pytest.mark.asyncio
async def test_handshake_cancellation_still_cleans_up():
    """Cancellation during the handshake cleans up and still propagates."""
    connection, cleaned = make_connection(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await connection.handle_connection()

    assert cleaned["ran"]
    assert connection._is_connected is False
    assert connection.audio_queue.get_nowait() is None


@pytest.mark.asyncio
async def test_unexpected_error_clears_connected_flag():
    """A non-disconnect error must not leave the connection marked live."""
    connection, cleaned = make_connection()

    async def fail():
        raise RuntimeError("engine died")

    connection.websocket.receive_text = fail

    await connection.handle_connection()

    assert cleaned["ran"]
    assert connection._is_connected is False


@pytest.mark.asyncio
async def test_cancellation_clears_connected_flag():
    """Cancelling the receive loop clears the flag and re-raises."""
    connection, cleaned = make_connection()

    async def cancel():
        raise asyncio.CancelledError()

    connection.websocket.receive_text = cancel

    with pytest.raises(asyncio.CancelledError):
        await connection.handle_connection()

    assert cleaned["ran"]
    assert connection._is_connected is False


@pytest.mark.asyncio
async def test_normal_disconnect_is_unchanged():
    """The path that already worked keeps working."""
    connection, cleaned = make_connection()

    await connection.handle_connection()

    assert cleaned["ran"]
    assert connection._is_connected is False
    assert connection.audio_queue.get_nowait() is None
