# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `/v1/realtime` idle-session watchdog.

Regression coverage for https://github.com/vllm-project/vllm/issues/46815:
a client that keeps the WebSocket open but stops sending audio (and never
sends a ``final`` commit) used to pin the engine request's KV cache forever.
The watchdog finalizes and closes such idle sessions.

These tests exercise the watchdog logic directly with mocked transports, so
they need neither a GPU nor a model.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.websockets import WebSocketDisconnect

from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection

# These tests use mocked transports and never initialize a device, so the
# global GPU/distributed cleanup fixture is unnecessary.
pytestmark = pytest.mark.skip_global_cleanup


def _make_websocket() -> SimpleNamespace:
    return SimpleNamespace(
        accept=AsyncMock(),
        send_text=AsyncMock(),
        receive_text=AsyncMock(),
        close=AsyncMock(),
    )


def _make_connection(
    idle_timeout: float = 0.1, grace: float = 0.1
) -> tuple[RealtimeConnection, SimpleNamespace]:
    websocket = _make_websocket()
    conn = RealtimeConnection(websocket, serving=Mock())
    conn._idle_timeout_s = idle_timeout
    conn._idle_finalize_grace_s = grace
    conn._is_connected = True
    conn._touch()
    return conn, websocket


@pytest.mark.asyncio
async def test_idle_timeout_closes_idle_session():
    """An idle session is notified, drained, and closed."""
    conn, websocket = _make_connection(idle_timeout=0.05)

    await asyncio.wait_for(conn._idle_watchdog(), timeout=2.0)

    # Client is notified, the socket is closed, and the connection is marked
    # disconnected so handle_connection()'s receive loop unwinds.
    websocket.close.assert_awaited_once()
    websocket.send_text.assert_awaited()  # idle_timeout error event
    assert conn._is_connected is False
    # The None sentinel was enqueued so a blocked generation can finalize.
    assert conn.audio_queue.get_nowait() is None


@pytest.mark.asyncio
async def test_activity_resets_idle_timer():
    """Recording activity keeps the session alive; stopping lets it time out."""
    conn, websocket = _make_connection(idle_timeout=0.3)

    watchdog = asyncio.create_task(conn._idle_watchdog())

    # Stay active for noticeably longer than the timeout.
    for _ in range(8):
        await asyncio.sleep(0.05)
        conn._touch()

    assert not watchdog.done()
    websocket.close.assert_not_awaited()

    # Stop touching; the watchdog should now fire.
    await asyncio.wait_for(watchdog, timeout=2.0)
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_timeout_finalizes_generation_gracefully():
    """A blocked generation that honors the None sentinel exits cleanly."""
    conn, websocket = _make_connection(idle_timeout=0.05, grace=2.0)

    finished = asyncio.Event()

    async def fake_generation():
        # Mirrors _run_generation: consume audio until the None sentinel.
        while True:
            chunk = await conn.audio_queue.get()
            if chunk is None:
                break
        finished.set()

    conn.generation_task = asyncio.create_task(fake_generation())
    await asyncio.sleep(0)  # let the task start and block on the queue

    await asyncio.wait_for(conn._idle_watchdog(), timeout=3.0)

    assert finished.is_set()
    assert conn.generation_task.done()
    assert not conn.generation_task.cancelled()
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_timeout_force_cancels_stuck_generation():
    """A generation that ignores the sentinel is force-cancelled after grace."""
    conn, websocket = _make_connection(idle_timeout=0.05, grace=0.1)

    async def stuck_generation():
        await asyncio.sleep(3600)

    conn.generation_task = asyncio.create_task(stuck_generation())
    await asyncio.sleep(0)

    await asyncio.wait_for(conn._idle_watchdog(), timeout=3.0)

    with pytest.raises(asyncio.CancelledError):
        await conn.generation_task
    assert conn.generation_task.cancelled()
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_watchdog_disabled_when_timeout_zero(monkeypatch):
    """Setting the timeout to 0 disables the watchdog entirely."""
    monkeypatch.setenv("VLLM_REALTIME_IDLE_TIMEOUT_S", "0")

    websocket = _make_websocket()
    websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect(code=1000))
    conn = RealtimeConnection(websocket, serving=Mock())
    assert conn._idle_timeout_s == 0

    # handle_connection() should run and tear down without ever spawning a
    # watchdog task.
    await asyncio.wait_for(conn.handle_connection(), timeout=2.0)

    assert conn._watchdog_task is None
    websocket.close.assert_not_awaited()
