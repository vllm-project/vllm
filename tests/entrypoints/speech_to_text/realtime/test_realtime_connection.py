# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from typing import cast

import numpy as np
import pybase64 as base64
import pytest

from vllm import envs
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection


class DummyWebSocket:
    def __init__(self):
        self.close_calls: list[tuple[int, str | None]] = []
        self.sent: list[dict] = []

    async def close(self, code: int = 1000, reason: str | None = None):
        self.close_calls.append((code, reason))

    async def send_text(self, data: str):
        self.sent.append(json.loads(data))


class DummyServing:
    def _is_model_supported(self, model: str | None) -> bool:
        return True


def _connection(
    monkeypatch: pytest.MonkeyPatch, idle_timeout_s: float = 0.05
) -> RealtimeConnection:
    monkeypatch.setattr(
        envs, "VLLM_REALTIME_AUDIO_IDLE_TIMEOUT_S", idle_timeout_s, raising=False
    )
    conn = RealtimeConnection(DummyWebSocket(), DummyServing())  # type: ignore[arg-type]
    conn._is_connected = True
    conn._is_model_validated = True
    return conn


@pytest.mark.asyncio
async def test_idle_watchdog_forces_final_and_closes(monkeypatch):
    conn = _connection(monkeypatch)

    async def wait_for_final():
        assert await conn.audio_queue.get() is None

    conn.generation_task = asyncio.create_task(wait_for_final())
    conn._reset_idle_watchdog()
    assert conn.idle_watchdog_task is not None

    await asyncio.wait_for(conn.idle_watchdog_task, timeout=1.0)

    websocket = cast(DummyWebSocket, conn.websocket)
    assert conn.generation_task.done()
    assert websocket.close_calls == [(1000, "Realtime session idle timeout.")]
    assert not conn._is_connected


@pytest.mark.asyncio
async def test_final_commit_cancels_idle_watchdog(monkeypatch):
    conn = _connection(monkeypatch)
    conn._reset_idle_watchdog()
    watchdog_task = conn.idle_watchdog_task
    assert watchdog_task is not None

    await conn.handle_event({"type": "input_audio_buffer.commit", "final": True})
    await asyncio.sleep(0)
    await asyncio.sleep(0.1)

    websocket = cast(DummyWebSocket, conn.websocket)
    assert conn.idle_watchdog_task is None
    assert watchdog_task.done()
    assert not websocket.close_calls


@pytest.mark.asyncio
async def test_noop_commit_does_not_refresh_idle_watchdog(monkeypatch):
    conn = _connection(monkeypatch, idle_timeout_s=1.0)
    conn.generation_task = asyncio.create_task(asyncio.sleep(10.0))
    conn._reset_idle_watchdog()
    watchdog_task = conn.idle_watchdog_task
    assert watchdog_task is not None

    await conn.handle_event({"type": "input_audio_buffer.commit"})

    assert conn.idle_watchdog_task is watchdog_task
    await conn.cleanup()


@pytest.mark.asyncio
async def test_append_refreshes_idle_watchdog(monkeypatch):
    conn = _connection(monkeypatch, idle_timeout_s=1.0)
    conn._reset_idle_watchdog()
    watchdog_task = conn.idle_watchdog_task
    assert watchdog_task is not None

    audio = np.array([1], dtype=np.int16)
    await conn.handle_event(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio.tobytes()).decode("utf-8"),
        }
    )
    await asyncio.sleep(0)

    assert watchdog_task.done()
    assert conn.idle_watchdog_task is not None
    assert conn.idle_watchdog_task is not watchdog_task
    await conn.cleanup()


@pytest.mark.asyncio
async def test_idle_watchdog_can_be_disabled(monkeypatch):
    conn = _connection(monkeypatch, idle_timeout_s=0.0)

    conn._reset_idle_watchdog()

    assert conn.idle_watchdog_task is None
    await conn.cleanup()
