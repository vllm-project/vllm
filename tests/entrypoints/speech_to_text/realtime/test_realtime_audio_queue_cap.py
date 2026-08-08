# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the realtime WebSocket audio path bounds memory.

Covers both the per-connection audio_queue cap and the cumulative
byte tracking against VLLM_MAX_AUDIO_CLIP_FILESIZE_MB.
"""

import asyncio
import json

import pybase64 as base64
import pytest

from vllm import envs
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection

MAX_AUDIO_QUEUE_SIZE = envs.VLLM_MAX_REALTIME_AUDIO_QUEUE_SIZE


class FakeWebSocket:
    def __init__(self):
        self.sent: list[str] = []

    async def accept(self):
        pass

    async def send_text(self, data: str):
        self.sent.append(data)


class StubServing:
    def _is_model_supported(self, model):
        return True

    def create_error_response(self, **kw):
        return None


class StubModelCls:
    realtime_max_tokens = 100


class StubEngineClient:
    async def generate(self, prompt, sampling_params, request_id):
        # Empty result stream: generation completes with no output.
        for _ in ():
            yield _


class StubServingWithEngine(StubServing):
    def __init__(self):
        self.model_cls = StubModelCls()
        self.engine_client = StubEngineClient()


def _append_event(num_samples: int = 1600) -> dict:
    payload = b"\x00\x01" * num_samples
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(payload).decode("ascii"),
    }


def _make_conn() -> tuple[RealtimeConnection, FakeWebSocket]:
    fake_ws = FakeWebSocket()
    conn = RealtimeConnection(fake_ws, StubServing())
    conn._is_connected = True
    conn._is_model_validated = True
    return conn, fake_ws


def _make_conn_with_engine() -> tuple[RealtimeConnection, FakeWebSocket]:
    fake_ws = FakeWebSocket()
    conn = RealtimeConnection(fake_ws, StubServingWithEngine())
    conn._is_connected = True
    conn._is_model_validated = True
    return conn, fake_ws


@pytest.mark.asyncio
async def test_audio_queue_is_capped():
    conn, _ = _make_conn()

    for _ in range(MAX_AUDIO_QUEUE_SIZE + 5):
        await conn.handle_event(_append_event())

    assert conn.audio_queue.qsize() == MAX_AUDIO_QUEUE_SIZE


@pytest.mark.asyncio
async def test_overflow_appends_emit_buffer_full_error():
    conn, fake_ws = _make_conn()

    overflow = 3
    for _ in range(MAX_AUDIO_QUEUE_SIZE + overflow):
        await conn.handle_event(_append_event())

    errors = [json.loads(m) for m in fake_ws.sent if "error" in m]
    buffer_full = [e for e in errors if e.get("code") == "buffer_full"]
    assert len(buffer_full) == overflow


@pytest.mark.asyncio
async def test_draining_queue_allows_new_appends():
    conn, _ = _make_conn()

    for _ in range(MAX_AUDIO_QUEUE_SIZE):
        await conn.handle_event(_append_event())
    assert conn.audio_queue.qsize() == MAX_AUDIO_QUEUE_SIZE

    while not conn.audio_queue.empty():
        conn.audio_queue.get_nowait()

    await conn.handle_event(_append_event())
    assert conn.audio_queue.qsize() == 1


@pytest.mark.asyncio
async def test_cumulative_bytes_rejects_past_threshold():
    conn, fake_ws = _make_conn()
    conn._max_audio_filesize_mb = 0.01

    # Each chunk is 1600 PCM16 samples = 3200 bytes = ~3.125 KB.
    # The 0.01 MB (10.24 KB) threshold is crossed on the 4th chunk.
    for _ in range(10):
        await conn.handle_event(_append_event())

    errors = [json.loads(m) for m in fake_ws.sent if "error" in m]
    assert any(e.get("code") == "invalid_audio" for e in errors)
    assert conn._accumulated_audio_bytes > 0.01 * 1024**2


@pytest.mark.asyncio
async def test_cumulative_bytes_reset_by_generation():
    conn, _ = _make_conn_with_engine()

    for _ in range(5):
        await conn.handle_event(_append_event())
    assert conn._accumulated_audio_bytes == 5 * 3200

    # Drive the real reset path: _run_generation clears the queue and the
    # byte counter in its finally block.
    audio_stream = conn.audio_stream_generator()
    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
    await conn._run_generation(audio_stream, input_stream)

    assert conn._accumulated_audio_bytes == 0
    assert conn.audio_queue.empty()


@pytest.mark.asyncio
async def test_rejected_chunks_do_not_count_toward_byte_limit():
    conn, _ = _make_conn()

    for _ in range(MAX_AUDIO_QUEUE_SIZE):
        await conn.handle_event(_append_event())
    at_capacity = conn._accumulated_audio_bytes
    assert at_capacity == MAX_AUDIO_QUEUE_SIZE * 3200

    # Overflow appends are rejected; their bytes must not be counted.
    for _ in range(3):
        await conn.handle_event(_append_event())
    assert conn._accumulated_audio_bytes == at_capacity


@pytest.mark.asyncio
async def test_cleanup_enqueues_sentinel_on_full_queue():
    conn, _ = _make_conn()

    for _ in range(MAX_AUDIO_QUEUE_SIZE):
        await conn.handle_event(_append_event())
    assert conn.audio_queue.full()

    # cleanup must not raise QueueFull and must still enqueue the sentinel.
    await conn.cleanup()

    items = []
    while not conn.audio_queue.empty():
        items.append(conn.audio_queue.get_nowait())
    assert items[-1] is None


@pytest.mark.asyncio
async def test_commit_final_enqueues_sentinel_on_full_queue():
    conn, _ = _make_conn()

    for _ in range(MAX_AUDIO_QUEUE_SIZE):
        await conn.handle_event(_append_event())
    assert conn.audio_queue.full()

    # A final commit on a full queue must not raise QueueFull.
    await conn.handle_event({"type": "input_audio_buffer.commit", "final": True})

    items = []
    while not conn.audio_queue.empty():
        items.append(conn.audio_queue.get_nowait())
    assert items[-1] is None
