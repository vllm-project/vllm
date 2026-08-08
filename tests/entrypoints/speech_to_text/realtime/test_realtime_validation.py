# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import numpy as np
import pybase64 as base64
import pytest
import websockets

from tests.entrypoints.speech_to_text.conftest import add_attention_backend
from tests.utils import ROCM_ENV_OVERRIDES, ROCM_EXTRA_ARGS, RemoteOpenAIServer
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

# Increase engine iteration timeout for ROCm where first-use JIT compilation
# can exceed the default 60s, causing a silent deadlock in feed_tokens.
REALTIME_ENV_OVERRIDES = {
    **ROCM_ENV_OVERRIDES,
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "600",
}

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
] + ROCM_EXTRA_ARGS

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def _get_websocket_url(server: RemoteOpenAIServer) -> str:
    """Convert HTTP URL to WebSocket URL for realtime endpoint."""
    http_url = server.url_root
    ws_url = http_url.replace("http://", "ws://")
    return f"{ws_url}/v1/realtime"


async def receive_event(ws, timeout: float = 60.0) -> dict:
    """Receive and parse JSON event from WebSocket."""
    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(message)


async def send_event(ws, event: dict) -> None:
    """Send JSON event to WebSocket."""
    await ws.send(json.dumps(event))


@pytest.fixture
def mary_had_lamb_audio_chunks() -> list[str]:
    """Audio split into ~1 second chunks for streaming."""
    path = AudioAsset("mary_had_lamb").get_local_path()
    audio, _ = load_audio(str(path), sr=16000, mono=True)

    # Split into ~0.1 second chunks (1600 samples at 16kHz)
    chunk_size = 1600
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()
        chunks.append(base64.b64encode(chunk_bytes).decode("utf-8"))

    return chunks


async def _start_session(
    ws, model_name: str, timestamp_granularities: list[str] | None = None
) -> dict:
    """Open a session and return the session.updated acknowledgement."""
    event = await receive_event(ws, timeout=30.0)
    assert event["type"] == "session.created"

    update: dict = {"type": "session.update", "model": model_name}
    if timestamp_granularities is not None:
        update["timestamp_granularities"] = timestamp_granularities
    await send_event(ws, update)

    event = await receive_event(ws, timeout=10.0)
    assert event["type"] == "session.updated"
    return event


async def _stream_utterance(ws, chunks: list[str], timeout: float = 60.0) -> list[dict]:
    """Stream one utterance and return every event up to transcription.done."""
    await send_event(ws, {"type": "input_audio_buffer.commit"})
    for chunk in chunks:
        await send_event(ws, {"type": "input_audio_buffer.append", "audio": chunk})
    await send_event(ws, {"type": "input_audio_buffer.commit", "final": True})

    events = []
    while True:
        event = await receive_event(ws, timeout=timeout)
        if event["type"] == "error":
            pytest.fail(f"Received error: {event}")
        events.append(event)
        if event["type"] == "transcription.done":
            return events


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_segment_timestamps(
    model_name, mary_had_lamb_audio_chunks, rocm_aiter_fa_attention
):
    """Segment timestamps are opt-in, aligned, and invisible when not asked for.

    The alignment bound is the point of this test: the token index of a
    generated token leads the audio it describes by the streaming prefix
    minus the left pad minus the transcription delay. Subtracting only the
    delay - as https://github.com/vllm-project/vllm/issues/39735 proposes -
    leaves the 32-frame left pad in and puts every timestamp 2.56 s late,
    which the upper bound catches.
    """
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    chunk_duration_s = 1600 / 16000
    duration_s = len(mary_had_lamb_audio_chunks) * chunk_duration_s

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)

        # --- Not opted in: the wire must be byte-identical to before ------
        async with websockets.connect(ws_url) as ws:
            ack = await _start_session(ws, model_name)
            assert ack["timestamp_granularities"] == []

            # (ROCm) generous timeout: first use triggers aiter JIT.
            events = await _stream_utterance(
                ws, mary_had_lamb_audio_chunks, timeout=600.0
            )
            done = events[-1]
            assert set(done) == {"type", "text", "usage"}
            assert all("segments" not in event for event in events)
            baseline_text = done["text"]

        # --- Opted in: same text, plus timestamps -------------------------
        async with websockets.connect(ws_url) as ws:
            ack = await _start_session(ws, model_name, ["segment"])
            assert ack["timestamp_granularities"] == ["segment"]

            events = await _stream_utterance(ws, mary_had_lamb_audio_chunks)
            done = events[-1]
            segments = done["segments"]

            # Opting in must not change what was transcribed.
            assert done["text"] == baseline_text
            assert segments

            # done repeats the deltas' segments, plus the trailing segment
            # that generation ended before any delta could carry.
            streamed = [
                segment
                for event in events
                if event["type"] == "transcription.delta"
                for segment in event["segments"]
            ]
            assert segments[: len(streamed)] == streamed
            assert 0 <= len(segments) - len(streamed) <= 1

            ends = [segment["end"] for segment in segments]
            assert ends == sorted(ends)
            assert all(end >= 0.08 for end in ends)
            assert all(abs(end / 0.08 - round(end / 0.08)) < 1e-6 for end in ends)

            # Bounded on both sides: +32 frames of left pad would overshoot,
            # a negative offset would undershoot.
            assert 0.5 * duration_s <= ends[-1] <= duration_s + 0.5

            # Entries are emission groups, so there are at most as many as
            # there are words, and together they reconstruct the transcript.
            assert len(segments) <= len(baseline_text.split())
            reconstructed = "".join(segment["text"] for segment in segments)
            assert baseline_text.endswith(reconstructed)
            assert len(reconstructed) >= 0.9 * len(baseline_text)

        # --- The clock restarts on every utterance ------------------------
        async with websockets.connect(ws_url) as ws:
            await _start_session(ws, model_name, ["segment"])
            short_chunks = mary_had_lamb_audio_chunks[:40]

            first = await _stream_utterance(ws, short_chunks)
            second = await _stream_utterance(ws, short_chunks)

            assert first[-1]["segments"]
            assert second[-1]["segments"]
            # Not "greater than the first utterance's last end": each commit
            # is a new engine request with a fresh prompt and left pad.
            assert second[-1]["segments"][0]["end"] < 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_multi_chunk_streaming(
    model_name, mary_had_lamb_audio_chunks, rocm_aiter_fa_attention
):
    """Test streaming multiple audio chunks before committing."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await send_event(ws, {"type": "session.update", "model": model_name})

            # Wait for the server to acknowledge the session update.
            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "session.updated"

            # (ROCm) Warm-up: send a non-final commit (required to start
            # transcription) with a small audio chunk to trigger aiter
            # compilation on first use.
            await send_event(ws, {"type": "input_audio_buffer.commit"})
            await send_event(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": mary_had_lamb_audio_chunks[0],
                },
            )
            await send_event(ws, {"type": "input_audio_buffer.commit", "final": True})

            # (ROCm) Drain all warm-up responses with generous timeout for
            # JIT compilation
            warmup_done = False
            while not warmup_done:
                event = await receive_event(ws, timeout=600.0)
                if event["type"] in ("transcription.done", "error"):
                    warmup_done = True

            # Now send the real test audio
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            # Send multiple audio chunks
            for chunk in mary_had_lamb_audio_chunks:
                await send_event(
                    ws, {"type": "input_audio_buffer.append", "audio": chunk}
                )

            # Send commit to end
            await send_event(ws, {"type": "input_audio_buffer.commit", "final": True})

            # Collect transcription deltas
            full_text = ""
            done_received = False

            while not done_received:
                event = await receive_event(ws, timeout=60.0)

                if event["type"] == "transcription.delta":
                    full_text += event["delta"]
                elif event["type"] == "transcription.done":
                    done_received = True
                    assert "text" in event
                elif event["type"] == "error":
                    pytest.fail(f"Received error: {event}")

            # Verify transcription contains expected content
            assert event["type"] == "transcription.done"
            assert event["text"] == full_text
            assert full_text == (
                " First words I spoke in the original phonograph."
                " A little piece of practical poetry. Mary had a little lamb,"
                " it sleeps with quite a flow, and everywhere that Mary went,"
                " the lamb was sure to go."
            ) or full_text == (
                " First words I spoke in the original phonograph."
                " A little piece of practical poetry. Mary had a little lamb,"
                " it squeaked with quite a flow, and everywhere that Mary went,"
                " the lamb was sure to go."
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_empty_commit_does_not_crash_engine(
    model_name, mary_had_lamb_audio_chunks, rocm_aiter_fa_attention
):
    """Test that committing without audio does not crash the engine.

    Regression test for https://github.com/vllm-project/vllm/issues/34532.
    An empty commit (no prior input_audio_buffer.append) used to trigger
    ``AssertionError: For realtime you must provide a multimodal_embedding
    at every step`` which killed the entire engine process, disconnecting
    every connected client.
    """
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)

        # --- First connection: empty commit (no audio appended) ----------
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await send_event(ws, {"type": "session.update", "model": model_name})

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "session.updated"

            # Start generation without sending any audio
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            # Immediately signal end-of-audio
            await send_event(ws, {"type": "input_audio_buffer.commit", "final": True})

            # We should get *some* response (error or empty transcription),
            # but the engine must NOT crash.
            # (ROCm) Use generous timeout for first request (aiter JIT compilation)
            event = await receive_event(ws, timeout=360.0)
            assert event["type"] in (
                "error",
                "transcription.done",
                "transcription.delta",
            )

        # --- Second connection: normal transcription ---------------------
        # Verifies the engine is still alive after the empty commit above.
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await send_event(ws, {"type": "session.update", "model": model_name})

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "session.updated"

            # Start transcription
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            for chunk in mary_had_lamb_audio_chunks:
                await send_event(
                    ws, {"type": "input_audio_buffer.append", "audio": chunk}
                )

            await send_event(ws, {"type": "input_audio_buffer.commit", "final": True})

            done_received = False
            while not done_received:
                event = await receive_event(ws, timeout=60.0)
                if event["type"] == "transcription.done":
                    done_received = True
                elif event["type"] == "error":
                    pytest.fail(f"Engine error after empty commit: {event}")
            assert done_received


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_session_update_invalid_model_returns_error(
    model_name, rocm_aiter_fa_attention
):
    """Test that session.update with an invalid model returns an error."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send session.update with a model that doesn't exist
            await send_event(
                ws,
                {"type": "session.update", "model": "nonexistent-model"},
            )

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "error"
            assert "nonexistent-model" in event["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_commit_without_session_update_returns_error(
    model_name, rocm_aiter_fa_attention
):
    """Test that committing before validating the model returns an error
    and does not fall through to processing."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send commit without sending session.update first
            await send_event(
                ws,
                {"type": "input_audio_buffer.commit", "final": True},
            )

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "error"
            assert "model_not_validated" in event.get("code", "")
