# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json
import warnings

import librosa
import numpy as np
import pytest
import websockets

from vllm.assets.audio import AudioAsset

from ...utils import RemoteOpenAIServer
from .conftest import add_attention_backend

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]

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
    audio, _ = librosa.load(str(path), sr=16000, mono=True)

    # Split into ~0.1 second chunks (1600 samples at 16kHz)
    chunk_size = 1600
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()
        chunks.append(base64.b64encode(chunk_bytes).decode("utf-8"))

    return chunks


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

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await send_event(ws, {"type": "session.update", "model": model_name})

            # Wait for the server to acknowledge the session update.
            try:
                while True:
                    event = await receive_event(ws, timeout=5.0)
                    if event["type"] == "session.updated":
                        break
            except TimeoutError:
                warnings.warn(
                    f"session.updated not received within {5.0}s after "
                    "session.update. The server may not implement this event.",
                    stacklevel=2,
                )

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
                event = await receive_event(ws, timeout=360.0)
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

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = _get_websocket_url(remote_server)

        # --- First connection: empty commit (no audio appended) ----------
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await send_event(ws, {"type": "session.update", "model": model_name})

            try:
                while True:
                    event = await receive_event(ws, timeout=5.0)
                    if event["type"] == "session.updated":
                        break
            except TimeoutError:
                warnings.warn(
                    f"session.updated not received within {5.0}s after "
                    "session.update. The server may not implement this event.",
                    stacklevel=2,
                )

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

            try:
                while True:
                    event = await receive_event(ws, timeout=5.0)
                    if event["type"] == "session.updated":
                        break
            except TimeoutError:
                warnings.warn(
                    f"session.updated not received within {5.0}s after "
                    "session.update. The server may not implement this event.",
                    stacklevel=2,
                )

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
