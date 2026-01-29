# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json

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

MODEL_NAME = "mistralai/Voxtral-Mini-3B-Realtime-2602"


def audio_to_base64_pcm16(path: str, target_sr: int = 16000) -> str:
    """Load audio file, convert to PCM16 @ target sample rate, base64 encode."""
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return base64.b64encode(audio_bytes).decode("utf-8")


def get_websocket_url(server: RemoteOpenAIServer) -> str:
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
def mary_had_lamb_audio() -> str:
    """Full audio as base64 PCM16."""
    path = AudioAsset("mary_had_lamb").get_local_path()
    return audio_to_base64_pcm16(str(path))


@pytest.fixture
def mary_had_lamb_audio_chunks() -> list[str]:
    """Audio split into ~1 second chunks for streaming."""
    path = AudioAsset("mary_had_lamb").get_local_path()
    audio, _ = librosa.load(str(path), sr=16000, mono=True)

    # Split into ~1 second chunks (16000 samples at 16kHz)
    chunk_size = 16000
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()
        chunks.append(base64.b64encode(chunk_bytes).decode("utf-8"))

    return chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_session_created_on_connect(model_name, rocm_aiter_fa_attention):
    """Verify session.created event is received on connection."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Should receive session.created immediately
            event = await receive_event(ws, timeout=30.0)

            assert event["type"] == "session.created"
            assert "id" in event
            assert event["id"].startswith("sess-")
            assert "created" in event


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic_transcription(
    model_name, mary_had_lamb_audio, rocm_aiter_fa_attention
):
    """Test full transcription flow with append, commit, delta, and done events."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send audio append
            await send_event(
                ws,
                {"type": "input_audio_buffer.append", "audio": mary_had_lamb_audio},
            )

            # Send commit to start transcription
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            # Collect transcription deltas
            full_text = ""
            done_received = False

            while not done_received:
                event = await receive_event(ws, timeout=60.0)

                if event["type"] == "transcription.delta":
                    assert "delta" in event
                    full_text += event["delta"]
                elif event["type"] == "transcription.done":
                    done_received = True
                    assert "text" in event
                    assert "usage" in event
                elif event["type"] == "error":
                    pytest.fail(f"Received error: {event}")

            # Verify transcription contains expected content
            assert "Mary" in full_text or "mary" in full_text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_invalid_json_error(model_name, rocm_aiter_fa_attention):
    """Test that invalid JSON returns an error with code=invalid_json."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send invalid JSON
            await ws.send("this is not valid json{{{")

            # Should receive error event
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "error"
            assert event["code"] == "invalid_json"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_empty_audio_error(model_name, rocm_aiter_fa_attention):
    """Test that empty audio returns an error with code=invalid_audio."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send empty audio (base64 of empty bytes)
            empty_audio = base64.b64encode(b"").decode("utf-8")
            await send_event(
                ws, {"type": "input_audio_buffer.append", "audio": empty_audio}
            )

            # Should receive error event
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "error"
            assert event["code"] == "invalid_audio"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_unknown_event_type_error(model_name, rocm_aiter_fa_attention):
    """Test that unknown event type returns an error with code=unknown_event."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send unknown event type
            await send_event(ws, {"type": "unknown.event.type"})

            # Should receive error event
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "error"
            assert event["code"] == "unknown_event"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_invalid_audio_data_error(model_name, rocm_aiter_fa_attention):
    """Test that invalid base64 audio data returns an error."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send invalid base64 data
            await send_event(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": "this is not valid base64!!!",
                },
            )

            # Should receive error event
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "error"
            assert event["code"] == "invalid_audio"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_multi_chunk_streaming(
    model_name, mary_had_lamb_audio_chunks, rocm_aiter_fa_attention
):
    """Test streaming multiple audio chunks before committing."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send multiple audio chunks
            for chunk in mary_had_lamb_audio_chunks:
                await send_event(
                    ws, {"type": "input_audio_buffer.append", "audio": chunk}
                )
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)

            # Send commit to start transcription
            await send_event(ws, {"type": "input_audio_buffer.commit"})

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
            assert "Mary" in full_text or "mary" in full_text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_session_update(model_name, rocm_aiter_fa_attention):
    """Test that session.update event is handled without error."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send session.update
            await send_event(ws, {"type": "session.update", "model": model_name})

            # Send a valid audio append to verify connection still works
            # Generate a short silence as test audio
            silence = np.zeros(1600, dtype=np.float32)  # 0.1 seconds at 16kHz
            silence_int16 = (silence * 32767).astype(np.int16)
            silence_b64 = base64.b64encode(silence_int16.tobytes()).decode("utf-8")

            await send_event(
                ws, {"type": "input_audio_buffer.append", "audio": silence_b64}
            )

            # If we get here without error, session.update was handled correctly
            # The silence append should succeed (no error expected for valid audio)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_final_commit_flag(
    model_name, mary_had_lamb_audio, rocm_aiter_fa_attention
):
    """Test commit with final=True flag behavior."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            # Receive session.created
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send audio append
            await send_event(
                ws,
                {"type": "input_audio_buffer.append", "audio": mary_had_lamb_audio},
            )

            # Send commit with final=True
            await send_event(ws, {"type": "input_audio_buffer.commit", "final": True})

            # The final flag signals that audio input is finished
            # This should set _is_input_finished on the connection
            # No immediate response expected from just setting the flag
            # (transcription starts on commit without final flag)
