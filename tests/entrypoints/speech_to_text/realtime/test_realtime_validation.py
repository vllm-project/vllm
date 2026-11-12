# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import signal
import threading
import warnings
from types import SimpleNamespace

import numpy as np
import pybase64 as base64
import pytest
import websockets
from fastapi import FastAPI

from tests.utils import ROCM_EXTRA_ARGS, RemoteOpenAIServer
from vllm.assets.audio import AudioAsset
from vllm.entrypoints.launchers.launcher import create_server_socket, serve_http
from vllm.entrypoints.speech_to_text.realtime.api_router import (
    router as realtime_router,
)
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection
from vllm.multimodal.media.audio import load_audio
from vllm.outputs import CompletionOutput, RequestOutput

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
] + ROCM_EXTRA_ARGS

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def test_realtime_generation_preserves_watermarking(monkeypatch):
    transcription_outputs = []

    class EngineClient:
        def generate(self, **kwargs):
            async def outputs():
                output = SimpleNamespace(
                    outputs=[],
                    prompt_token_ids=[],
                    is_watermarked=kwargs["sampling_params"].watermarking,
                )
                transcription_outputs.append(output)
                yield output

            return outputs()

    async def streaming_input():
        chunk: np.ndarray
        for chunk in ():
            yield chunk

    async def send(_event):
        return None

    connection = object.__new__(RealtimeConnection)
    connection.connection_id = "test"
    connection.serving = SimpleNamespace(
        model_cls=SimpleNamespace(realtime_max_tokens=1),
        engine_client=EngineClient(),
    )
    connection.audio_queue = asyncio.Queue()
    connection._is_connected = True
    monkeypatch.setattr(connection, "send", send)

    asyncio.run(connection._run_generation(streaming_input(), asyncio.Queue()))

    assert len(transcription_outputs) == 1
    assert transcription_outputs[0].is_watermarked is None


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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_multi_chunk_streaming(model_name, mary_had_lamb_audio_chunks):
    """Test streaming multiple audio chunks before committing."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

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
    model_name, mary_had_lamb_audio_chunks
):
    """Test that committing without audio does not crash the engine.

    Regression test for https://github.com/vllm-project/vllm/issues/34532.
    An empty commit (no prior input_audio_buffer.append) used to trigger
    `AssertionError: For realtime you must provide a multimodal_embedding
    at every step` which killed the entire engine process, disconnecting
    every connected client.
    """
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_session_update_invalid_model_returns_error(model_name):
    """Test that session.update with an invalid model returns an error."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
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
async def test_commit_without_session_update_returns_error(model_name):
    """Test that committing before validating the model returns an error
    and does not fall through to processing."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
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


async def _connect_until_refused(ws_url: str, timeout: float = 30.0):
    """Open sessions until one is refused; the SIGTERM handler runs async."""
    async with asyncio.timeout(timeout):
        while True:
            async with websockets.connect(ws_url) as ws:
                try:
                    event = await receive_event(ws, timeout=timeout)
                except websockets.ConnectionClosed:
                    return ws.close_code, ws.close_reason
                assert event["type"] == "session.created"
            await asyncio.sleep(0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_realtime_shutdown_refuses_new_sessions(
    model_name, mary_had_lamb_audio_chunks
):
    """After SIGTERM, new sessions are closed with 1001 (Going Away), and a
    session opened before SIGTERM is still served."""
    server_args = [
        "--enforce-eager",
        "--max-model-len",
        "2048",
        "--shutdown-timeout",
        "60",
        *MISTRAL_FORMAT_ARGS,
    ]

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as open_ws:
            assert (await receive_event(open_ws, timeout=30.0))[
                "type"
            ] == "session.created"
            await send_event(open_ws, {"type": "session.update", "model": model_name})
            await send_event(open_ws, {"type": "input_audio_buffer.commit"})
            for chunk in mary_had_lamb_audio_chunks[:20]:
                await send_event(
                    open_ws, {"type": "input_audio_buffer.append", "audio": chunk}
                )
            first = await receive_event(open_ws, timeout=600.0)
            assert first["type"] == "transcription.delta", first

            remote_server.proc.send_signal(signal.SIGTERM)

            code, reason = await _connect_until_refused(ws_url)
            assert (code, reason) == (1001, "Server shutting down")

            await send_event(open_ws, {"type": "not.an.event"})
            event = await receive_event(open_ws, timeout=30.0)
            while event["type"] == "transcription.delta":
                event = await receive_event(open_ws, timeout=30.0)
            assert event == {
                "type": "error",
                "error": "Unknown event type: not.an.event",
                "code": "unknown_event",
            }


class _DrainingEngineClient:
    """Engine stand-in whose drain lasts until the test releases it."""

    def __init__(self):
        self.vllm_config = SimpleNamespace(shutdown_timeout=30)
        self.errored = False
        self.is_running = True
        self.should_keep_api_server_alive = False
        self.drain_started = threading.Event()
        self.drain_released = threading.Event()

    def shutdown(self, timeout=None):
        self.drain_started.set()
        self.drain_released.wait(timeout=30)

    def generate(self, prompt, sampling_params, request_id):
        async def outputs():
            async for _ in prompt:
                completion = CompletionOutput(
                    index=0,
                    text=" lamb",
                    token_ids=[7],
                    cumulative_logprob=None,
                    logprobs=None,
                )
                yield RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    prompt_token_ids=[1],
                    prompt_logprobs=None,
                    outputs=[completion],
                    finished=False,
                )

        return outputs()

    async def abort(self, request_id):
        pass


class _RealtimeServing:
    def __init__(self, engine_client):
        self.engine_client = engine_client
        self.model_cls = SimpleNamespace(realtime_max_tokens=1)

    def _is_model_supported(self, model):
        return model == MODEL_NAME

    async def transcribe_realtime(self, audio_stream, input_stream):
        async for chunk in audio_stream:
            yield chunk


async def _wait_for(condition, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_realtime_refuses_new_sessions_while_draining():
    """After SIGTERM, a new session is closed with 1001 (Going Away) while a
    session opened before the drain keeps transcribing until it finishes."""
    engine_client = _DrainingEngineClient()
    app = FastAPI()
    app.include_router(realtime_router)
    app.state.engine_client = engine_client
    app.state.openai_serving_realtime = _RealtimeServing(engine_client)

    sock = create_server_socket(("127.0.0.1", 0), reuse_port=False)
    port = sock.getsockname()[1]
    url = f"ws://127.0.0.1:{port}/v1/realtime"
    audio = base64.b64encode(np.ones(160, dtype=np.int16).tobytes()).decode()

    loop = asyncio.get_running_loop()
    serve_task = asyncio.create_task(
        serve_http(app, sock, host="127.0.0.1", port=port, log_level="warning")
    )
    try:
        await _wait_for(lambda: getattr(app.state, "server", None) is not None)
        await _wait_for(lambda: app.state.server.started)

        async with websockets.connect(url) as open_ws:
            assert (await receive_event(open_ws))["type"] == "session.created"
            await send_event(open_ws, {"type": "session.update", "model": MODEL_NAME})
            await send_event(open_ws, {"type": "input_audio_buffer.commit"})

            signal.raise_signal(signal.SIGTERM)
            await _wait_for(engine_client.drain_started.is_set)

            async with websockets.connect(url) as new_ws:
                with pytest.raises(websockets.ConnectionClosed):
                    await receive_event(new_ws)
                assert new_ws.close_code == 1001
                assert new_ws.close_reason == "Server shutting down"

            await send_event(
                open_ws, {"type": "input_audio_buffer.append", "audio": audio}
            )
            await send_event(
                open_ws, {"type": "input_audio_buffer.commit", "final": True}
            )
            delta = await receive_event(open_ws)
            done = await receive_event(open_ws)
            assert delta == {"type": "transcription.delta", "delta": " lamb"}
            assert done["type"] == "transcription.done"
            assert done["text"] == " lamb"

        engine_client.drain_released.set()
        await (await asyncio.wait_for(serve_task, timeout=10))
    finally:
        engine_client.drain_released.set()
        serve_task.cancel()
        loop.remove_signal_handler(signal.SIGTERM)
        loop.remove_signal_handler(signal.SIGINT)
