# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming transcription must report the input audio duration.

Clients that bill or budget by audio length cannot recover the duration from the
transcribed text, so every stream chunk carries it.
"""

import json
from collections.abc import AsyncGenerator

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "openai/whisper-large-v3-turbo"


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.2",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


async def _post_transcription(server, audio_file, data) -> dict:
    audio_file.seek(0)
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            server.url_for("v1/audio/transcriptions"),
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"},
            data=data,
            files={"file": audio_file},
        )
    response.raise_for_status()
    return response.json()


async def _stream_transcription(server, audio_file, data) -> AsyncGenerator[dict, None]:
    audio_file.seek(0)
    async with (
        httpx.AsyncClient() as http_client,
        http_client.stream(
            "POST",
            server.url_for("v1/audio/transcriptions"),
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"},
            data=data,
            files={"file": audio_file},
        ) as response,
    ):
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                break
            yield json.loads(payload)


@pytest.mark.asyncio
async def test_stream_chunks_report_audio_duration(server, winning_call):
    request = {"model": MODEL_NAME, "language": "en", "temperature": 0.0}

    stream = _stream_transcription(server, winning_call, {**request, "stream": True})
    chunks = [chunk async for chunk in stream]
    verbose = await _post_transcription(
        server, winning_call, {**request, "response_format": "verbose_json"}
    )

    assert chunks
    durations = [chunk.get("duration") for chunk in chunks]
    assert all(isinstance(duration, float) and duration > 0 for duration in durations)
    # A single stream transcribes one request's audio, so every chunk agrees,
    # and the value matches the non-streaming `verbose_json` response.
    assert set(durations) == {verbose["duration"]}
