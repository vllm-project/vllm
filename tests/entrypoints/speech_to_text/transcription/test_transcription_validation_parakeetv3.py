# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import math

import numpy as np
import pytest
import soundfile as sf

from tests.entrypoints.speech_to_text.conftest import add_attention_backend
from tests.utils import ROCM_ENV_OVERRIDES, RemoteOpenAIServer
from vllm.multimodal.media.audio import load_audio

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

SERVER_ARGS = [
    "--max-model-len",
    "512",
    "--max-num-batched-tokens",
    "512",
    "--max-num-seqs",
    "1",
]


def make_long_audio(file, *, repeats: int) -> tuple[io.BytesIO, int]:
    file.seek(0)
    audio, sr = load_audio(file)
    # Add small silence after each repeat for repeatable chunk splitting.
    audio = np.pad(audio, (0, int(sr * 0.1)))
    repeated_audio = np.tile(audio, repeats)

    buffer = io.BytesIO()
    buffer.name = "long_audio.wav"
    sf.write(buffer, repeated_audio, sr, format="WAV")
    buffer.seek(0)

    expected_seconds = math.ceil(len(repeated_audio) / sr)
    return buffer, expected_seconds


@pytest.mark.asyncio
async def test_basic_audio(mary_had_lamb, rocm_aiter_fa_attention):
    server_args = [*SERVER_ARGS]
    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        max_wait_seconds=480,
        env_dict=ROCM_ENV_OVERRIDES,
    ) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0,
        )

    out = json.loads(transcription)
    assert "Mary had a little lamb" in out["text"], out["text"]
    assert out["usage"]["seconds"] == 16


@pytest.mark.asyncio
async def test_streaming_audio_strips_eos(mary_had_lamb, rocm_aiter_fa_attention):
    server_args = [*SERVER_ARGS]
    add_attention_backend(server_args, rocm_aiter_fa_attention)

    transcription = ""
    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        max_wait_seconds=480,
        env_dict=ROCM_ENV_OVERRIDES,
    ) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=mary_had_lamb,
            language="en",
            temperature=0.0,
            stream=True,
        )

        async for chunk in res:
            transcription += chunk.choices[0]["delta"]["content"]

    assert "Mary had a little lamb" in transcription
    assert "<|endoftext|>" not in transcription


@pytest.mark.asyncio
async def test_long_audio(mary_had_lamb, rocm_aiter_fa_attention):
    server_args = [*SERVER_ARGS]
    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        max_wait_seconds=480,
        env_dict=ROCM_ENV_OVERRIDES,
    ) as remote_server:
        client = remote_server.get_async_client()
        long_audio, expected_seconds = make_long_audio(mary_had_lamb, repeats=3)
        transcription = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=long_audio,
            language="en",
            response_format="text",
            temperature=0.0,
        )

    out = json.loads(transcription)
    out_text = out["text"]
    count = out_text.lower().count("mary had a little lamb")
    assert count == 3, f"Expected 3 repeats, found {count}: {out_text!r}"
    assert out["usage"]["seconds"] == expected_seconds
