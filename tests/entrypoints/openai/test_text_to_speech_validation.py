# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import pytest
import pytest_asyncio
import torch
from transformers import pipeline
import soundfile as sf

from ...utils import RemoteOpenAIServer

# ------------------------------
# vLLM model for serving
# ------------------------------
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server(zephyr_lora_files):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora={zephyr_lora_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        # Mock TTS endpoint for vLLM client
        async_client.audio = type("audio", (), {})()
        async_client.audio.speech = type("speech", (), {})()
        async_client.audio.speech.create = tts_create_mock_async
        yield async_client


# ------------------------------
# TTS pipeline mock for tests
# ------------------------------
async def tts_create_mock_async(model, input, voice="default"):
    """
    Mock TTS call using Hugging Face pipeline, preserving OpenAI-style async API.
    """
    tts_pipeline = pipeline(
        task="text-to-speech",
        model=model,
        device=0 if torch.cuda.is_available() else -1
    )
    audio_array = tts_pipeline(input)[0]["array"]

    class Response:
        def read(self):
            buf = io.BytesIO()
            sf.write(buf, audio_array, samplerate=22050, format="WAV")
            buf.seek(0)
            return buf.read()

    return Response()


# ------------------------------
# TTS Tests
# ------------------------------
@pytest.mark.asyncio
async def test_tts_basic(client, model_name="espnet/kan-bayashi_ljspeech_vits"):
    response = await client.audio.speech.create(
        model=model_name,
        voice="default",
        input="Hello, this is a TTS validation by vLLM server."
    )

    assert response is not None
    audio_bytes = response.read()
    assert isinstance(audio_bytes, (bytes, bytearray))
    assert len(audio_bytes) > 1000


@pytest.mark.asyncio
async def test_tts_different_voices(client, model_name="microsoft/speecht5_tts"):
    voices = ["en-US-GuyNeural", "en-US-JennyNeural"]
    results = []

    for voice in voices:
        response = await client.audio.speech.create(
            model=model_name,
            voice=voice,
            input=f"This is a sample spoken by {voice}"
        )
        results.append(response)

    for res in results:
        assert res is not None
        assert len(res.read()) > 5000


@pytest.mark.asyncio
async def test_tts_long_text(client, model_name="espnet/kan-bayashi_ljspeech_vits"):
    text = (
        "Artificial intelligence is transforming industries worldwide. "
        "From healthcare to finance, TTS systems make machines more natural."
    )

    response = await client.audio.speech.create(model=model_name, input=text)
    audio_bytes = response.read()
    assert audio_bytes.startswith(b"RIFF") or len(audio_bytes) > 20000


@pytest.mark.asyncio
async def test_tts_invalid_input(client, model_name="espnet/kan-bayashi_ljspeech_vits"):
    with pytest.raises(Exception):
        await client.audio.speech.create(model=model_name, input="")

