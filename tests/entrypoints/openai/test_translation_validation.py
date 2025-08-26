# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
# imports for guided decoding tests
import json

import httpx
import librosa
import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf

from vllm.assets.audio import AudioAsset

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/whisper-small"
SERVER_ARGS = ["--enforce-eager"]


@pytest.fixture
def foscolo():
    # Test translation it->en
    path = AudioAsset('azacinto_foscolo').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_non_asr_model(foscolo):
    # text to text model
    model_name = "JackFram/llama-68m"
    with RemoteOpenAIServer(model_name, SERVER_ARGS) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.translations.create(model=model_name,
                                                     file=foscolo,
                                                     temperature=0.0)
        err = res.error
        assert err["code"] == 400 and not res.text
        assert err["message"] == "The model does not support Translations API"


# NOTE: (NickLucche) the large-v3-turbo model was not trained on translation!
@pytest.mark.asyncio
async def test_basic_audio(foscolo, client):
    translation = await client.audio.translations.create(
        model=MODEL_NAME,
        file=foscolo,
        response_format="text",
        # TODO remove once language detection is implemented
        extra_body=dict(language="it"),
        temperature=0.0)
    out = json.loads(translation)['text'].strip().lower()
    assert "greek sea" in out


@pytest.mark.asyncio
async def test_audio_prompt(foscolo, client):
    # Condition whisper on starting text
    prompt = "Nor have I ever"
    transcription = await client.audio.translations.create(
        model=MODEL_NAME,
        file=foscolo,
        prompt=prompt,
        extra_body=dict(language="it"),
        response_format="text",
        temperature=0.0)
    out = json.loads(transcription)['text']
    assert "Nor will I ever touch the sacred" not in out
    assert prompt not in out


@pytest.mark.asyncio
async def test_streaming_response(foscolo, client, server):
    translation = ""
    res_no_stream = await client.audio.translations.create(
        model=MODEL_NAME,
        file=foscolo,
        response_format="json",
        extra_body=dict(language="it"),
        temperature=0.0)
    # Stream via HTTPX since OpenAI translation client doesn't expose streaming
    url = server.url_for("v1/audio/translations")
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    data = {
        "model": MODEL_NAME,
        "language": "it",
        "stream": True,
        "temperature": 0.0,
    }
    foscolo.seek(0)
    async with httpx.AsyncClient() as http_client:
        files = {"file": foscolo}
        async with http_client.stream("POST",
                                      url,
                                      headers=headers,
                                      data=data,
                                      files=files) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line.strip() == "[DONE]":
                    break
                chunk = json.loads(line)
                text = chunk["choices"][0].get("delta", {}).get("content")
                translation += text or ""

    assert translation == res_no_stream.text


@pytest.mark.asyncio
async def test_stream_options(foscolo, client, server):
    url = server.url_for("v1/audio/translations")
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    data = {
        "model": MODEL_NAME,
        "language": "it",
        "stream": True,
        "stream_include_usage": True,
        "stream_continuous_usage_stats": True,
        "temperature": 0.0,
    }
    foscolo.seek(0)
    final = False
    continuous = True
    async with httpx.AsyncClient() as http_client:
        files = {"file": foscolo}
        async with http_client.stream("POST",
                                      url,
                                      headers=headers,
                                      data=data,
                                      files=files) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line.strip() == "[DONE]":
                    break
                chunk = json.loads(line)
                choices = chunk.get("choices", [])
                if not choices:
                    # final usage sent
                    final = True
                else:
                    continuous = continuous and ("usage" in chunk)
    assert final and continuous


@pytest.mark.asyncio
async def test_long_audio_request(foscolo, client):
    foscolo.seek(0)
    audio, sr = librosa.load(foscolo)
    repeated_audio = np.tile(audio, 2)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format='WAV')
    buffer.seek(0)
    translation = await client.audio.translations.create(
        model=MODEL_NAME,
        file=buffer,
        extra_body=dict(language="it"),
        response_format="text",
        temperature=0.0)
    out = json.loads(translation)['text'].strip().lower()
    assert out.count("greek sea") == 2
