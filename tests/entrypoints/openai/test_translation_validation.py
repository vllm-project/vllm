# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
# imports for structured outputs tests
import json

import httpx
import librosa
import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf

from ...utils import RemoteOpenAIServer

SERVER_ARGS = ["--enforce-eager"]


@pytest.fixture(scope="module",
                params=["openai/whisper-small", "google/gemma-3n-E2B-it"])
def server(request):
    # Parametrize over model name
    with RemoteOpenAIServer(request.param, SERVER_ARGS) as remote_server:
        yield remote_server, request.param


@pytest_asyncio.fixture
async def client_and_model(server):
    server, model_name = server
    async with server.get_async_client() as async_client:
        yield async_client, model_name


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
async def test_basic_audio(foscolo, client_and_model):
    client, model_name = client_and_model
    translation = await client.audio.translations.create(
        model=model_name,
        file=foscolo,
        response_format="text",
        # TODO remove `language="it"` once language detection is implemented
        extra_body=dict(language="it", to_language="en"),
        temperature=0.0)
    out = json.loads(translation)['text'].strip().lower()
    assert "greek sea" in out


@pytest.mark.asyncio
async def test_audio_prompt(foscolo, client_and_model):
    client, model_name = client_and_model
    # Condition whisper on starting text
    prompt = "Nor have I ever"
    transcription = await client.audio.translations.create(
        model=model_name,
        file=foscolo,
        prompt=prompt,
        extra_body=dict(language="it", to_language="en"),
        response_format="text",
        temperature=0.0)
    out = json.loads(transcription)['text']
    assert "Nor will I ever touch the sacred" not in out
    assert prompt not in out


@pytest.mark.asyncio
async def test_streaming_response(foscolo, client_and_model, server):
    client, model_name = client_and_model
    translation = ""
    res_no_stream = await client.audio.translations.create(
        model=model_name,
        file=foscolo,
        response_format="json",
        extra_body=dict(language="it", to_language="en", seed=42),
        temperature=0.0)

    # Stream via HTTPX since OpenAI translation client doesn't expose streaming
    server, model_name = server
    url = server.url_for("v1/audio/translations")
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    data = {
        "model": model_name,
        "language": "it",
        "to_language": "en",
        "stream": True,
        "temperature": 0.0,
        "seed": 42,
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

    res_stream = translation.split()
    # NOTE There's a small non-deterministic issue here, likely in the attn
    # computation, which will cause a few tokens to be different, while still
    # being very close semantically.
    assert sum([
        x == y for x, y in zip(res_stream, res_no_stream.text.split())
    ]) >= len(res_stream) * 0.9


@pytest.mark.asyncio
async def test_stream_options(foscolo, server):
    server, model_name = server
    url = server.url_for("v1/audio/translations")
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    data = {
        "model": model_name,
        "language": "it",
        "to_language": "en",
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
async def test_long_audio_request(foscolo, client_and_model):
    client, model_name = client_and_model
    if model_name == "google/gemma-3n-E2B-it":
        pytest.skip("Gemma3n does not support long audio requests")
    foscolo.seek(0)
    audio, sr = librosa.load(foscolo)
    repeated_audio = np.tile(audio, 2)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format='WAV')
    buffer.seek(0)
    translation = await client.audio.translations.create(
        model=model_name,
        file=buffer,
        extra_body=dict(language="it", to_language="en"),
        response_format="text",
        temperature=0.0)
    out = json.loads(translation)['text'].strip().lower()
    assert out.count("greek sea") == 2
