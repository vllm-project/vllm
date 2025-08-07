# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
# imports for guided decoding tests
import json
from unittest.mock import patch

import librosa
import numpy as np
import pytest
import soundfile as sf
from openai._base_client import AsyncAPIClient

from vllm.assets.audio import AudioAsset

from ...utils import RemoteOpenAIServer


@pytest.fixture
def foscolo():
    # Test translation it->en
    path = AudioAsset('azacinto_foscolo').get_local_path()
    with open(str(path), "rb") as f:
        yield f


# NOTE: (NickLucche) the large-v3-turbo model was not trained on translation!
@pytest.mark.asyncio
async def test_basic_audio(foscolo):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        translation = await client.audio.translations.create(
            model=model_name,
            file=foscolo,
            response_format="text",
            # TODO remove once language detection is implemented
            extra_body=dict(language="it"),
            temperature=0.0)
        out = json.loads(translation)['text'].strip().lower()
        assert "greek sea" in out


@pytest.mark.asyncio
async def test_audio_prompt(foscolo):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    # Condition whisper on starting text
    prompt = "Nor have I ever"
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.translations.create(
            model=model_name,
            file=foscolo,
            prompt=prompt,
            extra_body=dict(language="it"),
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        assert "Nor will I ever touch the sacred" not in out
        assert prompt not in out


@pytest.mark.asyncio
async def test_non_asr_model(foscolo):
    # text to text model
    model_name = "JackFram/llama-68m"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.translations.create(model=model_name,
                                                     file=foscolo,
                                                     temperature=0.0)
        err = res.error
        assert err["code"] == 400 and not res.text
        assert err["message"] == "The model does not support Translations API"


@pytest.mark.asyncio
async def test_streaming_response(foscolo):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    translation = ""
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res_no_stream = await client.audio.translations.create(
            model=model_name,
            file=foscolo,
            response_format="json",
            extra_body=dict(language="it"),
            temperature=0.0)
        # Unfortunately this only works when the openai client is patched
        # to use streaming mode, not exposed in the translation api.
        original_post = AsyncAPIClient.post

        async def post_with_stream(*args, **kwargs):
            kwargs['stream'] = True
            return await original_post(*args, **kwargs)

        with patch.object(AsyncAPIClient, "post", new=post_with_stream):
            client = remote_server.get_async_client()
            res = await client.audio.translations.create(model=model_name,
                                                         file=foscolo,
                                                         temperature=0.0,
                                                         extra_body=dict(
                                                             stream=True,
                                                             language="it"))
            # Reconstruct from chunks and validate
            async for chunk in res:
                # just a chunk
                text = chunk.choices[0]['delta']['content']
                translation += text

        assert translation == res_no_stream.text


@pytest.mark.asyncio
async def test_stream_options(foscolo):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        original_post = AsyncAPIClient.post

        async def post_with_stream(*args, **kwargs):
            kwargs['stream'] = True
            return await original_post(*args, **kwargs)

        with patch.object(AsyncAPIClient, "post", new=post_with_stream):
            client = remote_server.get_async_client()
            res = await client.audio.translations.create(
                model=model_name,
                file=foscolo,
                temperature=0.0,
                extra_body=dict(language="it",
                                stream=True,
                                stream_include_usage=True,
                                stream_continuous_usage_stats=True))
            final = False
            continuous = True
            async for chunk in res:
                if not len(chunk.choices):
                    # final usage sent
                    final = True
                else:
                    continuous = continuous and hasattr(chunk, 'usage')
            assert final and continuous


@pytest.mark.asyncio
async def test_long_audio_request(foscolo):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]

    foscolo.seek(0)
    audio, sr = librosa.load(foscolo)
    repeated_audio = np.tile(audio, 2)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format='WAV')
    buffer.seek(0)
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        translation = await client.audio.translations.create(
            model=model_name,
            file=buffer,
            extra_body=dict(language="it"),
            response_format="text",
            temperature=0.0)
        out = json.loads(translation)['text'].strip().lower()
        assert out.count("greek sea") == 2
