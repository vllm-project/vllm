# SPDX-License-Identifier: Apache-2.0

# imports for guided decoding tests
import io
import json

import librosa
import numpy as np
import openai
import pytest
import soundfile as sf

from vllm.assets.audio import AudioAsset

from ...utils import RemoteOpenAIServer


@pytest.fixture
def mary_had_lamb():
    path = AudioAsset('mary_had_lamb').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def winning_call():
    path = AudioAsset('winning_call').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.mark.asyncio
async def test_basic_audio(mary_had_lamb):
    model_name = "openai/whisper-large-v3-turbo"
    server_args = ["--enforce-eager"]
    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    prompt = "THE FIRST WORDS I SPOKE"
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        assert "Mary had a little lamb," in out
        # This should "force" whisper to continue prompt in all caps
        transcription_wprompt = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            prompt=prompt,
            temperature=0.0)
        out_capital = json.loads(transcription_wprompt)['text']
        assert prompt not in out_capital


@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # invalid language
        with pytest.raises(openai.BadRequestError):
            await client.audio.transcriptions.create(model=model_name,
                                                     file=mary_had_lamb,
                                                     language="hh",
                                                     temperature=0.0)

        # Expect audio too long: repeat the timeseries
        mary_had_lamb.seek(0)
        audio, sr = librosa.load(mary_had_lamb)
        repeated_audio = np.tile(audio, 10)
        # Repeated audio to buffer
        buffer = io.BytesIO()
        sf.write(buffer, repeated_audio, sr, format='WAV')
        buffer.seek(0)
        with pytest.raises(openai.BadRequestError):
            await client.audio.transcriptions.create(model=model_name,
                                                     file=buffer,
                                                     language="en",
                                                     temperature=0.0)


@pytest.mark.asyncio
async def test_non_asr_model(winning_call):
    # text to text model
    model_name = "JackFram/llama-68m"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.transcriptions.create(model=model_name,
                                                       file=winning_call,
                                                       language="en",
                                                       temperature=0.0)
        assert res.code == 400 and not res.text
        assert res.message == "The model does not support Transcriptions API"


@pytest.mark.asyncio
async def test_completion_endpoints():
    # text to text model
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }])
        assert res.code == 400
        assert res.message == "The model does not support Chat Completions API"

        res = await client.completions.create(model=model_name, prompt="Hello")
        assert res.code == 400
        assert res.message == "The model does not support Completions API"
