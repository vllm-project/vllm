# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for guided decoding tests
import io
import json

import librosa
import numpy as np
import openai
import pytest
import pytest_asyncio
import soundfile as sf

from vllm.assets.audio import AudioAsset

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/whisper-large-v3-turbo"
SERVER_ARGS = ["--enforce-eager"]
MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode", "mistral", "--config_format", "mistral",
    "--load_format", "mistral"
]


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


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["openai/whisper-large-v3-turbo", "mistralai/Voxtral-Mini-3B-2507"])
async def test_basic_audio(mary_had_lamb, model_name):
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
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


@pytest.mark.asyncio
async def test_non_asr_model(winning_call):
    # text to text model
    model_name = "JackFram/llama-68m"
    with RemoteOpenAIServer(model_name, SERVER_ARGS) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.transcriptions.create(model=model_name,
                                                       file=winning_call,
                                                       language="en",
                                                       temperature=0.0)
        err = res.error
        assert err["code"] == 400 and not res.text
        assert err[
            "message"] == "The model does not support Transcriptions API"


@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb, client):
    # invalid language
    with pytest.raises(openai.BadRequestError):
        await client.audio.transcriptions.create(model=MODEL_NAME,
                                                 file=mary_had_lamb,
                                                 language="hh",
                                                 temperature=0.0)


@pytest.mark.asyncio
async def test_long_audio_request(mary_had_lamb, client):
    mary_had_lamb.seek(0)
    audio, sr = librosa.load(mary_had_lamb)
    # Add small silence after each audio for repeatability in the split process
    audio = np.pad(audio, (0, 1600))
    repeated_audio = np.tile(audio, 10)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format='WAV')
    buffer.seek(0)
    transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=buffer,
        language="en",
        response_format="text",
        temperature=0.0)
    out = json.loads(transcription)['text']
    counts = out.count("Mary had a little lamb")
    assert counts == 10, counts


@pytest.mark.asyncio
async def test_completion_endpoints(client):
    # text to text model
    res = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }])
    err = res.error
    assert err["code"] == 400
    assert err["message"] == "The model does not support Chat Completions API"

    res = await client.completions.create(model=MODEL_NAME, prompt="Hello")
    err = res.error
    assert err["code"] == 400
    assert err["message"] == "The model does not support Completions API"


@pytest.mark.asyncio
async def test_streaming_response(winning_call, client):
    transcription = ""
    res_no_stream = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=winning_call,
        response_format="json",
        language="en",
        temperature=0.0)
    res = await client.audio.transcriptions.create(model=MODEL_NAME,
                                                   file=winning_call,
                                                   language="en",
                                                   temperature=0.0,
                                                   stream=True,
                                                   timeout=30)
    # Reconstruct from chunks and validate
    async for chunk in res:
        text = chunk.choices[0]['delta']['content']
        transcription += text

    assert transcription == res_no_stream.text


@pytest.mark.asyncio
async def test_stream_options(winning_call, client):
    res = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=winning_call,
        language="en",
        temperature=0.0,
        stream=True,
        extra_body=dict(stream_include_usage=True,
                        stream_continuous_usage_stats=True),
        timeout=30)
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
async def test_sampling_params(mary_had_lamb, client):
    """
    Compare sampling with params and greedy sampling to assert results
    are different when extreme sampling parameters values are picked. 
    """
    transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        temperature=0.8,
        extra_body=dict(seed=42,
                        repetition_penalty=1.9,
                        top_k=12,
                        top_p=0.4,
                        min_p=0.5,
                        frequency_penalty=1.8,
                        presence_penalty=2.0))

    greedy_transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        temperature=0.0,
        extra_body=dict(seed=42))

    assert greedy_transcription.text != transcription.text


@pytest.mark.asyncio
async def test_audio_prompt(mary_had_lamb, client):
    prompt = "This is a speech, recorded in a phonograph."
    #Prompts should not omit the part of original prompt while transcribing.
    prefix = "The first words I spoke in the original phonograph"
    transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0)
    out = json.loads(transcription)['text']
    assert prefix in out
    transcription_wprompt = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        prompt=prompt,
        temperature=0.0)
    out_prompt = json.loads(transcription_wprompt)['text']
    assert prefix in out_prompt
