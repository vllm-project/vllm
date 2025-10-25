# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import io
import json

import librosa
import numpy as np
import openai
import pytest
import pytest_asyncio
import soundfile as sf

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/whisper-large-v3-turbo"
SERVER_ARGS = ["--enforce-eager"]
MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]


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
    "model_name", ["openai/whisper-large-v3-turbo", "mistralai/Voxtral-Mini-3B-2507"]
)
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
            temperature=0.0,
        )
        out = json.loads(transcription)
        out_text = out["text"]
        out_usage = out["usage"]
        assert "Mary had a little lamb," in out_text
        assert out_usage["seconds"] == 16, out_usage["seconds"]


@pytest.mark.asyncio
async def test_basic_audio_with_lora(mary_had_lamb):
    """Ensure STT (transcribe) requests can pass LoRA through to generate."""
    model_name = "ibm-granite/granite-speech-3.3-2b"
    lora_model_name = "speech"
    server_args = [
        "--enforce-eager",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        f"{lora_model_name}={model_name}",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
    ]

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=lora_model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0,
        )
    out = json.loads(transcription)
    out_text = out["text"]
    out_usage = out["usage"]
    assert "mary had a little lamb" in out_text
    assert out_usage["seconds"] == 16, out_usage["seconds"]


@pytest.mark.asyncio
async def test_basic_audio_gemma(foscolo):
    # Gemma accuracy on some of the audio samples we use is particularly bad,
    # hence we use a different one here. WER is evaluated separately.
    model_name = "google/gemma-3n-E2B-it"
    server_args = ["--enforce-eager"]

    with RemoteOpenAIServer(
        model_name, server_args, max_wait_seconds=480
    ) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=foscolo,
            language="it",
            response_format="text",
            temperature=0.0,
        )
        out = json.loads(transcription)["text"]
        assert "da cui vergine nacque Venere" in out


@pytest.mark.asyncio
async def test_non_asr_model(winning_call):
    # text to text model
    model_name = "JackFram/llama-68m"
    with RemoteOpenAIServer(model_name, SERVER_ARGS) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.transcriptions.create(
            model=model_name, file=winning_call, language="en", temperature=0.0
        )
        err = res.error
        assert err["code"] == 400 and not res.text
        assert err["message"] == "The model does not support Transcriptions API"


@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb, client):
    # invalid language
    with pytest.raises(openai.BadRequestError):
        await client.audio.transcriptions.create(
            model=MODEL_NAME, file=mary_had_lamb, language="hh", temperature=0.0
        )


@pytest.mark.asyncio
async def test_long_audio_request(mary_had_lamb, client):
    mary_had_lamb.seek(0)
    audio, sr = librosa.load(mary_had_lamb)
    # Add small silence after each audio for repeatability in the split process
    audio = np.pad(audio, (0, 1600))
    repeated_audio = np.tile(audio, 10)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format="WAV")
    buffer.seek(0)
    transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=buffer,
        language="en",
        response_format="text",
        temperature=0.0,
    )
    out = json.loads(transcription)
    out_text = out["text"]
    out_usage = out["usage"]
    counts = out_text.count("Mary had a little lamb")
    assert counts == 10, counts
    assert out_usage["seconds"] == 161, out_usage["seconds"]


@pytest.mark.asyncio
async def test_completion_endpoints(client):
    # text to text model
    res = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."}],
    )
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
        temperature=0.0,
    )
    res = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=winning_call,
        language="en",
        temperature=0.0,
        stream=True,
        timeout=30,
    )
    # Reconstruct from chunks and validate
    async for chunk in res:
        text = chunk.choices[0]["delta"]["content"]
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
        extra_body=dict(stream_include_usage=True, stream_continuous_usage_stats=True),
        timeout=30,
    )
    final = False
    continuous = True
    async for chunk in res:
        if not len(chunk.choices):
            # final usage sent
            final = True
        else:
            continuous = continuous and hasattr(chunk, "usage")
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
        extra_body=dict(
            seed=42,
            repetition_penalty=1.9,
            top_k=12,
            top_p=0.4,
            min_p=0.5,
            frequency_penalty=1.8,
            presence_penalty=2.0,
        ),
    )

    greedy_transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        temperature=0.0,
        extra_body=dict(seed=42),
    )

    assert greedy_transcription.text != transcription.text


@pytest.mark.asyncio
async def test_audio_prompt(mary_had_lamb, client):
    prompt = "This is a speech, recorded in a phonograph."
    # Prompts should not omit the part of original prompt while transcribing.
    prefix = "The first words I spoke in the original phonograph"
    transcription = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
    )
    out = json.loads(transcription)["text"]
    assert prefix in out
    transcription_wprompt = await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        prompt=prompt,
        temperature=0.0,
    )
    out_prompt = json.loads(transcription_wprompt)["text"]
    assert prefix in out_prompt
