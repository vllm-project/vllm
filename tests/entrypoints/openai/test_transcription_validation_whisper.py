# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import asyncio
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


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def whisper_client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_basic_audio(whisper_client, mary_had_lamb):
    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
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
async def test_basic_audio_batched(mary_had_lamb, winning_call, whisper_client):
    transcription = whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
    )
    transcription2 = whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=winning_call,
        language="en",
        response_format="text",
        temperature=0.0,
    )
    # Await both transcriptions by scheduling coroutines together
    transcription, transcription2 = await asyncio.gather(transcription, transcription2)
    out = json.loads(transcription)
    out_text = out["text"]
    assert "Mary had a little lamb," in out_text
    out2 = json.loads(transcription2)
    out_text2 = out2["text"]
    assert "Edgar Martinez" in out_text2


@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb, whisper_client):
    # invalid language
    with pytest.raises(openai.BadRequestError):
        await whisper_client.audio.transcriptions.create(
            model=MODEL_NAME, file=mary_had_lamb, language="hh", temperature=0.0
        )


@pytest.mark.asyncio
async def test_long_audio_request(mary_had_lamb, whisper_client):
    mary_had_lamb.seek(0)
    audio, sr = librosa.load(mary_had_lamb)
    # Add small silence after each audio for repeatability in the split process
    audio = np.pad(audio, (0, 1600))
    repeated_audio = np.tile(audio, 10)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format="WAV")
    buffer.seek(0)
    transcription = await whisper_client.audio.transcriptions.create(
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
async def test_completion_endpoints(whisper_client):
    # text to text model
    with pytest.raises(openai.NotFoundError):
        await whisper_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a helpful assistant."}],
        )

    with pytest.raises(openai.NotFoundError):
        await whisper_client.completions.create(model=MODEL_NAME, prompt="Hello")


@pytest.mark.asyncio
async def test_streaming_response(winning_call, whisper_client):
    transcription = ""
    res_no_stream = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=winning_call,
        response_format="json",
        language="en",
        temperature=0.0,
    )
    res = await whisper_client.audio.transcriptions.create(
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
async def test_stream_options(winning_call, whisper_client):
    res = await whisper_client.audio.transcriptions.create(
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
async def test_sampling_params(mary_had_lamb, whisper_client):
    """
    Compare sampling with params and greedy sampling to assert results
    are different when extreme sampling parameters values are picked.
    """
    transcription = await whisper_client.audio.transcriptions.create(
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

    greedy_transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        temperature=0.0,
        extra_body=dict(seed=42),
    )

    assert greedy_transcription.text != transcription.text


@pytest.mark.asyncio
async def test_audio_prompt(mary_had_lamb, whisper_client):
    prompt = "This is a speech, recorded in a phonograph."
    # Prompts should not omit the part of original prompt while transcribing.
    prefix = "The first words I spoke in the original phonograph"
    transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
    )
    out = json.loads(transcription)["text"]
    assert prefix in out
    transcription_wprompt = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        prompt=prompt,
        temperature=0.0,
    )
    out_prompt = json.loads(transcription_wprompt)["text"]
    assert prefix in out_prompt


@pytest.mark.asyncio
async def test_audio_with_timestamp(mary_had_lamb, whisper_client):
    transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="verbose_json",
        temperature=0.0,
    )
    assert transcription.segments is not None
    assert len(transcription.segments) > 0
    assert transcription.segments[0].avg_logprob is not None
    assert transcription.segments[0].compression_ratio is not None


@pytest.mark.asyncio
async def test_audio_with_max_tokens(whisper_client, mary_had_lamb):
    transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
        extra_body={"max_completion_tokens": 1},
    )
    out = json.loads(transcription)
    out_text = out["text"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    out_tokens = tok(out_text, add_special_tokens=False)["input_ids"]
    assert len(out_tokens) == 1
    # max_completion_tokens > max_model_len
    transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
        extra_body={"max_completion_tokens": int(1e6)},
    )
    out = json.loads(transcription)
    out_text = out["text"]
    out_tokens = tok(out_text, add_special_tokens=False)["input_ids"]
    assert len(out_tokens) < 450  # ~Whisper max output len
