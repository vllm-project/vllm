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

import openai
from ...utils import RemoteOpenAIServer
from .conftest import add_attention_backend

SERVER_ARGS = ["--enforce-eager"]


def _get_server_args(attention_config):
    """Get server args with attention backend if specified."""
    args = SERVER_ARGS.copy()
    add_attention_backend(args, attention_config)
    return args


@pytest.fixture(
    scope="module", params=["openai/whisper-small", "google/gemma-3n-E2B-it"]
)
def server(request, rocm_aiter_fa_attention):
    # Parametrize over model name
    with RemoteOpenAIServer(
        request.param, _get_server_args(rocm_aiter_fa_attention)
    ) as remote_server:
        yield remote_server, request.param


@pytest_asyncio.fixture
async def client_and_model(server):
    server, model_name = server
    async with server.get_async_client() as async_client:
        yield async_client, model_name


@pytest.mark.asyncio
async def test_non_asr_model(foscolo, rocm_aiter_fa_attention):
    # text to text model
    model_name = "JackFram/llama-68m"
    with RemoteOpenAIServer(
        model_name, _get_server_args(rocm_aiter_fa_attention)
    ) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.NotFoundError):
            await client.audio.translations.create(
                model=model_name, file=foscolo, temperature=0.0
            )


@pytest.mark.asyncio
async def test_basic_audio_with_lora(mary_had_lamb, rocm_aiter_fa_attention):
    """Ensure STT (translate) requests can pass LoRA through to generate."""
    # ROCm SPECIFIC CONFIGURATION:
    # To ensure the test passes on ROCm, we modify the max model length to 512.
    # We DO NOT apply this to other platforms to maintain strict upstream parity.
    from vllm.platforms import current_platform

    # NOTE - careful to call this test before the module scoped server
    # fixture, otherwise it'll OOMkill the CI
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
        "512" if current_platform.is_rocm() else "2048",
        "--max-num-seqs",
        "1",
    ]

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        translation = await client.audio.translations.create(
            model=lora_model_name,
            file=mary_had_lamb,
            extra_body=dict(language="en", to_language="es"),
            response_format="text",
            temperature=0.0,
        )
    out = json.loads(translation)["text"].strip().lower()
    assert "pequeño" in out.split(" ")


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
        temperature=0.0,
    )
    out = json.loads(translation)["text"].strip().lower()
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
        temperature=0.0,
    )
    out = json.loads(transcription)["text"]
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
        temperature=0.0,
    )

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
        async with http_client.stream(
            "POST", url, headers=headers, data=data, files=files
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: ") :]
                if line.strip() == "[DONE]":
                    break
                chunk = json.loads(line)
                text = chunk["choices"][0].get("delta", {}).get("content")
                translation += text or ""

    res_stream = translation.split()
    # NOTE There's a small non-deterministic issue here, likely in the attn
    # computation, which will cause a few tokens to be different, while still
    # being very close semantically.
    assert (
        sum([x == y for x, y in zip(res_stream, res_no_stream.text.split())])
        >= len(res_stream) * 0.9
    )


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
        async with http_client.stream(
            "POST", url, headers=headers, data=data, files=files
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: ") :]
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
    sf.write(buffer, repeated_audio, sr, format="WAV")
    buffer.seek(0)
    translation = await client.audio.translations.create(
        model=model_name,
        file=buffer,
        extra_body=dict(language="it", to_language="en"),
        response_format="text",
        temperature=0.0,
    )
    out = json.loads(translation)["text"].strip().lower()
    assert out.count("greek sea") == 2


@pytest.mark.asyncio
async def test_audio_with_max_tokens(mary_had_lamb, client_and_model):
    client, model_name = client_and_model
    transcription = await client.audio.translations.create(
        model=model_name,
        file=mary_had_lamb,
        response_format="text",
        temperature=0.0,
        extra_body={"max_completion_tokens": 1},
    )
    out = json.loads(transcription)
    out_text = out["text"]
    print(out_text)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    out_tokens = tok(out_text, add_special_tokens=False)["input_ids"]
    assert len(out_tokens) == 1
    # max_completion_tokens > max_model_len
    # max_model_len=32768 for Gemma-3n-E2B-it
    transcription = await client.audio.transcriptions.create(
        model=model_name,
        file=mary_had_lamb,
        response_format="text",
        temperature=0.0,
        extra_body={
            "max_completion_tokens": int(1e6),
            "repetition_penalty": 1.3,
        },
    )
    out = json.loads(transcription)
    out_text = out["text"]
    print(out_text)
    out_tokens = tok(out_text, add_special_tokens=False)["input_ids"]
    assert len(out_tokens) < 450  # ~Whisper max output len
