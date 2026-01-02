# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import json

import pytest

from ...utils import RemoteOpenAIServer
from .conftest import add_attention_backend

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["mistralai/Voxtral-Mini-3B-2507"])
async def test_basic_audio(mary_had_lamb, model_name, rocm_aiter_fa_attention):
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

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
async def test_basic_audio_with_lora(mary_had_lamb, rocm_aiter_fa_attention):
    """Ensure STT (transcribe) requests can pass LoRA through to generate."""
    # ROCm SPECIFIC CONFIGURATION:
    # To ensure the test passes on ROCm, we modify the max model length to 512.
    # We DO NOT apply this to other platforms to maintain strict upstream parity.
    from vllm.platforms import current_platform

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
async def test_basic_audio_gemma(foscolo, rocm_aiter_fa_attention):
    # Gemma accuracy on some of the audio samples we use is particularly bad,
    # hence we use a different one here. WER is evaluated separately.
    model_name = "google/gemma-3n-E2B-it"
    server_args = ["--enforce-eager"]

    add_attention_backend(server_args, rocm_aiter_fa_attention)

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
@pytest.mark.parametrize("model_name", ["mistralai/Voxtral-Mini-3B-2507"])
async def test_transcription_with_logprobs(mary_had_lamb, model_name):
    """Test that logprobs are returned when requested (vLLM extension)."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # Test with logprobs=5 - should return top-5 alternatives
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="json",
            temperature=0.0,
            extra_body={"logprobs": 5},
        )
        out = json.loads(transcription)
        assert "text" in out
        assert "Mary had a little lamb" in out["text"]

        # Verify logprobs structure matches OpenAI format
        # (list of objects with token, bytes, logprob fields)
        assert "logprobs" in out
        assert out["logprobs"] is not None
        logprobs = out["logprobs"]

        # logprobs should be a list of token objects
        assert isinstance(logprobs, list)
        assert len(logprobs) > 0

        # Each entry should have token, bytes, and logprob fields
        for lp in logprobs:
            assert "token" in lp
            assert "bytes" in lp
            assert "logprob" in lp
            # token should be a string
            assert isinstance(lp["token"], str)
            # bytes should be a list of integers (UTF-8 encoding)
            if lp["bytes"] is not None:
                assert isinstance(lp["bytes"], list)
            # logprob should be a float or None
            if lp["logprob"] is not None:
                assert isinstance(lp["logprob"], float)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["mistralai/Voxtral-Mini-3B-2507"])
async def test_transcription_without_logprobs(mary_had_lamb, model_name):
    """Test that logprobs are not returned when not requested."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # Test without logprobs - should return null
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="json",
            temperature=0.0,
        )
        out = json.loads(transcription)
        assert "text" in out
        assert "Mary had a little lamb" in out["text"]

        # logprobs should be null or not present
        assert out.get("logprobs") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["mistralai/Voxtral-Mini-3B-2507"])
async def test_transcription_verbose_json_with_logprobs(mary_had_lamb, model_name):
    """Test that logprobs work with verbose_json response format."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="verbose_json",
            temperature=0.0,
            extra_body={"logprobs": 3},
        )
        out = json.loads(transcription)

        # Verify verbose_json fields
        assert "text" in out
        assert "language" in out
        assert "duration" in out

        # Verify logprobs in verbose_json format (OpenAI-compatible list format)
        assert "logprobs" in out
        assert out["logprobs"] is not None
        logprobs = out["logprobs"]

        # logprobs should be a list of token objects
        assert isinstance(logprobs, list)
        assert len(logprobs) > 0

        # Each entry should have token, bytes, and logprob fields
        for lp in logprobs:
            assert "token" in lp
            assert "bytes" in lp
            assert "logprob" in lp
