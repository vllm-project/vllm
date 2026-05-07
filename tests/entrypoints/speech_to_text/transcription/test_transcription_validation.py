# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import io
import json
import math
import os

import numpy as np
import pytest
import soundfile as sf

from tests.entrypoints.speech_to_text.conftest import add_attention_backend
from tests.utils import ROCM_ENV_OVERRIDES, ROCM_EXTRA_ARGS, RemoteOpenAIServer
from vllm.multimodal.media.audio import load_audio

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]
PARAKEET_HF_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_MODEL_NAME = os.environ.get("PARAKEET_TEST_MODEL", PARAKEET_HF_MODEL_NAME)


def make_long_audio(file, *, repeats: int) -> tuple[io.BytesIO, int]:
    file.seek(0)
    audio, sr = load_audio(file)
    # Add small silence after each repeat for repeatable chunk splitting.
    audio = np.pad(audio, (0, int(sr * 0.1)))
    repeated_audio = np.tile(audio, repeats)

    buffer = io.BytesIO()
    buffer.name = "long_audio.wav"
    sf.write(buffer, repeated_audio, sr, format="WAV")
    buffer.seek(0)

    expected_seconds = math.ceil(len(repeated_audio) / sr)
    return buffer, expected_seconds


async def transcribe_and_check(
    client,
    model_name: str,
    file,
    *,
    language: str,
    expected_text: str,
    expected_seconds: int | None = None,
    case_sensitive: bool = False,
):
    """Run a transcription request and assert the output contains
    *expected_text* and optionally that usage reports *expected_seconds*.

    Provides detailed failure messages with the actual transcription output.
    """
    transcription = await client.audio.transcriptions.create(
        model=model_name,
        file=file,
        language=language,
        response_format="text",
        temperature=0.0,
    )
    out = json.loads(transcription)
    out_text = out["text"]
    out_usage = out["usage"]

    if case_sensitive:
        assert expected_text in out_text, (
            f"Expected {expected_text!r} in transcription output, got: {out_text!r}"
        )
    else:
        assert expected_text.lower() in out_text.lower(), (
            f"Expected {expected_text!r} (case-insensitive) in transcription "
            f"output, got: {out_text!r}"
        )

    if expected_seconds is not None:
        assert out_usage["seconds"] == expected_seconds, (
            f"Expected {expected_seconds}s of audio, "
            f"got {out_usage['seconds']}s. Full usage: {out_usage!r}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["mistralai/Voxtral-Mini-3B-2507", "Qwen/Qwen3-ASR-0.6B"]
)
async def test_basic_audio(mary_had_lamb, model_name, rocm_aiter_fa_attention):
    server_args = ["--enforce-eager", *ROCM_EXTRA_ARGS]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(
        model_name, server_args, env_dict=ROCM_ENV_OVERRIDES
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            model_name,
            mary_had_lamb,
            language="en",
            expected_text="Mary had a little lamb",
            expected_seconds=16,
        )


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
    with RemoteOpenAIServer(
        model_name, server_args, env_dict=ROCM_ENV_OVERRIDES
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            lora_model_name,
            mary_had_lamb,
            language="en",
            expected_text="mary had a little lamb",
            expected_seconds=16,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["google/gemma-3n-E2B-it", "Qwen/Qwen3-ASR-0.6B"]
)
async def test_basic_audio_foscolo(foscolo, rocm_aiter_fa_attention, model_name):
    # Gemma accuracy on some of the audio samples we use is particularly bad,
    # hence we use a different one here. WER is evaluated separately.
    server_args = ["--enforce-eager", *ROCM_EXTRA_ARGS]

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name,
        server_args,
        max_wait_seconds=480,
        env_dict=ROCM_ENV_OVERRIDES,
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            model_name,
            foscolo,
            language="it",
            expected_text="ove il mio corpo fanciulletto",
        )


@pytest.fixture(scope="module")
def parakeet_server(rocm_aiter_fa_attention):
    server_args = [
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "1",
    ]
    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        PARAKEET_MODEL_NAME,
        server_args,
        max_wait_seconds=480,
        env_dict=ROCM_ENV_OVERRIDES,
    ) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_basic_audio_parakeet(mary_had_lamb, parakeet_server):
    async with parakeet_server.get_async_client() as client:
        await transcribe_and_check(
            client,
            PARAKEET_MODEL_NAME,
            mary_had_lamb,
            language="en",
            expected_text="Mary had a little lamb",
            expected_seconds=16,
        )


@pytest.mark.asyncio
async def test_streaming_audio_parakeet_strips_eos(mary_had_lamb, parakeet_server):
    transcription = ""
    async with parakeet_server.get_async_client() as client:
        res = await client.audio.transcriptions.create(
            model=PARAKEET_MODEL_NAME,
            file=mary_had_lamb,
            language="en",
            temperature=0.0,
            stream=True,
        )

        async for chunk in res:
            transcription += chunk.choices[0]["delta"]["content"]

    assert "Mary had a little lamb" in transcription
    assert "<|endoftext|>" not in transcription


@pytest.mark.asyncio
async def test_long_audio_parakeet(mary_had_lamb, parakeet_server):
    long_audio, expected_seconds = make_long_audio(mary_had_lamb, repeats=3)

    async with parakeet_server.get_async_client() as client:
        transcription = await client.audio.transcriptions.create(
            model=PARAKEET_MODEL_NAME,
            file=long_audio,
            language="en",
            response_format="text",
            temperature=0.0,
        )

    out = json.loads(transcription)
    out_text = out["text"]
    count = out_text.lower().count("mary had a little lamb")
    assert count == 3, f"Expected 3 repeats, found {count}: {out_text!r}"
    assert out["usage"]["seconds"] == expected_seconds
