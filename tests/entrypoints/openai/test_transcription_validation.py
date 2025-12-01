# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import json

import pytest

from ...utils import RemoteOpenAIServer

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
