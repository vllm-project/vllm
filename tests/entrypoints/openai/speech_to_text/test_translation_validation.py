# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

# imports for structured outputs tests
import json

import httpx
import librosa
import numpy as np
import openai
import pytest
import pytest_asyncio
import soundfile as sf

from tests.entrypoints.openai.conftest import add_attention_backend
from tests.utils import RemoteOpenAIServer
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

SEED = 42
TEMPERATURE = 0.0
SERVER_ARGS = ["--enforce-eager"]
if current_platform.is_rocm():
    SERVER_ARGS.append("--no-enable-prefix-caching")


def _get_rocm_attention_config(model_name):
    """Return appropriate ROCm attention config for the given model.

    Whisper uses cross-attention (ENCODER_DECODER) which ROCM_AITER_FA does
    not support. For Whisper we use ROCM_AITER_UNIFIED_ATTN (or TRITON_ATTN
    as fallback); other models can use ROCM_AITER_FA.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return None

    if "whisper" in model_name.lower():
        try:
            from vllm.platforms.rocm import _ON_MI3XX

            if _ON_MI3XX:
                return {"backend": "ROCM_AITER_UNIFIED_ATTN"}
        except ImportError:
            logger.warning(
                "Could not import _ON_MI3XX from rocm platform, "
                "falling back to TRITON_ATTN for Whisper."
            )
        return {"backend": "TRITON_ATTN"}

    return {"backend": "ROCM_AITER_FA"}


def _get_server_args(attention_config):
    """Get server args with attention backend if specified."""
    args = SERVER_ARGS.copy()
    add_attention_backend(args, attention_config)
    return args


@pytest.fixture(
    scope="module", params=["openai/whisper-small", "google/gemma-3n-E2B-it"]
)
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name):
    attention_config = _get_rocm_attention_config(model_name)
    with RemoteOpenAIServer(
        model_name, _get_server_args(attention_config)
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_non_asr_model(foscolo):
    # text to text model
    model_name = "JackFram/llama-68m"
    attention_config = _get_rocm_attention_config(model_name)
    with RemoteOpenAIServer(
        model_name, _get_server_args(attention_config)
    ) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.NotFoundError):
            await client.audio.translations.create(
                model=model_name,
                file=foscolo,
                temperature=TEMPERATURE,
                extra_body={"seed": SEED},
            )


@pytest.mark.asyncio
async def test_basic_audio_with_lora(mary_had_lamb):
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

    add_attention_backend(server_args, _get_rocm_attention_config(model_name))

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        translation = await client.audio.translations.create(
            model=lora_model_name,
            file=mary_had_lamb,
            extra_body={"seed": SEED, "language": "en", "to_language": "es"},
            response_format="text",
            temperature=TEMPERATURE,
        )

    out = json.loads(translation)["text"].strip()
    assert len(out) > 0, "LoRA translation returned empty text"
    out_lower = out.lower()
    assert "pequeño" in out_lower, (
        f"Expected 'pequeño' in Spanish translation, got: {out!r}"
    )
    assert "mary" in out_lower, f"Expected 'mary' in Spanish translation, got: {out!r}"


# NOTE: (NickLucche) the large-v3-turbo model was not trained on translation!
@pytest.mark.asyncio
async def test_basic_audio(foscolo, client, model_name):
    translation = await client.audio.translations.create(
        model=model_name,
        file=foscolo,
        response_format="text",
        # TODO remove `language="it"` once language detection is implemented
        extra_body={"seed": SEED, "language": "it", "to_language": "en"},
        temperature=TEMPERATURE,
    )
    out = json.loads(translation)["text"].strip()
    assert len(out) > 0, "Translation returned empty text"
    out_lower = out.lower()
    assert "greek sea" in out_lower, (
        f"Expected 'greek sea' in translation, got: {out!r}"
    )
    assert "exile" in out_lower, f"Expected 'exile' in translation, got: {out!r}"


@pytest.mark.asyncio
async def test_audio_prompt(foscolo, client, model_name):
    # Condition whisper on starting text
    prompt = "Nor have I ever"
    translation = await client.audio.translations.create(
        model=model_name,
        file=foscolo,
        prompt=prompt,
        extra_body={"seed": SEED, "language": "it", "to_language": "en"},
        response_format="text",
        temperature=TEMPERATURE,
    )
    out = json.loads(translation)["text"].strip()
    assert len(out) > 0, "Prompted translation returned empty text"
    if model_name == "openai/whisper-small":
        assert "Nor will I ever touch the sacred" not in out, (
            f"Prompt conditioning failed; unprompted text leaked: {out!r}"
        )
        assert prompt not in out, f"Prompt text echoed back in output: {out!r}"
    out_lower = out.lower()
    assert "greek sea" in out_lower, (
        f"Expected 'greek sea' in prompted translation, got: {out!r}"
    )


@pytest.mark.asyncio
async def test_streaming_response(foscolo, client, model_name, server):
    """Streaming output must match non-streaming output."""
    res_no_stream = await client.audio.translations.create(
        model=model_name,
        file=foscolo,
        response_format="json",
        extra_body={"seed": SEED, "language": "it", "to_language": "en"},
        temperature=TEMPERATURE,
    )
    expected_text = res_no_stream.text

    # Stream via HTTPX since OpenAI translation client doesn't expose streaming
    url = server.url_for("v1/audio/translations")
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    data = {
        "model": model_name,
        "language": "it",
        "to_language": "en",
        "stream": True,
        "temperature": str(TEMPERATURE),
        "seed": str(SEED),
    }
    foscolo.seek(0)
    streamed_text = ""
    async with (
        httpx.AsyncClient() as http_client,
        http_client.stream(
            "POST",
            url,
            headers=headers,
            data=data,
            files={"file": foscolo},
        ) as response,
    ):
        async for line in response.aiter_lines():
            if not line:
                continue

            if line.startswith("data: "):
                data = line[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                content = chunk["choices"][0].get("delta", {}).get("content")
                streamed_text += content or ""
            elif line.strip() == "[DONE]":
                break
    # NOTE Run is expected to be deterministic with temperature set to 0.0
    assert streamed_text == expected_text, (
        f"Streaming/non-streaming mismatch.\n"
        f"  Non-stream: {expected_text!r}\n"
        f"  Streamed:   {streamed_text!r}"
    )


@pytest.mark.asyncio
async def test_stream_options(foscolo, model_name, server):
    """stream_include_usage and stream_continuous_usage_stats must work."""
    url = server.url_for("v1/audio/translations")
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    data = {
        "model": model_name,
        "language": "it",
        "to_language": "en",
        "stream": True,
        "stream_include_usage": True,
        "stream_continuous_usage_stats": True,
        "temperature": str(TEMPERATURE),
        "seed": str(SEED),
    }
    foscolo.seek(0)
    got_final_usage = False
    chunks_with_choices = 0
    chunks_missing_usage = 0
    async with (
        httpx.AsyncClient() as http_client,
        http_client.stream(
            "POST",
            url,
            headers=headers,
            data=data,
            files={"file": foscolo},
        ) as response,
    ):
        async for line in response.aiter_lines():
            if not line:
                continue

            if line.startswith("data: "):
                data = line[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    assert "usage" in chunk
                    got_final_usage = True
                else:
                    chunks_with_choices += 1
                    if "usage" not in chunk:
                        chunks_missing_usage += 1
            elif line.strip() == "[DONE]":
                break

    assert got_final_usage, "Never received final usage-only chunk"
    assert chunks_with_choices > 0, "No content chunks received"
    assert chunks_missing_usage == 0, (
        f"{chunks_missing_usage}/{chunks_with_choices} content chunks "
        f"missing continuous usage stats"
    )


@pytest.mark.asyncio
async def test_long_audio_request(foscolo, client, model_name):
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
        extra_body={"seed": SEED, "language": "it", "to_language": "en"},
        response_format="text",
        temperature=TEMPERATURE,
    )
    out = json.loads(translation)["text"].strip()
    assert len(out) > 0, "Long audio translation returned empty text"
    out_lower = out.lower()
    assert out_lower.count("greek sea") == 2, (
        f"Expected 'greek sea' exactly twice in repeated audio translation, "
        f"found {out_lower.count('greek sea')}: {out!r}"
    )


@pytest.mark.asyncio
async def test_audio_with_max_tokens(mary_had_lamb, client, model_name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    transcription = await client.audio.translations.create(
        model=model_name,
        file=mary_had_lamb,
        response_format="text",
        temperature=TEMPERATURE,
        extra_body={"seed": SEED, "max_completion_tokens": 1},
    )
    out = json.loads(transcription)
    out_tokens = tok(out["text"], add_special_tokens=False)["input_ids"]
    assert len(out_tokens) == 1, (
        f"Expected exactly 1 token, got {len(out_tokens)}: {out_tokens}"
    )

    # max_completion_tokens >> max_model_len should not crash
    # max_model_len=32768 for Gemma-3n-E2B-it
    mary_had_lamb.seek(0)
    transcription = await client.audio.transcriptions.create(
        model=model_name,
        file=mary_had_lamb,
        response_format="text",
        temperature=TEMPERATURE,
        extra_body={
            "seed": SEED,
            "max_completion_tokens": int(1e6),
            "repetition_penalty": 1.3,
        },
    )
    out = json.loads(transcription)
    out_tokens = tok(out["text"], add_special_tokens=False)["input_ids"]
    assert len(out_tokens) > 0, "Unconstrained transcription returned no tokens"
    assert len(out_tokens) < 450, (
        f"Unexpectedly long output: {len(out_tokens)} tokens"
    )  # ~Whisper max output len
