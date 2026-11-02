# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from io import BytesIO

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi import UploadFile
from transformers import WhisperConfig

from tests.utils import ROCM_EXTRA_ARGS, RemoteOpenAIServer
from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.multimodal.audio import AudioResampler

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]
MING_ASR_ARGS = [
    "--trust-remote-code",
    "--max-model-len",
    "8192",
    *(ROCM_EXTRA_ARGS or ["--max-num-seqs", "4"]),
    "--max-num-batched-tokens",
    "1024",
    "--logits-processors",
    "vllm.model_executor.models.ming_v3_asr_logits_processor:NoRepeatNGramLogitsProcessor",
]


@pytest.fixture
def ming_server(model_name, request):
    args = ["--enforce-eager", *MING_ASR_ARGS]
    options = getattr(request, "param", {})
    if options.get("enable_prefix_caching", False):
        args = [arg for arg in args if arg != "--no-enable-prefix-caching"]
        args.append("--enable-prefix-caching")
    with RemoteOpenAIServer(model_name, args) as remote:
        yield remote


@pytest.fixture(scope="module")
def ming_audio():
    base_url = (
        "https://raw.githubusercontent.com/inclusionAI/Ming-3.0-asr/"
        "6141e8c552815db724d8df0f72c9d865c547a1ea/examples/cases/audio/"
    )
    with httpx.Client() as client:
        clips = {}
        for name in (
            "chinese_common_voice.wav",
            "context_same_sentence_pcm16.wav",
        ):
            response = client.get(base_url + name)
            response.raise_for_status()
            clips[name] = response.content
        return clips


@pytest.mark.parametrize("encoding", ["absent", "dict", "json"])
def test_request_extensions_reach_prompt_and_sampling_params(tmp_path, encoding):
    """Both request consumers must receive extensions, including form JSON."""
    WhisperConfig().save_pretrained(tmp_path)
    config = ModelConfig(model=str(tmp_path))
    extensions = {"frontend_window": 128, "custom_tags": ["speech", 1]}
    options = {}
    if encoding != "absent":
        options["vllm_xargs"] = (
            json.dumps(extensions) if encoding == "json" else extensions
        )
    request = TranscriptionRequest(file=UploadFile(BytesIO()), **options)
    params = request.build_stt_params(
        audio=np.zeros(160, dtype=np.float32),
        stt_config=SpeechToTextConfig(),
        model_config=config,
        task_type="transcribe",
    )

    expected = None if encoding == "absent" else extensions
    assert params.extra_args == expected
    assert (request.to_sampling_params(16).extra_args or {}) == (expected or {})


async def transcribe_and_check(
    client,
    model_name: str,
    file,
    *,
    language: str | None,
    expected_text: str,
    expected_seconds: int | None = None,
    case_sensitive: bool = False,
    response_format: str = "text",
    extra_body: dict | None = None,
):
    """Run a transcription request and assert the output contains
    *expected_text* and optionally that usage reports *expected_seconds*.

    Provides detailed failure messages with the actual transcription output.
    """
    transcription = await client.audio.transcriptions.create(
        model=model_name,
        file=file,
        language=language,
        response_format=response_format,
        temperature=0.0,
        extra_body=extra_body,
    )
    out = (
        json.loads(transcription)
        if isinstance(transcription, str)
        else transcription.model_dump()
    )
    out_text = out["text"]

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
        out_usage = out["usage"]
        assert out_usage["seconds"] == expected_seconds, (
            f"Expected {expected_seconds}s of audio, "
            f"got {out_usage['seconds']}s. Full usage: {out_usage!r}"
        )

    return out_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "mistralai/Voxtral-Mini-3B-2507",
        "Qwen/Qwen3-ASR-0.6B",
        "inclusionAI/Ming-3.0-asr-tiny",
    ],
)
async def test_basic_audio(mary_had_lamb, model_name):
    server_args = ["--enforce-eager", *ROCM_EXTRA_ARGS]
    extra_body = None
    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS
    elif model_name == "inclusionAI/Ming-3.0-asr-tiny":
        server_args = ["--enforce-eager", *MING_ASR_ARGS]
        extra_body = {
            "max_completion_tokens": 256,
            "vllm_xargs": json.dumps({"no_repeat_ngram_size": 10}),
        }

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            model_name,
            mary_had_lamb,
            language="en",
            expected_text="Mary had a little lamb",
            expected_seconds=16,
            extra_body=extra_body,
        )


@pytest.mark.asyncio
async def test_basic_audio_with_lora(mary_had_lamb):
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

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
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
async def test_basic_audio_foscolo(foscolo, model_name):
    # Gemma accuracy on some of the audio samples we use is particularly bad,
    # hence we use a different one here. WER is evaluated separately.
    server_args = ["--enforce-eager", *ROCM_EXTRA_ARGS]

    with RemoteOpenAIServer(
        model_name,
        server_args,
        max_wait_seconds=480,
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            model_name,
            foscolo,
            language="it",
            expected_text="ove il mio corpo fanciulletto",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["inclusionAI/Ming-3.0-asr-tiny"])
async def test_ming_context(model_name, ming_server, ming_audio):
    """Topic and hotword hints must reach the native ASR prompt over HTTP."""
    async with ming_server.get_async_client() as client:
        text = await transcribe_and_check(
            client,
            model_name,
            ("context.wav", ming_audio["context_same_sentence_pcm16.wav"], "audio/wav"),
            language=None,
            expected_text="同囚",
            response_format="json",
            extra_body={
                "max_completion_tokens": 256,
                "vllm_xargs": json.dumps({"no_repeat_ngram_size": 10}),
                "prompt": "This is an audio about 同囚.",
                "hotwords": (
                    "同囚,黄国权,游学修,关楚耀,赵永洪,赵咏瑶,狱警,长镜头,"
                    "吞玻璃戏,糖化玻璃,真实案例数据."
                ),
            },
        )
    assert "\t" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["inclusionAI/Ming-3.0-asr-tiny"])
@pytest.mark.parametrize(
    "ming_server", [{"enable_prefix_caching": True}], indirect=True
)
async def test_ming_chunk_policy_prefix_cache(model_name, ming_server, ming_audio):
    """Restoring a long request's chunk policy must reuse its cached prefix."""
    waveform, rate = sf.read(
        BytesIO(ming_audio["chinese_common_voice.wav"]), dtype="float32"
    )
    waveform = AudioResampler(16000).resample(waveform, orig_sr=rate)
    waveform = np.pad(waveform, (0, 64 * 16000 - len(waveform)))
    clip = BytesIO()
    sf.write(clip, waveform, 16000, format="WAV", subtype="PCM_16")
    async with httpx.AsyncClient() as metrics, ming_server.get_async_client() as client:

        async def prefix_hits():
            response = await metrics.get(ming_server.url_for("metrics"))
            response.raise_for_status()
            return sum(
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith("vllm:prefix_cache_hits_total{")
            )

        for frames in (3000, 6000, 3000):
            before = await prefix_hits()
            text = await transcribe_and_check(
                client,
                model_name,
                ("audio.wav", clip.getvalue(), "audio/wav"),
                language="zh",
                expected_text="漓江和湘江有秦代开凿的灵渠沟通",
                response_format="json",
                extra_body={
                    "max_completion_tokens": 256,
                    "vllm_xargs": json.dumps(
                        {"audio_chunk_frames": frames, "no_repeat_ngram_size": 10}
                    ),
                },
            )
            assert "\t" not in text
        after = await prefix_hits()
        assert after > before, (
            f"Restoring the chunk policy added no prefix hits: {before=} {after=}"
        )
