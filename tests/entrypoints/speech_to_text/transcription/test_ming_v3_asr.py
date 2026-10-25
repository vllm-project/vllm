# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight Ming ASR API coverage using the vendor's public examples."""

import asyncio
import json
import os
from io import BytesIO

import httpx
import numpy as np
import pytest
import soundfile as sf

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.audio import AudioResampler

MODEL = os.environ.get("VLLM_TEST_MING_ASR_MODEL", "inclusionAI/Ming-3.0-asr-tiny")
SERVED_MODEL = "inclusionAI/Ming-3.0-asr-tiny"
ASSETS = (
    "https://raw.githubusercontent.com/inclusionAI/Ming-3.0-asr/"
    "6141e8c552815db724d8df0f72c9d865c547a1ea/examples/cases/audio/"
)


@pytest.fixture(scope="module")
def server():
    args = [
        "--trust-remote-code",
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "1024",
        "--served-model-name",
        SERVED_MODEL,
        "--logits-processors",
        "vllm.model_executor.models.ming_v3_asr_logits_processor:NoRepeatNGramLogitsProcessor",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote:
        yield remote


@pytest.fixture(scope="module")
def audio():
    with httpx.Client() as client:
        clips = {}
        for name in (
            "english_common_voice.wav",
            "chinese_common_voice.wav",
            "context_same_sentence_pcm16.wav",
        ):
            response = client.get(ASSETS + name)
            response.raise_for_status()
            clips[name] = response.content
        return clips


async def _transcribe(client, clip, **options):
    xargs = options.pop("vllm_xargs", {"no_repeat_ngram_size": 10})
    response = await client.audio.transcriptions.create(
        model=SERVED_MODEL,
        file=("audio.wav", clip, "audio/wav"),
        temperature=0,
        extra_body={
            "max_completion_tokens": 256,
            "vllm_xargs": json.dumps(xargs),
            **options,
        },
    )
    assert "\t" not in response.text
    return response.text


@pytest.mark.asyncio
async def test_public_examples(server, audio):
    """Recognize both languages and apply the vendor's context hints through HTTP."""
    cases = [
        ("english_common_voice.wav", {}, "same way you did"),
        ("chinese_common_voice.wav", {}, "漓江和湘江有秦代开凿的灵渠沟通"),
        (
            "context_same_sentence_pcm16.wav",
            {
                "language": "zh",
                "prompt": "This is an audio about 同囚.",
                "hotwords": (
                    "同囚,黄国权,游学修,关楚耀,赵永洪,赵咏瑶,狱警,长镜头,"
                    "吞玻璃戏,糖化玻璃,真实案例数据."
                ),
            },
            "同囚",
        ),
    ]
    async with server.get_async_client() as client:
        texts = await asyncio.gather(
            *(_transcribe(client, audio[name], **options) for name, options, _ in cases)
        )
    for text, (_, _, expected) in zip(texts, cases):
        assert expected in text


@pytest.mark.asyncio
async def test_streaming(server, audio):
    """The public SDK must receive transcript deltas without a language envelope."""
    async with server.get_async_client() as client:
        response = await client.audio.transcriptions.create(
            model=SERVED_MODEL,
            file=("audio.wav", audio["english_common_voice.wav"], "audio/wav"),
            language="en",
            temperature=0,
            stream=True,
            extra_body={"vllm_xargs": json.dumps({"no_repeat_ngram_size": 10})},
        )
        deltas = []
        async for chunk in response:
            if chunk.choices:
                deltas.append(chunk.choices[0]["delta"].get("content") or "")
    assert "".join(deltas) == "same way you did"


@pytest.mark.asyncio
async def test_long_audio_prefix_cache(server, audio):
    """Long requests must increase prefix hits across audio chunk policies."""
    waveform, rate = sf.read(
        BytesIO(audio["chinese_common_voice.wav"]), dtype="float32"
    )
    waveform = AudioResampler(16000).resample(waveform, orig_sr=rate)
    waveform = np.pad(waveform, (0, 64 * 16000 - len(waveform)))
    clip = BytesIO()
    sf.write(clip, waveform, 16000, format="WAV", subtype="PCM_16")
    async with httpx.AsyncClient() as metrics, server.get_async_client() as client:

        async def prefix_hits():
            response = await metrics.get(server.url_for("metrics"))
            response.raise_for_status()
            return sum(
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith("vllm:prefix_cache_hits_total{")
            )

        before = await prefix_hits()
        for frames in (3000, 3000, 6000, 3000):
            text = await _transcribe(
                client,
                clip.getvalue(),
                language="zh",
                vllm_xargs={"audio_chunk_frames": frames, "no_repeat_ngram_size": 10},
            )
            assert "漓江和湘江有秦代开凿的灵渠沟通" in text
        after = await prefix_hits()
        assert after > before, f"Long requests added no prefix hits: {before=} {after=}"
