# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.audio import AudioAsset
from vllm.multimodal.audio import AudioResampler
from vllm.platforms import CpuArchEnum, current_platform

if (
    not current_platform.is_cpu()
    or current_platform.get_cpu_architecture() != CpuArchEnum.ARM
):
    pytest.skip("skipping Arm CPU-only tests", allow_module_level=True)

MODEL = "Arm/whisper-large-v3-quantized.w4a8"
PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
WHISPER_SAMPLE_RATE = 16000


def test_cpu_int8_w4a8_whisper(vllm_runner, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    audio, sample_rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    if sample_rate != WHISPER_SAMPLE_RATE:
        audio = AudioResampler(target_sr=WHISPER_SAMPLE_RATE).resample(
            audio, orig_sr=sample_rate
        )

    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_model_len=448,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
    ) as llm:
        outputs = llm.generate_greedy(
            [PROMPT],
            max_tokens=8,
            audios=[(audio, WHISPER_SAMPLE_RATE)],
        )

    assert "the first words" in outputs[0][1].lower()
