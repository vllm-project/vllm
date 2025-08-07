# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for applying default registered multimodal loras.
"""

import os

from huggingface_hub import snapshot_download

from vllm.lora.request import LoRARequest

from ..conftest import AudioTestAssets, VllmRunner

MODEL_PATH = snapshot_download("microsoft/Phi-4-multimodal-instruct")
AUDIO_LORA_PATH = os.path.join(MODEL_PATH, "speech-lora")
IMAGE_LORA_PATH = os.path.join(MODEL_PATH, "vision-lora")

AUDIO_PROMPT = "<|user|><|audio_1|>Can you transcribe this audio?<|end|><|assistant|>"  # noqa: E501

# Responses are greedy decoded; we just check the end of
# the generated text. If the lora is inactive, this model
# generates commentary on the transcription.
RESPONSE_SUFFIX_WITH_LORA = "Spoken text: The first words I spoke in the original chronograph, a little piece of practical poetry. Mary had a little lamb, it slept with quite a snow, and everywhere that Mary went, the lamb was sure to go."  # noqa: E501
RESPONSE_SUFFIX_WITHOUT_LORA = "Certainly! Here is the transcription of the audio you provided:\n\nThe first words I spoke in the original phonograph record: A little piece of practical poetry. Mary had a little lamb; its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go."  # noqa: E501

VLLM_RUNNER_BASE_KWARGS = {
    "model_name": MODEL_PATH,
    "dtype": "half",
    "enable_lora": "True",
    "max_num_seqs": 2,
    "max_lora_rank": 320,
    "max_model_len": 12800,
    "gpu_memory_utilization": 0.8,
    "limit_mm_per_prompt": {
        "audio": 1
    },
    "enforce_eager": True,
}


def run_test(vllm_runner, audio_assets, lora_request, expected_suffix,
             **kwargs):
    inputs = [([AUDIO_PROMPT], [audio_assets[0].audio_and_sample_rate[0]])]

    # Apply any additional kwargs as overrides to the base kwargs
    vllm_runner_kwargs = {**VLLM_RUNNER_BASE_KWARGS, **kwargs}

    with vllm_runner(**vllm_runner_kwargs) as vllm_model:
        vllm_outputs_with_default_lora = [
            vllm_model.generate_greedy(
                prompts,
                max_tokens=128,
                audios=audios,
                lora_request=lora_request,
            ) for prompts, audios in inputs
        ]

        assert vllm_outputs_with_default_lora[-1][-1][-1].endswith(
            expected_suffix)


def test_active_default_mm_lora(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    """Ensure that we can use the default audio lora."""
    run_test(
        vllm_runner,
        audio_assets,
        lora_request=None,
        default_mm_loras={"audio": AUDIO_LORA_PATH},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )


def test_inactive_default_mm_lora(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    """Ensure that modalities are filtered properly."""
    # Default image lora won't be active since we only pass audio
    run_test(
        vllm_runner,
        audio_assets,
        lora_request=None,
        default_mm_loras={"image": IMAGE_LORA_PATH},
        expected_suffix=RESPONSE_SUFFIX_WITHOUT_LORA,
    )


def test_default_mm_lora_succeeds_with_redundant_lora_request(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    """Ensure that redundantly providing the lora works."""
    run_test(
        vllm_runner,
        audio_assets,
        lora_request=LoRARequest("audio", 1, AUDIO_LORA_PATH),
        default_mm_loras={"audio": AUDIO_LORA_PATH},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )


def test_default_mm_lora_fails_with_overridden_lora_request(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    """Ensure that if the lora_request conflicts with default_mm_loras,
    we use the lora_request."""
    run_test(
        vllm_runner,
        audio_assets,
        lora_request=LoRARequest("speech", 2, AUDIO_LORA_PATH),
        default_mm_loras={"audio": IMAGE_LORA_PATH},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )
