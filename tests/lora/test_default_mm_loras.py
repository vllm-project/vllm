# SPDX-License-Identifier: Apache-2.0
"""
Tests for applying default registered multimodal loras.
"""

import pytest

from vllm.lora.request import LoRARequest

from ..conftest import AudioTestAssets, VllmRunner

MODEL = "ibm-granite/granite-speech-3.3-2b"
AUDIO_PROMPT = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|><|audio|>can you transcribe the speech into a written format?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"  # noqa: E501

# Responses are greedy decoded; we just check the end of
# the generated text. If the lora is inactive, this model
# generates commentary on the transcription.
RESPONSE_SUFFIX_WITH_LORA = "the first words i spoke in the original chorus a little piece of practical poetry mary had a little lamb its fleece was white as snow and everywhere that mary went the lamb would surely go"  # noqa: E501
RESPONSE_SUFFIX_WITHOUT_LORA = "This is a simplified transcription of the given text, which appears to be a poetic description or narrative."  # noqa: E501


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


VLLM_RUNNER_BASE_KWARGS = {
    "model_name": MODEL,
    "dtype": "half",
    "enable_lora": "True",
    "max_lora_rank": 64,
    "max_model_len": 2048,
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
        default_mm_loras={"audio": MODEL},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )


def test_inactive_default_mm_lora(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    """Ensure that modalities are filtered properly."""
    run_test(
        vllm_runner,
        audio_assets,
        lora_request=None,
        default_mm_loras={"image": MODEL},
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
        lora_request=LoRARequest("audio", 1, MODEL),
        default_mm_loras={"audio": MODEL},
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
        lora_request=LoRARequest("speech", 2, MODEL),
        default_mm_loras={"audio": "this path is overridden"},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )
