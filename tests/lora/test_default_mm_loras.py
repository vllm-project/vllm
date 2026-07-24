# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for applying default registered multimodal loras.
"""

import unittest.mock as mock

import pytest

from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

from ..conftest import AudioTestAssets, VllmRunner
from ..utils import create_new_process_for_each_test

AUDIO_PROMPT = "<|user|><|audio_1|>Can you transcribe this audio?<|end|><|assistant|>"  # noqa: E501

# Responses are greedy decoded; we just check the end of
# the generated text. If the lora is inactive, this model
# generates commentary on the transcription.
RESPONSE_SUFFIX_WITH_LORA = "Spoken text: The first words I spoke in the original chronograph, a little piece of practical poetry. Mary had a little lamb, it slept with quite a snow, and everywhere that Mary went, the lamb was sure to go."  # noqa: E501
RESPONSE_SUFFIX_WITHOUT_LORA = "Certainly! Here is the transcription of the audio you provided:\n\nThe first words I spoke in the original phonograph record: A little piece of practical poetry. Mary had a little lamb; its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go."  # noqa: E501

VLLM_RUNNER_BASE_KWARGS = {
    "dtype": "half",
    "enable_lora": "True",
    "max_num_seqs": 2,
    "max_lora_rank": 320,
    # Keep these LoRA tests on short-RoPE for determinism post-LongRoPE change.
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.8,
    "limit_mm_per_prompt": {"audio": 1},
    "enforce_eager": True,
}


def run_test(
    vllm_runner,
    audio_assets,
    monkeypatch,
    model_path,
    lora_request,
    expected_suffix,
    **kwargs,
):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    inputs = [([AUDIO_PROMPT], [audio_assets[0].audio_and_sample_rate[0]])]

    # Apply any additional kwargs as overrides to the base kwargs
    vllm_runner_kwargs = {
        "model_name": model_path,
        **VLLM_RUNNER_BASE_KWARGS,
        **kwargs,
    }

    with vllm_runner(**vllm_runner_kwargs) as vllm_model:
        vllm_outputs_with_default_lora = [
            vllm_model.generate_greedy(
                prompts,
                max_tokens=128,
                audios=audios,
                lora_request=lora_request,
            )
            for prompts, audios in inputs
        ]

        assert vllm_outputs_with_default_lora[-1][-1][-1].endswith(expected_suffix)


@create_new_process_for_each_test()
def test_active_default_mm_lora(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    monkeypatch: pytest.MonkeyPatch,
    phi4_multimodal_model_path: str,
    phi4_multimodal_audio_lora_path: str,
):
    """Ensure that we can use the default audio lora."""
    run_test(
        vllm_runner,
        audio_assets,
        monkeypatch,
        model_path=phi4_multimodal_model_path,
        lora_request=None,
        default_mm_loras={"audio": phi4_multimodal_audio_lora_path},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@create_new_process_for_each_test()
def test_inactive_default_mm_lora(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    monkeypatch: pytest.MonkeyPatch,
    phi4_multimodal_model_path: str,
    phi4_multimodal_vision_lora_path: str,
):
    """Ensure that modalities are filtered properly."""
    # Default image lora won't be active since we only pass audio
    run_test(
        vllm_runner,
        audio_assets,
        monkeypatch,
        model_path=phi4_multimodal_model_path,
        lora_request=None,
        default_mm_loras={"image": phi4_multimodal_vision_lora_path},
        expected_suffix=RESPONSE_SUFFIX_WITHOUT_LORA,
    )


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@create_new_process_for_each_test()
def test_default_mm_lora_succeeds_with_redundant_lora_request(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    monkeypatch: pytest.MonkeyPatch,
    phi4_multimodal_model_path: str,
    phi4_multimodal_audio_lora_path: str,
):
    """Ensure that redundantly providing the lora works."""
    run_test(
        vllm_runner,
        audio_assets,
        monkeypatch,
        model_path=phi4_multimodal_model_path,
        lora_request=LoRARequest("audio", 1, phi4_multimodal_audio_lora_path),
        default_mm_loras={"audio": phi4_multimodal_audio_lora_path},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@create_new_process_for_each_test()
def test_default_mm_lora_fails_with_overridden_lora_request(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    monkeypatch: pytest.MonkeyPatch,
    phi4_multimodal_model_path: str,
    phi4_multimodal_audio_lora_path: str,
    phi4_multimodal_vision_lora_path: str,
):
    """Ensure that if the lora_request conflicts with default_mm_loras,
    we use the lora_request."""
    run_test(
        vllm_runner,
        audio_assets,
        monkeypatch,
        model_path=phi4_multimodal_model_path,
        lora_request=LoRARequest("speech", 2, phi4_multimodal_audio_lora_path),
        default_mm_loras={"audio": phi4_multimodal_vision_lora_path},
        expected_suffix=RESPONSE_SUFFIX_WITH_LORA,
    )


@create_new_process_for_each_test()
def test_default_mm_lora_does_not_expand_string_reqs(
    vllm_runner,
    monkeypatch,
    phi4_multimodal_model_path: str,
    phi4_multimodal_audio_lora_path: str,
):
    # See run_test: force spawn to avoid the forked-child CUDA re-init crash.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    class MockEngineException(Exception):
        pass

    # Regression test for ensuring default multimodal lora resolution
    # does not expand the lora req if the prompt type is a string.
    vllm_runner_kwargs = {
        "model_name": phi4_multimodal_model_path,
        **VLLM_RUNNER_BASE_KWARGS,
        **{"default_mm_loras": {"audio": phi4_multimodal_audio_lora_path}},
    }

    # Avoid the full generation call since these tests are expensive;
    # just check what lora request is actually submitted to the engine
    mock_err = "Engine is mocked for this test"

    with (
        mock.patch(
            "vllm.v1.engine.llm_engine.LLMEngine.add_request",
            side_effect=MockEngineException(mock_err),
        ) as mock_add_request,
        vllm_runner(**vllm_runner_kwargs) as vllm_model,
    ):
        # Die once we actually submit the request to the engine
        with pytest.raises(MockEngineException):
            vllm_model.llm.generate(prompts=AUDIO_PROMPT)

        # Then check to make sure the submitted lora request
        # and text prompt were zipped together correctly
        engine_args, engine_kwargs = mock_add_request.call_args
        assert engine_args[1]["prompt"] == AUDIO_PROMPT
        assert engine_kwargs["lora_request"] is None
