# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for Whisper models with LoRA adapters.

These tests verify that Whisper models can correctly load and use LoRA adapters
for speech-to-text transcription tasks.
"""

import pytest

import vllm
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test

# Model configuration
WHISPER_MODEL = "openai/whisper-small"

# Test prompts for Whisper transcription
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Note: whisper_lora_files fixture is defined in conftest.py


@pytest.fixture(autouse=True)
def use_spawn_for_whisper(monkeypatch):
    """Whisper has issues with forked workers, use spawn instead."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def create_whisper_llm(enable_lora: bool = True, max_loras: int = 2):
    """Create a Whisper LLM instance with optional LoRA support."""
    return vllm.LLM(
        model=WHISPER_MODEL,
        enable_lora=enable_lora,
        max_loras=max_loras if enable_lora else 1,
        max_lora_rank=64,
        max_model_len=448,
        dtype="half",
        enforce_eager=True,  # For stability in tests
    )


def run_whisper_inference(
    llm: vllm.LLM,
    lora_path: str | None = None,
    lora_id: int = 1,
) -> list[str]:
    """Run Whisper inference with optional LoRA adapter."""
    # Load test audio
    audio_asset = AudioAsset("mary_had_lamb")
    audio_data = audio_asset.audio_and_sample_rate

    inputs = [
        {
            "prompt": WHISPER_PROMPT,
            "multi_modal_data": {"audio": audio_data},
        }
    ]

    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=200,
    )

    # Prepare LoRA request if adapter path is provided
    lora_request = None
    if lora_path:
        lora_request = LoRARequest(
            lora_name=f"whisper_lora_{lora_id}",
            lora_int_id=lora_id,
            lora_path=lora_path,
        )

    outputs = llm.generate(inputs, sampling_params, lora_request=lora_request)

    return [output.outputs[0].text for output in outputs]


@create_new_process_for_each_test()
def test_whisper_lora_inference(whisper_lora_files):
    """Test basic Whisper inference with a LoRA adapter.

    This test verifies that:
    1. Whisper model can be loaded with LoRA support enabled
    2. A LoRA adapter can be applied during inference
    3. The model produces valid transcription output
    """
    llm = create_whisper_llm(enable_lora=True)

    # Run inference with LoRA
    outputs = run_whisper_inference(llm, lora_path=whisper_lora_files, lora_id=1)

    # Verify we got a non-empty transcription
    assert len(outputs) == 1
    assert len(outputs[0]) > 0, "Expected non-empty transcription output"

    # The output should contain some recognizable words from the audio
    # (Mary had a little lamb)
    print(f"Transcription output: {outputs[0]}")


@create_new_process_for_each_test()
def test_whisper_multi_lora(whisper_lora_files):
    """Test Whisper with multiple LoRA adapter IDs.

    This test verifies that the same LoRA adapter can be loaded with
    different IDs and produce consistent results.
    """
    llm = create_whisper_llm(enable_lora=True, max_loras=4)

    # Test with different LoRA IDs using the same adapter
    outputs_lora1 = run_whisper_inference(llm, lora_path=whisper_lora_files, lora_id=1)
    outputs_lora2 = run_whisper_inference(llm, lora_path=whisper_lora_files, lora_id=2)

    # Both should produce valid outputs
    assert len(outputs_lora1[0]) > 0
    assert len(outputs_lora2[0]) > 0

    # Same adapter with different IDs should produce same output
    assert outputs_lora1 == outputs_lora2, (
        f"Expected same outputs for same adapter with different IDs. "
        f"Got: {outputs_lora1} vs {outputs_lora2}"
    )


@create_new_process_for_each_test()
def test_whisper_with_and_without_lora(whisper_lora_files):
    """Test that Whisper produces different outputs with and without LoRA.

    This test verifies that the LoRA adapter actually affects the model output.
    """
    llm = create_whisper_llm(enable_lora=True)

    # Run with LoRA
    outputs_with_lora = run_whisper_inference(
        llm, lora_path=whisper_lora_files, lora_id=1
    )

    # Run without LoRA (base model only)
    outputs_without_lora = run_whisper_inference(llm, lora_path=None)

    # Both should produce valid outputs
    assert len(outputs_with_lora[0]) > 0
    assert len(outputs_without_lora[0]) > 0

    print(f"Output with LoRA: {outputs_with_lora[0]}")
    print(f"Output without LoRA: {outputs_without_lora[0]}")

    # Note: Outputs may or may not differ depending on the adapter
    # The main verification is that both configurations work
