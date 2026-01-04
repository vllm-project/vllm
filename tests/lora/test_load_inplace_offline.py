# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that load_inplace works in offline mode (using LLM class directly).
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def llm_instance():
    """Create a shared LLM instance for all tests."""
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_model_len=512,
    )
    return llm


def test_load_inplace_offline_reload(
    llm_instance, qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """
    Test that load_inplace=True allows reloading LoRA adapters with the same ID
    in offline mode (using LLM class directly).
    """
    llm = llm_instance
    adapter_id = 1
    messages = [
        {"content": "Follow the instructions to make animal noises", "role": "system"},
        {"content": "Make your favorite animal noise.", "role": "user"},
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # Load meowing LoRA with load_inplace=True
    meowing_request = LoRARequest(
        lora_name="test-adapter",
        lora_int_id=adapter_id,
        lora_path=qwen3_meowing_lora_files,
    )

    outputs = llm.chat([messages], sampling_params, lora_request=meowing_request)
    first_output = outputs[0].outputs[0].text.strip()
    assert "Meow Meow Meow" in first_output, (
        f"Expected meowing output, got: {first_output}"
    )

    # Reload with woofing LoRA (same ID, different weights, load_inplace=True)
    woofing_request = LoRARequest(
        lora_name="test-adapter-woof",
        lora_int_id=adapter_id,  # Same ID
        lora_path=qwen3_woofing_lora_files,  # Different weights
        load_inplace=True,  # Force reload
    )

    outputs = llm.chat([messages], sampling_params, lora_request=woofing_request)
    second_output = outputs[0].outputs[0].text.strip()
    assert "Woof Woof Woof" in second_output, (
        f"Expected woofing output, got: {second_output}"
    )


def test_load_inplace_false_no_reload(
    llm_instance, qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """
    Test that load_inplace=False prevents reloading when an adapter
    with the same ID already exists.
    """
    llm = llm_instance
    adapter_id = 2
    messages = [
        {"content": "Follow the instructions to make animal noises", "role": "system"},
        {"content": "Make your favorite animal noise.", "role": "user"},
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # Load meowing LoRA first with load_inplace=True
    meowing_request_initial = LoRARequest(
        lora_name="test-adapter-2",
        lora_int_id=adapter_id,
        lora_path=qwen3_meowing_lora_files,
    )

    outputs = llm.chat(
        [messages], sampling_params, lora_request=meowing_request_initial
    )
    first_output = outputs[0].outputs[0].text.strip()
    assert "Meow Meow Meow" in first_output, (
        f"Expected meowing output, got: {first_output}"
    )

    # Try to load woofing LoRA with same ID but load_inplace=False
    # This should NOT reload (adapter 2 already exists)
    woofing_request_no_reload = LoRARequest(
        lora_name="test-adapter-2-woof",
        lora_int_id=adapter_id,  # Same ID
        lora_path=qwen3_woofing_lora_files,
    )

    outputs = llm.chat(
        [messages], sampling_params, lora_request=woofing_request_no_reload
    )
    second_output = outputs[0].outputs[0].text.strip()
    # Should still get meowing output because it didn't reload
    assert "Meow Meow Meow" in second_output, (
        f"Expected meowing output (no reload), got: {second_output}"
    )
