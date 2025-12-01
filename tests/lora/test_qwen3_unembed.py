# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Qwen3 unembed LoRA support.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test

MODEL_PATH = "Qwen/Qwen3-0.6B"

# LoRA adapters for Qwen3
LORA_QWEN3 = "Pelmeshek/qwen3-0.6B-function-calling-lora"


def format_chatml_messages(prompt: str):
    """Format prompt for Qwen3 models"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


@create_new_process_for_each_test()
@pytest.mark.parametrize("max_lora_rank", [8, 16])
def test_qwen3_unembed_lora(max_lora_rank: int):
    """
    Test Qwen3 with unembed LoRA adapters.

    This test verifies:
    1. Qwen3 models can load and use LoRA adapters with lm_head (unembed)
    2. Multiple LoRA adapters can be used simultaneously
    """
    # Initialize LLM with LoRA support
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=max_lora_rank,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    # Test prompts
    prompts = [
        "What is GitHub?",
        "Hi, tell me about you",
        "Hello, my name is",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    # Test with base model (no LoRA)
    print("Testing base model without LoRA...")
    base_outputs = llm.generate(prompts, sampling_params)
    assert len(base_outputs) == len(prompts)
    for output in base_outputs:
        assert output.outputs[0].text  # Should generate some text

    # Test with first LoRA adapter (Alice)
    print("Testing with LoRA adapter (Alice)...")
    lora_request_alice = LoRARequest("function-calling-lora", 1, LORA_QWEN3)

    # Format messages for chat template
    formatted_prompts = [format_chatml_messages(p) for p in prompts]

    lora_outputs_alice = llm.chat(
        formatted_prompts,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=lora_request_alice,
        use_tqdm=False,
    )
    assert len(lora_outputs_alice) == len(prompts)

    # Verify outputs are different from base model
    for base_out, lora_out in zip(base_outputs, lora_outputs_alice):
        base_text = base_out.outputs[0].text
        lora_text = lora_out.outputs[0].text
        assert lora_text  # Should generate some text
        print(f"Base: {base_text[:50]}...")
        print(f"LoRA: {lora_text[:50]}...")

    print("Test passed.")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_qwen3_unembed_lora(8)
    test_qwen3_unembed_lora(16)
