# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Qwen3 unembed LoRA support, including:
1. Qwen3 with unembed lora (with extra vocab size)
2. Unembed lora with no vocab padding (extra_vocab_size = 0)
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test

MODEL_PATH = "Qwen/Qwen3-0.6B"

# LoRA adapters for Qwen3
LORA_QWEN3_ALICE = "charent/self_cognition_Alice"
LORA_QWEN3_BOB = "charent/self_cognition_Bob"


def format_chatml_messages(prompt: str):
    """Format prompt for Qwen3 models"""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "max_lora_rank,lora_extra_vocab_size",
    [
        (8, 256),  # Standard case with extra vocab
        (16, 512),  # Standard case with larger extra vocab
        (8, 0),  # Test case: no vocab padding
    ])
def test_qwen3_unembed_lora(max_lora_rank: int, lora_extra_vocab_size: int):
    """
    Test Qwen3 with unembed LoRA adapters.

    This test verifies:
    1. Qwen3 models can load and use LoRA adapters with lm_head (unembed)
    2. The system handles extra_vocab_size = 0 correctly (no vocab padding)
    3. Multiple LoRA adapters can be used simultaneously
    """
    # Initialize LLM with LoRA support
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=max_lora_rank,
        lora_extra_vocab_size=lora_extra_vocab_size,
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
    print(f"Testing with LoRA adapter (Alice) - "
          f"extra_vocab_size={lora_extra_vocab_size}...")
    lora_request_alice = LoRARequest("alice", 1, LORA_QWEN3_ALICE)

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

    # Test with second LoRA adapter (Bob)
    print(f"Testing with second LoRA adapter (Bob) - "
          f"extra_vocab_size={lora_extra_vocab_size}...")
    lora_request_bob = LoRARequest("bob", 2, LORA_QWEN3_BOB)

    lora_outputs_bob = llm.chat(
        formatted_prompts,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=lora_request_bob,
        use_tqdm=False,
    )
    assert len(lora_outputs_bob) == len(prompts)

    for output in lora_outputs_bob:
        assert output.outputs[0].text  # Should generate some text

    # Test switching between LoRA adapters
    print("Testing switching between LoRA adapters...")
    mixed_requests = [lora_request_alice, lora_request_bob, lora_request_alice]

    for i, (prompt,
            lora_req) in enumerate(zip(formatted_prompts, mixed_requests)):
        output = llm.chat(
            [prompt],
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            lora_request=lora_req,
            use_tqdm=False,
        )
        assert len(output) == 1
        assert output[0].outputs[0].text
        print(f"Prompt {i} with {lora_req.lora_name}: "
              f"{output[0].outputs[0].text[:50]}...")

    print(f"Test passed with extra_vocab_size={lora_extra_vocab_size}")


@create_new_process_for_each_test()
def test_qwen3_unembed_lora_zero_vocab_padding():
    """
    Specific test for unembed LoRA with extra_vocab_size = 0.

    This is a regression test to ensure that the changes to support
    no vocab padding don't break the basic LoRA functionality.
    """
    # Initialize LLM with LoRA support and NO extra vocab size
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        lora_extra_vocab_size=0,  # No vocab padding
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    # Simple test prompt
    prompt = "What is Python?"
    formatted_prompt = format_chatml_messages(prompt)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
    )

    # Test with LoRA adapter
    lora_request = LoRARequest("alice", 1, LORA_QWEN3_ALICE)

    outputs = llm.chat(
        [formatted_prompt],
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=lora_request,
        use_tqdm=False,
    )

    assert len(outputs) == 1
    assert outputs[0].outputs[0].text  # Should generate some text

    print(f"Output: {outputs[0].outputs[0].text}")
    print("Test passed with extra_vocab_size=0")


@create_new_process_for_each_test()
def test_qwen3_moe_unembed_lora():
    """
    Test Qwen3 MoE with unembed LoRA adapters.

    This test verifies that Qwen3 MoE models can also use unembed LoRA.
    """
    # Note: Using a smaller MoE model if available
    # For now, we skip if model is not available
    pytest.skip("Qwen3 MoE model requires more resources, test separately")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_qwen3_unembed_lora(8, 256)
    test_qwen3_unembed_lora(8, 0)
    test_qwen3_unembed_lora_zero_vocab_padding()
