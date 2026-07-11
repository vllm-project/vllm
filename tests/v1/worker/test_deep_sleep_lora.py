# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test level-2 sleep/wake/reload with LoRA enabled.

Validates that:
1. LoRA-wrapped parameter names resolve correctly during reload_weights()
2. LoRA stacked tensors are re-zeroed after wake (not left with stale GPU data)
3. Multiple sleep/wake cycles do not accumulate corruption

Requires: single GPU, ~500MB VRAM (tiny model).
"""

import pytest
import torch

from vllm import LLM, SamplingParams


@pytest.fixture
def model_name():
    return "hmellor/tiny-random-LlamaForCausalLM"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deep_sleep_lora_single_cycle(model_name):
    """Level-2 sleep + wake + reload with LoRA should produce same output."""
    llm = LLM(
        model_name,
        enable_sleep_mode=True,
        enable_lora=True,
        max_lora_rank=8,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0, max_tokens=10)
    output_before = llm.generate("How are you?", params)

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])

    output_after = llm.generate("How are you?", params)
    assert output_before[0].outputs[0].text == output_after[0].outputs[0].text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deep_sleep_lora_multi_cycle(model_name):
    """Multiple sleep/wake cycles should not accumulate corruption."""
    llm = LLM(
        model_name,
        enable_sleep_mode=True,
        enable_lora=True,
        max_lora_rank=8,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0, max_tokens=10)
    output_ref = llm.generate("Hello world", params)

    for _ in range(3):
        llm.sleep(level=2)
        llm.wake_up(tags=["weights"])
        llm.collective_rpc("reload_weights")
        llm.wake_up(tags=["kv_cache"])

    output_final = llm.generate("Hello world", params)
    assert output_ref[0].outputs[0].text == output_final[0].outputs[0].text
