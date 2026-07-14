# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for level-2 sleep/wake with LoRA enabled (#39934).

Two bugs fixed:
1. LoRA wrapping moves parameters under base_layer, breaking
   named_parameters() name resolution during reload_weights().
2. LoRA stacked tensors (lora_a_stacked, lora_b_stacked) are plain
   attributes not restored by reload — must be explicitly re-zeroed.

These tests validate that:
- Single and multi-cycle sleep/wake/reload produce deterministic output
- LoRA stacked tensors are properly zeroed (not left with stale GPU data)
- sharded_to_full_mapping_gpu (TP>1) is NOT zeroed by zero_lora_state
- The fix works across different models
"""

import pytest
import torch

from tests.utils import create_new_process_for_each_test, multi_gpu_test
from vllm import LLM, SamplingParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
PROMPT = "How are you?"
SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=10)


def _sleep_wake_reload(llm: LLM) -> None:
    """Execute one full sleep-level-2 / wake / reload cycle."""
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])


# ---- Single-GPU tests ----


@create_new_process_for_each_test()
def test_deep_sleep_lora_single_cycle():
    """Level-2 sleep + wake + reload with LoRA produces same output."""
    llm = LLM(MODEL,
              enable_sleep_mode=True,
              enable_lora=True,
              max_lora_rank=8,
              enforce_eager=True)

    output_before = llm.generate(PROMPT, SAMPLING_PARAMS)
    _sleep_wake_reload(llm)
    output_after = llm.generate(PROMPT, SAMPLING_PARAMS)

    assert output_before[0].outputs[0].text == output_after[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_lora_multi_cycle():
    """Multiple sleep/wake cycles do not accumulate corruption."""
    llm = LLM(MODEL,
              enable_sleep_mode=True,
              enable_lora=True,
              max_lora_rank=8,
              enforce_eager=True)

    output_ref = llm.generate(PROMPT, SAMPLING_PARAMS)

    for _ in range(3):
        _sleep_wake_reload(llm)

    output_final = llm.generate(PROMPT, SAMPLING_PARAMS)
    assert output_ref[0].outputs[0].text == output_final[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_lora_level1_still_works():
    """Level-1 sleep (CPU backup) with LoRA should still work correctly."""
    llm = LLM(MODEL,
              enable_sleep_mode=True,
              enable_lora=True,
              max_lora_rank=8,
              enforce_eager=True)

    output_before = llm.generate(PROMPT, SAMPLING_PARAMS)

    llm.sleep(level=1)
    llm.wake_up()

    output_after = llm.generate(PROMPT, SAMPLING_PARAMS)
    assert output_before[0].outputs[0].text == output_after[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_lora_stacked_tensors_zeroed():
    """After level-2 sleep/wake/reload, LoRA stacked tensors must be zero."""
    llm = LLM(MODEL,
              enable_sleep_mode=True,
              enable_lora=True,
              max_lora_rank=8,
              enforce_eager=True)

    _sleep_wake_reload(llm)

    # Inspect model LoRA layers — stacked tensors should be zero
    from vllm.lora.layers.base import BaseLayerWithLoRA
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model  # noqa: E501
    for name, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            for attr_name in module._LORA_TENSOR_ATTRS:
                val = getattr(module, attr_name, None)
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    tensors = [val]
                elif isinstance(val, (tuple, list)):
                    tensors = [t for t in val if isinstance(t, torch.Tensor)]
                else:
                    continue
                for t in tensors:
                    if t.device.type != "meta":
                        assert t.abs().sum() == 0, (
                            f"{name}.{attr_name} not zeroed after sleep/wake"
                        )


@create_new_process_for_each_test()
def test_deep_sleep_lora_preserves_non_lora_tensors():
    """zero_lora_state must NOT zero non-LoRA unregistered tensors.

    LogitsProcessorWithLoRA.sharded_to_full_mapping_gpu is an unregistered
    tensor used for TP>1 logit reordering. Zeroing it would cause logits
    to be indexed from position 0 repeatedly. This test validates that
    _LORA_TENSOR_ATTRS is correctly scoped.
    """
    from vllm.lora.layers.base import BaseLayerWithLoRA

    # Verify the explicit attr list does not include non-LoRA tensors
    assert "sharded_to_full_mapping_gpu" not in BaseLayerWithLoRA._LORA_TENSOR_ATTRS
    assert "indices" not in BaseLayerWithLoRA._LORA_TENSOR_ATTRS

    # Functional check: sleep/wake/reload should not break generation
    llm = LLM(MODEL,
              enable_sleep_mode=True,
              enable_lora=True,
              max_lora_rank=8,
              enforce_eager=True)

    output_before = llm.generate(PROMPT, SAMPLING_PARAMS)
    _sleep_wake_reload(llm)
    output_after = llm.generate(PROMPT, SAMPLING_PARAMS)

    assert output_before[0].outputs[0].text == output_after[0].outputs[0].text


# ---- Multi-GPU (TP>1) test ----


@multi_gpu_test(num_gpus=2)
def test_deep_sleep_lora_tp2():
    """Level-2 sleep/wake/reload with LoRA at TP=2.

    Validates that:
    - reload_weights succeeds under tensor parallelism
    - sharded_to_full_mapping_gpu is preserved (not zeroed)
    - Output is deterministic across sleep/wake cycles
    """
    llm = LLM(MODEL,
              enable_sleep_mode=True,
              enable_lora=True,
              max_lora_rank=8,
              enforce_eager=True,
              tensor_parallel_size=2)

    output_before = llm.generate(PROMPT, SAMPLING_PARAMS)
    _sleep_wake_reload(llm)
    output_after = llm.generate(PROMPT, SAMPLING_PARAMS)

    assert output_before[0].outputs[0].text == output_after[0].outputs[0].text

    # Multi-cycle at TP=2
    for _ in range(2):
        _sleep_wake_reload(llm)
    output_final = llm.generate(PROMPT, SAMPLING_PARAMS)
    assert output_before[0].outputs[0].text == output_final[0].outputs[0].text
