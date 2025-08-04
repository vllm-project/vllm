# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


@pytest.mark.parametrize(
    "model_path",
    [("nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized")])
def test_llama(vllm_runner, example_prompts, model_path):
    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                  max_tokens=20)
        print(vllm_outputs)
        assert vllm_outputs


@pytest.mark.parametrize(
    "model_path",
    [("nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized")])
def test_qwen(vllm_runner, example_prompts, model_path):
    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                  max_tokens=20)
        print(vllm_outputs)
        assert vllm_outputs
