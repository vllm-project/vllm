# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make sure ignore_eos works.

Run `pytest tests/samplers/test_ignore_eos.py`.
"""

import pytest

from vllm import SamplingParams


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    """We can run both engines for this test."""
    pass


# Test ignore_eos with models that have different eos_token_id configurations.
# This ensures vLLM correctly handles EOS tokens from generation_config.json
# (PR #4182).
#
# distilgpt2: Has single eos_token_id (int type: 50256)
# Llama-3.2-1B: Has multiple eos_token_ids (list type: [128001, 128008, 128009])
#
# Why both are needed:
# - The single int case tests the common scenario
# - The list/array case tests models like Llama 3 that use multiple EOS tokens
#   (e.g., <|end_of_text|> and <|eot_id|> for different stopping contexts)
MODELS = ["distilbert/distilgpt2", "meta-llama/Llama-3.2-1B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [512])
def test_ignore_eos(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         ignore_eos=True)

        for prompt in example_prompts:
            ignore_eos_output = vllm_model.llm.generate(
                prompt, sampling_params=sampling_params)
            output_length = len(ignore_eos_output[0].outputs[0].token_ids)
            assert output_length == max_tokens
