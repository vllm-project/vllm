# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

MODELS = ["distilbert/distilgpt2"]


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    """We can run both engines for this test."""
    pass


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_ranks(
    vllm_runner,
    model,
    dtype,
    example_prompts,
):
    max_tokens = 5
    num_top_logprobs = 5
    num_prompt_logprobs = 5

    with vllm_runner(model, dtype=dtype,
                     max_logprobs=num_top_logprobs) as vllm_model:

        ## Test greedy logprobs ranks
        vllm_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            logprobs=num_top_logprobs,
            prompt_logprobs=num_prompt_logprobs)
        vllm_results = vllm_model.generate_w_logprobs(example_prompts,
                                                      vllm_sampling_params)

        ## Test non-greedy logprobs ranks
        sampling_params = SamplingParams(temperature=1.0,
                                         top_p=1.0,
                                         max_tokens=max_tokens,
                                         logprobs=num_top_logprobs,
                                         prompt_logprobs=num_prompt_logprobs)
        res = vllm_model.generate_w_logprobs(example_prompts, sampling_params)

    for result in vllm_results:
        assert result[2] is not None
        assert len(result[2]) == len(result[0])
        # check whether all chosen tokens have ranks = 1
        for token, logprobs in zip(result[0], result[2]):
            assert token in logprobs
            assert logprobs[token].rank == 1

    for result in res:
        assert result[2] is not None
        assert len(result[2]) == len(result[0])
        # check whether all chosen tokens have ranks
        for token, logprobs in zip(result[0], result[2]):
            assert logprobs[token].rank >= 1
