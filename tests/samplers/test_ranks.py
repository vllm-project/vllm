# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

MODELS = ["distilbert/distilgpt2"]
MAX_TOKENS = 5
NUM_TOP_LOGPROBS = 5
NUM_PROMPT_LOGPROBS = 7
MAX_LOGPROBS = max(NUM_TOP_LOGPROBS, NUM_PROMPT_LOGPROBS)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("greedy", [True, False])
def test_ranks(
    vllm_runner,
    model,
    dtype,
    greedy,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_FLATTEN_LOGPROBS", "1")
    with vllm_runner(model, dtype=dtype, max_logprobs=MAX_LOGPROBS) as vllm_model:
        tokenizer = vllm_model.llm.get_tokenizer()
        example_prompt_tokens = [tokenizer.encode(prompt) for prompt in example_prompts]
        sampling_params = SamplingParams(
            temperature=0.0 if greedy else 1.0,
            top_p=1.0,
            max_tokens=MAX_TOKENS,
            logprobs=NUM_TOP_LOGPROBS,
            prompt_logprobs=NUM_PROMPT_LOGPROBS,
        )
        results = vllm_model.generate_w_logprobs(example_prompts, sampling_params)

    assert len(results) == len(example_prompt_tokens)
    for i, (result, prompt_tokens) in enumerate(zip(results, example_prompt_tokens)):
        decode_tokens, _, decode_logprobs, prompt_logprobs = result
        assert decode_logprobs is not None
        assert len(decode_logprobs) == len(decode_tokens)

        ########################
        # Check prompt logprobs
        ########################
        # Prompt logprob for first token is None
        assert prompt_logprobs[0] is None
        for token, logprobs in zip(prompt_tokens[1:], prompt_logprobs[1:]):
            # Ensure logprobs of prompt token is always returned
            logprob = logprobs.get(token)
            assert logprob is not None
            assert logprob.rank >= 1
            # Ensure # of returned logprobs should be
            # either NUM_PROMPT_LOGPROBS or NUM_PROMPT_LOGPROBS+1
            assert NUM_PROMPT_LOGPROBS <= len(logprobs) <= NUM_PROMPT_LOGPROBS + 1
            # Ensure top NUM_PROMPT_LOGPROBS is always extracted
            all_ranks = {logprob.rank for logprob in logprobs.values()}
            assert (
                len(all_ranks & set(range(1, NUM_PROMPT_LOGPROBS + 1)))
                == NUM_PROMPT_LOGPROBS
            )

        ########################
        # Check sample logprobs
        ########################
        for token, logprobs in zip(decode_tokens, decode_logprobs):
            logprob = logprobs.get(token)
            if greedy:
                # For greedy sampling, all chosen logprob should be top ranked
                assert logprob.rank == 1
            else:
                assert logprob.rank >= 1
            # Ensure # of returned logprobs should be
            # either NUM_TOP_LOGPROBS or NUM_TOP_LOGPROBS+1
            assert NUM_TOP_LOGPROBS <= len(logprobs) <= NUM_TOP_LOGPROBS + 1
            # Ensure top NUM_TOP_LOGPROBS logprobs is always extracted
            all_ranks = {logprob.rank for logprob in logprobs.values()}
            assert (
                len(all_ranks & set(range(1, NUM_TOP_LOGPROBS + 1))) == NUM_TOP_LOGPROBS
            )
