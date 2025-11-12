# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.logprobs import FlatLogprobs

MODELS = ["distilbert/distilgpt2"]
MAX_TOKENS = 5
NUM_TOP_LOGPROBS = 5
NUM_PROMPT_LOGPROBS = 7
MAX_LOGPROBS = max(NUM_TOP_LOGPROBS, NUM_PROMPT_LOGPROBS)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("flat_logprobs", [True, False])
def test_ranks(
    vllm_runner,
    model,
    dtype,
    greedy,
    flat_logprobs,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_FLAT_LOGPROBS", "1" if flat_logprobs else "0")
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

        # Ensure the return type of logprobs is accurate
        assert isinstance(prompt_logprobs, FlatLogprobs if flat_logprobs else list)
        assert isinstance(decode_logprobs, FlatLogprobs if flat_logprobs else list)

        ########################
        # Check prompt logprobs
        ########################
        assert len(prompt_tokens) == len(prompt_logprobs)
        # No logprob for first prompt token
        assert not prompt_logprobs[0]
        for position, (token, logprobs) in enumerate(
            zip(prompt_tokens[1:], prompt_logprobs[1:]), start=1
        ):
            # Ensure logprobs of prompt token is always returned
            logprob = logprobs.get(token)
            assert logprob is not None
            assert logprob.rank >= 1
            # Ensure # of returned logprobs should be
            # either NUM_PROMPT_LOGPROBS or NUM_PROMPT_LOGPROBS+1
            assert NUM_PROMPT_LOGPROBS <= len(logprobs) <= NUM_PROMPT_LOGPROBS + 1
            # Ensure top NUM_PROMPT_LOGPROBS is always extracted
            assert set(range(1, NUM_PROMPT_LOGPROBS + 1)).issubset(
                {logprob.rank for logprob in logprobs.values()}
            )

        ########################
        # Check sample logprobs
        ########################
        assert len(decode_tokens) == len(decode_logprobs)
        for position, (token, logprobs) in enumerate(
            zip(decode_tokens, decode_logprobs)
        ):
            # Ensure logprobs of chosen token is always returned
            logprob = logprobs.get(token)
            assert logprob is not None
            if greedy:
                # For greedy sampling, all chosen logprob should be top ranked
                assert logprob.rank == 1
            else:
                assert logprob.rank >= 1
            # Ensure # of returned logprobs should be
            # either NUM_TOP_LOGPROBS or NUM_TOP_LOGPROBS+1
            assert NUM_TOP_LOGPROBS <= len(logprobs) <= NUM_TOP_LOGPROBS + 1
            # Ensure top NUM_TOP_LOGPROBS logprobs is always extracted
            assert set(range(1, NUM_TOP_LOGPROBS + 1)).issubset(
                {logprob.rank for logprob in logprobs.values()}
            )
