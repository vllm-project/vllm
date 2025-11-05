# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.logprobs import (
    FlattenLogprobs,
    PromptLogprobs,
    SampleLogprobs,
    get_rank,
    num_logprobs_per_position,
    num_positions,
)


def all_ranks_per_position(
    logprobs: PromptLogprobs | SampleLogprobs, position: int
) -> set[int]:
    """Gets all ranks of a given position"""
    if isinstance(logprobs, FlattenLogprobs):
        return set(
            logprobs.ranks[
                logprobs.start_indices_per_position[
                    position
                ] : logprobs.start_indices_per_position[position + 1]
            ]
        )
    return (
        {logprob.rank for logprob in logprobs[position].values()}
        if logprobs[position] is not None
        else set()
    )


MODELS = ["distilbert/distilgpt2"]
MAX_TOKENS = 5
NUM_TOP_LOGPROBS = 5
NUM_PROMPT_LOGPROBS = 7
MAX_LOGPROBS = max(NUM_TOP_LOGPROBS, NUM_PROMPT_LOGPROBS)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("flatten_logprobs", [True, False])
def test_ranks(
    vllm_runner,
    model,
    dtype,
    greedy,
    flatten_logprobs,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_FLATTEN_LOGPROBS", "1" if flatten_logprobs else "0")
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
        assert isinstance(
            prompt_logprobs, FlattenLogprobs if flatten_logprobs else list
        )
        assert isinstance(
            decode_logprobs, FlattenLogprobs if flatten_logprobs else list
        )

        ########################
        # Check prompt logprobs
        ########################
        assert len(prompt_tokens) == num_positions(prompt_logprobs)
        # No logprob for first prompt token
        assert num_logprobs_per_position(prompt_logprobs, 0) == 0
        for position, token in enumerate(prompt_tokens[1:], start=1):
            # Ensure logprobs of prompt token is always returned
            rank = get_rank(prompt_logprobs, position, token)
            assert rank is not None
            assert rank >= 1
            # Ensure # of returned logprobs should be
            # either NUM_PROMPT_LOGPROBS or NUM_PROMPT_LOGPROBS+1
            assert (
                NUM_PROMPT_LOGPROBS
                <= num_logprobs_per_position(prompt_logprobs, position)
                <= NUM_PROMPT_LOGPROBS + 1
            )
            # Ensure top NUM_PROMPT_LOGPROBS is always extracted
            assert (
                len(
                    all_ranks_per_position(prompt_logprobs, position)
                    & set(range(1, NUM_PROMPT_LOGPROBS + 1))
                )
                == NUM_PROMPT_LOGPROBS
            )

        ########################
        # Check sample logprobs
        ########################
        assert len(decode_tokens) == num_positions(decode_logprobs)
        for position, token in enumerate(decode_tokens):
            # Ensure logprobs of chosen token is always returned
            rank = get_rank(decode_logprobs, position, token)
            assert rank is not None
            if greedy:
                # For greedy sampling, all chosen logprob should be top ranked
                assert rank == 1
            else:
                assert rank >= 1
            # Ensure # of returned logprobs should be
            # either NUM_TOP_LOGPROBS or NUM_TOP_LOGPROBS+1
            assert (
                NUM_TOP_LOGPROBS
                <= num_logprobs_per_position(decode_logprobs, position)
                <= NUM_TOP_LOGPROBS + 1
            )
            # Ensure top NUM_TOP_LOGPROBS logprobs is always extracted
            assert (
                len(
                    all_ranks_per_position(decode_logprobs, position)
                    & set(range(1, NUM_TOP_LOGPROBS + 1))
                )
                == NUM_TOP_LOGPROBS
            )
