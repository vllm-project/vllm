"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
from importlib import reload

import pytest

import vllm.model_executor.parallel_utils.parallel_state as parallel_state
from vllm import LLM, SamplingParams

prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("prefix_pool_memory_utilization", [0, 0.1, 0.2])
def test_prefix_caching(
    example_prompts,
    model: str,
    max_tokens: int,
    prefix_pool_memory_utilization: float,
):
    # IMPORTANT: If this line is removed from here, adding more than 1 item to
    # any of the parametrization lists above causes all tests but the first one
    # to fail with the message: "AssertionError: tensor model parallel group is
    # already initialized."
    reload(parallel_state)
    llm = LLM(model=model,
              prefix_pool_memory_utilization=prefix_pool_memory_utilization)
    # -1 since the last token can change when concatenating prompts.
    prefix_pos = len(llm.llm_engine.tokenizer.encode(prefix)) - 1
    prompts = [prefix + prompt for prompt in example_prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs_without_prefix = llm.generate(prompts, sampling_params)
    outputs_with_prefix = llm.generate(prompts,
                                       sampling_params,
                                       prefix_pos=[prefix_pos] * len(prompts))
    for output_without_prefix, output_with_prefix in zip(
            outputs_without_prefix, outputs_with_prefix):
        assert (output_without_prefix.outputs[0].token_ids ==
                output_with_prefix.outputs[0].token_ids)
    if prefix_pool_memory_utilization == 0:
        assert len(llm.llm_engine.scheduler.prefix_pool.prefixes) == 0


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("prefix_pool_memory_utilization", [0, 0.1, 0.3, 0.4])
def test_prefix_caching_with_multiple_prefixes(
        example_prompts, model: str, max_tokens: int,
        prefix_pool_memory_utilization: float):
    """
    Tests that the scheduler prefix pool size (length) does not go over the
    maximum capacity at any moment in time.
    """
    # IMPORTANT: If this line is removed from here, adding more than 1 item to
    # any of the parametrization lists above causes all tests but the first one
    # to fail with the message: "AssertionError: tensor model parallel group is
    # already initialized."
    reload(parallel_state)
    llm = LLM(model=model,
              prefix_pool_memory_utilization=prefix_pool_memory_utilization)

    # Use 10 different prefixes:
    for i in range(10):
        new_prefix = str(i) + ' ' + prefix
        # -1 since the last token can change when concatenating prompts.
        prefix_pos = len(llm.llm_engine.tokenizer.encode(new_prefix)) - 1
        prompts = [new_prefix + prompt for prompt in example_prompts]
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_tokens)
        outputs_with_prefix = llm.generate(prompts,
                                           sampling_params,
                                           prefix_pos=[prefix_pos] *
                                           len(prompts))
        outputs_without_prefix = llm.generate(prompts, sampling_params)
        for output_without_prefix, output_with_prefix in zip(
                outputs_without_prefix, outputs_with_prefix):
            assert (output_without_prefix.outputs[0].token_ids ==
                    output_with_prefix.outputs[0].token_ids)
