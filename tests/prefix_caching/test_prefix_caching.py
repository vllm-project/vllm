"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import pytest

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
def test_prefix_caching(
    example_prompts,
    model: str,
    max_tokens: int,
):
    llm = LLM(model=model)
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
    assert len(llm.llm_engine.scheduler.prefix_pool.prefixes) == 1
