# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import LLM, SamplingParams

from ...core.block.e2e.test_correctness_sliding_window import (check_answers,
                                                               prep_prompts)


@dataclass
class TestConfig:
    sliding_window: int
    ln_range: tuple[int, int]


model_config = {
    "bigcode/starcoder2-3b": TestConfig(4096, (800, 1100)),
    "google/gemma-3-1b-it": TestConfig(4096, (400, 800)),
}


@pytest.mark.parametrize(
    "model",
    [
        "bigcode/starcoder2-3b",  # sliding window only
        "google/gemma-3-1b-it",  # sliding window + full attention
    ])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_sliding_window_retrieval(monkeypatch, model, batch_size, seed):
    """
    The test does a bunch of assignments "x1 = 10\nx2 = 33\n..." and then
    asks for value of one of them (which is outside the sliding window).
    If we tell it upfront which we are going to be looking for, then
    it answers correctly (mostly).
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        test_config = model_config[model]

        llm = LLM(model=model)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

        prompts, answer, indices = prep_prompts(batch_size,
                                                ln_range=test_config.ln_range)

        check_length(prompts, llm, test_config.sliding_window)

        # Fresh generation
        responses = llm.generate(prompts, sampling_params)
        check_answers(indices,
                      answer,
                      [response.outputs[0].text for response in responses],
                      accept_rate=1.0)

        # Re-generate with the same prompts to test prefix caching
        responses = llm.generate(prompts, sampling_params)
        check_answers(indices,
                      answer,
                      [response.outputs[0].text for response in responses],
                      accept_rate=1.0)


def check_length(prompts: list[str], llm: LLM, sliding_window: int):
    """
    Check if the prompt length is valid, i.e., longer than the sliding window 
    size and shorter than the model's max length.

    Args:
        prompts: list of prompts
        llm: LLM object
        sliding_window: Sliding window size
    """
    tokenizer = llm.get_tokenizer()
    max_model_len = llm.llm_engine.model_config.max_model_len
    assert any(
        len(tokenizer.encode(prompt)) > sliding_window
        for prompt in prompts), "Prompt is too short for test"
    assert all(
        len(tokenizer.encode(prompt)) <= max_model_len
        for prompt in prompts), "Prompt is too long for test"
