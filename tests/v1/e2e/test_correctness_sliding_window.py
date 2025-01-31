import random
from typing import List
from vllm import LLM, SamplingParams
from ...core.block.e2e.test_correctness_sliding_window import (prep_prompts,
                                                               check_answers)
import pytest


@pytest.mark.parametrize("model", ["bigcode/starcoder2-3b"])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_sliding_window_retrival(monkeypatch, model, batch_size, seed):
    """
    The test does a bunch of assignments "x1 = 10\nx2 = 33\n..." and then
    asks for value of one of them (which is outside the sliding window).
    If we tell it upfront which we are going to be looking for, then
    it answers correctly (mostly).
    """
    # TODO: implement check_window
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(model=model, enable_prefix_caching=True)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

        prompts, answer, indices = prep_prompts(batch_size)

        responses = llm.generate(prompts, sampling_params)
        check_answers(indices, answer,
                      [response.outputs[0].text for response in responses])
