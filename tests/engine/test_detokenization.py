# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.config import LoadFormat
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

from ..conftest import MODEL_WEIGHTS_S3_BUCKET


@pytest.mark.parametrize("model",
                         [f"{MODEL_WEIGHTS_S3_BUCKET}/distilbert/distilgpt2"])
def test_computed_prefix_blocks(model: str):
    # This test checks if the engine generates completions both with and
    # without optional detokenization, that detokenization includes text
    # and no-detokenization doesn't, and that both completions have the same
    # token_ids.
    prompt = (
        "You are a helpful assistant. How do I build a car from cardboard and "
        "paper clips? Is there an easy to follow video tutorial available "
        "online for free?")

    llm = LLM(model=model, load_format=LoadFormat.RUNAI_STREAMER)
    sampling_params = SamplingParams(max_tokens=10,
                                     temperature=0.0,
                                     detokenize=False)

    outputs_no_detokenization = llm.generate(prompt,
                                             sampling_params)[0].outputs[0]
    sampling_params.detokenize = True
    outputs_with_detokenization = llm.generate(prompt,
                                               sampling_params)[0].outputs[0]

    assert outputs_no_detokenization.text == ''
    assert outputs_with_detokenization.text != ''
    assert outputs_no_detokenization.token_ids == \
        outputs_with_detokenization.token_ids
