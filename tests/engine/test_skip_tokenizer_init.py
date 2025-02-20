# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.config import LoadFormat
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

from ..conftest import MODEL_WEIGHTS_S3_BUCKET


@pytest.mark.parametrize("model",
                         [f"{MODEL_WEIGHTS_S3_BUCKET}/distilbert/distilgpt2"])
def test_skip_tokenizer_initialization(model: str):
    # This test checks if the flag skip_tokenizer_init skips the initialization
    # of tokenizer and detokenizer. The generated output is expected to contain
    # token ids.
    llm = LLM(model=model,
              skip_tokenizer_init=True,
              load_format=LoadFormat.RUNAI_STREAMER)
    sampling_params = SamplingParams(prompt_logprobs=True, detokenize=True)

    with pytest.raises(ValueError, match="cannot pass text prompts when"):
        llm.generate("abc", sampling_params)

    outputs = llm.generate({"prompt_token_ids": [1, 2, 3]},
                           sampling_params=sampling_params)
    assert len(outputs) > 0
    completions = outputs[0].outputs
    assert len(completions) > 0
    assert completions[0].text == ""
    assert completions[0].token_ids
