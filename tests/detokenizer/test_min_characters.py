# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

PROMPT = "Hello, my name is Lee, and I'm a student in the college of engineering"


@pytest.mark.parametrize(
    "min_characters,stop,truth",
    [
        (0, None, " is Lee, and I'm a student in the college of engineering"),
        (0, "e", " is L"),
        (7, "e", " is Lee, and I'm a stud"),
    ],
)
def test_min_characters_with_stop(min_characters: int, stop: str, truth: str):
    """Test for a specific min_characters and stop."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    all_prompt_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids

    # The prompt is "Hello, my name is"
    prompt_len = 4
    prompt_token_ids = all_prompt_ids[:prompt_len]
    params = SamplingParams(
        stop=stop,
        min_characters=min_characters,
    )
    request = EngineCoreRequest(
        request_id="",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    detokenizer = FastIncrementalDetokenizer(tokenizer, request)

    detokenizer.update(all_prompt_ids[prompt_len:], False)
    assert detokenizer.output_text == truth
