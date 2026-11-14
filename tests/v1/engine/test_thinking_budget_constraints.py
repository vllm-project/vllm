# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from vllm.exceptions import VLLMValidationError


def test_thinking_budget_token_constraints():
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="half",
        max_model_len=256,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        reasoning_config=ReasoningConfig(
            reasoning_parser="qwen3",
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        ),
    )
    tokenizer = llm.get_tokenizer()
    start = tokenizer.encode("<think>", add_special_tokens=False)[0]
    end = tokenizer.encode("</think>", add_special_tokens=False)[0]
    prompt = {"prompt_token_ids": [start]}

    for extra, field in [
        ({"allowed_token_ids": [59604]}, "allowed_token_ids"),
        ({"bad_words": ["</think>"]}, "bad_words"),
    ]:
        with pytest.raises(VLLMValidationError, match=field):
            llm.generate(
                prompt,
                SamplingParams(thinking_token_budget=0, max_tokens=1, **extra),
                use_tqdm=False,
            )

    result = llm.generate(
        prompt,
        SamplingParams(
            thinking_token_budget=0,
            allowed_token_ids=[end],
            max_tokens=1,
            temperature=0,
        ),
        use_tqdm=False,
    )[0]
    assert list(result.outputs[0].token_ids) == [end]
