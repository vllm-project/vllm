"""Ensure that a text-only Qwen model can be run without throwing an error.
We explicitly test this because Qwen is implemented as a multimodal and
supports a visual encoder for models like Qwen-VL.
"""
from typing import List, Type

import pytest

from ....conftest import VllmRunner

models = [
    "Qwen/Qwen-7B-Chat"  # Has no visual encoder
]


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_text_only_qwen_model_can_be_loaded_and_run(
    vllm_runner: Type[VllmRunner],
    example_prompts: List[str],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_model.generate_greedy_logprobs(
            example_prompts,
            max_tokens,
            num_logprobs=num_logprobs,
        )
