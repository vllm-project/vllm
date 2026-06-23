# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for GGUF plugin functionality.
"""

import os
from typing import NamedTuple

import pytest
from transformers import AutoTokenizer

from ...conftest import VllmRunner
from ...models.utils import check_logprobs_close
from ...utils import multi_gpu_test

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024


class GGUFTestConfig(NamedTuple):
    original_model: str
    gguf_model_path: str  # Full path to .gguf file


QWEN3_CONFIG = GGUFTestConfig(
    original_model="Qwen/Qwen3-0.6B",
    gguf_model_path="unsloth/Qwen3-0.6B-GGUF:Q8_0",
)


OLMOE_CONFIG = GGUFTestConfig(
    original_model="allenai/OLMoE-1B-7B-0125",
    gguf_model_path="allenai/OLMoE-1B-7B-0125-GGUF:Q6_K",
)


MODELS = [
    QWEN3_CONFIG,
    OLMOE_CONFIG,
]


def check_model_outputs(
    vllm_runner: type[VllmRunner],
    prompts: list[str],
    model: GGUFTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model.original_model)
    if tokenizer.chat_template is not None:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Run gguf model.
    with vllm_runner(
        model_name=model.gguf_model_path,
        enforce_eager=True,
        tokenizer_name=model.original_model,
        dtype=dtype,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=tp_size,
    ) as gguf_model:
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            prompts[:-1], max_tokens, num_logprobs
        )

    # Run unquantized model.
    # Should run with tp=1, otherwise the test will stuck at
    # nccl initialization.
    with vllm_runner(
        model_name=model.original_model,
        enforce_eager=True,  # faster tests
        dtype=dtype,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
    ) as original_model:
        original_outputs = original_model.generate_greedy_logprobs(
            prompts[:-1], max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
        name_1="gguf",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [8])
@pytest.mark.parametrize("tp_size", [1])
def test_models(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: GGUFTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    check_model_outputs(
        vllm_runner, example_prompts, model, dtype, max_tokens, num_logprobs, tp_size
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tp_size", [2])
@multi_gpu_test(num_gpus=2)
def test_distributed(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: GGUFTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    check_model_outputs(
        vllm_runner, example_prompts, model, dtype, max_tokens, num_logprobs, tp_size
    )
