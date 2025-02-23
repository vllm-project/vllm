# SPDX-License-Identifier: Apache-2.0
"""Test the functionality of the Transformers backend.

Run `pytest tests/models/test_transformers.py`.
"""
from contextlib import nullcontext
from typing import Type

import pytest

from ..conftest import HfRunner, VllmRunner
from ..utils import multi_gpu_test
from .utils import check_logprobs_close


def check_implementation(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    example_prompts: list[str],
    model: str,
    **kwargs,
):
    max_tokens = 32
    num_logprobs = 5

    with vllm_runner(model, **kwargs) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "model,model_impl",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", "transformers"),
        ("openai-community/gpt2", "transformers"),
        ("ArthurZ/Ilama-3.2-1B", "auto"),  # CUSTOM CODE
    ])  # trust_remote_code=True by default
def test_models(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
) -> None:

    maybe_raises = nullcontext()
    if model == "openai-community/gpt2" and model_impl == "transformers":
        # Model is not backend compatible
        maybe_raises = pytest.raises(
            ValueError,
            match="The Transformers implementation.*not compatible with vLLM")

    with maybe_raises:
        check_implementation(hf_runner,
                             vllm_runner,
                             example_prompts,
                             model,
                             model_impl=model_impl)


@multi_gpu_test(num_gpus=2)
def test_distributed(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    example_prompts,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(hf_runner, vllm_runner, example_prompts,
                         "meta-llama/Llama-3.2-1B-Instruct", **kwargs)


@pytest.mark.parametrize("model, quantization_kwargs", [
    (
        "meta-llama/Llama-3.2-1B-Instruct",
        {
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        },
    ),
])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_quantization(
    vllm_runner: Type[VllmRunner],
    example_prompts: list[str],
    model: str,
    quantization_kwargs: dict[str, str],
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
            model, model_impl="auto", enforce_eager=True,
            **quantization_kwargs) as vllm_model:  # type: ignore[arg-type]
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs)

    with vllm_runner(
            model,
            model_impl="transformers",
            enforce_eager=True,
            **quantization_kwargs) as vllm_model:  # type: ignore[arg-type]
        transformers_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs)
    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )
