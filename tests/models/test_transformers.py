# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the functionality of the Transformers backend."""
from typing import Any, Optional, Union

import pytest

from vllm.platforms import current_platform

from ..conftest import HfRunner, VllmRunner
from ..core.block.e2e.test_correctness_sliding_window import prep_prompts
from ..utils import multi_gpu_test
from .utils import check_logprobs_close


def check_implementation(
    runner_ref: type[Union[HfRunner, VllmRunner]],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: Optional[dict[str, Any]] = None,
    kwargs_test: Optional[dict[str, Any]] = None,
    **kwargs,
):
    if kwargs_ref is None:
        kwargs_ref = {}
    if kwargs_test is None:
        kwargs_test = {}

    max_tokens = 32
    num_logprobs = 5

    args = (example_prompts, max_tokens, num_logprobs)

    with runner_test(model, **kwargs_test, **kwargs) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        outputs_test = model_test.generate_greedy_logprobs(*args)

    with runner_ref(model, **kwargs_ref) as model_ref:
        if isinstance(model_ref, VllmRunner):
            outputs_ref = model_ref.generate_greedy_logprobs(*args)
        else:
            outputs_ref = model_ref.generate_greedy_logprobs_limit(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="ref",
        name_1="test",
    )


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="Llama-3.2-1B-Instruct, Ilama-3.2-1B produce memory access fault.")
@pytest.mark.parametrize(
    "model,model_impl",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", "transformers"),
        ("hmellor/Ilama-3.2-1B", "auto"),  # CUSTOM CODE
    ])  # trust_remote_code=True by default
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
) -> None:
    check_implementation(hf_runner,
                         vllm_runner,
                         example_prompts,
                         model,
                         model_impl=model_impl)


def test_hybrid_attention(vllm_runner: type[VllmRunner]) -> None:
    prompts, _, _ = prep_prompts(4, (800, 801))
    kwargs_ref = {"max_model_len": 8192, "enforce_eager": True}
    kwargs_test = {"model_impl": "transformers", **kwargs_ref}
    check_implementation(vllm_runner,
                         vllm_runner,
                         prompts,
                         model="hmellor/tiny-random-Gemma2ForCausalLM",
                         kwargs_ref=kwargs_ref,
                         kwargs_test=kwargs_test)


@multi_gpu_test(num_gpus=2)
def test_distributed(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(hf_runner,
                         vllm_runner,
                         example_prompts,
                         "meta-llama/Llama-3.2-1B-Instruct",
                         kwargs_test=kwargs)


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="bitsandbytes quantization is currently not supported in rocm.")
@pytest.mark.parametrize("model, quantization_kwargs", [
    (
        "meta-llama/Llama-3.2-1B-Instruct",
        {
            "quantization": "bitsandbytes",
        },
    ),
])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_quantization(
    vllm_runner: type[VllmRunner],
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
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        transformers_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs)

    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "model",
    [
        # Layers live in `layers`
        "Qwen/Qwen3-Embedding-0.6B",
        # Layers live in `model.layers`
        "meta-llama/Llama-3.2-1B-Instruct"
    ],
)
def test_embed_loading(vllm_runner, model):
    with vllm_runner(model,
                     max_model_len=1024,
                     enforce_eager=True,
                     runner="pooling",
                     model_impl="transformers") as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()


@pytest.mark.parametrize(
    "model",
    ["jason9693/Qwen2.5-1.5B-apeach"],
)
@pytest.mark.parametrize("dtype", ["float"])
def test_classify(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification

    with vllm_runner(model,
                     max_model_len=512,
                     dtype=dtype,
                     model_impl="transformers") as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        vllm_outputs = vllm_model.classify(example_prompts)

    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)

        assert torch.allclose(hf_output, vllm_output,
                              1e-3 if dtype == "float" else 1e-2)
