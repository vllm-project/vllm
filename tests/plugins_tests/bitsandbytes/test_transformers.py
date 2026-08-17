# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers backend tests for bitsandbytes plugin models."""

import pytest

from vllm.platforms import current_platform

from ...models.utils import check_logprobs_close

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_cdna

    pytestmark = pytest.mark.skipif(
        on_cdna(),
        reason="bitsandbytes not supported on CDNA (warp size 64 limitation)",
    )


@pytest.mark.parametrize(
    "model, quantization_kwargs",
    [
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            {
                "quantization": "bitsandbytes",
            },
        ),
        ("unsloth/tinyllama-bnb-4bit", {}),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_transformers_backend_quantization(
    vllm_runner,
    example_prompts,
    model: str,
    quantization_kwargs: dict[str, str],
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        model,
        model_impl="auto",
        enforce_eager=True,
        **quantization_kwargs,  # type: ignore[arg-type]
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    with vllm_runner(
        model,
        model_impl="transformers",
        enforce_eager=True,
        **quantization_kwargs,  # type: ignore[arg-type]
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        transformers_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )
