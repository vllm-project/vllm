# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest

from tests.models.language.generation._hybrid_models import (
    APC_MULTIPLY_BY,
    HYBRID_MODELS,
    _get_vLLM_output,
    _get_vllm_runner_params,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close, check_outputs_equal

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_single_prompt(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts.
    generated_prompts = [APC_MULTIPLY_BY * example_prompts[0]]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size=tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"
    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    vllm_runner_kwargs["enable_prefix_caching"] = True
    vllm_outputs_cache_rep, _ = _get_vLLM_output(
        vllm_runner,
        vllm_runner_kwargs,
        generated_prompts,
        max_tokens,
        num_logprobs,
        n_repetitions,
    )

    for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
        # In the first repetition, the caches are filled
        # In the second repetition, these caches are reused

        compare_operator(
            outputs_0_lst=vllm_outputs_no_cache[0],
            outputs_1_lst=vllm_outputs_cache_itn,
            name_0="vllm_no_cache",
            name_1=f"vllm_cache_it_{r_idx + 1}",
        )


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_single_prompt_block_align_alignment(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts. This custom prompt is used, as it causes the most issues
    generated_prompts = ["The president of the United States is " * APC_MULTIPLY_BY]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size=tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"

    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    vllm_runner_kwargs["enable_prefix_caching"] = True
    with vllm_runner(**vllm_runner_kwargs) as vllm_model:
        # Retrieve the default mamba state block size
        vllm_config = vllm_model.llm.llm_engine.vllm_config
        mamba_block_size = vllm_config.cache_config.mamba_block_size

    # In case the hybrid model does not have the
    # "mamba_block_size" assume a fixed constant
    if mamba_block_size is None:
        mamba_block_size = 512

    mamba_block_size_multiplier = 10
    for offsets in [-3, 3, mamba_block_size // 4 + 3, mamba_block_size // 2 - 3]:
        vllm_runner_kwargs["max_num_batched_tokens"] = (
            mamba_block_size_multiplier * mamba_block_size - offsets
        )
        vllm_outputs_cache_rep, _ = _get_vLLM_output(
            vllm_runner,
            vllm_runner_kwargs,
            generated_prompts,
            max_tokens,
            num_logprobs,
            n_repetitions,
        )

        # Check alignment of the output logits when using APC
        for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            compare_operator(
                outputs_0_lst=vllm_outputs_no_cache[0],
                outputs_1_lst=vllm_outputs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )
