# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    ATTN_BACKEND,
    _get_vLLM_output,
    _get_vllm_runner_params,
)
from tests.models.utils import check_logprobs_close
from vllm import LLM
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.hybrid_model


# Test that outputs match whether prefix caching is enabled or not for mamba.
@pytest.mark.parametrize("model", ["tiiuae/falcon-mamba-7b"])
def test_same_mamba_output_apc_on_vs_off(
    vllm_runner,
    model: str,
) -> None:
    num_logprobs = 5
    prompts = [
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
    ]
    max_tokens = 20
    max_model_len = max(len(p) for p in prompts) + max_tokens + 64

    base_kwargs = _get_vllm_runner_params(model, max_model_len)
    base_kwargs.update(
        enforce_eager=True, block_size=16, seed=42, gpu_memory_utilization=0.8
    )

    # No prefix caching
    kwargs_no_apc = {**base_kwargs, "enable_prefix_caching": False}
    with vllm_runner(**kwargs_no_apc) as vllm_model:
        outputs_no_apc, _ = _get_vLLM_output(
            vllm_runner,
            kwargs_no_apc,
            prompts,
            max_tokens,
            num_logprobs=num_logprobs,
            vllm_model=vllm_model,
        )
    # With prefix caching
    kwargs_with_apc = {
        **base_kwargs,
        "enable_prefix_caching": True,
        "mamba_block_size": 16,
    }
    with vllm_runner(**kwargs_with_apc) as vllm_model:
        outputs_with_apc, _ = _get_vLLM_output(
            vllm_runner,
            kwargs_with_apc,
            prompts,
            max_tokens,
            num_logprobs=num_logprobs,
            vllm_model=vllm_model,
        )

    check_logprobs_close(
        outputs_0_lst=outputs_no_apc[0],
        outputs_1_lst=outputs_with_apc[0],
        name_0="vllm_no_apc",
        name_1="vllm_with_apc",
    )


# we have to use a real large model to get reasonable results
# the model can't be a hybrid model as we need block_size 16
@pytest.mark.parametrize("model", ["tiiuae/falcon-mamba-7b"])
def test_apc_common_prefix_same_batch(
    model: str,
    monkeypatch,
) -> None:
    # Required to put the two requests in the same batch
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=model,
        enforce_eager=True,
        block_size=16,
        mamba_block_size=16,
        enable_prefix_caching=True,
        seed=42,
        attention_backend=ATTN_BACKEND,
    )
    prompts = [
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        assert "two" in output.outputs[0].text
