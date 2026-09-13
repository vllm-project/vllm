# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    ATTN_BACKEND,
    HYBRID_MODELS,
    MAX_NUM_SEQS,
    SSM_MODELS,
    _set_conv_state_layout,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"])
def test_batching(
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    conv_state_layout: str,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    _set_conv_state_layout(monkeypatch, conv_state_layout)

    for_loop_outputs = []
    with vllm_runner(
        model, max_num_seqs=MAX_NUM_SEQS, enable_chunked_prefill=True
    ) as vllm_model:
        for prompt in example_prompts:
            (single_output,) = vllm_model.generate_greedy_logprobs(
                [prompt], max_tokens, num_logprobs
            )
            for_loop_outputs.append(single_output)

        batched_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=for_loop_outputs,
        outputs_1_lst=batched_outputs,
        name_0="for_loop_vllm",
        name_1="batched_vllm",
    )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [10])
@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"])
def test_chunked_prefill_with_parallel_sampling(
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    conv_state_layout: str,
) -> None:
    """
    Tests chunked prefill in conjunction with n > 1.

    In this case, prefill is populated with decoding tokens and
    we test that it doesn't fail.

    This test might fail if cache is not allocated correctly for n > 1
    decoding steps inside a chunked prefill forward pass
    (where we have both prefill and decode together)
    """
    _set_conv_state_layout(monkeypatch, conv_state_layout)

    sampling_params = SamplingParams(n=3, temperature=1, seed=0, max_tokens=max_tokens)
    with vllm_runner(
        model,
        enable_chunked_prefill=True,
        # forces prefill chunks with decoding
        max_num_batched_tokens=MAX_NUM_SEQS * 3,
        max_num_seqs=MAX_NUM_SEQS,
        attention_backend=ATTN_BACKEND,
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
