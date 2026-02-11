# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test whether spec decoding handles the max model length properly."""

import pytest

from tests.utils import get_attn_backend_list_based_on_platform
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams

_PROMPTS = [
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
    "Repeat the following sentence 10 times: Consistency is key to mastering any skill.",  # noqa: E501
    "Who won the Turing Award in 2018, and for what contribution? Describe in detail.",  # noqa: E501
]


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 10])
def test_ngram_max_len(num_speculative_tokens: int):
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=100,
        enforce_eager=True,  # For faster initialization.
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": num_speculative_tokens,
        },
    )
    sampling_params = SamplingParams(max_tokens=100, ignore_eos=True)
    llm.generate(_PROMPTS, sampling_params)


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 10])
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_max_len(
    monkeypatch: pytest.MonkeyPatch, num_speculative_tokens: int, attn_backend: str
):
    if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
        pytest.skip(
            "TRITON_ATTN does not support "
            "multi-token eagle spec decode on current platform"
        )

    if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        enforce_eager=True,  # For faster initialization.
        speculative_config={
            "method": "eagle",
            "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
            "num_speculative_tokens": num_speculative_tokens,
            "max_model_len": 80,
        },
        max_model_len=200,
        attention_config={"backend": attn_backend},
    )
    sampling_params = SamplingParams(max_tokens=200, ignore_eos=True)
    outputs = llm.generate(_PROMPTS, sampling_params)
    for o in outputs:
        assert o.outputs[0].finish_reason == "length", (
            "This test is only meaningful if the output is truncated due to max length"
        )

    sampling_params = SamplingParams(
        max_tokens=200,
        structured_outputs=StructuredOutputsParams(regex="^" + "a b c d e " * 15 + "$"),
    )
    output = llm.generate(_PROMPTS, sampling_params)
    for o in output:
        assert o.prompt_token_ids is not None
        assert (
            len(o.prompt_token_ids)
            < 80
            < len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
            <= 200
        ), (
            "This test is only meaningful if the output "
            "is longer than the eagle max length"
        )
        assert o.outputs[0].text == "a b c d e " * 15
