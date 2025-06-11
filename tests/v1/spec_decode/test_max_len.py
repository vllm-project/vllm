# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test whether spec decoding handles the max model length properly."""

import pytest

from vllm import LLM, SamplingParams

_PROMPTS = [
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
    "Repeat the following sentence 10 times: Consistency is key to mastering any skill.",  # noqa: E501
    "Who won the Turing Award in 2018, and for what contribution? Describe in detail.",  # noqa: E501
]


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 10])
def test_ngram_max_len(
    monkeypatch: pytest.MonkeyPatch,
    num_speculative_tokens: int,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

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
def test_eagle_max_len(
    monkeypatch: pytest.MonkeyPatch,
    num_speculative_tokens: int,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            enforce_eager=True,  # For faster initialization.
            speculative_config={
                "method": "eagle",
                "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
                "num_speculative_tokens": num_speculative_tokens,
            },
            max_model_len=100,
        )
        sampling_params = SamplingParams(max_tokens=100, ignore_eos=True)
        llm.generate(_PROMPTS, sampling_params)
