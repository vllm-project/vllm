# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

from .conftest import get_output_from_llm_generator


@pytest.mark.parametrize("common_llm_kwargs",
                         [{
                             "model": "meta-llama/Llama-3.2-1B-Instruct",
                         }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Speculative max model len > overridden max model len should raise.
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
                "max_model_len": 129,
            },
            "max_model_len": 128,
        },
        {
            # Speculative max model len > draft max model len should raise.
            # https://huggingface.co/JackFram/llama-68m/blob/3b606af5198a0b26762d589a3ee3d26ee6fa6c85/config.json#L12
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
                "max_model_len": 2048 + 1,
            },
        },
        {
            # Speculative max model len > target max model len should raise.
            # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/9213176726f574b556790deb65791e0c5aa438b6/config.json#L18
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
                "max_model_len": 131072 + 1,
            },
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail_spec_max_model_len(test_llm_generator):
    """Verify that speculative decoding validates speculative_max_model_len.
    """
    output_len = 128
    temperature = 0.0

    prompts = [
        "Hello, my name is",
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    with pytest.raises(ValueError, match="cannot be larger than"):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)
