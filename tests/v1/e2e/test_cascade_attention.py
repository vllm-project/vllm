# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams

from ...utils import create_new_process_for_each_test


@create_new_process_for_each_test()
@pytest.mark.parametrize("attn_backend", ["FLASH_ATTN", "FLASHINFER"])
def test_cascade_attention(example_system_message, attn_backend):
    prompt = "\n<User>: Implement fibonacci sequence in Python.\n<Claude>:"

    if attn_backend == "FLASHINFER":
        pytest.skip(
            "This test is failing with FlashInfer backend and "
            "needs investigation. See issue #25679."
        )

    llm = LLM(
        model="Qwen/Qwen2-1.5B-Instruct", attention_config={"backend": attn_backend}
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    # No cascade attention.
    single_prompt = [example_system_message + prompt]
    responses = llm.generate(single_prompt, sampling_params)
    ref_output = responses[0].outputs[0].text

    # (Probably) Use cascade attention.
    prompts = [example_system_message + prompt] * 64
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        assert response.outputs[0].text == ref_output
