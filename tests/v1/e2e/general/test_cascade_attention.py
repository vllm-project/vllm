# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

from ....utils import create_new_process_for_each_test

if current_platform.is_rocm():
    pytest.skip(
        "Cascade attention backends FLASH_ATTN and FLASHINFER are notsupported on ROCm",
        allow_module_level=True,
    )

GOLDEN_OUTPUTS = {
    "FLASH_ATTN": (
        " Sure, I can help you with that. The Fibonacci sequence is a series of "
        "numbers in which each number is the sum of the two preceding ones, "
        "usually starting with 0 and 1. Here's a simple Python function to "
        "generate the Fibonacci sequence up to a given number:\n\n"
        "```python\ndef fibonacci(n):\n    if n <= 0:\n"
        "        return []\n    elif n == 1:\n        return [0]\n"
        "    elif n == 2:\n        return [0, 1]\n"
    ),
}


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
        model="Qwen/Qwen2-1.5B-Instruct",
        attention_config={"backend": attn_backend},
        disable_cascade_attn=False,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, logprobs=5)

    prompts = [example_system_message + prompt] * 64
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        assert response.outputs[0].text == GOLDEN_OUTPUTS[attn_backend]
