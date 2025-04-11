# SPDX-License-Identifier: Apache-2.0
import pytest

import vllm
from vllm.lora.request import LoRARequest


@pytest.fixture(scope="function", autouse=True)
def use_v1_only(monkeypatch: pytest.MonkeyPatch):
    """
    Since Multi-LoRA is only supported on the v1 TPU backend, set VLLM_USE_V1=1
    for all tests in this file
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        yield


@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
def test_lora_e2e(num_loras: int):
    """
    This test ensures that we can run with LoRA adapters on the TPU backend.
    It verifies multiple capabilities:
        1. We can compile a model with LoRA adapters enabled
        2. We can run <num_loras> LoRA adapters
        3. We receive correct outputs when running with multiple LoRA adapters
        4. We can swap LoRA adapters between host and device
    """
    lora_name_template = \
        "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    lora_requests = [
        LoRARequest(f"lora_adapter_{i}", i, lora_name_template.format(i))
        for i in range(1, 5)
    ]

    llm = vllm.LLM(model="Qwen/Qwen2.5-3B-Instruct",
                   num_scheduler_steps=1,
                   max_model_len=256,
                   max_seq_len_to_capture=256,
                   max_num_seqs=8,
                   enable_lora=True,
                   max_loras=num_loras,
                   max_lora_rank=8)

    prompt = "What is 1+1? \n"

    for _ in range(2):
        for i, req in enumerate(lora_requests):
            output = llm.generate(prompt,
                                  sampling_params=vllm.SamplingParams(
                                      max_tokens=256, temperature=0),
                                  lora_request=req)[0].outputs[0].text
            assert int(output.strip()[0]) == i + 1
