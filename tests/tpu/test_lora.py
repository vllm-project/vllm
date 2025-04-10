# SPDX-License-Identifier: Apache-2.0
import vllm
from vllm.lora.request import LoRARequest


def test_lora_hotswapping():
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
                   max_loras=2,
                   max_lora_rank=8)

    prompt = "What is 1+1? \n"

    for _ in range(2):
        for i, req in enumerate(lora_requests):
            output = llm.generate(prompt,
                                  sampling_params=vllm.SamplingParams(
                                      max_tokens=256, temperature=0),
                                  lora_request=req)[0].outputs[0].text
            assert int(output.strip()[0]) == i + 1
