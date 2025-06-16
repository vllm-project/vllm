# SPDX-License-Identifier: Apache-2.0

import os

#        VLLM_ENABLE_V1_MULTIPROCESSING=0
#       VLLM_WORKER_MULTIPROC_METHOD=spawn
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

if __name__ == "__main__":

    context = "Hi " * 1000
    context2 = "Hey " * 1000
    context3 = "Hello " * 1000
    context4 = "How " * 1000
    prompts = [
        context + "Hello, my name is",
        context2 + "The capital of France is",
        context3 + "Your name is",
        context4 + "The capital of China is",
    ]

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig(
            kv_connector="CPUConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "host": "localhost",
                "port": 54321,
                "size": 4,
            },
        ),
        #load_format="dummy",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        block_size=128,
        tensor_parallel_size=1,
    )

    # 1ST generation (prefill instance)
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Write new_prompts to output.txt
    with open("output.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(new_prompts)} prompts to output.txt")

    # HACK: for offline single-process inference only
    # Wait for all send finishes
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    try:
        cpu_connector = get_kv_transfer_group()
        cpu_connector.close()
    except Exception:
        pass
