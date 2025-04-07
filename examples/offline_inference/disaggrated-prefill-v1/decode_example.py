# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Read prompts from output.txt
prompts = []
try:
    with open("output.txt") as f:
        for line in f:
            prompts.append(line.strip())
    print(f"Loaded {len(prompts)} prompts from output.txt")
except FileNotFoundError:
    print("Error: output.txt file not found")
    exit(-1)

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    kv_transfer_config=KVTransferConfig.from_cli(
        '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config": {"shared_storage_path": "local_storage"}}'
    ))  #, max_model_len=2048, max_num_batched_tokens=2048)

# 1ST generation (prefill instance)
outputs = llm.generate(prompts, sampling_params)

new_prompts = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    new_prompts.append(prompt + generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
