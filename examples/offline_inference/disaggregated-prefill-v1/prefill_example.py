# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

context = "Hi " * 1000
context2 = "Hey " * 500
prompts = [
    context + "Hello, my name is",
    context + "The capital of France is",
    context2 + "Your name is",
    context2 + "The capital of China is",
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct",
          enforce_eager=True,
          gpu_memory_utilization=0.8,
          kv_transfer_config=KVTransferConfig.from_cli(
              '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both", '
              '"kv_connector_extra_config": '
              '{"shared_storage_path": "local_storage"}}')
          )  #, max_model_len=2048, max_num_batched_tokens=2048)

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
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Write new_prompts to output.txt
with open("output.txt", "w") as f:
    for prompt in new_prompts:
        f.write(prompt + "\n")
print(f"Saved {len(new_prompts)} prompts to output.txt")
