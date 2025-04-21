# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

DIR = '/block/granite/granite-4.0-tiny-base-pipecleaner-hf'
# DIR = '/code/granite/granite-4_0-small-base-pipecleaner-hf'
# DIR = '/code/granite/granite-4_0-medium-base-pipecleaner-hf'

def main():
    tokenizer = AutoTokenizer.from_pretrained(DIR)
    inputs = tokenizer(prompts[1], return_tensors="pt").to("cuda")
    
    model = AutoModelForCausalLM.from_pretrained(DIR, torch_dtype=torch.float16).to("cuda")
    
    outputs_ids = model.generate(**inputs, max_new_tokens=20)

    # Print the outputs.
    outputs_str = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
    print("\nGenerated Outputs:\n" + "-" * 60)
    prompt = prompts[1]
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {outputs_str!r}")
    print("-" * 60)


if __name__ == "__main__":
    main()
