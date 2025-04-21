# SPDX-License-Identifier: Apache-2.0

import torch
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# sampling_params = SamplingParams(temperature=0.)

DIR = '/block/granite/granite-4.0-tiny-base-pipecleaner-hf'
# DIR = '/block/granite/granite-hybridmoe-7b-a1b-base-pipecleaner-hf'

def main():
    # Create an LLM.
    llm = LLM(model=DIR, dtype=torch.float16, gpu_memory_utilization=0.5)#, enforce_eager=True)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts[1], sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
