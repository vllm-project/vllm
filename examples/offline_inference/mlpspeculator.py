# SPDX-License-Identifier: Apache-2.0

import gc
import time
from typing import List

from vllm import LLM, SamplingParams


def time_generation(llm: LLM, prompts: List[str],
                    sampling_params: SamplingParams):
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    print((end - start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
    # Print the outputs.
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"text: {generated_text!r}")


if __name__ == "__main__":

    template = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n### Instruction:\n{}"
        "\n\n### Response:\n")

    # Sample prompts.
    prompts = [
        "Write about the president of the United States.",
    ]
    prompts = [template.format(prompt) for prompt in prompts]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # Create an LLM without spec decoding
    llm = LLM(model="meta-llama/Llama-2-13b-chat-hf")

    print("Without speculation")
    time_generation(llm, prompts, sampling_params)

    del llm
    gc.collect()

    # Create an LLM with spec decoding
    llm = LLM(
        model="meta-llama/Llama-2-13b-chat-hf",
        speculative_model="ibm-ai-platform/llama-13b-accelerator",
    )

    print("With speculation")
    time_generation(llm, prompts, sampling_params)
