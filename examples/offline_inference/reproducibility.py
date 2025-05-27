# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates how to achieve reproducibility in vLLM.

Main article: https://docs.vllm.ai/en/latest/usage/reproducibility.html
"""

import os
import random

from vllm import LLM, SamplingParams

# V1 only: Turn off multiprocessing to make the scheduling deterministic.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# V0 only: Set the global seed. The default seed is None, which is
# not reproducible.
SEED = 42

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    llm = LLM(model="facebook/opt-125m", seed=SEED)
    outputs = llm.generate(prompts, sampling_params)
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Try generating random numbers outside vLLM
    # The same number is output across runs, meaning that the random state
    # in the user code has been updated by vLLM
    print(random.randint(0, 100))


if __name__ == "__main__":
    main()
