from vllm import LLM, SamplingParams
import torch
import numpy as np

prompts = [
    "Hello, my name is",
    "A robot may not injure a human being",
    "To be or not to be,",
    "What is the meaning of life?",
    "It is only with the heart that one can see rightly",
    "The quick brown fox jumps over the lazy dog",
]
sampling_params = SamplingParams(n=4,
                                 use_beam_search=True,
                                 temperature=0.0,
                                 max_tokens=128)

llm = LLM(model="huggyllama/llama-7b", tokenizer="huggyllama/llama-7b")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    print(f"Prompt: {prompt!r}")
    for i, output in enumerate(output.outputs):
        print(f"Generated text {i}: {output.text!r}")
