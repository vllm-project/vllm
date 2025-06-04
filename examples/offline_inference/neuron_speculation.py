# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to run offline inference with a speculative
decoding model on neuron.
"""

import os

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, I am a language model and I can help",
    "The president of the United States is",
    "The capital of France is",
]


def config_buckets():
    """Configure context length and token gen buckets."""
    # creates XLA hlo graphs for all the context length buckets.
    os.environ["NEURON_CONTEXT_LENGTH_BUCKETS"] = "128,512,1024,2048"
    # creates XLA hlo graphs for all the token gen buckets.
    os.environ["NEURON_TOKEN_GEN_BUCKETS"] = "128,512,1024,2048"


def initialize_model():
    """Create an LLM with speculative decoding."""
    return LLM(
        model="openlm-research/open_llama_7b",
        speculative_config={
            "model": "openlm-research/open_llama_3b",
            "num_speculative_tokens": 4,
            "max_model_len": 2048,
        },
        max_num_seqs=4,
        max_model_len=2048,
        block_size=2048,
        use_v2_block_manager=True,
        device="neuron",
        tensor_parallel_size=32,
    )


def process_requests(model: LLM, sampling_params: SamplingParams):
    """Generate texts from prompts and print them."""
    outputs = model.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main():
    """Main function that sets up the model and processes prompts."""
    config_buckets()
    model = initialize_model()
    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=100, top_k=1)
    process_requests(model, sampling_params)


if __name__ == "__main__":
    main()
