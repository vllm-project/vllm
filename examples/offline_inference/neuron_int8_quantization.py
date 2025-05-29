# SPDX-License-Identifier: Apache-2.0

import os

from vllm import LLM, SamplingParams

# creates XLA hlo graphs for all the context length buckets.
os.environ["NEURON_CONTEXT_LENGTH_BUCKETS"] = "128,512,1024,2048"
# creates XLA hlo graphs for all the token gen buckets.
os.environ["NEURON_TOKEN_GEN_BUCKETS"] = "128,512,1024,2048"
# Quantizes neuron model weight to int8 ,
# The default config for quantization is int8 dtype.
os.environ["NEURON_QUANT_DTYPE"] = "s8"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=8,
        # The max_model_len and block_size arguments are required to be same as
        # max sequence length when targeting neuron device.
        # Currently, this is a known limitation in continuous batching support
        # in transformers-neuronx.
        # TODO(liangfu): Support paged-attention in transformers-neuronx.
        max_model_len=2048,
        block_size=2048,
        # ruff: noqa: E501
        # The device can be automatically detected when AWS Neuron SDK is installed.
        # The device argument can be either unspecified for automated detection,
        # or explicitly assigned.
        device="neuron",
        quantization="neuron_quant",
        override_neuron_config={
            "cast_logits_dtype": "bfloat16",
        },
        tensor_parallel_size=2,
    )
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
