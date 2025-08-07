# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        max_model_len=1024,
        block_size=1024,
        # ruff: noqa: E501
        # The device can be automatically detected when AWS Neuron SDK is installed.
        # The device argument can be either unspecified for automated detection,
        # or explicitly assigned.
        device="neuron",
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
