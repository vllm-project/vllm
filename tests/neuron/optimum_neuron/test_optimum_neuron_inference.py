# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams


def test_optimum_neuron_greedy_expectations():
    llm = LLM(
        model="unsloth/Llama-3.2-1B-Instruct",
        max_num_seqs=4,
        max_model_len=4096,
        tensor_parallel_size=2,
        device="neuron",
    )

    # Send more prompts than the compiled batch size (4) and request
    # varying generation lengths to test accuracy related to Neuron
    # specific sequence id sorting.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "It was a bright cold day in April, and the clocks"
        " were striking thirteen.",
        "I believe the meaning of life is",
        "The colour of the sky is",
        "One of my fondest memory is",
    ]

    sampling_params = [
        SamplingParams(top_k=1, max_tokens=10),
        SamplingParams(top_k=1, max_tokens=20),
        SamplingParams(top_k=1, max_tokens=30),
        SamplingParams(top_k=1, max_tokens=40),
        SamplingParams(top_k=1, max_tokens=10),
        SamplingParams(top_k=1, max_tokens=20),
    ]

    outputs = llm.generate(prompts, sampling_params)

    expected_outputs = [
        " the head of state and government of the United States",
        " Paris. The Eiffel Tower is located in Paris."
        " The Louvre Museum is also located in",
        " The world was holding its breath as the world's top scientists"
        " and engineers gathered at the secret underground facility"
        " to witness the unveiling of the ultimate time machine.\n",
        " to find happiness and fulfillment in the present moment."
        " It's a simple yet profound concept that can bring joy and peace"
        " to our lives.\n\nAs I reflect on my own life, I realize that I've",
        " blue, but what about the colour of the sky",
        " of my grandmother's kitchen, where I spent countless hours helping"
        " her in the kitchen. She was a",
    ]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert expected_output == generated_text
