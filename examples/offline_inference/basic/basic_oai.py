# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "How are you?",
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=100)


def main():
    # Create an LLM.
    llm = LLM(
        model=
        "../../../../models/real-weight/pytorch-os-mini-final-quantized-moe-sharded",
        tokenizer="../../../../models/hf-converted",
        tensor_parallel_size=4,
        # Set these to make dummy run faster
        enforce_eager=True,
        # max_num_seqs=1,
        # max_num_batched_tokens=100,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
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
