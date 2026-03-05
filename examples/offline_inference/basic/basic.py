# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Basic vLLM text generation example")
    parser.add_argument("--model",
                        type=str,
                        default="facebook/opt-125m",
                        help="Name or path of the model to use")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.8,
                        help="Temperature for sampling")
    parser.add_argument("--top-p",
                        type=float,
                        default=0.95,
                        help="Top-p for sampling")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=100,
                        help="Maximum number of tokens to generate")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=args.temperature,
                                     top_p=args.top_p,
                                     max_tokens=args.max_tokens)

    # Create an LLM.
    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model)

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    print("Generating text...")
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
