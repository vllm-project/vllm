# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Validates the loading of a model saved with the sharded_state format.
This script demonstrates how to load a model that was previously saved
using save_sharded_state.py and validates it by running inference.
Example usage:
(First need to save a sharded_state mode)

python save_sharded_state.py \
    --model /path/to/load \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /path/to/save/sharded/model

python load_sharded_state.py \
    --model /path/to/saved/sharded/model \
    --load-format sharded_state \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --prompt "Hello, my name is" \
    --max-tokens 50
"""

import dataclasses

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    # Add engine arguments
    EngineArgs.add_cli_args(parser)

    # Override default load_format for clarity
    parser.set_defaults(load_format="sharded_state")

    # Add validation arguments
    parser.add_argument(
        "--prompt", type=str, default="Hello, world!", help="Prompt for validation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p sampling parameter"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    engine_args = EngineArgs.from_cli_args(args)

    print(
        f"Loading model from {engine_args.model} using format {engine_args.load_format}"
    )
    print(f"Tensor parallel size: {engine_args.tensor_parallel_size}")

    # Load the model using engine args
    llm = LLM(**dataclasses.asdict(engine_args))

    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print("\nRunning inference:")
    print(f"Prompt: {args.prompt}")

    # Generate completion
    outputs = llm.generate(args.prompt, sampling_params)

    # Display generated text
    print("\nGenerated outputs:")
    for output in outputs:
        generated_text = output.outputs[0].text
        print("-" * 50)
        print(f"Full output: {args.prompt}{generated_text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
