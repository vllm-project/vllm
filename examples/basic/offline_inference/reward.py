# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.print_utils import print_embeddings


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="internlm/internlm2-1_8b-reward",
        runner="pooling",
        enforce_eager=True,
        max_model_len=1024,
        trust_remote_code=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass runner="pooling" for reward models
    llm = LLM(**vars(args))

    # Generate rewards. The output is a list of PoolingRequestOutput.
    outputs = llm.reward(prompts)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        rewards = output.outputs.data
        print(f"Prompt: {prompt!r}")
        print_embeddings(rewards, prefix="Reward")
        print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
