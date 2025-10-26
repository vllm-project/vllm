# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="BAAI/bge-m3",
        runner="pooling",
        enforce_eager=True,
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
    # You should pass runner="pooling" for embedding models
    llm = LLM(**vars(args))

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = llm.embed(prompts)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        print(len(embeds))

    # Generate embedding for each token. The output is a list of PoolingRequestOutput.
    outputs = llm.encode(prompts, pooling_task="token_embed")

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        multi_vector = output.outputs.data
        print(multi_vector.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
