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
        model="BAAI/bge-reranker-v2-m3",
        runner="pooling",
        enforce_eager=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    text_1 = "What is the capital of France?"
    texts_2 = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    # Create an LLM.
    # You should pass runner="pooling" for cross-encoder models
    llm = LLM(**vars(args))

    # Generate scores. The output is a list of ScoringRequestOutputs.
    outputs = llm.score(text_1, texts_2)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for text_2, output in zip(texts_2, outputs):
        score = output.outputs.score
        print(f"Pair: {[text_1, text_2]!r} \nScore: {score}")
        print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
