# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
from argparse import Namespace
from pathlib import Path
from typing import Any

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

"""
Example of using score with API SentenceTransformers V6 reranker config.
Users can correctly use the latest powerful rerank model without manually 
setting any hf_overrides or loading any templates.

e.g.
    cross-encoder-testing/ctxl-rerank-v2-instruct-multilingual-1b-STv6
    cross-encoder-testing/Qwen3-Reranker-0.6B-STv6
    cross-encoder-testing/Qwen3-Reranker-0.6B-seq-cls-STv6
    cross-encoder-testing/mxbai-rerank-large-v2-STv6
    cross-encoder-testing/mxbai-rerank-base-v2-STv6
"""


def parse_args():
    """Parse command line arguments for the reranking example.

    This function sets up the argument parser with default values
    specific to reranking models, including the model name and
    runner type.
    """
    parser = FlexibleArgumentParser()
    # Add all EngineArgs command line arguments to the parser
    parser = EngineArgs.add_cli_args(parser)

    # Set default values specific to this reranking example
    # These defaults ensure the script works out-of-the-box for reranking tasks
    parser.set_defaults(
        model="cross-encoder-testing/Qwen3-Reranker-0.6B-seq-cls-STv6",  # Default reranking model
        runner="pooling",  # Required for cross-encoder/reranking models
        trust_remote_code=True,  # Allow loading models with custom code
    )
    return parser.parse_args()

def main(args: Namespace):
    """Main execution function for the reranking example."""

    # Initialize the LLM with all provided arguments
    llm = LLM(**vars(args))

    # Example query for demonstration
    query = "how much protein should a female eat?"

    # Example documents to be reranked based on relevance to the query
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision of a health professional.",
    ]

    # Score documents based on relevance to the query
    # The score method returns relevance scores for each document
    outputs = llm.score(query, documents)

    # Display the relevance scores
    # Higher scores indicate more relevant documents
    print("-" * 30)
    print([output.outputs.score for output in outputs])
    print("-" * 30)


if __name__ == "__main__":
    args = parse_args()
    main(args)
