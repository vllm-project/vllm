# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
from argparse import Namespace
from pathlib import Path
from typing import Any

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


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
        model="nvidia/llama-nemotron-rerank-1b-v2",  # Default reranking model
        runner="pooling",  # Required for cross-encoder/reranking models
        trust_remote_code=True,  # Allow loading models with custom code
    )
    return parser.parse_args()


def get_chat_template(model: str) -> str:
    """Load the appropriate chat template for the specified model.

    Reranking models require specific prompt templates to format
    query-document pairs correctly. This function maps model names
    to their corresponding template files.
    """
    # Directory containing all chat template files
    template_home = Path(__file__).parent / "template"

    # Mapping from model names to their corresponding template files
    # Each reranking model has its own specific prompt format
    model_name_to_template_path_map = {
        "BAAI/bge-reranker-v2-gemma": "bge-reranker-v2-gemma.jinja",
        "Qwen/Qwen3-Reranker-0.6B": "qwen3_reranker.jinja",
        "Qwen/Qwen3-Reranker-4B": "qwen3_reranker.jinja",
        "Qwen/Qwen3-Reranker-8B": "qwen3_reranker.jinja",
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": "qwen3_reranker.jinja",
        "tomaarsen/Qwen3-Reranker-4B-seq-cls": "qwen3_reranker.jinja",
        "tomaarsen/Qwen3-Reranker-8B-seq-cls": "qwen3_reranker.jinja",
        "mixedbread-ai/mxbai-rerank-base-v2": "mxbai_rerank_v2.jinja",
        "mixedbread-ai/mxbai-rerank-large-v2": "mxbai_rerank_v2.jinja",
        "nvidia/llama-nemotron-rerank-1b-v2": "nemotron-rerank.jinja",
    }

    # Get the template filename for the specified model
    template_path = model_name_to_template_path_map.get(model)

    if template_path is None:
        raise ValueError(f"This demo does not support model name: {model}.")

    # Read and return the template content
    return (template_home / template_path).read_text()


def get_hf_overrides(model: str) -> dict[str, Any]:
    """Convert Large Language Models (LLMs) to Sequence Classification models.

    note:
        Some reranking models require special configuration overrides to work
        correctly with vLLM's score API.
        Reference: https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/qwen3_reranker_offline.py
        Reference: https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/convert_model_to_seq_cls.py
    """

    model_name_to_hf_overrides_map = {
        "BAAI/bge-reranker-v2-gemma": {
            "architectures": ["GemmaForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        },
        "Qwen/Qwen3-Reranker-0.6B": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        "Qwen/Qwen3-Reranker-4B": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        "Qwen/Qwen3-Reranker-8B": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": {},
        "tomaarsen/Qwen3-Reranker-4B-seq-cls": {},
        "tomaarsen/Qwen3-Reranker-8B-seq-cls": {},
        "mixedbread-ai/mxbai-rerank-base-v2": {
            "architectures": ["Qwen2ForSequenceClassification"],
            "classifier_from_token": ["0", "1"],
            "method": "from_2_way_softmax",
        },
        "mixedbread-ai/mxbai-rerank-large-v2": {
            "architectures": ["Qwen2ForSequenceClassification"],
            "classifier_from_token": ["0", "1"],
            "method": "from_2_way_softmax",
        },
        "nvidia/llama-nemotron-rerank-1b-v2": {},
    }

    hf_overrides = model_name_to_hf_overrides_map.get(model)

    if hf_overrides is None:
        raise ValueError(f"This demo does not support model name: {model}.")

    return hf_overrides


def main(args: Namespace):
    """Main execution function for the reranking example."""

    # Get the overrides for the specified model
    args.hf_overrides = get_hf_overrides(args.model)

    # Load the appropriate chat template for the selected model
    # The template formats query-document pairs for the reranking model
    args.chat_template = get_chat_template(args.model)

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
