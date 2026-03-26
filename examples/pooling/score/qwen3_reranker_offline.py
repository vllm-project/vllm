# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""
What is the difference between the official original version and one
that has been converted into a sequence classification model?

Qwen3-Reranker is a language model that doing reranker by using the
logits of "no" and "yes" tokens.
This requires computing logits for all 151,669 tokens in the vocabulary,
making it inefficient and incompatible with vLLM's score() API.

A conversion method has been proposed to transform the original model into a
sequence classification model. This converted model:
1. Is significantly more efficient
2. Fully supports vLLM's score() API
3. Simplifies initialization parameters
Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
Reference: https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/convert_model_to_seq_cls.py

For the converted model, initialization would simply be:
llm = LLM(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", runner="pooling")

This example demonstrates loading the ORIGINAL model with special overrides
to make it compatible with vLLM's score API.
"""

from pathlib import Path

from vllm import LLM

model_name = "Qwen/Qwen3-Reranker-0.6B"


def get_llm() -> LLM:
    """
    Initializes and returns the LLM model for Qwen3-Reranker.

    Returns:
        LLM: Configured vLLM instance for reranking tasks.

    Note:
        This function loads the ORIGINAL Qwen3-Reranker model with specific
        overrides to make it compatible with vLLM's score API.
    """
    return LLM(
        # Specify the original model from HuggingFace
        model=model_name,
        # Use pooling runner for score task
        runner="pooling",
        # HuggingFace model configuration overrides required for compatibility
        hf_overrides={
            # Manually route to sequence classification architecture
            # This tells vLLM to use Qwen3ForSequenceClassification instead of
            # the default Qwen3ForCausalLM
            "architectures": ["Qwen3ForSequenceClassification"],
            # Specify which token logits to extract from the language model head
            # The original reranker uses "no" and "yes" token logits for scoring
            "classifier_from_token": ["no", "yes"],
            # Enable special handling for original Qwen3-Reranker models
            # This flag triggers conversion logic that transforms the two token
            # vectors into a single classification vector
            "is_original_qwen3_reranker": True,
        },
    )


def main() -> None:
    # Load the Jinja template for formatting query-document pairs
    # The template ensures proper formatting for the reranker model
    template_home = Path(__file__).parent / "template"
    template_path = "qwen3_reranker.jinja"
    chat_template = (template_home / template_path).read_text()

    # Sample queries for testing the reranker
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    # Corresponding documents to be scored against each query
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    # Initialize the LLM model with the original Qwen3-Reranker configuration
    llm = get_llm()

    # Compute relevance scores for each query-document pair
    # The score() method returns a relevance score for each pair
    # Higher scores indicate better relevance
    outputs = llm.score(queries, documents, chat_template=chat_template)

    # Extract and print the relevance scores from the outputs
    # Each output contains a score representing query-document relevance
    print("-" * 30)
    print("Relevance scores:", [output.outputs.score for output in outputs])
    print("-" * 30)


if __name__ == "__main__":
    main()
