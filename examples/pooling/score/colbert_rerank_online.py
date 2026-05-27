# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example of using ColBERT late interaction models for reranking and scoring.

ColBERT (Contextualized Late Interaction over BERT) uses per-token embeddings
and MaxSim scoring for document reranking, providing better accuracy than
single-vector models while being more efficient than cross-encoders.

vLLM supports ColBERT with multiple encoder backbones. Start the server
with one of the following:

    # BERT backbone (works out of the box)
    vllm serve answerdotai/answerai-colbert-small-v1

    # ModernBERT backbone
    vllm serve lightonai/GTE-ModernColBERT-v1 \
        --hf-overrides '{"architectures": ["ColBERTModernBertModel"]}'

    # Jina XLM-RoBERTa backbone
    vllm serve jinaai/jina-colbert-v2 \
        --hf-overrides '{"architectures": ["ColBERTJinaRobertaModel"]}' \
        --trust-remote-code

Then run this script:
    python colbert_rerank_online.py
"""

import json

import requests

# Change this to match the model you started the server with
MODEL = "answerdotai/answerai-colbert-small-v1"
BASE_URL = "http://127.0.0.1:8000"

headers = {"accept": "application/json", "Content-Type": "application/json"}

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a programming language.",
    "Deep learning uses neural networks for complex tasks.",
    "The weather today is sunny.",
]


def rerank_example():
    """Use the /rerank endpoint to rank documents by query relevance."""
    print("=== Rerank Example ===")

    data = {
        "model": MODEL,
        "query": "What is machine learning?",
        "documents": documents,
    }

    response = requests.post(f"{BASE_URL}/rerank", headers=headers, json=data)
    result = response.json()
    print(json.dumps(result, indent=2))

    print("\nRanked documents (most relevant first):")
    for item in result["results"]:
        doc_idx = item["index"]
        score = item["relevance_score"]
        print(f"  Score {score:.4f}: {documents[doc_idx]}")


def score_example():
    """Use the /score endpoint for pairwise query-document scoring."""
    print("\n=== Score Example ===")

    data = {
        "model": MODEL,
        "text_1": "What is machine learning?",
        "text_2": [
            "Machine learning is a subset of AI.",
            "The weather is sunny.",
        ],
    }

    response = requests.post(f"{BASE_URL}/score", headers=headers, json=data)
    result = response.json()
    print(json.dumps(result, indent=2))


def main():
    rerank_example()
    score_example()


if __name__ == "__main__":
    main()
