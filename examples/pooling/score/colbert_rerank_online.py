# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example of using ColBERT late interaction model for reranking.

ColBERT (Contextualized Late Interaction over BERT) uses per-token embeddings
and MaxSim scoring for document reranking, providing better accuracy than
single-vector models while being more efficient than cross-encoders.

Start the server with:
    vllm serve answerdotai/answerai-colbert-small-v1

Then run this script:
    python colbert_rerank_online.py
"""

import json

import requests

url = "http://127.0.0.1:8000/rerank"

headers = {"accept": "application/json", "Content-Type": "application/json"}

data = {
    "model": "answerdotai/answerai-colbert-small-v1",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks for complex tasks.",
        "The weather today is sunny.",
    ],
}


def main():
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("ColBERT Rerank Request successful!")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Show ranked results
        print("\nRanked documents (most relevant first):")
        for item in result["results"]:
            doc_idx = item["index"]
            score = item["relevance_score"]
            print(f"  Score {score:.4f}: {data['documents'][doc_idx]}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
