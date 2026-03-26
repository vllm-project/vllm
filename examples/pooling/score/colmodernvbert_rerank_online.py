# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example of using ColModernVBERT late interaction model for reranking.

ColModernVBERT is a multi-modal ColBERT-style model combining a SigLIP
vision encoder with a ModernBERT text encoder. It produces per-token
embeddings and uses MaxSim scoring for retrieval and reranking.
Supports both text and image inputs.

Start the server with:
    vllm serve ModernVBERT/colmodernvbert-merged --max-model-len 8192

Then run this script:
    python colmodernvbert_rerank_online.py
"""

import requests

MODEL = "ModernVBERT/colmodernvbert-merged"
BASE_URL = "http://127.0.0.1:8000"

headers = {"accept": "application/json", "Content-Type": "application/json"}

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"  # noqa: E501


def rerank_text():
    """Text-only reranking via /rerank endpoint."""
    print("=" * 60)
    print("1. Text reranking (/rerank)")
    print("=" * 60)

    data = {
        "model": MODEL,
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "Deep learning uses neural networks for complex tasks.",
            "The weather today is sunny.",
        ],
    }

    response = requests.post(f"{BASE_URL}/rerank", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print("\n  Ranked documents (most relevant first):")
        for item in result["results"]:
            doc_idx = item["index"]
            score = item["relevance_score"]
            print(f"    [{score:.4f}] {data['documents'][doc_idx]}")
    else:
        print(f"  Request failed: {response.status_code}")
        print(f"  {response.text[:300]}")


def score_text():
    """Text-only scoring via /score endpoint."""
    print()
    print("=" * 60)
    print("2. Text scoring (/score)")
    print("=" * 60)

    query = "What is the capital of France?"
    documents = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Python is a programming language.",
    ]

    data = {
        "model": MODEL,
        "text_1": query,
        "text_2": documents,
    }

    response = requests.post(f"{BASE_URL}/score", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"\n  Query: {query}\n")
        for item in result["data"]:
            idx = item["index"]
            score = item["score"]
            print(f"    Doc {idx} (score={score:.4f}): {documents[idx]}")
    else:
        print(f"  Request failed: {response.status_code}")
        print(f"  {response.text[:300]}")


def score_text_top_n():
    """Text reranking with top_n filtering via /rerank endpoint."""
    print()
    print("=" * 60)
    print("3. Text reranking with top_n=2 (/rerank)")
    print("=" * 60)

    data = {
        "model": MODEL,
        "query": "What is the capital of France?",
        "documents": [
            "The capital of France is Paris.",
            "Berlin is the capital of Germany.",
            "Python is a programming language.",
            "The Eiffel Tower is in Paris.",
        ],
        "top_n": 2,
    }

    response = requests.post(f"{BASE_URL}/rerank", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"\n  Top {data['top_n']} results:")
        for item in result["results"]:
            doc_idx = item["index"]
            score = item["relevance_score"]
            print(f"    [{score:.4f}] {data['documents'][doc_idx]}")
    else:
        print(f"  Request failed: {response.status_code}")
        print(f"  {response.text[:300]}")


def rerank_multimodal():
    """Multimodal reranking with text and image documents via /rerank."""
    print()
    print("=" * 60)
    print("4. Multimodal reranking: text query vs image document (/rerank)")
    print("=" * 60)

    data = {
        "model": MODEL,
        "query": "A colorful logo with transparency",
        "documents": [
            {"content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}}]},
            "Python is a programming language.",
            "The weather today is sunny.",
        ],
    }

    response = requests.post(f"{BASE_URL}/rerank", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print("\n  Ranked documents (most relevant first):")
        labels = ["[image]", "Python doc", "Weather doc"]
        for item in result["results"]:
            doc_idx = item["index"]
            score = item["relevance_score"]
            print(f"    [{score:.4f}] {labels[doc_idx]}")
    else:
        print(f"  Request failed: {response.status_code}")
        print(f"  {response.text[:300]}")


def main():
    rerank_text()
    score_text()
    score_text_top_n()
    rerank_multimodal()


if __name__ == "__main__":
    main()
