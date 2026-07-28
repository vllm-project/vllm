# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example of using ColQwen3.5 late interaction model for reranking.

ColQwen3.5 is a multi-modal ColBERT-style model based on Qwen3.5.
It produces per-token embeddings and uses MaxSim scoring for retrieval
and reranking. Supports both text and image inputs.

Works for any ColQwen3.5 checkpoint, e.g. `athrael-soju/colqwen3.5-4.5B-v3`
or `vultr/VultronRetrieverPrime-Qwen3.5-8B`.

Start the server with:
    vllm serve athrael-soju/colqwen3.5-4.5B-v3 --max-model-len 4096 \
        --mm-processor-kwargs '{"min_pixels": 65536, "max_pixels": 1835008}'

Then run this script:
    python colqwen3_5_rerank_online.py

Parity note (matching the native colpali ColQwen3_5Processor pipeline):
  - Visual-token budget: ColQwen3_5Processor uses max_num_visual_tokens=1792,
    i.e. max_pixels = 1792 * (patch_size*merge_size)^2 = 1792 * 32^2 = 1835008
    (with min_pixels = shortest_edge = 65536). Pass these via --mm-processor-kwargs
    as above; the default budget gives fewer visual tokens and lower retrieval ndcg.
  - When you build prompts yourself (token_embed), reproduce the processor exactly:
      image (document): wrap in the instruction template
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image.<|im_end|><|endoftext|>"
      query: append the augmentation suffix  <text> + "<|endoftext|>" * 10
    Omitting these reproduces a silent ~2.5 ndcg@10 drop vs the native pipeline.
"""

import requests

MODEL = "athrael-soju/colqwen3.5-4.5B"
BASE_URL = "http://127.0.0.1:8000"

headers = {"accept": "application/json", "Content-Type": "application/json"}


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


def main():
    rerank_text()
    score_text()
    score_text_top_n()


if __name__ == "__main__":
    main()
