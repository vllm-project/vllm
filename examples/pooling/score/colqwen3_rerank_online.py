# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Example of using ColQwen3 late interaction model for reranking and scoring.

ColQwen3 is a multi-modal ColBERT-style model based on Qwen3-VL.
It produces per-token embeddings and uses MaxSim scoring for retrieval
and reranking. Supports both text and image inputs.

Start the server with:
    vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 50000

Then run this script:
    python colqwen3_rerank_online.py
"""

import base64
from io import BytesIO

import requests
from PIL import Image

MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"
BASE_URL = "http://127.0.0.1:8000"

headers = {"accept": "application/json", "Content-Type": "application/json"}

# ── Image helpers ──────────────────────────────────────────


def load_image(url: str) -> Image.Image:
    """Download an image from URL (handles Wikimedia 403)."""
    for hdrs in (
        {},
        {"User-Agent": "Mozilla/5.0 (compatible; ColQwen3-demo/1.0)"},
    ):
        resp = requests.get(url, headers=hdrs, timeout=15)
        if resp.status_code == 403:
            continue
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    raise RuntimeError(f"Could not fetch image from {url}")


def encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL image to a base64 data URI."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_image_content(image_url: str, text: str = "Describe the image.") -> dict:
    """Build a ScoreMultiModalParam dict from an image URL."""
    image = load_image(image_url)
    return {
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_base64(image)},
            },
            {"type": "text", "text": text},
        ]
    }


# ── Sample image URLs ─────────────────────────────────────

IMAGE_URLS = {
    "beijing": "https://upload.wikimedia.org/wikipedia/commons/6/61/Beijing_skyline_at_night.JPG",
    "london": "https://upload.wikimedia.org/wikipedia/commons/4/49/London_skyline.jpg",
    "singapore": "https://upload.wikimedia.org/wikipedia/commons/2/27/Singapore_skyline_2022.jpg",
}

# ── Text-only examples ────────────────────────────────────


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


# ── Multi-modal examples (text query × image documents) ──


def score_text_vs_images():
    """Score a text query against image documents via /score."""
    print()
    print("=" * 60)
    print("4. Multi-modal scoring: text query vs image docs (/score)")
    print("=" * 60)

    query = "Retrieve the city of Beijing"
    labels = list(IMAGE_URLS.keys())
    print(f"\n  Loading {len(labels)} images...")
    image_contents = [make_image_content(IMAGE_URLS[name]) for name in labels]

    data = {
        "model": MODEL,
        "data_1": query,
        "data_2": image_contents,
    }

    response = requests.post(f"{BASE_URL}/score", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f'\n  Query: "{query}"\n')
        for item in result["data"]:
            idx = item["index"]
            print(f"    Doc {idx} [{labels[idx]}] score={item['score']:.4f}")
    else:
        print(f"  Request failed: {response.status_code}")
        print(f"  {response.text[:300]}")


def rerank_text_vs_images():
    """Rerank image documents by a text query via /rerank."""
    print()
    print("=" * 60)
    print("5. Multi-modal reranking: text query vs image docs (/rerank)")
    print("=" * 60)

    query = "Retrieve the city of London"
    labels = list(IMAGE_URLS.keys())
    print(f"\n  Loading {len(labels)} images...")
    image_contents = [make_image_content(IMAGE_URLS[name]) for name in labels]

    data = {
        "model": MODEL,
        "query": query,
        "documents": image_contents,
        "top_n": 2,
    }

    response = requests.post(f"{BASE_URL}/rerank", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f'\n  Query: "{query}"')
        print(f"  Top {data['top_n']} results:\n")
        for item in result["results"]:
            idx = item["index"]
            print(f"    [{item['relevance_score']:.4f}] {labels[idx]}")
    else:
        print(f"  Request failed: {response.status_code}")
        print(f"  {response.text[:300]}")


# ── Main ──────────────────────────────────────────────────


def main():
    # Text-only
    rerank_text()
    score_text()
    score_text_top_n()

    # Multi-modal (text query × image documents)
    score_text_vs_images()
    rerank_text_vs_images()


if __name__ == "__main__":
    main()
