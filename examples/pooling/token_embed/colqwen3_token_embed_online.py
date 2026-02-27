# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""
Example online usage of Pooling API for ColQwen3 multi-vector retrieval.

ColQwen3 is a multi-modal late interaction model based on Qwen3-VL that
produces per-token embeddings (320-dim, L2-normalized) for both text and
image inputs. Similarity is computed via MaxSim scoring.

This example mirrors the official TomoroAI inference code
(https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) but uses the
vLLM serving API instead of local HuggingFace model loading.

Start the server with:
    vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 4096

Then run this script:
    python colqwen3_token_embed_online.py
"""

import argparse
import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image

# ── Helpers ─────────────────────────────────────────────────


def post_http_request(payload: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    return requests.post(api_url, headers=headers, json=payload)


def load_image(url: str) -> Image.Image:
    """Download an image from URL (handles Wikimedia 403)."""
    for hdrs in ({}, {"User-Agent": "Mozilla/5.0 (compatible; ColQwen3-demo/1.0)"}):
        resp = requests.get(url, headers=hdrs, timeout=10)
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


def compute_maxsim(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
    """Compute ColBERT-style MaxSim score between query and document."""
    sim = q_emb @ d_emb.T
    return float(sim.max(axis=-1).sum())


# ── Encode functions ────────────────────────────────────────


def encode_queries(texts: list[str], model: str, api_url: str) -> list[np.ndarray]:
    """Encode text queries → list of multi-vector embeddings."""
    resp = post_http_request({"model": model, "input": texts}, api_url)
    return [np.array(item["data"]) for item in resp.json()["data"]]


def encode_images(image_urls: list[str], model: str, api_url: str) -> list[np.ndarray]:
    """Encode image documents → list of multi-vector embeddings.

    Images are sent via the chat-style `messages` field so that the
    vLLM multimodal processor handles them correctly.
    """
    embeddings = []
    for url in image_urls:
        print(f"  Loading: {url.split('/')[-1]}...")
        image = load_image(url)
        image_uri = encode_image_base64(image)
        resp = post_http_request(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": "Describe the image."},
                        ],
                    }
                ],
            },
            api_url,
        )
        result = resp.json()
        if resp.status_code != 200 or "data" not in result:
            print(f"    Error ({resp.status_code}): {str(result)[:200]}")
            continue
        embeddings.append(np.array(result["data"][0]["data"]))
    return embeddings


# ── Main ────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="TomoroAI/tomoro-colqwen3-embed-4b",
    )
    return parser.parse_args()


def main(args):
    pooling_url = f"http://{args.host}:{args.port}/pooling"
    score_url = f"http://{args.host}:{args.port}/score"
    model = args.model

    # Same sample data as the official TomoroAI example
    queries = [
        "Retrieve the city of Singapore",
        "Retrieve the city of Beijing",
        "Retrieve the city of London",
    ]
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Singapore_skyline_2022.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/61/Beijing_skyline_at_night.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/4/49/London_skyline.jpg",
    ]

    # ── 1) Text query embeddings ────────────────────────────
    print("=" * 60)
    print("1. Encode text queries (multi-vector)")
    print("=" * 60)
    query_embeddings = encode_queries(queries, model, pooling_url)
    for i, emb in enumerate(query_embeddings):
        norm = float(np.linalg.norm(emb[0]))
        print(f'  Query {i}: {emb.shape}  (L2 norm: {norm:.4f})  "{queries[i]}"')

    # ── 2) Image document embeddings ────────────────────────
    print()
    print("=" * 60)
    print("2. Encode image documents (multi-vector)")
    print("=" * 60)
    doc_embeddings = encode_images(image_urls, model, pooling_url)
    for i, emb in enumerate(doc_embeddings):
        print(f"  Doc {i}:   {emb.shape}  {image_urls[i].split('/')[-1]}")

    # ── 3) Cross-modal MaxSim scoring ───────────────────────
    if doc_embeddings:
        print()
        print("=" * 60)
        print("3. Cross-modal MaxSim scores (text queries × image docs)")
        print("=" * 60)
        # Header
        print(f"{'':>35s}", end="")
        for j in range(len(doc_embeddings)):
            print(f"  Doc {j:>2d}", end="")
        print()
        # Score matrix
        for i, q_emb in enumerate(query_embeddings):
            print(f"  {queries[i]:<33s}", end="")
            for j, d_emb in enumerate(doc_embeddings):
                score = compute_maxsim(q_emb, d_emb)
                print(f"  {score:6.2f}", end="")
            print()

    # ── 4) Text-only /score endpoint ────────────────────────
    print()
    print("=" * 60)
    print("4. Text-only late interaction scoring (/score endpoint)")
    print("=" * 60)
    text_query = "What is the capital of France?"
    text_docs = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Python is a programming language.",
    ]
    resp = post_http_request(
        {"model": model, "text_1": text_query, "text_2": text_docs},
        score_url,
    )
    print(f'  Query: "{text_query}"\n')
    for item in resp.json()["data"]:
        idx = item["index"]
        print(f"  Doc {idx} (score={item['score']:.4f}): {text_docs[idx]}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
