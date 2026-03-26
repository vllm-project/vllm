# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity test between Cohere /v2/embed and OpenAI /v1/embeddings.

Verifies that both endpoints produce identical float embeddings when
no prompt prefix is applied (input_type omitted for Cohere /v2/embed).
"""

import numpy as np
import pytest
import requests

from tests.utils import ROCM_EXTRA_ARGS, RemoteOpenAIServer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DTYPE = "bfloat16"


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.02",
    ] + ROCM_EXTRA_ARGS
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _cohere_embed(
    server: RemoteOpenAIServer,
    texts: list[str],
) -> list[list[float]]:
    body = {
        "model": MODEL_NAME,
        "texts": texts,
        "embedding_types": ["float"],
    }
    resp = requests.post(server.url_for("/v2/embed"), json=body)
    resp.raise_for_status()
    return resp.json()["embeddings"]["float"]


def _openai_embed(
    server: RemoteOpenAIServer,
    texts: list[str],
) -> list[list[float]]:
    body = {"model": MODEL_NAME, "input": texts, "encoding_format": "float"}
    resp = requests.post(server.url_for("/v1/embeddings"), json=body)
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def test_single_text_parity(server: RemoteOpenAIServer):
    """A single text should produce identical embeddings via both APIs."""
    texts = ["the quick brown fox jumps over the lazy dog"]
    v2 = _cohere_embed(server, texts)
    v1 = _openai_embed(server, texts)
    np.testing.assert_allclose(v2[0], v1[0], rtol=1e-5)


def test_batch_parity(server: RemoteOpenAIServer):
    """A batch of texts should produce identical embeddings via both APIs,
    in the same order."""
    texts = [
        "machine learning",
        "deep learning",
        "natural language processing",
    ]
    v2 = _cohere_embed(server, texts)
    v1 = _openai_embed(server, texts)
    assert len(v2) == len(v1) == 3
    for i in range(3):
        np.testing.assert_allclose(v2[i], v1[i], rtol=1e-5, err_msg=f"index {i}")


def test_token_count_parity(server: RemoteOpenAIServer):
    """Both APIs should report the same prompt token count."""
    texts = ["hello world"]
    v2_resp = requests.post(
        server.url_for("/v2/embed"),
        json={
            "model": MODEL_NAME,
            "texts": texts,
            "embedding_types": ["float"],
        },
    )
    v1_resp = requests.post(
        server.url_for("/v1/embeddings"),
        json={"model": MODEL_NAME, "input": texts, "encoding_format": "float"},
    )
    v2_resp.raise_for_status()
    v1_resp.raise_for_status()
    v2_tokens = v2_resp.json()["meta"]["billed_units"]["input_tokens"]
    v1_tokens = v1_resp.json()["usage"]["prompt_tokens"]
    assert v2_tokens == v1_tokens
