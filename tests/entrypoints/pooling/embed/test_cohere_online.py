# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Cohere /v2/embed API with generic (non-Cohere) models.

Validates that the Cohere v2 embed endpoint works correctly with standard
embedding models, covering text embedding, embedding type conversions,
response structure, batching, normalisation, and semantic similarity.
"""

import struct

import numpy as np
import pybase64 as base64
import pytest
import requests

from tests.utils import RemoteOpenAIServer

DTYPE = "bfloat16"

MODELS: list[tuple[str, list[str]]] = [
    ("intfloat/multilingual-e5-small", []),
    (
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        [
            "--trust_remote_code",
            "--hf_overrides",
            '{"matryoshka_dimensions":[256]}',
        ],
    ),
]


@pytest.fixture(scope="module", params=MODELS, ids=lambda m: m[0])
def model_config(request):
    return request.param


@pytest.fixture(scope="module")
def model_name(model_config):
    return model_config[0]


@pytest.fixture(scope="module")
def server(model_config):
    name, extra_args = model_config
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
    ] + extra_args
    with RemoteOpenAIServer(name, args) as remote_server:
        yield remote_server


def _cohere_embed(
    server: RemoteOpenAIServer,
    model_name: str,
    texts: list[str] | None = None,
    images: list[str] | None = None,
    input_type: str | None = None,
    embedding_types: list[str] | None = None,
) -> dict:
    body: dict = {"model": model_name}
    if input_type is not None:
        body["input_type"] = input_type
    if texts is not None:
        body["texts"] = texts
    if images is not None:
        body["images"] = images
    if embedding_types is not None:
        body["embedding_types"] = embedding_types
    resp = requests.post(server.url_for("/v2/embed"), json=body)
    resp.raise_for_status()
    return resp.json()


def _openai_embed(
    server: RemoteOpenAIServer, model_name: str, texts: list[str]
) -> dict:
    body = {"model": model_name, "input": texts, "encoding_format": "float"}
    resp = requests.post(server.url_for("/v1/embeddings"), json=body)
    resp.raise_for_status()
    return resp.json()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


# -----------------------------------------------------------
# Text embedding tests
# -----------------------------------------------------------


def test_basic_embed(server: RemoteOpenAIServer, model_name: str):
    r = _cohere_embed(
        server, model_name, texts=["hello world"], embedding_types=["float"]
    )
    assert "embeddings" in r
    assert len(r["embeddings"]["float"]) == 1
    assert len(r["embeddings"]["float"][0]) > 0


def test_unsupported_input_type_rejected(server: RemoteOpenAIServer, model_name: str):
    """An input_type not defined in the model's prompt config should be
    rejected with a 400 error."""
    body = {
        "model": model_name,
        "input_type": "nonexistent_type",
        "texts": ["hello world"],
        "embedding_types": ["float"],
    }
    resp = requests.post(server.url_for("/v2/embed"), json=body)
    assert resp.status_code == 400
    assert "Unsupported input_type" in resp.json()["error"]["message"]


def test_omitted_input_type_accepted(server: RemoteOpenAIServer, model_name: str):
    """Omitting input_type should always work (no prompt prefix applied)."""
    body = {
        "model": model_name,
        "texts": ["hello world"],
        "embedding_types": ["float"],
    }
    resp = requests.post(server.url_for("/v2/embed"), json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embeddings"]["float"]) == 1


def test_v1_v2_parity(server: RemoteOpenAIServer, model_name: str):
    """v1 (OpenAI) and v2 (Cohere) endpoints should produce the same
    float embeddings for a generic model."""
    texts = ["hello world"]
    v2 = _cohere_embed(server, model_name, texts=texts, embedding_types=["float"])
    v1 = _openai_embed(server, model_name, texts)
    cos = _cosine_sim(v2["embeddings"]["float"][0], v1["data"][0]["embedding"])
    assert cos > 0.9999, f"v1/v2 parity failed, cosine={cos}"


def test_embedding_types(server: RemoteOpenAIServer, model_name: str):
    r = _cohere_embed(
        server,
        model_name,
        texts=["test"],
        embedding_types=["float", "binary", "ubinary"],
    )
    dim = len(r["embeddings"]["float"][0])
    assert len(r["embeddings"]["binary"][0]) == dim // 8
    assert len(r["embeddings"]["ubinary"][0]) == dim // 8


def test_response_structure(server: RemoteOpenAIServer, model_name: str):
    r = _cohere_embed(server, model_name, texts=["test"], embedding_types=["float"])
    assert "id" in r
    assert "embeddings" in r
    assert "texts" in r
    assert r["texts"] == ["test"]
    assert "meta" in r
    assert r["meta"]["api_version"]["version"] == "2"
    assert "billed_units" in r["meta"]
    assert r["meta"]["billed_units"]["input_tokens"] > 0
    assert r["meta"]["billed_units"]["image_tokens"] == 0


def test_batch(server: RemoteOpenAIServer, model_name: str):
    texts = ["apple", "banana", "cherry"]
    r = _cohere_embed(server, model_name, texts=texts, embedding_types=["float"])
    assert len(r["embeddings"]["float"]) == 3
    dim = len(r["embeddings"]["float"][0])
    for emb in r["embeddings"]["float"]:
        assert len(emb) == dim


def test_l2_normalized(server: RemoteOpenAIServer, model_name: str):
    r = _cohere_embed(
        server, model_name, texts=["hello world"], embedding_types=["float"]
    )
    emb = np.array(r["embeddings"]["float"][0])
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 0.01


def test_semantic_similarity(server: RemoteOpenAIServer, model_name: str):
    r = _cohere_embed(
        server,
        model_name,
        texts=["machine learning", "deep learning", "chocolate cake recipe"],
        embedding_types=["float"],
    )
    embs = r["embeddings"]["float"]
    cos_related = _cosine_sim(embs[0], embs[1])
    cos_unrelated = _cosine_sim(embs[0], embs[2])
    assert cos_related > cos_unrelated


def test_missing_input_returns_error(server: RemoteOpenAIServer, model_name: str):
    body = {"model": model_name}
    resp = requests.post(server.url_for("/v2/embed"), json=body)
    assert resp.status_code == 400


def test_base64_embedding_type(server: RemoteOpenAIServer, model_name: str):
    r = _cohere_embed(
        server,
        model_name,
        texts=["test encoding"],
        embedding_types=["float", "base64"],
    )
    float_emb = r["embeddings"]["float"][0]
    b64_str = r["embeddings"]["base64"][0]
    decoded = struct.unpack(f"<{len(float_emb)}f", base64.b64decode(b64_str))
    np.testing.assert_allclose(float_emb, decoded, rtol=1e-5)


# -----------------------------------------------------------
# Truncation tests
# -----------------------------------------------------------


def _cohere_embed_raw(
    server: RemoteOpenAIServer,
    body: dict,
) -> requests.Response:
    return requests.post(server.url_for("/v2/embed"), json=body)


def test_truncate_end_succeeds(server: RemoteOpenAIServer, model_name: str):
    """truncate=END should silently truncate long input."""
    long_text = " ".join(["word"] * 2000)
    body = {
        "model": model_name,
        "texts": [long_text],
        "embedding_types": ["float"],
        "truncate": "END",
    }
    resp = _cohere_embed_raw(server, body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embeddings"]["float"]) == 1


def test_truncate_start_succeeds(server: RemoteOpenAIServer, model_name: str):
    """truncate=START should silently truncate long input from the start."""
    long_text = " ".join(["word"] * 2000)
    body = {
        "model": model_name,
        "texts": [long_text],
        "embedding_types": ["float"],
        "truncate": "START",
    }
    resp = _cohere_embed_raw(server, body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embeddings"]["float"]) == 1


def test_truncate_none_rejects_long_input(server: RemoteOpenAIServer, model_name: str):
    """truncate=NONE should error when input exceeds model context."""
    long_text = " ".join(["word"] * 2000)
    body = {
        "model": model_name,
        "texts": [long_text],
        "embedding_types": ["float"],
        "truncate": "NONE",
    }
    resp = _cohere_embed_raw(server, body)
    assert resp.status_code == 400


def test_truncate_start_vs_end_differ(server: RemoteOpenAIServer, model_name: str):
    """START and END truncation should produce different embeddings
    when the input is long enough to actually be truncated.

    We construct input with distinct tokens at the start vs end
    so that keeping different halves produces different embeddings.
    """
    start_words = " ".join([f"alpha{i}" for i in range(300)])
    end_words = " ".join([f"omega{i}" for i in range(300)])
    long_text = start_words + " " + end_words

    body_end = {
        "model": model_name,
        "texts": [long_text],
        "embedding_types": ["float"],
        "truncate": "END",
    }
    body_start = {
        "model": model_name,
        "texts": [long_text],
        "embedding_types": ["float"],
        "truncate": "START",
    }
    r_end = _cohere_embed_raw(server, body_end).json()
    r_start = _cohere_embed_raw(server, body_start).json()

    emb_end = r_end["embeddings"]["float"][0]
    emb_start = r_start["embeddings"]["float"][0]
    cos = _cosine_sim(emb_end, emb_start)
    assert cos < 0.99, (
        f"START and END truncation should produce different embeddings "
        f"for long input, but cosine similarity was {cos}"
    )
