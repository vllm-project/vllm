# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Cohere /v2/embed API with a multimodal model (SigLIP).

Validates image embedding, batching, normalisation, and embedding type
conversions through the /v2/embed endpoint.
"""

import base64
import struct
import zlib

import numpy as np
import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "google/siglip-so400m-patch14-384"
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
        "64",
        "--gpu-memory-utilization",
        "0.3",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _make_tiny_png(r: int, g: int, b: int, w: int = 2, h: int = 2) -> str:
    raw = b""
    for _ in range(h):
        raw += b"\x00" + bytes([r, g, b]) * w
    compressed = zlib.compress(raw)

    def chunk(ctype: bytes, cdata: bytes) -> bytes:
        c = ctype + cdata
        return (
            struct.pack(">I", len(cdata))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _cohere_embed(
    server: RemoteOpenAIServer,
    texts: list[str] | None = None,
    images: list[str] | None = None,
    embedding_types: list[str] | None = None,
) -> dict:
    body: dict = {"model": MODEL_NAME}
    if texts is not None:
        body["texts"] = texts
    if images is not None:
        body["images"] = images
    if embedding_types is not None:
        body["embedding_types"] = embedding_types
    resp = requests.post(server.url_for("/v2/embed"), json=body)
    resp.raise_for_status()
    return resp.json()


def test_image_embed(server: RemoteOpenAIServer):
    img_uri = _make_tiny_png(255, 0, 0)
    r = _cohere_embed(
        server,
        images=[img_uri],
        embedding_types=["float"],
    )
    assert "embeddings" in r
    assert len(r["embeddings"]["float"]) == 1
    assert len(r["embeddings"]["float"][0]) > 0
    assert r["meta"]["billed_units"]["image_tokens"] > 0
    assert r["meta"]["billed_units"]["input_tokens"] == 0


def test_image_batch(server: RemoteOpenAIServer):
    red = _make_tiny_png(255, 0, 0)
    blue = _make_tiny_png(0, 0, 255)
    r = _cohere_embed(
        server,
        images=[red, blue],
        embedding_types=["float"],
    )
    assert len(r["embeddings"]["float"]) == 2


def test_image_l2_normalized(server: RemoteOpenAIServer):
    img_uri = _make_tiny_png(0, 255, 0)
    r = _cohere_embed(
        server,
        images=[img_uri],
        embedding_types=["float"],
    )
    emb = np.array(r["embeddings"]["float"][0])
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 0.01


def test_image_embedding_types(server: RemoteOpenAIServer):
    img_uri = _make_tiny_png(128, 128, 128)
    r = _cohere_embed(
        server,
        images=[img_uri],
        embedding_types=["float", "binary", "ubinary"],
    )
    dim = len(r["embeddings"]["float"][0])
    assert len(r["embeddings"]["binary"][0]) == dim // 8
    assert len(r["embeddings"]["ubinary"][0]) == dim // 8


def test_text_embed_on_multimodal(server: RemoteOpenAIServer):
    """SigLIP also supports text-only embedding via /v2/embed."""
    r = _cohere_embed(server, texts=["hello world"], embedding_types=["float"])
    assert "embeddings" in r
    assert len(r["embeddings"]["float"]) == 1
    assert len(r["embeddings"]["float"][0]) > 0
