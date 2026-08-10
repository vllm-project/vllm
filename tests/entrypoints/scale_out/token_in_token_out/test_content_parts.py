# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for content_parts on /inference/v1/generate.

Exercises raw multimodal input via content_parts, bypassing the /render
step entirely. The generate server resolves media internally.
"""

import os

import httpx
import pytest
import pytest_asyncio
from PIL import Image

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
GEN_ENDPOINT = "/inference/v1/generate"
RENDER_ENDPOINT = "/v1/chat/completions/render"
DETOKENIZE_ENDPOINT = "/detokenize"


@pytest.fixture(scope="module")
def test_image():
    return Image.new("RGB", (224, 224), color=(255, 0, 0))


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]

    envs = os.environ.copy()
    envs["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    async with httpx.AsyncClient(
        transport=transport,
        base_url=server.url_root,
        timeout=600,
        headers=headers,
    ) as c:
        yield c


def _get_token_ids_from_render(render_data: dict) -> list[int]:
    """Extract token_ids from render response, stripping features."""
    return render_data["token_ids"]


@pytest.mark.asyncio
async def test_content_parts_matches_render_generate(client, test_image):
    """content_parts should produce the same output as render → generate.

    1. Render a multimodal chat prompt to get token_ids
    2. Run generate with content_parts (raw media)
    3. Run generate with features (pre-processed from render)
    4. Both should produce identical token_ids
    """
    data_url = encode_image_url(test_image, format="PNG")

    render_resp = await client.post(
        RENDER_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "url": data_url},
                        {"type": "text", "text": "What color is this image? One word."},
                    ],
                }
            ],
        },
    )
    render_resp.raise_for_status()
    render_data = render_resp.json()
    token_ids = _get_token_ids_from_render(render_data)

    sampling_params = {"max_tokens": 10, "temperature": 0.0}

    # Path A: content_parts (raw media, server resolves)
    cp_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": token_ids,
            "content_parts": [
                {"type": "image_url", "url": data_url},
            ],
            "sampling_params": sampling_params,
        },
    )
    cp_resp.raise_for_status()
    cp_data = cp_resp.json()

    # Path B: features (pre-processed from render)
    feat_resp = await client.post(
        GEN_ENDPOINT,
        json={
            **render_data,
            "sampling_params": sampling_params,
        },
    )
    feat_resp.raise_for_status()
    feat_data = feat_resp.json()

    cp_tokens = cp_data["choices"][0]["token_ids"]
    feat_tokens = feat_data["choices"][0]["token_ids"]
    assert len(cp_tokens) > 0
    assert cp_tokens == feat_tokens, (
        f"content_parts and features paths diverged: {cp_tokens} vs {feat_tokens}"
    )


@pytest.mark.asyncio
async def test_content_parts_produces_correct_output(client, test_image):
    """content_parts with a red image should generate text containing 'red'."""
    data_url = encode_image_url(test_image, format="PNG")

    render_resp = await client.post(
        RENDER_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "url": data_url},
                        {
                            "type": "text",
                            "text": "What color is this image? Answer in one word.",
                        },
                    ],
                }
            ],
        },
    )
    render_resp.raise_for_status()
    token_ids = render_resp.json()["token_ids"]

    gen_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": token_ids,
            "content_parts": [
                {"type": "image_url", "url": data_url},
            ],
            "sampling_params": {"max_tokens": 10, "temperature": 0.0},
        },
    )
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()

    output_tokens = gen_data["choices"][0]["token_ids"]
    assert len(output_tokens) > 0

    detok_resp = await client.post(
        DETOKENIZE_ENDPOINT,
        json={"model": MODEL_NAME, "tokens": output_tokens},
    )
    detok_resp.raise_for_status()
    text = detok_resp.json()["prompt"]
    assert "red" in text.lower(), (
        f"Expected model to identify the red image, got: {text!r}"
    )


@pytest.mark.asyncio
async def test_content_parts_streaming(client, test_image):
    """content_parts should work with streaming responses."""
    import json

    data_url = encode_image_url(test_image, format="PNG")

    render_resp = await client.post(
        RENDER_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "url": data_url},
                        {"type": "text", "text": "Describe this image briefly."},
                    ],
                }
            ],
        },
    )
    render_resp.raise_for_status()
    token_ids = render_resp.json()["token_ids"]

    async with client.stream(
        "POST",
        GEN_ENDPOINT,
        json={
            "token_ids": token_ids,
            "content_parts": [
                {"type": "image_url", "url": data_url},
            ],
            "sampling_params": {"max_tokens": 16, "temperature": 0.0},
            "stream": True,
        },
    ) as resp:
        resp.raise_for_status()
        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))

    assert len(chunks) > 0, "Expected at least one streaming chunk"
    all_tokens = []
    for chunk in chunks:
        if chunk.get("choices"):
            tids = chunk["choices"][0].get("token_ids", [])
            all_tokens.extend(tids)
    assert len(all_tokens) > 0
