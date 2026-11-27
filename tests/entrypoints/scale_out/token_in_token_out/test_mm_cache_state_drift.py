# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end regression for multimodal processor cache state drift."""

from typing import Any

import httpx
import pytest
import pytest_asyncio
from PIL import Image

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
GEN_ENDPOINT = "/inference/v1/generate"
RENDER_ENDPOINT = "/v1/chat/completions/render"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--enable-scale-out",
        "--mm-processor-cache-gb",
        "0.005",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
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


@pytest.mark.asyncio
async def test_prerendered_features_do_not_evict_processor_cache(client):
    """Pre-rendered features must preserve an existing image cache entry."""
    original_url = encode_image_url(
        Image.new("RGB", (224, 224), color=(255, 0, 0)), format="PNG"
    )

    def image_messages(urls: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    *(
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in urls
                    ),
                    {"type": "text", "text": "Describe these images."},
                ],
            }
        ]

    original_messages = image_messages([original_url])

    # Capture the original hash and placeholder range for a cache-only lookup
    # after the pre-rendered request has been processed.
    original_render_resp = await client.post(
        RENDER_ENDPOINT,
        json={"model": MODEL_NAME, "messages": original_messages},
    )
    original_render_resp.raise_for_status()
    original_rendered = original_render_resp.json()

    # Warm the generate server's LRU through the regular multimodal endpoint.
    chat_resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": original_messages,
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )
    chat_resp.raise_for_status()

    prerendered_urls = [
        encode_image_url(Image.new("RGB", (512, 512), color=color), format="PNG")
        for color in [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    ]
    render_resp = await client.post(
        RENDER_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "messages": image_messages(prerendered_urls),
        },
    )
    render_resp.raise_for_status()
    rendered = render_resp.json()
    # With the small LRU, inserting these full tensors would evict the warmed
    # original image if InputProcessor drops the skip_mm_cache flag.
    prerendered_resp = await client.post(
        GEN_ENDPOINT,
        json={
            **rendered,
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
        },
    )
    prerendered_resp.raise_for_status()

    # A hash-only request requires resolving the original feature from the LRU.
    cache_only_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": original_rendered["token_ids"],
            "features": {
                "mm_hashes": original_rendered["features"]["mm_hashes"],
                "mm_placeholders": original_rendered["features"][
                    "mm_placeholders"
                ],
            },
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
        },
    )
    cache_only_resp.raise_for_status()
    cache_only = cache_only_resp.json()
    assert cache_only["choices"]
    assert cache_only["choices"][0]["token_ids"]
