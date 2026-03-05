# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multimodal tests for the /render endpoints that expose prompt preprocessing."""

import httpx
import pytest
import pytest_asyncio

from vllm.multimodal.utils import encode_image_url

from tests.utils import RemoteLaunchRenderServer

VISION_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def vision_server():
    """Vision-capable server used for multimodal /render tests."""

    args = [
        "--enforce-eager",
        "--max-model-len",
        "100",
        "--max-num-seqs",
        "1",
        "--limit-mm-per-prompt.image",
        "1",
        "--limit-mm-per-prompt.video",
        "0",
    ]

    env_overrides: dict[str, str] = {}

    with RemoteLaunchRenderServer(
        VISION_MODEL_NAME,
        args,
        env_dict=env_overrides,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def vision_client(vision_server):
    async with httpx.AsyncClient(
        base_url=vision_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_chat_completion_render_with_base64_image_url(
    vision_client,
    local_asset_server,
):
    """Render a multimodal chat request and verify tokens are returned."""

    image = local_asset_server.get_image_asset("RGBA_comp.png")
    data_url = encode_image_url(image, format="PNG")

    assert data_url.startswith("data:image/")
    assert ";base64," in data_url

    response = await vision_client.post(
        "/v1/chat/completions/render",
        json={
            "model": VISION_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "What's in this image?"},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0
