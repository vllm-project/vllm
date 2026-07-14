# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multimodal tests for the /render endpoints that expose prompt preprocessing."""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.utils import encode_image_url

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

    with RemoteOpenAIServer(
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

    # Verify multimodal features are populated
    assert "features" in data
    features = data["features"]
    assert features is not None

    # mm_hashes: should have an "image" key with a list of hash strings
    assert "mm_hashes" in features
    assert "image" in features["mm_hashes"]
    image_hashes = features["mm_hashes"]["image"]
    assert isinstance(image_hashes, list)
    assert len(image_hashes) > 0
    assert all(isinstance(h, str) for h in image_hashes)

    # mm_placeholders: should have an "image" key with offset/length dicts
    assert "mm_placeholders" in features
    assert "image" in features["mm_placeholders"]
    image_placeholders = features["mm_placeholders"]["image"]
    assert isinstance(image_placeholders, list)
    assert len(image_placeholders) > 0
    for p in image_placeholders:
        assert "offset" in p
        assert "length" in p
        assert isinstance(p["offset"], int)
        assert isinstance(p["length"], int)
        assert p["length"] > 0


@pytest.mark.asyncio
async def test_tokenize_matches_render_for_multimodal_input(
    vision_client,
    local_asset_server,
):
    """`/tokenize` should match `/v1/chat/completions/render` token output."""

    image = local_asset_server.get_image_asset("RGBA_comp.png")
    data_url = encode_image_url(image, format="PNG")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]

    render_response = await vision_client.post(
        "/v1/chat/completions/render",
        json={
            "model": VISION_MODEL_NAME,
            "messages": messages,
        },
    )
    assert render_response.status_code == 200
    render_data = render_response.json()

    tokenize_response = await vision_client.post(
        "/tokenize",
        json={
            "model": VISION_MODEL_NAME,
            "messages": messages,
        },
    )
    assert tokenize_response.status_code == 200
    tokenize_data = tokenize_response.json()

    assert tokenize_data["tokens"] == render_data["token_ids"]
    assert tokenize_data["count"] == len(render_data["token_ids"])
