# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def server():
    args: list[str] = ["--trust-request-chat-template",         "--api-server-count", "2"]

    with RemoteLaunchRenderServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_completion_render_basic(client):
    """Test basic completion render endpoint."""
    # Make request to render endpoint
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "When should a chat-completions handler return an empty string?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure - list of GenerateRequest
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify first prompt is a GenerateRequest
    first_prompt = data[0]
    assert "token_ids" in first_prompt
    assert "sampling_params" in first_prompt
    assert "model" in first_prompt
    assert "request_id" in first_prompt
    assert isinstance(first_prompt["token_ids"], list)
    assert len(first_prompt["token_ids"]) > 0
    assert first_prompt["model"] == MODEL_NAME
    assert first_prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_chat_completion_render_basic(client):
    """Test basic chat completion render endpoint."""
    # Make request to render endpoint
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Returning an empty string for the prompt may be confusing."
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure - should be a GenerateRequest
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0



@pytest.mark.asyncio
async def test_chat_completion_render_with_base64_image_url(
    client,
    local_asset_server,
):
    """Render a multimodal chat request and verify tokens are returned."""
    image = local_asset_server.get_image_asset("RGBA_comp.png")
    data_url = encode_image_url(image, format="PNG")

    assert data_url.startswith("data:image/")
    assert ";base64," in data_url

    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
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
