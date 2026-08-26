# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multimodal tests for the /render endpoints that expose prompt preprocessing."""

import httpx
import pybase64
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


@pytest.mark.asyncio
async def test_skip_pixel_values_returns_source_bytes_not_tensors(
    vision_client,
    local_asset_server,
):
    """``skip_pixel_values`` must swap the payload carrier without changing
    anything the downstream engine keys off: tokens, hashes and placeholders
    stay identical, only the tensors are replaced by the source bytes."""

    image = local_asset_server.get_image_asset("RGBA_comp.png")
    data_url = encode_image_url(image, format="PNG")
    source_bytes = pybase64.b64decode(data_url.split(",", 1)[1], validate=True)

    def payload(skip: bool) -> dict:
        return {
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
            "skip_pixel_values": skip,
        }

    full_resp = await vision_client.post(
        "/v1/chat/completions/render", json=payload(False)
    )
    skipped_resp = await vision_client.post(
        "/v1/chat/completions/render", json=payload(True)
    )
    full = full_resp.json()
    skipped = skipped_resp.json()

    assert skipped["token_ids"] == full["token_ids"]
    assert skipped["features"]["mm_hashes"] == full["features"]["mm_hashes"]
    assert skipped["features"]["mm_placeholders"] == full["features"]["mm_placeholders"]

    assert full["features"]["kwargs_data"] is not None
    assert skipped["features"]["kwargs_data"] is None

    raw_images = skipped["features"]["raw_images"]
    assert list(raw_images) == ["image"]
    assert len(raw_images["image"]) == len(skipped["features"]["mm_hashes"]["image"])
    # The exact bytes the renderer loaded, so the /generate side recomputes
    # the same hash when it reloads them.
    assert pybase64.b64decode(raw_images["image"][0], validate=True) == source_bytes

    # The point of the flag: source bytes are far cheaper than pixel_values.
    assert len(skipped_resp.content) < len(full_resp.content)
