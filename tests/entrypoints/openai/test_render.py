# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /render endpoints that expose prompt preprocessing."""

import json

import httpx
import pytest
import pytest_asyncio

from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform

from ...utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def server():
    args: list[str] = []

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def vision_server():
    """Vision-capable server used for multimodal /render tests."""

    args = [
        "--runner",
        "generate",
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "2",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
    ]

    # ROCm: Increase timeouts to handle potential network delays and slower
    # image processing.
    env_overrides: dict[str, str] = {}
    if current_platform.is_rocm():
        env_overrides = {
            "VLLM_VIDEO_FETCH_TIMEOUT": "120",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "300",
        }

    with RemoteOpenAIServer(
        VISION_MODEL_NAME, args, env_dict=env_overrides
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def vision_client(vision_server):
    async with httpx.AsyncClient(
        base_url=vision_server.url_for(""), timeout=60.0
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

    # Verify response structure
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify first prompt
    first_prompt = data[0]
    assert "prompt_token_ids" in first_prompt
    assert "prompt" in first_prompt
    assert isinstance(first_prompt["prompt_token_ids"], list)
    assert len(first_prompt["prompt_token_ids"]) > 0
    assert isinstance(first_prompt["prompt"], str)

    # Verify prompt text is preserved
    assert (
        "When should a chat-completions handler return an empty string?"
        in first_prompt["prompt"]
    )


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

    # Verify token IDs are integers and BOS token is present
    token_ids = data["token_ids"]
    assert all(isinstance(tid, int) for tid in token_ids)
    assert token_ids[0] == 1


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts(client):
    """Test completion render with multiple prompts."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": ["Hello world", "Goodbye world"],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should return two prompts
    assert isinstance(data, list)
    assert len(data) == 2

    # Verify both prompts have required fields
    for prompt in data:
        assert "prompt_token_ids" in prompt
        assert "prompt" in prompt
        assert len(prompt["prompt_token_ids"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_render_multi_turn(client):
    """Test chat completion render with multi-turn conversation."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify tokenization occurred
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_render_with_stream_true(client):
    """Render accepts stream params but still returns JSON (non-streamed)."""

    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Stream options should be accepted by /render.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith(
        "application/json"
    )

    data = response.json()
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0

    # /render should preserve stream fields on the returned token-in request.
    assert data.get("stream") is True
    assert isinstance(data.get("stream_options"), dict)
    assert data["stream_options"].get("include_usage") is True
    assert data["stream_options"].get("continuous_usage_stats") is True


@pytest.mark.asyncio
async def test_completion_render_error_invalid_model(client):
    """Test completion render with invalid model returns error."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": "invalid-model-name",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_chat_completion_render_error_invalid_model(client):
    """Test chat completion render with invalid model returns error."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": "invalid-model-name",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_completion_render_no_generation(client):
    """Verify render endpoint does not generate text."""
    # This test verifies that calling render is fast (no generation)
    import time

    start = time.perf_counter()
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Tell me a very long story about " * 10,
        },
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    # Render should be fast (< 1 second) since no generation
    assert elapsed < 1.0

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

    # Expect successful render response
    assert response.status_code == 200

    data = response.json()

    # Validate token_ids field exists
    assert "token_ids" in data, "Response must contain 'token_ids' field"
    assert isinstance(data["token_ids"], list), "'token_ids' must be a list"

    # Ensure non-empty token output
    assert len(data["token_ids"]) > 0, "Token list must not be empty"
