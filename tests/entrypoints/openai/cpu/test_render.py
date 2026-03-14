# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /render endpoints that expose prompt preprocessing."""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    args: list[str] = []

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

    # Should return two GenerateRequest items
    assert isinstance(data, list)
    assert len(data) == 2

    # Verify both prompts have GenerateRequest fields
    for prompt in data:
        assert "token_ids" in prompt
        assert "sampling_params" in prompt
        assert "model" in prompt
        assert "request_id" in prompt
        assert len(prompt["token_ids"]) > 0
        assert prompt["request_id"].startswith("cmpl-")


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
    assert response.headers.get("content-type", "").startswith("application/json")

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
async def test_chat_completion_render_with_sampling_params(client):
    """Verify sampling params are correctly returned by /render."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Test sampling params"}],
            "temperature": 0.123,
            "top_p": 0.456,
            "frequency_penalty": 1.1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "sampling_params" in data
    sampling_params = data["sampling_params"]

    assert sampling_params.get("temperature") == 0.123
    assert sampling_params.get("top_p") == 0.456
    assert sampling_params.get("frequency_penalty") == 1.1

    # Check that internal fields are not present
    assert "_all_stop_token_ids" not in sampling_params
