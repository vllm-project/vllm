# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /render endpoints that expose prompt preprocessing."""

import httpx
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


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

    # Verify response structure - should be [conversation, engine_prompts]
    assert isinstance(data, list)
    assert len(data) == 2

    conversation, engine_prompts = data

    # Verify conversation
    assert isinstance(conversation, list)
    assert len(conversation) > 0
    assert conversation[0]["role"] == "user"
    assert "empty string" in conversation[0]["content"]

    # Verify engine_prompts
    assert isinstance(engine_prompts, list)
    assert len(engine_prompts) > 0

    first_prompt = engine_prompts[0]
    assert "prompt_token_ids" in first_prompt
    assert "prompt" in first_prompt
    assert isinstance(first_prompt["prompt_token_ids"], list)
    assert len(first_prompt["prompt_token_ids"]) > 0

    # Verify chat template was applied (should have instruction markers)
    assert "[INST]" in first_prompt["prompt"]
    assert "[/INST]" in first_prompt["prompt"]

    # Verify token IDs are correctly preserved as integers
    token_ids = first_prompt["prompt_token_ids"]
    assert all(isinstance(tid, int) for tid in token_ids)
    # Verify BOS token (usually 1 for LLaMA models)
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

    conversation, engine_prompts = data

    # Verify all messages preserved
    assert len(conversation) == 3
    assert conversation[0]["role"] == "user"
    assert conversation[1]["role"] == "assistant"
    assert conversation[2]["role"] == "user"

    # Verify tokenization occurred
    assert len(engine_prompts) > 0
    assert len(engine_prompts[0]["prompt_token_ids"]) > 0


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
