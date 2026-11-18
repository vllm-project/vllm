# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for render endpoints via `vllm launch` (GPU-less serving)."""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    args: list[str] = []
    with RemoteLaunchRenderServer(MODEL_NAME, args, max_wait_seconds=120) as srv:
        yield srv


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


# -- Chat Completion Render --


@pytest.mark.asyncio
async def test_chat_render_basic(client):
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Response should be a GenerateRequest dict
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0
    assert all(isinstance(t, int) for t in data["token_ids"])


@pytest.mark.asyncio
async def test_chat_render_multi_turn(client):
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

    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0


@pytest.mark.asyncio
async def test_chat_render_multiple_chat_templates():
    # Ships chat_template.jinja and additional_chat_templates/reranker.jinja
    model = "Qwen/Qwen3-VL-Reranker-2B"
    with RemoteLaunchRenderServer(model, [], max_wait_seconds=120) as srv:
        async with httpx.AsyncClient(
            base_url=srv.url_for(""), timeout=30.0
        ) as http_client:
            response = await http_client.post(
                "/v1/chat/completions/render",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    token_ids = response.json()["token_ids"]
    assert token_ids == [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]


# -- Completion Render --


@pytest.mark.asyncio
async def test_completion_render_basic(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Once upon a time",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0

    first_prompt = data[0]
    assert "token_ids" in first_prompt
    assert "sampling_params" in first_prompt
    assert "model" in first_prompt
    assert "request_id" in first_prompt
    assert isinstance(first_prompt["token_ids"], list)
    assert len(first_prompt["token_ids"]) > 0
    assert first_prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": ["Hello world", "Goodbye world"],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    for prompt in data:
        assert "token_ids" in prompt
        assert "sampling_params" in prompt
        assert "model" in prompt
        assert "request_id" in prompt
        assert len(prompt["token_ids"]) > 0
        assert prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,payload",
    [
        pytest.param(
            "/v1/chat/completions/render",
            {"messages": [{"role": "user", "content": "Hello"}]},
            id="chat",
        ),
        pytest.param("/v1/completions/render", {"prompt": "Hello"}, id="completion"),
        pytest.param("/v1/responses/render", {"input": "Hello"}, id="responses"),
        pytest.param(
            "/v1/messages/render",
            {
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            id="messages",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_name,display_name",
    [
        pytest.param("nonexistent-model", "nonexistent-model", id="short"),
        pytest.param("x" * 256, "x" * 256, id="boundary"),
        pytest.param("x" * 257, "x" * 256 + "...", id="truncated"),
        pytest.param("x" * 16384, "x" * 256 + "...", id="oversized"),
        pytest.param("\u00e9" * 257, "\u00e9" * 256 + "...", id="unicode"),
    ],
)
async def test_render_invalid_model_error_is_bounded(
    client, endpoint, payload, model_name, display_name
):
    response = await client.post(
        endpoint,
        json={"model": model_name, **payload},
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "message": (
            f"The model `{display_name}` does not exist. Valid aliases: {MODEL_NAME}."
        ),
        "type": "NotFoundError",
        "param": "model",
        "code": 404,
    }


@pytest.mark.asyncio
async def test_render_is_fast(client):
    """Render should complete quickly since there is no inference."""
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
    assert elapsed < 2.0


# -- Health & Models --


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_endpoint(client):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert MODEL_NAME in model_ids
