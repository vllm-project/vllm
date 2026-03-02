# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for render endpoints via `vllm online` (GPU-less serving)."""

import os
import subprocess
import sys

import httpx
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


class RemoteOnlineServer(RemoteOpenAIServer):
    """Launches `vllm online` subprocess instead of `vllm serve`."""

    def _start_server(
        self,
        model: str,
        vllm_serve_args: list[str],
        env_dict: dict[str, str] | None,
    ) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict is not None:
            env.update(env_dict)
        serve_cmd = ["vllm", "online", model, *vllm_serve_args]
        print(f"Launching RemoteOnlineServer with: {' '.join(serve_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(
            serve_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def _wait_for_gpu_memory_release(self, timeout: float = 30.0):
        pass  # No GPU used


@pytest.fixture(scope="module")
def server():
    args: list[str] = []
    with RemoteOnlineServer(MODEL_NAME, args, max_wait_seconds=120) as srv:
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

    assert isinstance(data, list)
    assert len(data) == 2

    conversation, engine_prompts = data

    assert isinstance(conversation, list)
    assert conversation[0]["role"] == "user"

    assert isinstance(engine_prompts, list)
    assert len(engine_prompts) > 0
    first_prompt = engine_prompts[0]
    assert "prompt_token_ids" in first_prompt
    assert "prompt" in first_prompt
    assert isinstance(first_prompt["prompt_token_ids"], list)
    assert all(isinstance(t, int) for t in first_prompt["prompt_token_ids"])


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
    conversation, engine_prompts = response.json()

    assert len(conversation) == 3
    assert conversation[0]["role"] == "user"
    assert conversation[1]["role"] == "assistant"
    assert conversation[2]["role"] == "user"
    assert len(engine_prompts) > 0
    assert len(engine_prompts[0]["prompt_token_ids"]) > 0


@pytest.mark.asyncio
async def test_chat_render_invalid_model(client):
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    assert "error" in response.json()


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
    assert "prompt_token_ids" in first_prompt
    assert "prompt" in first_prompt
    assert isinstance(first_prompt["prompt_token_ids"], list)
    assert len(first_prompt["prompt_token_ids"]) > 0
    assert "Once upon a time" in first_prompt["prompt"]


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
        assert "prompt_token_ids" in prompt
        assert "prompt" in prompt
        assert len(prompt["prompt_token_ids"]) > 0


@pytest.mark.asyncio
async def test_completion_render_invalid_model(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": "nonexistent-model",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 404
    assert "error" in response.json()


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
