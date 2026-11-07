# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for render endpoints via `vllm launch` (GPU-less serving)."""

import httpx
import pytest
import pytest_asyncio
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, TokenizersBackend

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


@pytest.fixture(scope="module")
def deepseek_v41_server(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("deepseek_v41")
    LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=3,
        max_position_embeddings=128,
    ).save_pretrained(model_path)
    TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    ).save_pretrained(model_path)
    with RemoteLaunchRenderServer(
        str(model_path),
        ["--tokenizer-mode", "deepseek_v41", "--served-model-name", "deepseek-v41"],
        max_wait_seconds=120,
    ) as srv:
        yield srv


@pytest.mark.parametrize(
    "message", [{"role": "developer", "content": ""}, {"role": "developer"}]
)
def test_empty_developer_content_returns_400(deepseek_v41_server, message):
    response = httpx.post(
        deepseek_v41_server.url_for("v1/chat/completions/render"),
        json={
            "model": "deepseek-v41",
            "messages": [message],
            "chat_template_kwargs": {"thinking": False},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["message"] == (
        "Developer message content must not be empty."
    )
    assert "_image_indices" not in response.text
    assert str(message) not in response.text


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
