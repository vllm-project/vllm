# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for render endpoints via `vllm launch` (GPU-less serving)."""

import httpx
import pybase64 as base64
import pytest
import pytest_asyncio

from tests.multimodal.utils import create_long_gop_video
from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
QWEN2_VL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


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
def qwen2vl_server():
    args = [
        "--max-model-len",
        "2048",
        "--limit-mm-per-prompt",
        '{"video": 1}',
        "--media-io-kwargs",
        '{"video": {"backend": "opencv"}}',
        "--trust-request-mm-kwargs",
    ]
    with RemoteLaunchRenderServer(QWEN2_VL_MODEL_NAME, args) as srv:
        yield srv


@pytest.mark.asyncio
@pytest.mark.parametrize("sampling_kwargs", [{}, {"num_frames": 4}])
async def test_qwen2vl_single_frame_video_is_bad_request(
    qwen2vl_server, sampling_kwargs
):
    """Reject an empty sample over HTTP and still preprocess a valid clip."""
    async with httpx.AsyncClient(
        base_url=qwen2vl_server.url_for(""), timeout=30.0
    ) as client:
        for frame_count in (1, 2):
            video = create_long_gop_video(num_frames=frame_count, width=32, height=32)
            url = "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")
            response = await client.post(
                "/v1/chat/completions/render",
                json={
                    "model": QWEN2_VL_MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this video."},
                                {"type": "video_url", "video_url": {"url": url}},
                            ],
                        }
                    ],
                    "media_io_kwargs": {"video": sampling_kwargs},
                },
            )
            if frame_count == 1:
                assert response.status_code == 400, response.text
                error = response.json()["error"]
                assert error["param"] == "video"
                assert "no frames after temporal alignment" in error["message"]
            else:
                assert response.status_code == 200, response.text
                assert response.json()["token_ids"]


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
