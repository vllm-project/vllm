# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for render endpoints via `vllm launch` (GPU-less serving)."""

import httpx
import numpy as np
import pybase64 as base64
import pytest
import pytest_asyncio
from PIL import Image

from tests.multimodal.utils import create_video_from_image
from tests.utils import RemoteLaunchRenderServer
from vllm.multimodal.utils import encode_video_url

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"


@pytest.mark.parametrize(
    ("model", "budget", "cache_gb", "device_normalize"),
    [
        ("Qwen/Qwen3-VL-Reranker-2B", 512 * 1024, 0, True),
        ("Qwen/Qwen3-VL-Reranker-2B", 2 * 1024**2, 0, False),
        ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 3 * 1024**2, 1, True),
        ("google/gemma-4-E2B-it", 6 * 1024**2, 0, True),
    ],
)
def test_chat_render_rejects_over_budget_video_tensors(
    model: str, budget: int, cache_gb: int, device_normalize: bool
):
    """Each clip fits alone, but the combined preprocessing tensors do not."""
    args = [
        "--host",
        "127.0.0.1",
        "--dtype",
        "float32",
        "--max-model-len",
        "4096",
        "--limit-mm-per-prompt",
        '{"image": 0, "video": 2}',
        "--mm-processor-cache-gb",
        str(cache_gb),
        (
            "--mm-device-do-normalize"
            if device_normalize
            else "--no-mm-device-do-normalize"
        ),
    ]
    videos = [
        {"type": "video_url", "video_url": {"url": encode_video_url(frames)}}
        for frames in (
            np.zeros((4, 128, 128, 3), dtype=np.uint8),
            np.full((4, 128, 128, 3), 128, dtype=np.uint8),
        )
    ]
    with RemoteLaunchRenderServer(
        model,
        args,
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": str(budget)},
    ) as server:
        url = server.url_for("v1/chat/completions/render")
        response = httpx.post(
            url,
            json={"model": model, "messages": [{"role": "user", "content": videos}]},
            timeout=30,
        )
        assert response.status_code == 400, response.text[:1000]
        assert (
            "VLLM_MAX_MEDIA_PREPROCESSING_BYTES" in response.json()["error"]["message"]
        )

        for video in videos:
            response = httpx.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": [video]}],
                },
                timeout=30,
            )
            assert response.status_code == 200, response.text[:1000]

        if cache_gb:
            response = httpx.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": videos}],
                },
                timeout=30,
            )
            assert response.status_code == 200, response.text[:1000]


def test_chat_render_llava_video_budget_uses_effective_options():
    model = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    # Two float32 frames fit without resizing, but exceed the budget at 384x384.
    with RemoteLaunchRenderServer(
        model,
        [
            "--host",
            "127.0.0.1",
            "--dtype",
            "float32",
            "--mm-processor-cache-gb",
            "0",
            "--no-mm-device-do-normalize",
            "--trust-request-mm-kwargs",
        ],
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": str(64 * 1024)},
    ) as server:
        url = server.url_for("v1/chat/completions/render")
        for frame_count, videos_kwargs, status in (
            (2, {"do_resize": False}, 200),
            (2, {"size": {"shortest_edge": 24}}, 200),
            (2, {}, 400),
            (2, {"size": {"height": "invalid", "width": 24}}, 400),
            (6, {"do_resize": False, "do_sample_frames": True, "num_frames": 2}, 200),
        ):
            video_url = encode_video_url(
                np.zeros((frame_count, 32, 48, 3), dtype=np.uint8)
            )
            response = httpx.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video_url", "video_url": {"url": video_url}}
                            ],
                        }
                    ],
                    "mm_processor_kwargs": {"videos_kwargs": videos_kwargs},
                },
                timeout=30,
            )
            assert response.status_code == status, response.text[:1000]
            if not videos_kwargs:
                error = response.json()["error"]["message"]
                assert "VLLM_MAX_MEDIA_PREPROCESSING_BYTES" in error


def test_chat_render_gemma4_video_budget_uses_image_processor_options():
    model = "google/gemma-4-E2B-it"
    video_url = encode_video_url(np.zeros((2, 96, 96, 3), dtype=np.uint8))
    # With device normalization both frames are uint8. Float32 frames would
    # exceed the same budget because the output is concatenated.
    with RemoteLaunchRenderServer(
        model,
        [
            "--host",
            "127.0.0.1",
            "--dtype",
            "float32",
            "--max-model-len",
            "4096",
            "--mm-processor-cache-gb",
            "0",
            "--mm-device-do-normalize",
            "--trust-request-mm-kwargs",
        ],
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": str(2 * 1024**2)},
    ) as server:
        response = httpx.post(
            server.url_for("v1/chat/completions/render"),
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_url}}
                        ],
                    }
                ],
                "mm_processor_kwargs": {"videos_kwargs": {"do_normalize": True}},
            },
            timeout=30,
        )

    assert response.status_code == 200, response.text[:1000]


def test_chat_render_video_options_keep_default_count_limit():
    model = "Qwen/Qwen3-VL-Reranker-2B"
    video_url = encode_video_url(np.zeros((4, 16, 16, 3), dtype=np.uint8))
    content = [{"type": "video_url", "video_url": {"url": video_url}}] * 33
    with RemoteLaunchRenderServer(
        model,
        ["--limit-mm-per-prompt", '{"video": {"num_frames": 4}}'],
    ) as server:
        response = httpx.post(
            server.url_for("v1/chat/completions/render"),
            json={"model": model, "messages": [{"role": "user", "content": content}]},
            timeout=30,
        )

    assert response.status_code == 400, response.text[:1000]
    assert "At most 32 video" in response.json()["error"]["message"]


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


def test_dots3_note_render_rejects_video_exceeding_request_decode_budget(tmp_path):
    model = "dots-studio/dots3-note-prev"
    image_path = tmp_path / "frame.png"
    video_path = tmp_path / "video.mp4"
    Image.new("RGB", (64, 64), color="red").save(image_path)
    create_video_from_image(str(image_path), str(video_path), num_frames=8)
    video_url = "data:video/mp4;base64," + base64.b64encode(
        video_path.read_bytes()
    ).decode("ascii")
    frame_bytes = 64 * 64 * 3

    with RemoteLaunchRenderServer(
        model,
        ["--mm-processor-cache-gb", "0", "--max-model-len", "4096"],
        env_dict={"VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES": str(9 * frame_bytes)},
    ) as server:
        response = httpx.post(
            server.url_for("v1/chat/completions/render"),
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_url}}
                        ],
                    }
                ],
                "mm_payload": "none",
            },
            timeout=120,
        )
        health = httpx.get(server.url_for("health"))

    assert response.status_code == 400
    assert "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES" in response.text
    assert health.status_code == 200


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
