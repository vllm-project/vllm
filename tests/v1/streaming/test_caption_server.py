# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for streaming admission, image limits, and HTTP validation.

The model-serving path is exercised by the real-server integration test in
tests/entrypoints/multimodal/test_streaming_sessions.py.
"""

import io
from argparse import Namespace

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.streaming.api_router import (
    _MAX_FRAME_BYTES,
    attach_router,
)
from vllm.entrypoints.openai.streaming.chunking import (
    decode_frame,
    render_incremental_turn,
)
from vllm.entrypoints.openai.streaming.config import configure_streaming
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.renderers import renderer_from_config
from vllm.renderers.params import ChatParams

pytestmark = pytest.mark.cpu_test


def _png(width: int = 20, height: int = 12) -> bytes:
    with Image.new("RGB", (width, height), "red") as image:
        encoded = io.BytesIO()
        image.save(encoded, "PNG")
        return encoded.getvalue()


def _app() -> FastAPI:
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False)
    init_exception_handler(app)
    attach_router(app)
    return app


def test_decoded_image_limit_is_checked_before_loading_pixels():
    # A valid PNG header with truncated pixels raises OSError if decoding starts.
    with pytest.raises(ValueError, match="per-frame limit"):
        decode_frame(_png(2049, 2049)[:64])


_MIXED_CONTENT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\n"
    "{{ server_preamble }}<|im_end|>\n"
    "{% endif %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['role'] == 'user' and message['content'] is not string %}"
    "{% for part in message['content'] %}{{ part['text'] }}{% endfor %}"
    "{% else %}{{ message['content'] }}{% endif %}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{{ server_prefill }}{% endif %}"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "generation_prefill"),
    [
        pytest.param(
            ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
            "",
            id="default",
        ),
        pytest.param(
            ChatParams(
                chat_template_kwargs={
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                }
            ),
            "<think>\n\n</think>\n\n",
            id="generation-prefill",
        ),
        pytest.param(
            ChatParams(
                chat_template=_MIXED_CONTENT_TEMPLATE,
                chat_template_content_format="openai",
                chat_template_kwargs={
                    "add_generation_prompt": True,
                    "server_preamble": "SERVER-SYSTEM",
                    "server_prefill": "SERVER-PREFILL",
                },
            ),
            "SERVER-PREFILL",
            id="mixed-content",
        ),
    ],
)
async def test_incremental_transcript_contract(params, generation_prefill):
    config = VllmConfig(
        model_config=ModelConfig(model="Qwen/Qwen3-0.6B", max_model_len=512)
    )
    renderer = renderer_from_config(config)
    try:
        tokenizer = renderer.get_tokenizer()
        replay = tokenizer.encode("A", add_special_tokens=False)
        assert len(replay) == 1
        chunk = await render_incremental_turn(
            renderer,
            [{"role": "user", "content": "Output B."}],
            params,
            replay_token=replay[0],
        )
        assert tokenizer.decode(chunk["prompt_token_ids"]) == (
            "A<|im_end|>\n<|im_start|>user\nOutput B.<|im_end|>\n"
            "<|im_start|>assistant\n" + generation_prefill
        )
    finally:
        renderer.shutdown()


@pytest.mark.parametrize(
    ("eager", "asynchronous", "api_servers", "error"),
    [
        (False, None, 1, "enforce-eager"),
        (True, True, 1, "no-async-scheduling"),
        (True, False, 2, "api-server-count 1"),
    ],
)
def test_unsupported_startup_options_fail_before_engine_creation(
    eager, asynchronous, api_servers, error
):
    args = AsyncEngineArgs(enforce_eager=eager, async_scheduling=asynchronous)
    with pytest.raises(ValueError, match=error):
        configure_streaming(Namespace(api_server_count=api_servers), args)


def test_streaming_defaults_to_synchronous_scheduling():
    args = AsyncEngineArgs(enforce_eager=True)
    configure_streaming(Namespace(api_server_count=1), args)
    assert args.async_scheduling is False


@pytest.mark.asyncio
@pytest.mark.parametrize("chunked", [False, True])
async def test_http_rejects_oversized_body_before_session_lookup(chunked):
    app = _app()

    async def body():
        yield b"x" * _MAX_FRAME_BYTES
        yield b"x"

    content = body() if chunked else b"x" * (_MAX_FRAME_BYTES + 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/streaming/sessions/missing/frame", content=content
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == 413


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"[]",
        b'{"system_prompt": "task", "fps": 1}',
        b'{"system_prompt":"task","sampling":{"max_tokens":0}}',
        b'{"system_prompt":"task","sampling":{"temperature":NaN}}',
        b'{"system_prompt":"task","sampling":{"guided_choice":["yes","no"]}}',
        (
            b'{"system_prompt":"task","sampling":{"structured_outputs":'
            b'{"choice":["red","blue"],"regex":"red|blue"}}}'
        ),
    ],
)
async def test_http_rejects_invalid_session_without_starting_engine(body):
    app = _app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/streaming/sessions",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == 400
    assert response.json()["error"]["message"]
