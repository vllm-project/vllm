# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise completion-stream success accounting over a real HTTP connection."""

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

pytestmark = pytest.mark.skip_global_cleanup

DONE = b"data: [DONE]\n\n"


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _choice(backend: str, **kwargs) -> bytes:
    content: dict = {"delta": {"content": "Hello 世界"}}
    if backend == "openai":
        content = {"text": "Hello 世界"}
    return _sse({"choices": [{"index": 0, **content, **kwargs}]})


async def _request(backend: str, stream: bytes) -> RequestFuncOutput:
    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        # Split both UTF-8 characters and the terminal marker across writes.
        for offset in range(0, len(stream), 3):
            await response.write(stream[offset : offset + 3])
        await response.write_eof()
        return response

    endpoint = "/v1/completions" if backend == "openai" else "/v1/chat/completions"
    app = web.Application()
    app.router.add_post(endpoint, handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        return await ASYNC_REQUEST_FUNCS[backend](
            RequestFuncInput(
                prompt="Say hello",
                api_url=str(server.make_url(endpoint)),
                prompt_len=2,
                output_len=4,
                model="test-model",
            ),
            session,
        )


@pytest.mark.parametrize("backend", ["openai", "openai-chat"])
@pytest.mark.parametrize("finish_reason", [None, "stop", "length"])
def test_eof_without_done_is_not_a_success(backend: str, finish_reason: str | None):
    """A proxy can close HTTP 200 cleanly after a token or even final usage."""
    stream = _choice(backend)
    if finish_reason:
        stream += _sse({"choices": [{"index": 0, "finish_reason": finish_reason}]})
        stream += _sse({"choices": [], "usage": {"completion_tokens": 4}})
    output = asyncio.run(_request(backend, stream))
    assert not output.success
    assert "[DONE]" in output.error
    assert output.generated_text == "Hello 世界"


@pytest.mark.parametrize("backend", ["openai", "openai-chat"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "has_output", [False, True], ids=["before-output", "after-output"]
)
def test_stream_error_is_not_a_success(backend: str, nested: bool, has_output: bool):
    error = {"object": "error", "message": "generation failed", "type": "server_error"}
    stream = (
        (_choice(backend) if has_output else b"")
        + _sse({"error": error} if nested else error)
        + DONE
    )
    output = asyncio.run(_request(backend, stream))
    assert not output.success
    assert "generation failed" in output.error
    assert output.generated_text == ("Hello 世界" if has_output else "")


@pytest.mark.parametrize("backend", ["openai", "openai-chat"])
def test_complete_empty_rendered_token_is_a_success(backend: str):
    choice = {"text": ""} if backend == "openai" else {"delta": {"content": ""}}
    stream = (
        _sse({"choices": [{"index": 0, **choice, "finish_reason": "stop"}]})
        + _sse({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 1}})
        + DONE
    )
    output = asyncio.run(_request(backend, stream))
    assert output.success, output.error
    assert output.generated_text == ""
    assert output.output_tokens == 1
    assert output.ttft > 0


@pytest.mark.parametrize("backend", ["openai", "openai-chat"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_complete_stream_preserves_usage(backend: str, finish_reason: str):
    stream = _choice(backend, finish_reason=finish_reason)
    stream += (
        _sse({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4}})
        + DONE
    )
    output = asyncio.run(_request(backend, stream))
    assert output.success, output.error
    assert output.generated_text == "Hello 世界"
    assert output.prompt_len == 7
    assert output.output_tokens == 4
    assert output.ttft > 0
    assert output.latency == output.ttft
    assert output.itl == []


@pytest.mark.parametrize("backend", ["openai", "openai-chat"])
def test_done_without_output_still_fails(backend: str):
    output = asyncio.run(_request(backend, DONE))
    assert not output.success
    assert "TTFT" in output.error


def test_complete_tool_only_chat_stream_is_a_success():
    stream = (
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_test",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + DONE
    )
    output = asyncio.run(_request("openai-chat", stream))
    assert output.success, output.error
    assert output.generated_text == ""
    assert output.ttft > 0
