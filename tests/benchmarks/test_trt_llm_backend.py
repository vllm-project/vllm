# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from types import SimpleNamespace

import pytest

# Import from the top-level benchmarks module in the repo
from benchmarks.backend_request_func import (
    RequestFuncInput,
    async_request_trt_llm,
)


class _AsyncContext:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AsyncIterable:
    def __init__(self, chunks_bytes):
        self._chunks = chunks_bytes

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeResponse:
    def __init__(self, status, chunks=None, json_obj=None, headers=None):
        self.status = status
        self._chunks = chunks or []
        self._json_obj = json_obj
        self.headers = headers or {"content-type": "text/event-stream"}
        self.content = _AsyncIterable(self._chunks)

    async def json(self, content_type=None):
        return self._json_obj

    async def text(self):
        return json.dumps(self._json_obj) if self._json_obj is not None else ""


class _FakeSession:
    def __init__(self, responses):
        # responses: list of (status, chunks, json_obj)
        self._responses = responses
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append(SimpleNamespace(url=url, json=json, headers=headers))
        status, chunks, json_obj = self._responses.pop(0)
        resp = _FakeResponse(status=status, chunks=chunks, json_obj=json_obj)
        return _AsyncContext(resp)


def _sse_chunk(obj):
    return ("data: " + json.dumps(obj)).encode("utf-8")


@pytest.mark.asyncio
async def test_trt_llm_streaming_content(monkeypatch):
    # Simulate streaming content tokens via delta.content
    sse_chunks = [
        _sse_chunk(
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Hello"}}],
            }
        ),
        _sse_chunk(
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": " World"}}],
            }
        ),
        b"data: [DONE]",
    ]

    fake = _FakeSession(responses=[(200, sse_chunks, None)])

    # Patch ClientSession in module under test
    import benchmarks.backend_request_func as mod

    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda **_: fake)

    req = RequestFuncInput(
        prompt="hi",
        api_url="https://x/v1/chat/completions",
        prompt_len=2,
        output_len=8,
        model="m",
        extra_body={"response_format": {"type": "json", "schema": {}}},
        api_key=None,
        debug=False,
    )
    out = await async_request_trt_llm(req)
    assert out.success is True
    assert out.generated_text == "Hello World"
    # Ensure only one POST and it was streaming
    assert len(fake.calls) == 1
    assert fake.calls[0].json.get("stream") is True


@pytest.mark.asyncio
async def test_trt_llm_role_only_deltas_fallback(monkeypatch):
    # First call: role-only deltas, no content
    sse_chunks = [
        _sse_chunk(
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"role": "assistant"}}],
            }
        ),
        _sse_chunk(
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]",
    ]
    # Fallback non-streaming: full response JSON
    fallback_json = {
        "choices": [{"message": {"content": '{"name": "Alice"}'}}],
        "usage": {"completion_tokens": 10},
    }

    fake = _FakeSession(responses=[(200, sse_chunks, None), (200, [], fallback_json)])

    import benchmarks.backend_request_func as mod

    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda **_: fake)

    req = RequestFuncInput(
        prompt="hi",
        api_url="https://x/v1/chat/completions",
        prompt_len=2,
        output_len=8,
        model="m",
        extra_body={"response_format": {"type": "json", "schema": {}}},
        api_key=None,
        debug=False,
    )
    out = await async_request_trt_llm(req)
    assert out.success is True
    assert out.generated_text == '{"name": "Alice"}'
    # Two calls: initial stream + fallback non-stream
    assert len(fake.calls) == 2
    assert fake.calls[0].json.get("stream") is True
    assert fake.calls[1].json.get("stream") is False


@pytest.mark.asyncio
async def test_trt_llm_headers_auth_optional(monkeypatch):
    sse_done = [b"data: [DONE]"]
    fake = _FakeSession(responses=[(200, sse_done, None)])

    import benchmarks.backend_request_func as mod

    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda **_: fake)

    req = RequestFuncInput(
        prompt="hi",
        api_url="https://x/v1/completions",
        prompt_len=2,
        output_len=8,
        model="m",
        extra_body=None,
        api_key=None,
        debug=False,
    )
    _ = await async_request_trt_llm(req)
    hdrs = fake.calls[0].headers
    assert "Authorization" not in hdrs

    # With api_key
    fake2 = _FakeSession(responses=[(200, sse_done, None)])
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda **_: fake2)
    req2 = RequestFuncInput(
        prompt="hi",
        api_url="https://x/v1/completions",
        prompt_len=2,
        output_len=8,
        model="m",
        extra_body=None,
        api_key="tok",
        debug=False,
    )
    _ = await async_request_trt_llm(req2)
    hdrs2 = fake2.calls[0].headers
    assert hdrs2.get("Authorization") == "Bearer tok"


@pytest.mark.asyncio
async def test_trt_llm_payload_shapes_chat_vs_completions(monkeypatch):
    sse_done = [b"data: [DONE]"]
    fake = _FakeSession(responses=[(200, sse_done, None)])
    import benchmarks.backend_request_func as mod

    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda **_: fake)

    # Chat
    req_chat = RequestFuncInput(
        prompt="hi",
        api_url="https://x/v1/chat/completions",
        prompt_len=2,
        output_len=8,
        model="m",
        extra_body=None,
        api_key=None,
        debug=False,
    )
    _ = await async_request_trt_llm(req_chat)
    assert "messages" in fake.calls[0].json
    assert "prompt" not in fake.calls[0].json

    # Completions
    fake2 = _FakeSession(responses=[(200, sse_done, None)])
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda **_: fake2)
    req_comp = RequestFuncInput(
        prompt="hi",
        api_url="https://x/v1/completions",
        prompt_len=2,
        output_len=8,
        model="m",
        extra_body=None,
        api_key=None,
        debug=False,
    )
    _ = await async_request_trt_llm(req_comp)
    assert "prompt" in fake2.calls[0].json
    assert "messages" not in fake2.calls[0].json
