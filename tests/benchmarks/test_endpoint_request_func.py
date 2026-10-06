# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import aiohttp
import pytest

from benchmarks import backend_request_func
from benchmarks.backend_request_func import (
    RequestFuncInput as LegacyRequestFuncInput,
)
from vllm.benchmarks.lib import endpoint_request_func
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_chat_completions,
)


class _FakeStreamContent:
    def __init__(self, messages: list[dict[str, Any] | str]):
        self.messages = messages

    async def iter_any(self):
        for message in self.messages:
            if isinstance(message, str):
                yield f"data: {message}\n\n".encode()
            else:
                yield f"data: {json.dumps(message)}\n\n".encode()

    def __aiter__(self):
        return self.iter_any()


class _FakeResponse:
    status = 200
    reason = ""

    def __init__(self, messages: list[dict[str, Any] | str]):
        self.content = _FakeStreamContent(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    def __init__(self, messages: list[dict[str, Any] | str]):
        self.messages = messages
        self.payload = None

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
        self.payload = json
        return _FakeResponse(self.messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeAiohttp:
    def __init__(self, session: _FakeSession):
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(aiohttp, name)

    def ClientSession(self, **kwargs: Any) -> _FakeSession:  # noqa: N802
        del kwargs
        return self._session


def _make_input() -> RequestFuncInput:
    return RequestFuncInput(
        prompt="hello",
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=5,
        output_len=8,
        model="test-model",
    )


def _make_legacy_input() -> LegacyRequestFuncInput:
    return LegacyRequestFuncInput(
        prompt="hello",
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=5,
        output_len=8,
        model="test-model",
    )


@pytest.mark.asyncio
async def test_openai_chat_itl_uses_continuous_usage_token_deltas(monkeypatch):
    messages = [
        {
            "choices": [{"delta": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 0,
                "total_tokens": 7,
            },
        },
        {
            "choices": [{"delta": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        {
            "choices": [{"delta": {"content": " world"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    timings = iter([0.0, 0.05, 0.35, 0.75, 0.8])
    monkeypatch.setattr(
        endpoint_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await async_request_openai_chat_completions(_make_input(), session)

    assert output.success
    assert output.generated_text == "hello world"
    assert output.prompt_len == 7
    assert output.output_tokens == 5
    assert output.ttft == pytest.approx(0.35)
    assert output.itl == pytest.approx([0.0, 0.0, 0.2, 0.2])
    assert output.latency - output.ttft == pytest.approx(sum(output.itl))
    # Trailing usage chunk carries no new tokens, so it does not extend latency.
    # Matches upstream #55508, which keeps latency - ttft == sum(itl).
    assert output.latency == pytest.approx(0.75)
    assert session.payload["stream_options"] == {
        "include_usage": True,
        "continuous_usage_stats": True,
    }


@pytest.mark.asyncio
async def test_openai_chat_final_usage_does_not_double_count_fallback_timing(
    monkeypatch,
):
    messages = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    timings = iter([0.0, 0.05, 0.1, 0.3, 0.5])
    monkeypatch.setattr(
        endpoint_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await async_request_openai_chat_completions(_make_input(), session)

    assert output.success
    assert output.output_tokens == 5
    assert output.ttft == pytest.approx(0.1)
    assert output.itl == pytest.approx([0.2])


@pytest.mark.asyncio
async def test_openai_chat_latency_invariant_holds_with_continuous_usage(
    monkeypatch,
):
    messages = [
        {
            "choices": [{"delta": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 0,
                "total_tokens": 7,
            },
        },
        {
            "choices": [{"delta": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
            },
        },
        {
            "choices": [{"delta": {"content": " world"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    timings = iter([0.0, 0.05, 0.35, 0.75, 0.8])
    monkeypatch.setattr(
        endpoint_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await async_request_openai_chat_completions(_make_input(), session)

    assert output.success
    assert output.latency - output.ttft == pytest.approx(sum(output.itl))


@pytest.mark.asyncio
async def test_openai_chat_usage_with_tokens_satisfies_first_chunk(monkeypatch):
    messages = [
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    timings = iter([0.0, 0.25])
    monkeypatch.setattr(
        endpoint_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await async_request_openai_chat_completions(_make_input(), session)

    assert output.success
    assert output.output_tokens == 3
    assert output.ttft == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_openai_chat_first_chunk_multi_token_preserves_latency_invariant(
    monkeypatch,
):
    messages = [
        {
            "choices": [{"delta": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 0,
                "total_tokens": 7,
            },
        },
        {
            "choices": [{"delta": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        },
        {
            "choices": [{"delta": {"content": " world"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    timings = iter([0.0, 0.05, 0.35, 0.75, 0.8])
    monkeypatch.setattr(
        endpoint_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await async_request_openai_chat_completions(_make_input(), session)

    assert output.success
    assert output.output_tokens == 3
    assert len(output.itl) == output.output_tokens - 1
    assert output.latency - output.ttft == pytest.approx(sum(output.itl))
    assert output.itl[0] == 0


@pytest.mark.asyncio
async def test_legacy_openai_chat_itl_uses_continuous_usage_token_deltas(
    monkeypatch,
):
    messages = [
        {
            "choices": [{"delta": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 0,
                "total_tokens": 7,
            },
        },
        {
            "choices": [{"delta": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        {
            "choices": [{"delta": {"content": " world"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    monkeypatch.setattr(backend_request_func, "aiohttp", _FakeAiohttp(session))
    timings = iter([0.0, 0.05, 0.35, 0.75, 0.8])
    monkeypatch.setattr(
        backend_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await backend_request_func.async_request_openai_chat_completions(
        _make_legacy_input()
    )

    assert output.success
    assert output.generated_text == "hello world"
    assert output.prompt_len == 7
    assert output.output_tokens == 5
    assert output.ttft == pytest.approx(0.35)
    assert output.itl == pytest.approx([0.0, 0.0, 0.2, 0.2])
    assert output.latency - output.ttft == pytest.approx(sum(output.itl))
    # Trailing usage chunk carries no new tokens, so it does not extend latency.
    # Matches upstream #55508, which keeps latency - ttft == sum(itl).
    assert output.latency == pytest.approx(0.75)
    assert session.payload["stream_options"] == {
        "include_usage": True,
        "continuous_usage_stats": True,
    }


@pytest.mark.asyncio
async def test_legacy_openai_chat_final_usage_does_not_double_count_fallback_timing(
    monkeypatch,
):
    messages = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    monkeypatch.setattr(backend_request_func, "aiohttp", _FakeAiohttp(session))
    timings = iter([0.0, 0.05, 0.1, 0.3, 0.5])
    monkeypatch.setattr(
        backend_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await backend_request_func.async_request_openai_chat_completions(
        _make_legacy_input()
    )

    assert output.success
    assert output.output_tokens == 5
    assert output.ttft == pytest.approx(0.1)
    assert output.itl == pytest.approx([0.2])


@pytest.mark.asyncio
async def test_legacy_openai_chat_first_chunk_multi_token_preserves_latency_invariant(
    monkeypatch,
):
    messages = [
        {
            "choices": [{"delta": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 0,
                "total_tokens": 7,
            },
        },
        {
            "choices": [{"delta": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        },
        {
            "choices": [{"delta": {"content": " world"}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        "[DONE]",
    ]
    session = _FakeSession(messages)
    monkeypatch.setattr(backend_request_func, "aiohttp", _FakeAiohttp(session))
    timings = iter([0.0, 0.05, 0.35, 0.75, 0.8])
    monkeypatch.setattr(
        backend_request_func.time, "perf_counter", lambda: next(timings)
    )

    output = await backend_request_func.async_request_openai_chat_completions(
        _make_legacy_input()
    )

    assert output.success
    assert output.output_tokens == 3
    assert len(output.itl) == output.output_tokens - 1
    assert output.latency - output.ttft == pytest.approx(sum(output.itl))
    assert output.itl[0] == 0
