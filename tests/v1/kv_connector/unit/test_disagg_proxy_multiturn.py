# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _load_proxy_module(monkeypatch):
    class DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            return lambda func: func

        def get(self, *args, **kwargs):
            return lambda func: func

    class DummyRequest:
        pass

    class DummyResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyAsyncClient:
        pass

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(AsyncClient=DummyAsyncClient),
    )
    monkeypatch.setitem(
        sys.modules,
        "fastapi",
        types.SimpleNamespace(FastAPI=DummyFastAPI, Request=DummyRequest),
    )
    monkeypatch.setitem(
        sys.modules,
        "fastapi.responses",
        types.SimpleNamespace(
            JSONResponse=DummyResponse,
            StreamingResponse=DummyResponse,
        ),
    )

    module_name = "_disagg_proxy_multiturn_test"
    sys.modules.pop(module_name, None)
    module_path = (
        Path(__file__).resolve().parents[4]
        / "examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def proxy(monkeypatch):
    return _load_proxy_module(monkeypatch)


def _kv_params():
    return {
        "remote_engine_id": "decode-engine",
        "remote_request_id": "decode-request",
        "remote_block_ids": [[1, 2, 3]],
        "remote_host": "decode-host",
        "remote_port": 8200,
    }


def test_chat_continuation_reuses_cached_kv(proxy):
    cache = proxy.ConversationKVCache(ttl_seconds=60.0)
    previous = {"messages": [{"role": "user", "content": "What is vLLM?"}]}
    cache.put("conv", _kv_params(), previous, "vLLM is a serving engine.")

    current = {
        "messages": [
            {"role": "user", "content": "What is vLLM?"},
            {"role": "assistant", "content": "vLLM is a serving engine."},
            {"role": "user", "content": "How does prefill work?"},
        ]
    }

    assert cache.get("conv", current) == _kv_params()


def test_chat_rewritten_history_misses_cached_kv(proxy):
    cache = proxy.ConversationKVCache(ttl_seconds=60.0)
    previous = {"messages": [{"role": "user", "content": "Solve it."}]}
    cache.put("conv", _kv_params(), previous, "<think>hidden</think>Final")

    current = {
        "messages": [
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": "Final"},
            {"role": "user", "content": "Now explain it."},
        ]
    }

    assert cache.get("conv", current) is None


def test_completion_continuation_reuses_cached_kv(proxy):
    cache = proxy.ConversationKVCache(ttl_seconds=60.0)
    cache.put("conv", _kv_params(), {"prompt": "Question:"}, " Answer.")

    assert cache.get("conv", {"prompt": "Question: Answer. Follow-up:"}) == _kv_params()


def test_completion_rewritten_prompt_misses_cached_kv(proxy):
    cache = proxy.ConversationKVCache(ttl_seconds=60.0)
    cache.put("conv", _kv_params(), {"prompt": "Question:"}, " Answer.")

    assert cache.get("conv", {"prompt": "Different question:"}) is None


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeHTTPClient:
    def __init__(self, lines):
        self.lines = lines

    def stream(self, *args, **kwargs):
        return FakeStreamResponse(self.lines)


def _sse_line(chunk):
    return "data: " + json.dumps(chunk)


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning": "hidden"},
        {"reasoning_content": "hidden"},
        {"tool_calls": [{"id": "call-1"}]},
        {"function_call": {"name": "tool"}},
    ],
)
def test_decode_stream_with_untracked_delta_does_not_cache(proxy, delta):
    async def run_test():
        kv_params = _kv_params()
        lines = [
            _sse_line(
                {
                    "id": "resp",
                    "model": "model",
                    "created": 1,
                    "choices": [{"delta": delta}],
                }
            ),
            _sse_line(
                {"choices": [{"delta": {"content": "Final"}, "finish_reason": "stop"}]}
            ),
            _sse_line({"kv_transfer_params": kv_params}),
            "data: [DONE]",
        ]
        proxy.kv_cache = proxy.ConversationKVCache(ttl_seconds=60.0)
        client = proxy.ServiceClient(
            client=FakeHTTPClient(lines),
            host="decode-host",
            port=8200,
            id=0,
        )

        text, _, captured_kv, _, _, _ = await proxy._stream_from_decode(
            client,
            "/chat/completions",
            {"messages": [{"role": "user", "content": "Solve it."}]},
            "request-id",
            "conv",
        )

        assert text == "Final"
        assert captured_kv is not None
        assert proxy.kv_cache.size == 0

    asyncio.run(run_test())


def test_decode_stream_with_include_reasoning_false_does_not_cache(proxy):
    async def run_test():
        kv_params = _kv_params()
        lines = [
            _sse_line(
                {
                    "id": "resp",
                    "model": "model",
                    "created": 1,
                    "choices": [
                        {"delta": {"content": "Final"}, "finish_reason": "stop"}
                    ],
                }
            ),
            _sse_line({"kv_transfer_params": kv_params}),
            "data: [DONE]",
        ]
        proxy.kv_cache = proxy.ConversationKVCache(ttl_seconds=60.0)
        client = proxy.ServiceClient(
            client=FakeHTTPClient(lines),
            host="decode-host",
            port=8200,
            id=0,
        )

        text, _, captured_kv, _, _, _ = await proxy._stream_from_decode(
            client,
            "/chat/completions",
            {
                "messages": [{"role": "user", "content": "Solve it."}],
                "include_reasoning": False,
            },
            "request-id",
            "conv",
        )

        assert text == "Final"
        assert captured_kv is not None
        assert proxy.kv_cache.size == 0

    asyncio.run(run_test())
