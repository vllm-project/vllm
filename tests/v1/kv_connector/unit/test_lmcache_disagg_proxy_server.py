# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill must request one non-streaming token without changing decode arguments."""

import runpy
from pathlib import Path

import pytest

_PROXY_SERVER = (
    Path(__file__).parents[4]
    / "examples"
    / "disaggregated"
    / "lmcache"
    / "disagg_prefill_lmcache_v1"
    / "disagg_proxy_server.py"
)

_CLIENT_PAYLOAD = {
    "model": "test-model",
    "prompt": "hello",
    "stream": True,
    "stream_options": {"include_usage": True},
    "min_tokens": 10,
    "min_completion_tokens": 10,
    "max_completion_tokens": 100,
}

_ENDPOINTS = [
    ("handle_completions", "/completions"),
    ("handle_chat_completions", "/chat/completions"),
]


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_bytes(self):
        yield b'data: {"choices": []}\n\n'


class _FakeStream:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _RecordingClient:
    """A stand-in for `httpx.AsyncClient` that records what it was sent."""

    def __init__(self):
        self.sent: list[dict] = []

    async def post(self, endpoint, json=None, headers=None) -> _FakeResponse:
        self.sent.append({"endpoint": endpoint, "json": json, "headers": headers})
        return _FakeResponse()

    def stream(self, method, endpoint, json=None, headers=None) -> _FakeStream:
        self.sent.append({"endpoint": endpoint, "json": json, "headers": headers})
        return _FakeStream(_FakeResponse())


class _FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


@pytest.fixture(scope="module")
def proxy() -> dict:
    return runpy.run_path(str(_PROXY_SERVER))


@pytest.fixture
def clients(proxy) -> tuple[_RecordingClient, _RecordingClient]:
    prefill_client = _RecordingClient()
    decode_client = _RecordingClient()
    proxy["app"].state.prefill_client = prefill_client
    proxy["app"].state.decode_client = decode_client
    return prefill_client, decode_client


@pytest.mark.parametrize(("handler_name", "endpoint"), _ENDPOINTS)
@pytest.mark.asyncio
async def test_prefill_payload_is_a_valid_single_token_request(
    proxy, clients, handler_name, endpoint
):
    prefill_client, _ = clients

    await proxy[handler_name](_FakeRequest(dict(_CLIENT_PAYLOAD)))

    assert len(prefill_client.sent) == 1
    sent = prefill_client.sent[0]
    assert sent["endpoint"] == endpoint
    assert sent["json"]["max_tokens"] == 1
    assert sent["json"]["max_completion_tokens"] == 1
    assert sent["json"]["stream"] is False
    assert "stream_options" not in sent["json"]
    # P is pinned to one token, and `min_tokens > max_tokens` is a validation
    # error on the prefill instance.
    assert "min_tokens" not in sent["json"]
    assert "min_completion_tokens" not in sent["json"]


@pytest.mark.parametrize(("handler_name", "endpoint"), _ENDPOINTS)
@pytest.mark.asyncio
async def test_decode_leg_keeps_the_original_sampling_args(
    proxy, clients, handler_name, endpoint
):
    _, decode_client = clients

    response = await proxy[handler_name](_FakeRequest(dict(_CLIENT_PAYLOAD)))
    # Draining the stream is what actually issues the decode request.
    async for _ in response.body_iterator:
        pass

    assert len(decode_client.sent) == 1
    sent = decode_client.sent[0]
    assert sent["endpoint"] == endpoint
    for key in ("min_tokens", "min_completion_tokens", "max_completion_tokens"):
        assert sent["json"][key] == _CLIENT_PAYLOAD[key]
    assert sent["json"]["stream"] is True
