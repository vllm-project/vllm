# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-rewriting contract of the NIXL toy proxy server.

`send_request_to_service` rewrites the client payload before handing it to the
prefill instance. P is only asked to populate its KV cache for D, so the
response is thrown away afterwards. Two invariants are easy to break by hand:

1. The payload sent to P must still be a *valid* request. P is forced to
   `max_tokens=1`, and `SamplingParams` rejects `min_tokens > max_tokens`, so
   the client's `min_tokens` has to be dropped. `stream` must be pinned to
   `False` and `stream_options` removed, otherwise P answers with a stream that
   nobody consumes for a "one token" request.
2. The *caller's* payload must keep those sampling arguments, because the
   decode leg is served from it. `send_request_to_service` works on a copy, so
   the two are decoupled - deleting the copy or "restoring" the popped keys
   into the local copy silently changes which of the two the network sees.
"""

import runpy
from pathlib import Path

import pytest

_PROXY_SERVER = (
    Path(__file__).parents[4]
    / "tests"
    / "v1"
    / "kv_connector"
    / "nixl_integration"
    / "toy_proxy_server.py"
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


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Records the last request it was asked to send."""

    def __init__(self):
        self.sent: dict | None = None

    async def post(self, endpoint, json=None, headers=None):
        self.sent = {"endpoint": endpoint, "json": json, "headers": headers}
        return _FakeResponse({"kv_transfer_params": {}})


@pytest.fixture(scope="module")
def proxy() -> dict:
    return runpy.run_path(str(_PROXY_SERVER))


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


@pytest.mark.asyncio
async def test_prefill_payload_is_a_valid_single_token_request(proxy, fake_client):
    req_data = dict(_CLIENT_PAYLOAD)

    await proxy["send_request_to_service"](
        {"client": fake_client}, "/completions", req_data, "req-1"
    )

    sent = fake_client.sent["json"]
    assert sent["max_tokens"] == 1
    assert sent["max_completion_tokens"] == 1
    assert sent["stream"] is False
    assert "stream_options" not in sent
    # Dropped from P only: `min_tokens > max_tokens` would be a validation
    # error, and the single token P produces is discarded anyway.
    assert "min_tokens" not in sent
    assert "min_completion_tokens" not in sent
    # P has to keep the KV around for D.
    assert sent["kv_transfer_params"]["do_remote_decode"] is True
    assert sent["kv_transfer_params"]["do_remote_prefill"] is False


@pytest.mark.asyncio
async def test_caller_payload_keeps_the_args_the_decode_leg_needs(proxy, fake_client):
    req_data = dict(_CLIENT_PAYLOAD)

    await proxy["send_request_to_service"](
        {"client": fake_client}, "/completions", req_data, "req-2"
    )

    # The rewrite happens on a copy, and the decode leg is built from the
    # caller's dict, so the sampling args must survive untouched.
    assert req_data == _CLIENT_PAYLOAD
