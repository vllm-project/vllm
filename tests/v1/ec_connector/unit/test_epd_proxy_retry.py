# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Body handling across the EPD proxy's decode retries.

Exercises the REAL helpers loaded from the ``examples/`` proxy, so a future
change to them is what these tests catch.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

PROXY_REL = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"


@pytest.fixture(scope="module")
def proxy():
    path = Path(__file__).parents[4] / PROXY_REL
    spec = importlib.util.spec_from_file_location("disagg_epd_proxy_retry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, params):
        self._params = params

    async def json(self):
        return {"kv_transfer_params": self._params}


def test_maybe_prefill_leaves_the_caller_body_untouched(proxy, monkeypatch):
    """A decode retry re-enters this function with the body it was given.

    Mutating that body in place let one attempt's `remote_block_ids` survive
    into the next, so a retry whose prefill returns nothing sent decode blocks
    the prefiller may already have freed.
    """
    served = [{"remote_block_ids": [1, 2]}, {}]

    async def _stage(req_data, p_url, req_id):
        assert "kv_transfer_params" not in req_data
        return _Response(served.pop(0))

    monkeypatch.setattr(proxy, "process_prefill_stage", _stage)

    body = {"messages": [], "stream": False}
    first = asyncio.run(proxy.maybe_prefill(body, "http://prefill", "r1"))
    assert first["kv_transfer_params"] == {"remote_block_ids": [1, 2]}
    assert "kv_transfer_params" not in body

    second = asyncio.run(proxy.maybe_prefill(body, "http://prefill", "r1"))
    assert "kv_transfer_params" not in second


class _EncoderResponse:
    def __init__(self, params):
        self.status = 200
        self._params = params

    async def json(self):
        return {"ec_transfer_params": self._params}

    async def text(self):
        return ""


class _EncoderSession:
    """Serve one canned encoder reply per attempt."""

    def __init__(self, replies):
        self._replies = list(replies)

    async def post(self, url, json=None, headers=None):
        return _EncoderResponse(self._replies.pop(0))


def test_a_decode_retry_does_not_inherit_the_previous_handles(proxy, monkeypatch):
    """The retry loop re-enters `prepare_for_decode` with the same body.

    Recording the encoder's connector handles on that body in place let
    attempt 1's handle survive into attempt 2, so a second encode that
    reported nothing still sent decode a handle on an embedding the encoder
    no longer publishes -- the exact state the retry exists to leave behind.
    """
    handle = {"metadata": {"image_grid_thw": [1, 2, 2]}, "peer_port": 1234}
    monkeypatch.setattr(
        proxy,
        "encode_session",
        _EncoderSession([{"encoder-side-hash": handle}, {}]),
    )

    async def _no_prefill(req_data, p_url, req_id):
        return req_data

    monkeypatch.setattr(proxy, "maybe_prefill", _no_prefill)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ],
        "stream": False,
    }
    args = ("r1", ["http://encoder"], "http://prefill", None)

    first, _, _ = asyncio.run(proxy.prepare_for_decode(body, *args))
    reported = first["ec_transfer_params"]
    assert [handle] == [value for key, value in reported.items() if key != "ec_items"]
    assert "ec_transfer_params" not in body

    # Attempt 2's encode reports nothing: decode must be told nothing.
    second, _, _ = asyncio.run(proxy.prepare_for_decode(body, *args))
    assert "ec_transfer_params" not in second


def test_raw_media_keeps_encoder_transfer_identity(proxy, monkeypatch):
    handle = {"metadata": {"image_grid_thw": [1, 2, 2]}}
    monkeypatch.setattr(proxy, "NO_REWRITE", True)
    monkeypatch.setattr(
        proxy, "encode_session", _EncoderSession([{"encoded-hash": handle}])
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ]
    }
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body, "request", ["http://encoder"], "", "tcp://consumer:1"
        )
    )
    assert prepared["messages"] == body["messages"]
    assert "ec_transfer_params" not in body
    params = prepared["ec_transfer_params"]
    assert params["encoded-hash"] == handle
    assert params["ec_items"][0]["mm_hash"] == "encoded-hash"
    assert params["ec_items"][0]["transfer_id"]


@pytest.mark.parametrize("server_keep_alive", ["0.1", "1", "5", "2", "30"])
def test_pooled_connections_are_retired_before_the_server_closes_them(
    proxy, monkeypatch, server_keep_alive
):
    """The proxy must not hand a request a connection the server has dropped.

    vLLM closes idle keep-alive connections at `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`
    seconds while aiohttp pools them for 15, so a slow hop leaves a dead
    connection in the pool. The next request fails with
    ServerDisconnectedError and the server logs nothing, because it closed the
    socket before the request arrived.
    """
    monkeypatch.setenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", server_keep_alive)
    monkeypatch.setattr(proxy.app.state, "p_urls", [], raising=False)

    asyncio.run(proxy.on_startup())
    try:
        # No public accessor for the pool's idle timeout.
        pooled_for = proxy.encode_session.connector._keepalive_timeout
        assert pooled_for < float(server_keep_alive)
        assert proxy.decode_session.connector._keepalive_timeout == pooled_for
    finally:
        asyncio.run(proxy.on_shutdown())
