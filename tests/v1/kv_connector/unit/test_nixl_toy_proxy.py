# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import json
import runpy
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.cpu_test

_PROXY = Path(__file__).parents[1] / "nixl_integration/toy_proxy_server.py"
_TIMEOUT_S = 5

_P_COORDINATES = {
    "remote_engine_id": "p-engine",
    "remote_host": "p-host",
    "remote_port": 5600,
    "tp_size": 2,
    "dcp_size": 1,
    "pp_size": 1,
}


def _prefill_params(transfer_mode: str, request_id: str) -> dict:
    return {
        **_P_COORDINATES,
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [[1, 2]],
        "remote_request_id": f"cmpl-{request_id}-0-abcd1234",
        "remote_num_tokens": 7,
        "transfer_mode": transfer_mode,
    }


class _Backends:
    """Fake P and D servers behind the real toy proxy app."""

    def __init__(self, transfer_mode: str):
        self.transfer_mode = transfer_mode
        self.prefill_requests: list[httpx.Request] = []
        self.decode_requests: list[dict] = []
        self.decode_arrived = asyncio.Event()
        # Set to hold P's response until D has received its request.
        self.prefill_waits_for_decode = False
        self.prefill_status = 200
        self.decode_hangs = False

        proxy = runpy.run_path(str(_PROXY))["_handle_completions"].__globals__
        self.app = proxy["app"]
        self.app.state.prefill_clients = [self._client(self._prefill)]
        self.app.state.decode_clients = [self._client(self._decode)]
        self.app.state.prefill_iterator = itertools.cycle([0])
        self.app.state.decode_iterator = itertools.cycle([0])

    @staticmethod
    def _client(handler) -> dict:
        return {
            "client": httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://backend/v1"
            )
        }

    async def _prefill(self, request: httpx.Request) -> httpx.Response:
        self.prefill_requests.append(request)
        if self.prefill_waits_for_decode:
            await asyncio.wait_for(self.decode_arrived.wait(), _TIMEOUT_S)
        if self.prefill_status != 200:
            return httpx.Response(self.prefill_status, json={"error": "prefill"})
        params = _prefill_params(self.transfer_mode, request.headers["X-Request-Id"])
        return httpx.Response(200, json={"kv_transfer_params": params})

    async def _decode(self, request: httpx.Request) -> httpx.Response:
        self.decode_requests.append(json.loads(request.content))
        self.decode_arrived.set()
        if self.decode_hangs:
            await asyncio.Event().wait()
        return httpx.Response(200, json={"choices": [{"text": "hello"}]})

    async def post(self) -> httpx.Response:
        self.decode_arrived.clear()
        transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy"
        ) as client:
            return await asyncio.wait_for(
                client.post("/v1/completions", json={"prompt": "hi"}), _TIMEOUT_S
            )


@pytest.mark.asyncio
async def test_push_decode_starts_before_prefill_finishes():
    """In push mode D must register its blocks while P is still prefilling."""
    backends = _Backends("push")

    # P's coordinates are learned from its first response.
    assert (await backends.post()).status_code == 200

    backends.prefill_waits_for_decode = True
    response = await backends.post()

    assert response.status_code == 200
    assert response.json() == {"choices": [{"text": "hello"}]}
    request_id = backends.prefill_requests[1].headers["X-Request-Id"]
    assert backends.decode_requests[1]["kv_transfer_params"] == {
        **_P_COORDINATES,
        "transfer_mode": "push",
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_request_id": request_id,
    }


@pytest.mark.asyncio
async def test_push_prefill_failure_fails_request():
    """A failed prefill must not leave the client waiting on D for a push."""
    backends = _Backends("push")
    assert (await backends.post()).status_code == 200

    backends.prefill_status = 500
    backends.decode_hangs = True
    response = await backends.post()

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_pull_decode_gets_prefill_response():
    """In pull mode D needs the block ids from P's response for every request."""
    backends = _Backends("pull")

    for _ in range(2):
        assert (await backends.post()).status_code == 200

    for prefill_request, decode_request in zip(
        backends.prefill_requests, backends.decode_requests
    ):
        assert decode_request["kv_transfer_params"] == _prefill_params(
            "pull", prefill_request.headers["X-Request-Id"]
        )
