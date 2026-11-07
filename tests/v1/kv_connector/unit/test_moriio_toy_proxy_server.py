# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import runpy
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "MoRIIO toy proxy tests run only in the ROCm image.",
        allow_module_level=True,
    )

pytest.importorskip("quart")

_PROXY_SERVER = (
    Path(__file__).parents[4]
    / "examples"
    / "disaggregated"
    / "disaggregated_serving"
    / "moriio_toy_proxy_server.py"
)


def _load_proxy():
    namespace = runpy.run_path(str(_PROXY_SERVER))
    return namespace["handle_request"].__globals__


@pytest.mark.asyncio
async def test_health_route_returns_ok():
    proxy = _load_proxy()

    response = await proxy["app"].test_client().get("/health")

    assert response.status_code == 200
    assert await response.get_data(as_text=True) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["READ", "WRITE"])
@pytest.mark.parametrize("prefill_dp,decode_dp", [(2, 1), (2, 4), (2, 2)])
async def test_decode_dispatch_uses_its_own_dp_rank(
    monkeypatch, unused_tcp_port, mode, prefill_dp, decode_dp
):
    proxy = _load_proxy()
    received: dict[str, list[tuple[str | None, dict[str, Any]]]] = {
        "prefill": [],
        "decode": [],
    }
    prefill_received: asyncio.Queue[None] = asyncio.Queue()

    async def backend(request):
        role = request.match_info["role"]
        body = await request.json()
        received[role].append((request.headers.get("X-data-parallel-rank"), body))
        if role == "prefill":
            prefill_received.put_nowait(None)
            return web.json_response(
                {
                    "kv_transfer_params": {
                        "remote_engine_id": "prefill",
                        "remote_block_ids": [0],
                        "transfer_id": body["kv_transfer_params"]["transfer_id"],
                    }
                }
            )
        return web.Response(text="data: [DONE]\n\n", content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/{role}/completions", backend)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
        for role, dp in [("prefill", prefill_dp), ("decode", decode_dp)]:
            monkeypatch.setitem(
                proxy,
                f"{role}_instances",
                [
                    {
                        "request_address": f"http://127.0.0.1:{unused_tcp_port}/{role}",
                        "zmq_address": "host:127.0.0.1,handshake:6301,notify:61005",
                        "dp_size": dp,
                        "tp_size": 1,
                    }
                ],
            )
        monkeypatch.setitem(proxy, "TRANSFER_TYPE", mode)
        client = proxy["app"].test_client()
        for _ in range(max(prefill_dp, decode_dp)):
            response = await client.post(
                "/v1/completions", json={"prompt": "hello", "max_tokens": 4}
            )
            assert response.status_code == 200
            assert await response.get_data(as_text=True) == "data: [DONE]\n\n"
            await asyncio.wait_for(prefill_received.get(), timeout=10)
        for index, (rank, body) in enumerate(received["decode"]):
            expected = str(index % decode_dp) if decode_dp > 1 else None
            assert rank == expected
            assert body["kv_transfer_params"]["remote_dp_rank"] == index % prefill_dp
        assert len(received["decode"]) == max(prefill_dp, decode_dp)
        assert [rank for rank, _ in received["prefill"]] == [
            str(index % prefill_dp) for index in range(max(prefill_dp, decode_dp))
        ]
    finally:
        await runner.cleanup()
