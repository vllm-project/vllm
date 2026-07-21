# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Network-level end-to-end tests for prefix-aware request routing."""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
import pytest
import uvicorn
from aiohttp import web
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.distributed.prefix_scheduler import (
    GlobalPrefixScheduler,
    PrefixCacheSnapshot,
)
from vllm.entrypoints.openai.prefix_routing import (
    PrefixRoutingMiddleware,
    PrefixRoutingNode,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    maybe_convert_block_hash,
)

pytestmark = pytest.mark.skip_global_cleanup


def _hash(value: int) -> BlockHash:
    return BlockHash(bytes([value]) * 32)


def _external_hash(value: BlockHash) -> ExternalBlockHash:
    return maybe_convert_block_hash(value)


def _stored_event(value: BlockHash) -> BlockStored:
    return BlockStored(
        block_hashes=[_external_hash(value)],
        parent_block_hash=None,
        token_ids=[],
        block_size=16,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=0,
    )


class _NetworkRoutingProxy:
    def __init__(
        self,
        nodes: list[PrefixRoutingNode],
        routes: dict[str, tuple[list[BlockHash], int]],
    ) -> None:
        self.config = SimpleNamespace(
            routing_token="router-secret",
            max_request_body_size=1024 * 1024,
        )
        self.nodes = {node.node_id: node for node in nodes}
        self.routes = routes
        self.scheduler = GlobalPrefixScheduler()
        self.session: aiohttp.ClientSession | None = None

    async def choose_node_for_request(self, path, payload):
        del path
        block_hashes, prompt_num_tokens = self.routes[payload["route"]]
        return self.scheduler.choose_node(block_hashes, prompt_num_tokens)


@asynccontextmanager
async def _serve_upstream(
    node_id: str,
    token: str,
    *,
    interrupt_stream: bool = False,
) -> AsyncIterator[str]:
    async def completions(request: web.Request) -> web.StreamResponse:
        assert request.headers["x-vllm-prefix-routing"] == token
        rank = request.headers.get("x-data-parallel-rank")
        if interrupt_stream:
            response = web.StreamResponse(
                status=200,
                headers={"content-type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(b'data: {"node":"remote"}\n\n')
            assert request.transport is not None
            request.transport.close()
            return response
        return web.json_response({"node": node_id, "rank": rank})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _serve_router(
    proxy: _NetworkRoutingProxy,
) -> AsyncIterator[tuple[str, list[dict[str, str | None]]]]:
    local_requests: list[dict[str, str | None]] = []
    app = FastAPI()

    @app.post("/v1/completions")
    async def local_completion(request: Request) -> JSONResponse:
        local_requests.append({"rank": request.headers.get("x-data-parallel-rank")})
        return JSONResponse({"node": "local"})

    app.state.prefix_routing_proxy = proxy
    app.add_middleware(PrefixRoutingMiddleware)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            lifespan="off",
            log_level="warning",
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if task.done():
            await task
        await asyncio.sleep(0.01)

    proxy.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))
    try:
        yield f"http://127.0.0.1:{port}", local_requests
    finally:
        await proxy.session.close()
        proxy.session = None
        server.should_exit = True
        await task


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    route: str,
) -> tuple[int, dict]:
    async with session.post(
        f"{url}/v1/completions",
        json={"model": "test-model", "prompt": "hello", "route": route},
        headers={
            "x-vllm-prefix-routing": "forged",
            "x-data-parallel-rank": "99",
        },
    ) as response:
        return response.status, await response.json()


def test_prefix_routing_multi_node_network_e2e():
    async def exercise() -> None:
        first_hash = _hash(1)
        second_hash = _hash(2)
        third_hash = _hash(3)
        async with (
            _serve_upstream("node-a", "node-a-secret") as node_a_url,
            _serve_upstream("node-b", "node-b-secret") as node_b_url,
        ):
            proxy = _NetworkRoutingProxy(
                [
                    PrefixRoutingNode(
                        node_id="node-a",
                        url=node_a_url,
                        routing_token="node-a-secret",
                    ),
                    PrefixRoutingNode(
                        node_id="node-b",
                        url=node_b_url,
                        routing_token="node-b-secret",
                    ),
                ],
                {
                    "node-a": ([first_hash], 32),
                    "node-b": ([second_hash, third_hash], 48),
                },
            )
            proxy.scheduler.update_snapshot(
                PrefixCacheSnapshot(
                    node_id="node-a",
                    hash_block_size=16,
                    group_block_sizes={0: 16},
                    group_hashes={0: {_external_hash(first_hash)}},
                    data_parallel_rank=0,
                )
            )
            proxy.scheduler.update_snapshot(
                PrefixCacheSnapshot(
                    node_id="node-b",
                    hash_block_size=16,
                    group_block_sizes={0: 16},
                    group_hashes={
                        0: {
                            _external_hash(second_hash),
                            _external_hash(third_hash),
                        }
                    },
                    data_parallel_rank=1,
                )
            )

            async with (
                _serve_router(proxy) as (router_url, local_requests),
                aiohttp.ClientSession() as session,
            ):
                first_status, first = await _post_json(session, router_url, "node-a")
                second_status, second = await _post_json(session, router_url, "node-b")

            assert first_status == second_status == 200
            assert first == {"node": "node-a", "rank": "0"}
            assert second == {"node": "node-b", "rank": "1"}
            assert not local_requests

    asyncio.run(exercise())


def test_prefix_routing_rank_isolation_network_e2e():
    async def exercise() -> None:
        rank_zero_hash = _hash(4)
        rank_one_hash = _hash(5)
        async with _serve_upstream("replica", "replica-secret") as replica_url:
            proxy = _NetworkRoutingProxy(
                [
                    PrefixRoutingNode(
                        node_id="replica",
                        url=replica_url,
                        routing_token="replica-secret",
                    )
                ],
                {
                    "rank-zero": ([rank_zero_hash], 32),
                    "rank-one": ([rank_one_hash], 32),
                },
            )
            proxy.scheduler.register_node("replica", hash_block_size=16)
            proxy.scheduler.apply_event_batch(
                "replica",
                KVEventBatch(
                    ts=1.0,
                    data_parallel_rank=0,
                    events=[_stored_event(rank_zero_hash)],
                ),
            )
            proxy.scheduler.apply_event_batch(
                "replica",
                KVEventBatch(
                    ts=2.0,
                    data_parallel_rank=1,
                    events=[_stored_event(rank_one_hash)],
                ),
            )

            async with (
                _serve_router(proxy) as (router_url, local_requests),
                aiohttp.ClientSession() as session,
            ):
                _, rank_zero = await _post_json(session, router_url, "rank-zero")
                _, rank_one = await _post_json(session, router_url, "rank-one")

            assert rank_zero == {"node": "replica", "rank": "0"}
            assert rank_one == {"node": "replica", "rank": "1"}
            assert not local_requests

    asyncio.run(exercise())


def test_prefix_routing_upstream_failure_network_e2e():
    async def exercise() -> None:
        selected_hash = _hash(6)
        closed_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        closed_socket.bind(("127.0.0.1", 0))
        unavailable_port = closed_socket.getsockname()[1]
        closed_socket.close()
        proxy = _NetworkRoutingProxy(
            [
                PrefixRoutingNode(
                    node_id="unavailable",
                    url=f"http://127.0.0.1:{unavailable_port}",
                    routing_token="unavailable-secret",
                )
            ],
            {"failure": ([selected_hash], 32)},
        )
        proxy.scheduler.update_snapshot(
            PrefixCacheSnapshot(
                node_id="unavailable",
                hash_block_size=16,
                group_block_sizes={0: 16},
                group_hashes={0: {_external_hash(selected_hash)}},
                data_parallel_rank=7,
            )
        )

        async with (
            _serve_router(proxy) as (router_url, local_requests),
            aiohttp.ClientSession() as session,
        ):
            status, response = await _post_json(session, router_url, "failure")

        assert status == 200
        assert response == {"node": "local"}
        assert local_requests == [{"rank": None}]

    asyncio.run(exercise())


def test_prefix_routing_stream_failure_network_e2e():
    async def exercise() -> None:
        selected_hash = _hash(7)
        async with _serve_upstream(
            "streaming",
            "streaming-secret",
            interrupt_stream=True,
        ) as streaming_url:
            proxy = _NetworkRoutingProxy(
                [
                    PrefixRoutingNode(
                        node_id="streaming",
                        url=streaming_url,
                        routing_token="streaming-secret",
                    )
                ],
                {"stream": ([selected_hash], 32)},
            )
            proxy.scheduler.update_snapshot(
                PrefixCacheSnapshot(
                    node_id="streaming",
                    hash_block_size=16,
                    group_block_sizes={0: 16},
                    group_hashes={0: {_external_hash(selected_hash)}},
                    data_parallel_rank=3,
                )
            )

            async with (
                _serve_router(proxy) as (router_url, local_requests),
                aiohttp.ClientSession() as session,
                session.post(
                    f"{router_url}/v1/completions",
                    json={
                        "model": "test-model",
                        "prompt": "hello",
                        "route": "stream",
                        "stream": True,
                    },
                ) as response,
            ):
                assert response.status == 200
                first_chunk = await response.content.readany()
                assert b'"node":"remote"' in first_chunk
                with pytest.raises(aiohttp.ClientPayloadError):
                    await response.read()

            assert not local_requests

    asyncio.run(exercise())
