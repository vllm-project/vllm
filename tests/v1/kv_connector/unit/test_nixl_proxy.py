# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import runpy
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

PROXY_PATH = (
    Path(__file__).parents[4]
    / "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
)
API_KEY = "nixl-proxy-test"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
MODEL = "facebook/opt-125m"


def load_proxy(monkeypatch, prefill, decode, enabled=True, api_key=API_KEY):
    if enabled:
        monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    else:
        monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    argv = [str(PROXY_PATH)]
    for role, instances in (("prefiller", prefill), ("decoder", decode)):
        argv.extend([f"--{role}-hosts", *(host for host, _ in instances)])
        argv.extend([f"--{role}-ports", *(str(port) for _, port in instances)])
    monkeypatch.setattr(sys, "argv", argv)
    proxy = runpy.run_path(str(PROXY_PATH))["parse_args"].__globals__
    proxy["global_args"] = proxy["parse_args"]()
    return proxy["app"], proxy["lifespan"]


@pytest_asyncio.fixture
async def proxy_client(request, monkeypatch):
    outcomes: dict[tuple[str, int], tuple[int, object]] = {}
    calls = []

    async def start_backend(stack, role, index):
        async def reset(request):
            calls.append((role, index, dict(request.query), request.headers))
            status, body = outcomes.get((role, index), (200, {"success": True}))
            if body == "disconnect":
                request.transport.close()
                return web.Response()
            return web.json_response(
                body,
                status=status,
                headers={"Location": "/redirected"} if status == 307 else {},
            )

        async def redirected(request):
            return web.json_response({"success": True})

        backend = web.Application()
        backend.router.add_post("/reset_prefix_cache", reset)
        backend.router.add_post("/redirected", redirected)
        server = await stack.enter_async_context(TestServer(backend))
        return "127.0.0.1", server.port

    async with contextlib.AsyncExitStack() as stack:
        prefill = [await start_backend(stack, "prefill", index) for index in range(2)]
        decode = [await start_backend(stack, "decode", index) for index in range(2)]
        app, lifespan = load_proxy(
            monkeypatch, prefill, decode, enabled=getattr(request, "param", True)
        )
        await stack.enter_async_context(lifespan(app))
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy"
            )
        )
        yield client, outcomes, calls


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize("options", [False, True])
async def test_reset_reaches_every_pool_at_the_root_url(proxy_client, options):
    """Reset all pools rather than only the next round-robin pair."""
    client, _, calls = proxy_client
    params = (
        {"reset_running_requests": "true", "reset_external": "true"} if options else {}
    )
    response = await client.post("/reset_prefix_cache", params=params, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "results": [
            {"role": role, "id": index, "status": 200, "success": True}
            for role in ("prefill", "decode")
            for index in range(2)
        ],
    }
    assert {(role, index) for role, index, _, _ in calls} == {
        (role, index) for role in ("prefill", "decode") for index in range(2)
    }
    expected = str(options).lower()
    for _, _, params, headers in calls:
        assert params == {
            "reset_running_requests": expected,
            "reset_external": expected,
        }
        assert headers["Authorization"] == HEADERS["Authorization"]


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, body",
    [
        (200, {"success": False}),
        (404, {}),
        (500, {}),
        (307, {}),
        (200, {"success": "true"}),
        (200, "disconnect"),
    ],
)
async def test_reset_reports_partial_failure_without_hiding_successes(
    proxy_client, status, body
):
    client, outcomes, calls = proxy_client
    outcomes["decode", 1] = status, body
    response = await client.post("/reset_prefix_cache", headers=HEADERS)
    assert response.status_code == (200 if body == {"success": False} else 502)
    result = response.json()
    assert result["success"] is False
    assert len(calls) == 4
    assert sum(item["success"] for item in result["results"]) == 3
    failed = result["results"][-1]
    assert (failed["role"], failed["id"], failed["success"]) == ("decode", 1, False)


@pytest.mark.cpu_test
@pytest.mark.asyncio
async def test_reset_rejects_unauthorized_or_invalid_requests_before_fanout(
    proxy_client,
):
    client, _, calls = proxy_client
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        response = await client.post("/reset_prefix_cache", headers=headers)
        assert response.status_code == 401
    response = await client.post(
        "/reset_prefix_cache?reset_external=invalid", headers=HEADERS
    )
    assert response.status_code == 422
    assert calls == []


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_client", [False], indirect=True)
async def test_reset_route_is_disabled_by_default(proxy_client):
    client, _, calls = proxy_client
    response = await client.post("/reset_prefix_cache", headers=HEADERS)
    assert response.status_code == 404
    assert calls == []


@pytest.fixture(scope="module")
def native_server():
    from tests.utils import RemoteOpenAIServer
    from vllm.platforms import current_platform

    api_key = RemoteOpenAIServer.DUMMY_API_KEY
    args = [
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "256",
        "--block-size",
        "32",
        "--gpu-memory-utilization",
        "0.2",
        "--kv-cache-memory-bytes",
        "67108864",
        "--dtype",
        "float32" if current_platform.is_cpu() else "auto",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--api-key",
        api_key,
    ]
    with RemoteOpenAIServer(
        MODEL, args, env_dict={"VLLM_SERVER_DEV_MODE": "1"}
    ) as server:
        yield server


@pytest.mark.cpu_test
@pytest.mark.asyncio
async def test_proxy_reset_reaches_a_real_backend(monkeypatch, native_server):
    """The root reset URL and response contract must work with native vLLM."""
    server = native_server
    app, lifespan = load_proxy(
        monkeypatch,
        [(server.host, server.port)],
        [(server.host, server.port)],
        api_key=server.DUMMY_API_KEY,
    )
    async with (
        lifespan(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client,
    ):
        response = await client.post(
            "/reset_prefix_cache",
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "results": [
                {"role": role, "id": 0, "status": 200, "success": True}
                for role in ("prefill", "decode")
            ],
        }


@pytest.mark.asyncio
async def test_proxy_reset_clears_native_prefix_cache(monkeypatch, native_server):
    """A successful proxy reset must make the next native prefill cold."""
    server = native_server
    api_key = server.DUMMY_API_KEY
    app, lifespan = load_proxy(
        monkeypatch,
        [(server.host, server.port)],
        [(server.host, server.port)],
        api_key=api_key,
    )
    async with server.get_async_client() as native:
        request = {
            "model": MODEL,
            "prompt": [100] * 64,
            "max_tokens": 1,
            "temperature": 0,
        }
        await native.completions.create(**request)
        warm = await native.completions.create(**request)
        assert warm.usage.prompt_tokens_details.cached_tokens > 0
        async with (
            lifespan(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy"
            ) as client,
        ):
            response = await client.post(
                "/reset_prefix_cache", headers={"Authorization": f"Bearer {api_key}"}
            )
            assert response.status_code == 200
            assert response.json()["success"] is True
        cold = await native.completions.create(**request)
        assert cold.usage.prompt_tokens_details.cached_tokens == 0
        print(
            "Cached prompt tokens:",
            warm.usage.prompt_tokens_details.cached_tokens,
            "->",
            cold.usage.prompt_tokens_details.cached_tokens,
        )
