# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator warm-prefetch manager (submit + status proxy)."""

# Standard
import json as _json
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.mp_coordinator.cache_control.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.registry import MPInstance


def _instance(instance_id: str, ip: str = "10.0.0.1", port: int = 8000) -> MPInstance:
    now = time.time()
    return MPInstance(
        instance_id=instance_id,
        ip=ip,
        http_port=port,
        registration_time=now,
        last_heartbeat_time=now,
    )


@pytest.mark.asyncio
async def test_submit_prefetch_posts_body_and_returns_reply():
    """submit_prefetch POSTs /cache/prefetches with the token body and returns the
    server's JSON reply verbatim."""
    mgr = PrefetchManager()
    target = _instance("mp-1", ip="10.0.0.7", port=8765)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["json"] = (request.read() or b"").decode()
        return httpx.Response(
            202, json={"request_id": "abc", "chunks": 2, "status": "submitted"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await mgr.submit_prefetch(
            target=target,
            http_client=client,
            model_name="m",
            world_size=2,
            token_ids=[1, 2, 3, 4],
            cache_salt="alice",
        )

    assert captured["method"] == "POST"
    assert captured["url"] == "http://10.0.0.7:8765/cache/prefetches"
    assert _json.loads(captured["json"]) == {
        "model_name": "m",
        "world_size": 2,
        "token_ids": [1, 2, 3, 4],
        "cache_salt": "alice",
    }
    assert result == {"request_id": "abc", "chunks": 2, "status": "submitted"}


@pytest.mark.asyncio
async def test_submit_prefetch_raises_on_http_error():
    """A non-2xx submit surfaces as an httpx error for the caller to map."""
    mgr = PrefetchManager()
    target = _instance("mp-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPError):
            await mgr.submit_prefetch(
                target=target,
                http_client=client,
                model_name="m",
                world_size=1,
                token_ids=[1, 2],
                cache_salt="",
            )


@pytest.mark.asyncio
async def test_get_status_proxies_code_and_body():
    """get_status returns the server's (status_code, body) verbatim."""
    mgr = PrefetchManager()
    target = _instance("mp-1", ip="10.0.0.7", port=8765)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200, json={"status": "completed", "found_keys": 4, "total_keys": 4}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        code, body = await mgr.get_status(
            target=target, http_client=client, request_id="abc"
        )

    assert captured["url"] == "http://10.0.0.7:8765/cache/prefetches/abc"
    assert code == 200
    assert body == {"status": "completed", "found_keys": 4, "total_keys": 4}


@pytest.mark.asyncio
async def test_get_status_relays_404():
    """An unknown id on the server is relayed as a 404 (not raised)."""
    mgr = PrefetchManager()
    target = _instance("mp-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        code, body = await mgr.get_status(
            target=target, http_client=client, request_id="x"
        )

    assert code == 404
    assert body == {"detail": "unknown"}
