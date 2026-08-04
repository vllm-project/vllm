# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/cache/*`` REST API (warm-prefetch dispatch).

Quota writes, usage events, and status reads moved to the ``/quota`` group --
see ``test_quota_api.py``.
"""

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return TestClient(create_app(config))


# -- Prefetch dispatch -------------------------------------------------------


def _prefetch_body(instance_id: str, salt: str = "alice") -> dict:
    return {
        "instance_id": instance_id,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4],
        "cache_salt": salt,
    }


def _mock_mp_server() -> httpx.AsyncClient:
    """An outbound client that emulates the target MP server's prefetch API."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/cache/prefetches":
            return httpx.Response(
                202, json={"request_id": "abc", "chunks": 2, "status": "submitted"}
            )
        if request.method == "GET" and request.url.path == "/cache/prefetches/abc":
            return httpx.Response(
                200, json={"status": "completed", "found_keys": 2, "total_keys": 2}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_prefetch_unknown_instance_returns_404():
    """Targeting an unregistered instance must 404 (before any dispatch)."""
    with _client() as client:
        resp = client.post("/cache/prefetches", json=_prefetch_body("does-not-exist"))
        assert resp.status_code == 404


def test_prefetch_submit_then_status_proxy():
    """A registered target: submit relays the server's request_id, and the
    status GET proxies the server's completion body."""
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        # Replace the lifespan's real outbound client with a mock MP server.
        client.app.state.ctx.outbound_client = _mock_mp_server()

        resp = client.post("/cache/prefetches", json=_prefetch_body("mp-1"))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "instance_id": "mp-1",
            "request_id": "abc",
            "chunks": 2,
            "status": "submitted",
        }

        status = client.get("/cache/prefetches/mp-1/abc")
        assert status.status_code == 200, status.text
        assert status.json() == {
            "status": "completed",
            "found_keys": 2,
            "total_keys": 2,
        }


def test_prefetch_status_unknown_instance_returns_404():
    """Status for an unregistered instance must 404."""
    with _client() as client:
        resp = client.get("/cache/prefetches/does-not-exist/abc")
        assert resp.status_code == 404


# -- Pin / unpin (coordinator-side L2 pin) -----------------------------------


def _pin_client() -> TestClient:
    """A coordinator with a small chunk_size so short token sequences resolve."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0, eviction_check_interval=0.0, chunk_size=4
    )
    return TestClient(create_app(config))


def _pin_body(salt: str = "alice") -> dict:
    return {
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "cache_salt": salt,
    }


def _resolve(ctx, salt: str = "alice") -> list[ObjectKey]:
    """Resolve the pin body's keys the same way the handler will."""
    keys, _ = resolve_object_keys(
        ctx.token_hasher, "m", 1, [1, 2, 3, 4, 5, 6, 7, 8], salt
    )
    return keys


def test_pin_then_unpin_tracks_l2_eviction():
    """Pin excludes the resolved keys from L2 eviction; unpin restores them."""
    with _pin_client() as client:
        ctx = client.app.state.ctx
        keys = _resolve(ctx)
        assert keys  # 2 chunks x world_size 1

        # Arm allowlist enforcement (unquota'd salts are exempt until the
        # default limit is set), then track the keys in the L2 eviction LRU
        # with no quota (evict-all), so the plan would evict them unless
        # pinned.
        assert (
            client.put("/quota/config", json={"default_limit_gb": 0}).status_code == 200
        )
        for k in keys:
            ctx.usage_manager.record_stored(k, 1000)
            ctx.eviction_manager.on_store(k)
        assert ctx.eviction_manager.compute_eviction_plan()["alice"]

        resp = client.post("/cache/pins", json=_pin_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 2,
            "affected": len(keys),
            "status": "pinned",
        }
        # Pinned: the keys drop out of the eviction plan.
        assert ctx.eviction_manager.compute_eviction_plan() == {}

        resp = client.request("DELETE", "/cache/pins", json=_pin_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 2,
            "affected": len(keys),
            "status": "unpinned",
        }
        # Unpinned: the keys are eligible for eviction again.
        assert ctx.eviction_manager.compute_eviction_plan()["alice"]


def test_pin_short_sequence_is_noop():
    """A sub-chunk sequence resolves to no keys (affected 0)."""
    with _pin_client() as client:
        body = {
            "model_name": "m",
            "world_size": 1,
            "token_ids": [1, 2],
            "cache_salt": "",
        }
        resp = client.post("/cache/pins", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"requested": 0, "affected": 0, "status": "pinned"}


def test_pin_invalid_cache_salt_returns_400():
    """An invalid cache_salt (forbidden char) is a 400."""
    with _pin_client() as client:
        resp = client.post("/cache/pins", json=_pin_body(salt="bad@salt"))
        assert resp.status_code == 400


# -- Delete dispatch (coordinator resolves; key-addressed L1 + L2 to the node) --


def _delete_client() -> TestClient:
    """A coordinator with a small chunk_size so short token sequences resolve."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0, eviction_check_interval=0.0, chunk_size=4
    )
    return TestClient(create_app(config))


def _delete_body(
    instance_id: str, salt: str = "alice", tier: str = "all", force: bool = False
) -> dict:
    return {
        "instance_id": instance_id,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "cache_salt": salt,
        "tier": tier,
        "force": force,
    }


def _resolve_delete(ctx, salt: str = "alice") -> list[ObjectKey]:
    """Resolve the delete body's keys the same way the handler will."""
    keys, _ = resolve_object_keys(
        ctx.token_hasher, "m", 1, [1, 2, 3, 4, 5, 6, 7, 8], salt
    )
    return keys


def _mock_delete_server(deletes: list) -> httpx.AsyncClient:
    """Emulate the node's unified key-addressed delete (``DELETE /cache/objects``).

    Records each request body ``{keys, tier, force}`` so tests can assert the
    coordinator's single-call dispatch and pin filtering, and reports the keys
    deleted: both tiers for ``all`` (n L1 + n L2), one tier otherwise.
    """
    # Standard
    import json as _json

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cache/objects" and request.method == "DELETE":
            body = _json.loads(request.content.decode())
            deletes.append(body)
            n = len(body["keys"])
            deleted = n * (2 if body.get("tier") == "all" else 1)
            return httpx.Response(
                200, json={"deleted": deleted, "skipped": 0, "ok": True}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_delete_unknown_instance_returns_404():
    """Targeting an unregistered instance must 404 (before any dispatch)."""
    with _delete_client() as client:
        resp = client.post("/cache/delete", json=_delete_body("does-not-exist"))
        assert resp.status_code == 404


def test_delete_short_sequence_is_noop():
    """A sub-chunk sequence resolves to nothing and dispatches no delete."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_delete_server(deletes)

        body = _delete_body("mp-1")
        body["token_ids"] = [1, 2]  # shorter than one chunk (chunk_size=4)
        resp = client.post("/cache/delete", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "instance_id": "mp-1",
            "requested": 0,
            "affected": 0,
            "skipped": 0,
            "status": "noop",
        }
        assert deletes == []


def test_delete_invalid_cache_salt_returns_400():
    """A bad cache_salt fails resolution on the coordinator with a 400."""
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        resp = client.post("/cache/delete", json=_delete_body("mp-1", salt="bad@salt"))
        assert resp.status_code == 400


def test_delete_all_tier_single_call_both_tiers():
    """tier=all issues one DELETE /cache/objects that removes L1 and L2."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        ctx.outbound_client = _mock_delete_server(deletes)
        n = len(_resolve_delete(ctx))
        assert n >= 1

        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["requested"] == 2  # 2 chunks (chunk_size=4, 8 tokens)
        assert body["affected"] == 2 * n  # n L1 + n L2
        assert body["skipped"] == 0
        # Exactly one node call, carrying tier=all and every resolved key.
        assert len(deletes) == 1
        assert deletes[0]["tier"] == "all"
        assert deletes[0]["force"] is False
        assert len(deletes[0]["keys"]) == n


def test_delete_non_force_holds_back_l2_pinned_key():
    """Non-force delete drops an L2-pinned key from the (single) delete set."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        ctx.outbound_client = _mock_delete_server(deletes)
        keys = _resolve_delete(ctx)
        ctx.eviction_manager.pin([keys[0]])  # protect one key at L2

        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 200, resp.text
        # The pinned key is reported skipped and never dispatched (retained).
        assert resp.json()["skipped"] == 1
        assert len(deletes) == 1
        assert len(deletes[0]["keys"]) == len(keys) - 1
        # The pin survives (non-force does not drop it).
        assert ctx.eviction_manager.filter_unpinned([keys[0]]) == []


def test_delete_force_removes_and_drops_l2_pin():
    """Force delete removes even an L2-pinned key and purges the pin."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        ctx.outbound_client = _mock_delete_server(deletes)
        keys = _resolve_delete(ctx)
        ctx.eviction_manager.pin([keys[0]])

        resp = client.post("/cache/delete", json=_delete_body("mp-1", force=True))
        assert resp.status_code == 200, resp.text
        # Force dispatched every key despite the pin...
        assert len(deletes) == 1
        assert deletes[0]["force"] is True
        assert len(deletes[0]["keys"]) == len(keys)
        assert resp.json()["skipped"] == 0
        # ...and the coordinator dropped the L2 pin.
        assert ctx.eviction_manager.filter_unpinned([keys[0]]) == [keys[0]]


def test_delete_l1_tier_ignores_l2_pins():
    """tier=l1 dispatches with tier=l1 and does not filter or drop L2 pins."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        ctx.outbound_client = _mock_delete_server(deletes)
        keys = _resolve_delete(ctx)
        ctx.eviction_manager.pin([keys[0]])

        resp = client.post("/cache/delete", json=_delete_body("mp-1", tier="l1"))
        assert resp.status_code == 200, resp.text
        # tier=l1 does not consult L2 pins: every key is dispatched.
        assert len(deletes) == 1
        assert deletes[0]["tier"] == "l1"
        assert len(deletes[0]["keys"]) == len(keys)
        assert resp.json()["affected"] == len(keys)  # L1 only
        assert ctx.eviction_manager.filter_unpinned([keys[0]]) == []  # pin untouched


def test_delete_server_unreachable_returns_502():
    """A transport error talking to the MP server surfaces as 502."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 502
