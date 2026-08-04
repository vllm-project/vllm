# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for the MP server's ``/cache/*`` surface (``cache_api``).

Covers the object endpoints (``/cache/objects``), the
prefetch endpoints (``/cache/prefetches``), and the diagnostics endpoints
(``/cache/clear``, ``/cache/checksums``). Handlers are thin over the typed
services resolved from the app context, so these inject a fake engine via
``build_context`` and exercise the HTTP layer without a real cache engine.
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import KeyEntry, KeyListPage, ObjectKey
from lmcache.v1.multiprocess.cache_control.object_service import MAX_DELETE_BATCH
from lmcache.v1.multiprocess.http_apis.cache_api import router as cache_router
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.http_apis.error_handlers import register_error_handlers


@dataclass
class _FakeDescriptor:
    """Minimal descriptor — only ``type_name`` is read."""

    type_name: str = "s3"


@dataclass
class _FakeAdapter:
    """Records calls and serves canned responses for adapter methods."""

    delete_calls: list[list[ObjectKey]] = field(default_factory=list)
    delete_raises: Optional[BaseException] = None

    list_page: Optional[KeyListPage] = None
    list_raises: Optional[BaseException] = None
    last_list_kwargs: dict[str, object] = field(default_factory=dict)

    def delete(self, keys: list[ObjectKey]) -> None:
        self.delete_calls.append(list(keys))
        if self.delete_raises is not None:
            raise self.delete_raises

    def list_l2_keys(
        self,
        model_name: Optional[str] = None,
        page_size: int = 500,
        cursor: Optional[str] = None,
    ) -> KeyListPage:
        self.last_list_kwargs = {
            "model_name": model_name,
            "page_size": page_size,
            "cursor": cursor,
        }
        if self.list_raises is not None:
            raise self.list_raises
        return self.list_page or KeyListPage(entries=(), next_page_token=None)


@dataclass
class _FakeStorageManager:
    """Implements ``l2_adapters()`` and ``delete_l1_keys()``. An empty adapter
    list reproduces the "no L2 adapters configured" condition."""

    adapters: list[tuple[str, _FakeAdapter]] = field(default_factory=list)
    # Number of L1 keys the next delete_l1_keys call reports as locked/skipped.
    l1_skip: int = 0
    # Records (keys, force) for each delete_l1_keys call.
    l1_delete_calls: list[tuple[list, bool]] = field(default_factory=list)

    def l2_adapters(self) -> list[tuple[_FakeDescriptor, _FakeAdapter]]:
        return [(_FakeDescriptor(type_name=n), a) for n, a in self.adapters]

    def delete_l1_keys(self, keys: list, force: bool = False) -> tuple[int, int]:
        self.l1_delete_calls.append((list(keys), force))
        deleted = max(0, len(keys) - self.l1_skip)
        return deleted, min(self.l1_skip, len(keys))


class _FakeEngine:
    def __init__(self, sm: _FakeStorageManager):
        self.storage_manager = sm


def _make_app(sm: Optional[_FakeStorageManager]) -> FastAPI:
    """Build an app with the cache router and a context wrapping a fake engine
    (or no context, reproducing the not-initialized condition)."""
    app = FastAPI()
    app.include_router(cache_router)
    register_error_handlers(app)
    if sm is not None:
        app.state.context = build_context(_FakeEngine(sm))
    return app


def _hex(n: int, width: int = 4) -> str:
    return n.to_bytes(width, "big").hex()


def _sm_with(*entries: tuple[str, _FakeAdapter]) -> _FakeStorageManager:
    return _FakeStorageManager(adapters=list(entries))


# =============================================================================
# Delete objects
# =============================================================================


class TestDeleteObjectsEndpoint:
    def test_happy_path_defaults_to_primary(self):
        primary = _FakeAdapter()
        secondary = _FakeAdapter()
        sm = _sm_with(("s3", primary), ("fs", secondary))
        client = TestClient(_make_app(sm))

        resp = client.request(
            "DELETE",
            "/cache/objects",
            json={
                "keys": [
                    {
                        "chunk_hash_hex": _hex(1),
                        "model_name": "llama",
                        "kv_rank": 0,
                        "cache_salt": "alice",
                    },
                    {"chunk_hash_hex": _hex(2), "model_name": "llama", "kv_rank": 0},
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        # Default tier is l2: 2 keys deleted from the primary adapter, no L1.
        assert resp.json() == {"deleted": 2, "skipped": 0, "ok": True}
        assert sm.l1_delete_calls == []
        assert len(primary.delete_calls) == 1
        assert secondary.delete_calls == []
        forwarded = primary.delete_calls[0]
        assert forwarded[0] == ObjectKey(
            chunk_hash=b"\x00\x00\x00\x01",
            model_name="llama",
            kv_rank=0,
            cache_salt="alice",
        )
        assert forwarded[1].cache_salt == ""

    def test_adapter_in_body_selects_non_primary(self):
        primary = _FakeAdapter()
        secondary = _FakeAdapter()
        sm = _sm_with(("s3", primary), ("fs", secondary))
        client = TestClient(_make_app(sm))

        resp = client.request(
            "DELETE",
            "/cache/objects",
            json={
                "adapter": "fs",
                "keys": [{"chunk_hash_hex": _hex(1), "model_name": "m", "kv_rank": 0}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] == 1
        assert primary.delete_calls == []
        assert len(secondary.delete_calls) == 1

    def test_404_when_adapter_no_match(self):
        client = TestClient(_make_app(_sm_with(("s3", _FakeAdapter()))))
        resp = client.request(
            "DELETE", "/cache/objects", json={"adapter": "nope", "keys": []}
        )
        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    def test_propagates_adapter_failure_in_body(self):
        adapter = _FakeAdapter(delete_raises=RuntimeError("s3 down"))
        client = TestClient(_make_app(_sm_with(("s3", adapter))))
        resp = client.request("DELETE", "/cache/objects", json={"keys": []})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is False
        assert "s3 down" in body["error"]

    def test_tier_l1_deletes_l1_only(self):
        """tier=l1 deletes from L1 and never touches an L2 adapter."""
        adapter = _FakeAdapter()
        sm = _sm_with(("s3", adapter))
        client = TestClient(_make_app(sm))
        resp = client.request(
            "DELETE",
            "/cache/objects",
            json={
                "tier": "l1",
                "keys": [{"chunk_hash_hex": _hex(1), "model_name": "m", "kv_rank": 0}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted": 1, "skipped": 0, "ok": True}
        assert len(sm.l1_delete_calls) == 1
        assert adapter.delete_calls == []  # L2 untouched

    def test_tier_all_deletes_both_and_force_plumbs(self):
        """tier=all deletes L1 + L2 and forwards ``force`` to the L1 delete."""
        adapter = _FakeAdapter()
        sm = _sm_with(("s3", adapter))
        client = TestClient(_make_app(sm))
        resp = client.request(
            "DELETE",
            "/cache/objects",
            json={
                "tier": "all",
                "force": True,
                "keys": [{"chunk_hash_hex": _hex(1), "model_name": "m", "kv_rank": 0}],
            },
        )
        assert resp.status_code == 200, resp.text
        # 1 L1 key + 1 L2 key removed.
        assert resp.json() == {"deleted": 2, "skipped": 0, "ok": True}
        assert sm.l1_delete_calls[0][1] is True  # force forwarded
        assert len(adapter.delete_calls) == 1

    def test_tier_l1_needs_no_l2_adapter(self):
        """A pure L1 delete works on a server with no L2 adapters."""
        sm = _sm_with()  # no L2 adapters
        client = TestClient(_make_app(sm))
        resp = client.request(
            "DELETE",
            "/cache/objects",
            json={
                "tier": "l1",
                "keys": [{"chunk_hash_hex": _hex(1), "model_name": "m", "kv_rank": 0}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] == 1

    def test_503_when_no_adapters_configured(self):
        """A delete that touches L2 needs an adapter (default tier is l2)."""
        client = TestClient(_make_app(_sm_with()))
        resp = client.request("DELETE", "/cache/objects", json={"keys": []})
        assert resp.status_code == 503
        assert "no L2 adapters" in resp.json()["detail"]

    def test_503_when_not_initialized(self):
        client = TestClient(_make_app(None))
        assert (
            client.request("DELETE", "/cache/objects", json={"keys": []}).status_code
            == 503
        )

    @pytest.mark.parametrize(
        "body",
        [
            "not-json-text",
            {},
            {"keys": "not-a-list"},
            {"keys": [{"chunk_hash_hex": _hex(1), "kv_rank": 0}]},
            {"keys": [{"chunk_hash_hex": _hex(1), "model_name": "m", "kv_rank": "x"}]},
        ],
    )
    def test_422_on_validation_failure(self, body):
        adapter = _FakeAdapter()
        client = TestClient(_make_app(_sm_with(("s3", adapter))))
        if isinstance(body, str):
            resp = client.request(
                "DELETE",
                "/cache/objects",
                content=body,
                headers={"content-type": "application/json"},
            )
        else:
            resp = client.request("DELETE", "/cache/objects", json=body)
        assert resp.status_code == 422, resp.text
        assert adapter.delete_calls == []

    @pytest.mark.parametrize(
        "body",
        [
            {"keys": [{"chunk_hash_hex": "zz", "model_name": "m", "kv_rank": 0}]},
            {
                "keys": [
                    {"chunk_hash_hex": _hex(1), "model_name": "bad@name", "kv_rank": 0}
                ]
            },
        ],
    )
    def test_400_on_object_key_invariant_violation(self, body):
        adapter = _FakeAdapter()
        client = TestClient(_make_app(_sm_with(("s3", adapter))))
        resp = client.request("DELETE", "/cache/objects", json=body)
        assert resp.status_code == 400, resp.text
        assert adapter.delete_calls == []

    def test_400_when_batch_exceeds_cap(self):
        adapter = _FakeAdapter()
        client = TestClient(_make_app(_sm_with(("s3", adapter))))
        oversized = [
            {"chunk_hash_hex": _hex(i), "model_name": "m", "kv_rank": 0}
            for i in range(MAX_DELETE_BATCH + 1)
        ]
        resp = client.request("DELETE", "/cache/objects", json={"keys": oversized})
        assert resp.status_code == 400, resp.text
        assert "too many keys" in resp.json()["detail"]
        assert adapter.delete_calls == []


# =============================================================================
# List objects
# =============================================================================


class TestListObjectsEndpoint:
    def test_happy_path_defaults_to_primary(self):
        k1 = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=2,
            cache_salt="alice",
        )
        primary = _FakeAdapter(
            list_page=KeyListPage(
                entries=(KeyEntry(key=k1.to_encoded_object_key(), size_bytes=4096),),
                next_page_token="opaque-cursor",
            )
        )
        secondary = _FakeAdapter()
        sm = _sm_with(("s3", primary), ("fs", secondary))
        client = TestClient(_make_app(sm))

        resp = client.get(
            "/cache/objects", params={"model_name": "llama", "page_size": 100}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["adapter"] == "s3"
        assert body["entries"] == [
            {
                "key": {
                    "chunk_hash_hex": "deadbeef",
                    "model_name": "llama",
                    "kv_rank": 2,
                    "object_group_id": 0,
                    "cache_salt": "alice",
                },
                "size_bytes": 4096,
            }
        ]
        assert body["next_page_token"] == "opaque-cursor"
        assert primary.last_list_kwargs == {
            "model_name": "llama",
            "page_size": 100,
            "cursor": None,
        }
        assert secondary.last_list_kwargs == {}

    def test_adapter_param_selects_non_primary(self):
        primary = _FakeAdapter()
        secondary = _FakeAdapter(
            list_page=KeyListPage(entries=(), next_page_token="from-fs")
        )
        sm = _sm_with(("s3", primary), ("fs", secondary))
        client = TestClient(_make_app(sm))
        resp = client.get("/cache/objects", params={"adapter": "fs"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["adapter"] == "fs"
        assert body["next_page_token"] == "from-fs"
        assert primary.last_list_kwargs == {}

    def test_404_when_adapter_no_match(self):
        client = TestClient(_make_app(_sm_with(("s3", _FakeAdapter()))))
        assert (
            client.get("/cache/objects", params={"adapter": "nope"}).status_code == 404
        )

    def test_page_token_threads_through_as_cursor(self):
        adapter = _FakeAdapter()
        client = TestClient(_make_app(_sm_with(("s3", adapter))))
        client.get("/cache/objects", params={"page_token": "abc"})
        assert adapter.last_list_kwargs["cursor"] == "abc"

    def test_400_on_unsupported_tier(self):
        client = TestClient(_make_app(_sm_with(("s3", _FakeAdapter()))))
        resp = client.get("/cache/objects", params={"tier": "l1"})
        assert resp.status_code == 400

    def test_503_when_no_adapters_configured(self):
        client = TestClient(_make_app(_sm_with()))
        assert client.get("/cache/objects").status_code == 503

    def test_503_when_listing_unsupported(self):
        adapter = _FakeAdapter(list_raises=NotImplementedError("no listing"))
        client = TestClient(_make_app(_sm_with(("fs", adapter))))
        resp = client.get("/cache/objects")
        assert resp.status_code == 503
        assert "does not support listing" in resp.json()["detail"]

    def test_400_on_malformed_page_token(self):
        adapter = _FakeAdapter(list_raises=ValueError("malformed cursor"))
        client = TestClient(_make_app(_sm_with(("s3", adapter))))
        assert (
            client.get("/cache/objects", params={"page_token": "garbage"}).status_code
            == 400
        )

    def test_503_when_not_initialized(self):
        assert TestClient(_make_app(None)).get("/cache/objects").status_code == 503


# =============================================================================
# Prefetch
# =============================================================================


@dataclass
class _FakeLayoutRegistry:
    layout: Optional[object] = None
    find_calls: list[tuple[str, int]] = field(default_factory=list)

    def find(self, model_name: str, world_size: int) -> Optional[object]:
        self.find_calls.append((model_name, world_size))
        return self.layout


class _PrefetchHandle:
    def __init__(self, total: int) -> None:
        self.total_requested_keys = total


class _PrefetchBitmap:
    def __init__(self, n: int) -> None:
        self._n = n

    def popcount(self) -> int:
        return self._n


@dataclass
class _PrefetchStorageManager:
    submit_calls: list[dict] = field(default_factory=list)

    def submit_prefetch_task(self, spec, **_) -> _PrefetchHandle:
        self.submit_calls.append({"keys": list(spec.keys), "mode": spec.mode})
        return _PrefetchHandle(len(spec.keys))

    def query_prefetch_status(self, handle) -> _PrefetchBitmap:
        return _PrefetchBitmap(handle.total_requested_keys)


@dataclass
class _FakeTokenHasher:
    chunk_size: int = 4

    def compute_chunk_hashes(self, token_ids: list[int]) -> list[bytes]:
        n = len(token_ids) // self.chunk_size
        return [i.to_bytes(4, "big") for i in range(n)]


@dataclass
class _FakeContext:
    layout_desc_registry: _FakeLayoutRegistry
    storage_manager: _PrefetchStorageManager
    token_hasher: _FakeTokenHasher = field(default_factory=_FakeTokenHasher)


class _PrefetchEngine:
    def __init__(self, ctx: _FakeContext):
        self.context = ctx
        self.storage_manager = ctx.storage_manager


def _make_prefetch_app(ctx: Optional[_FakeContext]) -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    register_error_handlers(app)
    if ctx is not None:
        app.state.context = build_context(_PrefetchEngine(ctx))
    return app


def _ctx(layout: Optional[object]) -> _FakeContext:
    return _FakeContext(
        layout_desc_registry=_FakeLayoutRegistry(layout=layout),
        storage_manager=_PrefetchStorageManager(),
    )


def _prefetch_body(token_ids: list[int], world_size: int = 2, salt: str = "") -> dict:
    return {
        "model_name": "m",
        "world_size": world_size,
        "token_ids": token_ids,
        "cache_salt": salt,
    }


class TestPrefetchEndpoint:
    def test_submit_returns_request_id_and_resolves_layout(self):
        ctx = _ctx(layout=object())
        client = TestClient(_make_prefetch_app(ctx))
        resp = client.post(
            "/cache/prefetches",
            json=_prefetch_body([1, 2, 3, 4, 5, 6, 7, 8], world_size=2),
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["chunks"] == 2
        assert body["request_id"]
        assert ("m", 2) in ctx.layout_desc_registry.find_calls

    def test_status_poll_completes_then_404(self):
        ctx = _ctx(layout=object())
        client = TestClient(_make_prefetch_app(ctx))
        rid = client.post(
            "/cache/prefetches",
            json=_prefetch_body([1, 2, 3, 4, 5, 6, 7, 8], world_size=2),
        ).json()["request_id"]

        resp = client.get(f"/cache/prefetches/{rid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "completed"
        assert body["found_keys"] == 4
        assert body["total_keys"] == 4
        assert client.get(f"/cache/prefetches/{rid}").status_code == 404

    def test_status_unknown_request_id_404(self):
        client = TestClient(_make_prefetch_app(_ctx(layout=object())))
        assert client.get("/cache/prefetches/nope").status_code == 404

    def test_short_sequence_is_noop(self):
        client = TestClient(_make_prefetch_app(_ctx(layout=object())))
        resp = client.post("/cache/prefetches", json=_prefetch_body([1, 2]))
        assert resp.status_code == 202, resp.text
        assert resp.json() == {"chunks": 0, "status": "noop"}

    def test_503_when_layout_not_registered(self):
        client = TestClient(_make_prefetch_app(_ctx(layout=None)))
        resp = client.post(
            "/cache/prefetches", json=_prefetch_body([1, 2, 3, 4], world_size=99)
        )
        assert resp.status_code == 503

    def test_400_on_invalid_cache_salt(self):
        client = TestClient(_make_prefetch_app(_ctx(layout=object())))
        resp = client.post(
            "/cache/prefetches", json=_prefetch_body([1, 2, 3, 4], salt="bad@salt")
        )
        assert resp.status_code == 400

    def test_400_on_unsupported_direction(self):
        client = TestClient(_make_prefetch_app(_ctx(layout=object())))
        body = _prefetch_body([1, 2, 3, 4])
        body["target_tier"] = "l2"
        resp = client.post("/cache/prefetches", json=body)
        assert resp.status_code == 400

    def test_503_when_not_initialized(self):
        client = TestClient(_make_prefetch_app(None))
        assert (
            client.post(
                "/cache/prefetches", json=_prefetch_body([1, 2, 3, 4])
            ).status_code
            == 503
        )


# =============================================================================
# Diagnostics (clear)
# =============================================================================


@dataclass
class _ClearEngine:
    clear_calls: int = 0
    cache_contexts: Optional[dict] = None

    def clear(self) -> None:
        self.clear_calls += 1


def _make_clear_app(engine: Optional[_ClearEngine]) -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    register_error_handlers(app)
    if engine is not None:
        app.state.context = build_context(engine)
    return app


class TestClearEndpoint:
    def test_clear_l1(self):
        engine = _ClearEngine()
        client = TestClient(_make_clear_app(engine))
        resp = client.post("/cache/clear", json={"tier": "l1"})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok", "cleared": {"tier": "l1"}}
        assert engine.clear_calls == 1

    def test_clear_no_body_defaults_to_l1(self):
        """The body is optional; an absent body defaults to tier l1."""
        engine = _ClearEngine()
        client = TestClient(_make_clear_app(engine))
        resp = client.post("/cache/clear")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok", "cleared": {"tier": "l1"}}
        assert engine.clear_calls == 1

    def test_clear_unsupported_tier(self):
        client = TestClient(_make_clear_app(_ClearEngine()))
        resp = client.post("/cache/clear", json={"tier": "l2"})
        assert resp.status_code == 400

    def test_clear_503_when_not_initialized(self):
        assert (
            TestClient(_make_clear_app(None)).post("/cache/clear", json={}).status_code
            == 503
        )

    def test_checksums_503_when_not_initialized(self):
        resp = TestClient(_make_clear_app(None)).post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        assert resp.status_code == 503

    def test_checksums_501_when_engine_unsupported(self):
        # _ClearEngine has cache_contexts=None -> 501.
        client = TestClient(_make_clear_app(_ClearEngine()))
        resp = client.post("/cache/checksums", json={"block_ids": [0], "chunk_size": 1})
        assert resp.status_code == 501
