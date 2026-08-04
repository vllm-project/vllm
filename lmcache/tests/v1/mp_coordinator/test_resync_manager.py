# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator L2 resync manager."""

# Standard
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.cache_control.eviction_manager import L2EvictionManager
from lmcache.v1.mp_coordinator.cache_control.resync_manager import L2ResyncManager
from lmcache.v1.mp_coordinator.cache_control.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance


def _make_components() -> tuple[L2UsageManager, L2EvictionManager, L2ResyncManager]:
    usage = L2UsageManager()
    eviction = L2EvictionManager(QuotaManager(), usage, eviction_ratio=1.0)
    resync = L2ResyncManager(usage, eviction, page_size=2)
    return usage, eviction, resync


def _instance(instance_id: str = "mp-1") -> MPInstance:
    now = time.time()
    return MPInstance(
        instance_id=instance_id,
        ip="10.0.0.1",
        http_port=8000,
        registration_time=now,
        last_heartbeat_time=now,
    )


def _entry(
    *,
    chunk_hash_hex: str = "aa",
    model_name: str = "llama",
    kv_rank: int = 0,
    object_group_id: int = 0,
    cache_salt: str = "alice",
    size_bytes: int = 100,
) -> dict[str, object]:
    """Wire shape of one ``KeyListPage`` entry: nested ``key`` plus
    a sibling ``size_bytes``."""
    return {
        "key": {
            "chunk_hash_hex": chunk_hash_hex,
            "model_name": model_name,
            "kv_rank": kv_rank,
            "object_group_id": object_group_id,
            "cache_salt": cache_salt,
        },
        "size_bytes": size_bytes,
    }


# =============================================================================
# resync_from — single instance, paginated walk
# =============================================================================


class TestResyncFrom:
    @pytest.mark.asyncio
    async def test_single_page_records_all_entries(self):
        usage, eviction, resync = _make_components()

        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://10.0.0.1:8000/cache/objects?page_size=2"
            return httpx.Response(
                200,
                json={
                    "entries": [
                        _entry(chunk_hash_hex="aa", size_bytes=100),
                        _entry(chunk_hash_hex="bb", size_bytes=200),
                    ],
                    "next_page_token": None,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            total = await resync.resync_from(_instance(), client)

        assert total == 2
        # Usage manager aggregated bytes per cache_salt.
        assert usage.get("alice") == 300
        # Eviction manager registered both keys — confirmed by
        # synthesizing an eviction (no quota ⇒ ratio=1.0 full sweep).
        eviction._quota_manager.set_quota("alice", 0)
        plan = eviction.compute_eviction_plan()
        assert len(plan.get("alice", [])) == 2

    @pytest.mark.asyncio
    async def test_paginates_through_continuation_tokens(self):
        usage, _eviction, resync = _make_components()

        pages = [
            {
                "entries": [_entry(chunk_hash_hex="aa", size_bytes=10)],
                "next_page_token": "T1",
            },
            {
                "entries": [_entry(chunk_hash_hex="bb", size_bytes=20)],
                "next_page_token": "T2",
            },
            {
                "entries": [_entry(chunk_hash_hex="cc", size_bytes=30)],
                "next_page_token": None,
            },
        ]
        seen_tokens: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            token = request.url.params.get("page_token")
            seen_tokens.append(token)
            return httpx.Response(200, json=pages.pop(0))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            total = await resync.resync_from(_instance(), client)

        assert total == 3
        # First page sends no token, follow-ups forward the previous
        # next_page_token verbatim.
        assert seen_tokens == [None, "T1", "T2"]
        assert usage.get("alice") == 60

    @pytest.mark.asyncio
    async def test_http_failure_stops_early_returns_partial_count(self):
        usage, _eviction, resync = _make_components()

        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "entries": [_entry(chunk_hash_hex="aa", size_bytes=10)],
                        "next_page_token": "T1",
                    },
                )
            return httpx.Response(503, json={"error": "down"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            total = await resync.resync_from(_instance(), client)

        # First page succeeded; second failed.
        assert total == 1
        assert usage.get("alice") == 10

    @pytest.mark.asyncio
    async def test_malformed_entries_skipped(self):
        usage, _eviction, resync = _make_components()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "entries": [
                        # Bad: ``@`` in model_name violates ObjectKey
                        # invariant → skipped.
                        _entry(model_name="bad@name", size_bytes=99),
                        # Bad: non-hex chunk_hash_hex → skipped.
                        _entry(chunk_hash_hex="not-hex", size_bytes=99),
                        # Bad: missing ``key`` field → skipped.
                        {"size_bytes": 99},
                        # Bad: missing ``size_bytes`` → skipped.
                        {
                            "key": {
                                "chunk_hash_hex": "aa",
                                "model_name": "m",
                                "kv_rank": 0,
                            }
                        },
                        # Good.
                        _entry(chunk_hash_hex="aa", size_bytes=50),
                    ],
                    "next_page_token": None,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            total = await resync.resync_from(_instance(), client)

        assert total == 1
        assert usage.get("alice") == 50

    @pytest.mark.asyncio
    async def test_constructor_rejects_non_positive_page_size(self):
        usage = L2UsageManager()
        eviction = L2EvictionManager(QuotaManager(), usage)
        with pytest.raises(ValueError):
            L2ResyncManager(usage, eviction, page_size=0)


# =============================================================================
# wait_and_resync — registry polling
# =============================================================================


class TestWaitAndResync:
    @pytest.mark.asyncio
    async def test_returns_zero_when_no_instance_within_max_wait(self):
        _usage, _eviction, resync = _make_components()
        registry = InstanceRegistry()

        # ``poll_interval`` and ``max_wait`` are both small enough that
        # the test finishes well under a second.
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: pytest.fail("HTTP must not be called")  # type: ignore[arg-type]
            )
        ) as client:
            total = await resync.wait_and_resync(
                registry=registry,
                http_client=client,
                poll_interval=0.05,
                max_wait=0.2,
            )

        assert total == 0

    @pytest.mark.asyncio
    async def test_resyncs_against_first_registered_instance(self):
        usage, _eviction, resync = _make_components()
        registry = InstanceRegistry()
        registry.register(_instance("mp-1"))

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "entries": [_entry(chunk_hash_hex="aa", size_bytes=25)],
                    "next_page_token": None,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            total = await resync.wait_and_resync(
                registry=registry,
                http_client=client,
                poll_interval=0.05,
                max_wait=1.0,
            )

        assert total == 1
        assert usage.get("alice") == 25
