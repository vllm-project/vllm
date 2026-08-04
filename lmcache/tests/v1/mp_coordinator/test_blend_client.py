# SPDX-License-Identifier: Apache-2.0
"""Tests for the blend-server side coordinator client.

The HTTP layer is replaced by an injected ``request_fn`` backed by a real
``GlobalBlendMatcher``, so the queue/daemon/poll state machine is exercised
end-to-end against the actual directory without a network or server.
"""

# Standard
from collections.abc import Callable
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.blend_client import (
    PENDING,
    BlendCoordinatorClient,
)
from lmcache.v1.mp_coordinator.blend_directory import (
    GlobalBlendMatcher,
    StoreRange,
)
from lmcache.v1.mp_coordinator.schemas import decode_tokens

CHUNK = 3
SCOPE = "model-a"


def _matcher_request(
    matcher: GlobalBlendMatcher,
) -> Callable[[str, str, dict], dict]:
    """request_fn that drives a real matcher, mirroring the coordinator router."""

    def request(method: str, path: str, payload: dict) -> dict:
        if method == "POST" and path == "/blend/fingerprints":
            ranges = [
                StoreRange(
                    model_scope=r["model_scope"],
                    tokens=r["tokens"],
                    object_keys=r["object_keys"],
                    old_st_base=r["old_st_base"],
                )
                for r in payload.get("ranges", [])
            ]
            return {"inserted": matcher.register(ranges) if ranges else 0}
        if method == "DELETE" and path == "/blend/fingerprints":
            keys = payload.get("object_keys", [])
            return {"removed": matcher.remove(keys) if keys else 0}
        if method == "POST" and path == "/blend/match":
            matches = matcher.match(
                payload["model_scope"], decode_tokens(payload["tokens_b64"])
            )
            return {
                "matches": [
                    {"object_key": m.object_key, "old_st": m.old_st, "cur_st": m.cur_st}
                    for m in matches
                ]
            }
        raise AssertionError(f"unexpected {method} {path}")

    return request


def _range(prefix: str, tokens: list[int]) -> dict:
    n_chunks = len(tokens) // CHUNK
    return {
        "model_scope": SCOPE,
        "tokens": tokens,
        "object_keys": [f"{prefix}{i}" for i in range(n_chunks)],
        "old_st_base": 0,
    }


def _store_range(prefix: str, tokens: list[int]) -> StoreRange:
    d = _range(prefix, tokens)
    return StoreRange(**d)


def _wait_match(client: BlendCoordinatorClient, rid: str, timeout: float = 2.0):
    end = time.time() + timeout
    while time.time() < end:
        v = client.poll_match(rid)
        if isinstance(v, list):
            return v
        time.sleep(0.005)
    return client.poll_match(rid)


def test_match_after_register():
    m = GlobalBlendMatcher(chunk_size=CHUNK)
    doc = [1, 2, 3, 4, 5, 6]
    m.register([_store_range("K", doc)])
    client = BlendCoordinatorClient(request_fn=_matcher_request(m))
    try:
        client.submit_match("r1", SCOPE, doc)
        matches = _wait_match(client, "r1")
        assert isinstance(matches, list)
        assert [(x.object_key, x.old_st, x.cur_st) for x in matches] == [
            ("K0", 0, 0),
            ("K1", 3, 3),
        ]
    finally:
        client.close()


def test_publish_reaches_matcher():
    m = GlobalBlendMatcher(chunk_size=CHUNK)
    client = BlendCoordinatorClient(request_fn=_matcher_request(m))
    try:
        doc = [1, 2, 3]
        client.enqueue_register([_range("K", doc)])
        end = time.time() + 2.0
        ok = False
        while time.time() < end:
            if m.match(SCOPE, doc):
                ok = True
                break
            time.sleep(0.005)
        assert ok
    finally:
        client.close()


def test_evict_reaches_matcher():
    m = GlobalBlendMatcher(chunk_size=CHUNK)
    doc = [1, 2, 3]
    m.register([_store_range("K", doc)])
    client = BlendCoordinatorClient(request_fn=_matcher_request(m))
    try:
        client.enqueue_evict(["K0"])
        end = time.time() + 2.0
        gone = False
        while time.time() < end:
            if not m.match(SCOPE, doc):
                gone = True
                break
            time.sleep(0.005)
        assert gone
    finally:
        client.close()


def test_poll_none_before_submit():
    client = BlendCoordinatorClient(request_fn=lambda mth, p, b: {})
    try:
        assert client.poll_match("never") is None
    finally:
        client.close()


def test_submit_is_idempotent():
    m = GlobalBlendMatcher(chunk_size=CHUNK)
    doc = [1, 2, 3]
    m.register([_store_range("K", doc)])
    client = BlendCoordinatorClient(request_fn=_matcher_request(m))
    try:
        client.submit_match("r1", SCOPE, doc)
        client.submit_match("r1", SCOPE, doc)  # no-op
        matches = _wait_match(client, "r1")
        assert isinstance(matches, list) and len(matches) == 1
    finally:
        client.close()


def test_match_error_degrades_to_empty():
    def boom(method: str, path: str, payload: dict) -> dict:
        raise RuntimeError("coordinator down")

    client = BlendCoordinatorClient(request_fn=boom)
    try:
        client.submit_match("r1", SCOPE, [1, 2, 3])
        matches = _wait_match(client, "r1")
        assert matches == []  # failure -> local-only, never hangs
    finally:
        client.close()


def test_take_match_clears():
    m = GlobalBlendMatcher(chunk_size=CHUNK)
    doc = [1, 2, 3]
    m.register([_store_range("K", doc)])
    client = BlendCoordinatorClient(request_fn=_matcher_request(m))
    try:
        client.submit_match("r1", SCOPE, doc)
        _wait_match(client, "r1")
        client.take_match("r1")
        assert client.poll_match("r1") is None
    finally:
        client.close()


def test_maybe_create(monkeypatch: pytest.MonkeyPatch):
    assert BlendCoordinatorClient.maybe_create("") is None
    assert BlendCoordinatorClient.maybe_create("   ") is None
    assert BlendCoordinatorClient.maybe_create(None) is None

    monkeypatch.delenv("LMCACHE_COORDINATOR_BLEND_TIMEOUT", raising=False)
    client = BlendCoordinatorClient.maybe_create("http://coord:9300")
    assert client is not None
    assert client.match_budget_s == 1.0  # default timeout
    client.close()

    monkeypatch.setenv("LMCACHE_COORDINATOR_BLEND_TIMEOUT", "1.5")
    client = BlendCoordinatorClient.maybe_create("http://coord:9300")
    assert client is not None
    assert client.match_budget_s == 1.5  # env override
    client.close()


def test_pending_sentinel_distinct():
    assert PENDING is not None and not isinstance(PENDING, list)
