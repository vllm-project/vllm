# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/directory`` REST API (cache-event
ingestion, placement lookup, and stats)."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return TestClient(create_app(config))


def _key(h: str = "aa", model: str = "m", rank: int = 0, salt: str = "") -> dict:
    return {
        "chunk_hash_hex": h,
        "model_name": model,
        "kv_rank": rank,
        "cache_salt": salt,
    }


def _batch(
    instance_id: str = "node-a",
    incarnation: int = 1,
    seq: int = 1,
    event_type: str = "store",
    tier: str = "l1",
    backend: str = "dram",
    entries: list[dict] | None = None,
) -> dict:
    if entries is None:
        entries = [{"key": _key(), "size_bytes": 1024}]
    return {
        "instance_id": instance_id,
        "incarnation": incarnation,
        "seq": seq,
        "event_type": event_type,
        "tier": tier,
        "backend": backend,
        "entries": entries,
    }


def _post_events(client: TestClient, batches: list[dict]) -> dict:
    resp = client.post("/directory/events", json={"batches": batches})
    assert resp.status_code == 200
    return resp.json()


def _lookup(client: TestClient, keys: list[dict]) -> dict:
    resp = client.post("/directory/lookup", json={"keys": keys})
    assert resp.status_code == 200
    return resp.json()


# -- Events + lookup ---------------------------------------------------------


def test_store_events_then_lookup():
    with _client() as client:
        data = _post_events(client, [_batch()])
        assert data == {"applied": 1, "duplicates": 0, "stale": 0}

        result = _lookup(client, [_key()])["results"]
        assert len(result) == 1
        assert result[0]["key"]["chunk_hash_hex"] == "aa"
        [placement] = result[0]["placements"]
        assert placement["instance_id"] == "node-a"
        assert placement["incarnation"] == 1
        assert placement["tier"] == "l1"
        assert placement["backend"] == "dram"
        assert placement["size_bytes"] == 1024


def test_lookup_unknown_key_returns_empty_placements():
    with _client() as client:
        result = _lookup(client, [_key(h="ff")])["results"]
        assert result[0]["key"]["chunk_hash_hex"] == "ff"
        assert result[0]["placements"] == []


def test_delete_event_removes_placement():
    with _client() as client:
        _post_events(client, [_batch(seq=1)])
        _post_events(
            client,
            [_batch(seq=2, event_type="delete", entries=[{"key": _key()}])],
        )
        result = _lookup(client, [_key()])["results"]
        assert result[0]["placements"] == []


def test_duplicate_and_stale_batches_are_counted():
    with _client() as client:
        _post_events(client, [_batch(incarnation=2, seq=1)])
        data = _post_events(
            client,
            [
                _batch(incarnation=2, seq=1),  # replay -> duplicate
                _batch(incarnation=1, seq=9),  # pre-restart -> stale
                _batch(incarnation=2, seq=2),  # fresh -> applied
            ],
        )
        assert data == {"applied": 1, "duplicates": 1, "stale": 1}


# -- Token -> placement lookup -------------------------------------------------


def _lookup_tokens_body(n_tokens: int, model: str = "m") -> dict:
    return {
        "model_name": model,
        "world_size": 1,
        "token_ids": list(range(n_tokens)),
        "cache_salt": "",
    }


def test_lookup_tokens_short_sequence_resolves_no_chunks():
    with _client() as client:
        resp = client.post("/directory/lookup_tokens", json=_lookup_tokens_body(10))
        assert resp.status_code == 200
        assert resp.json() == {"chunks": 0, "results": []}


def test_lookup_tokens_roundtrip():
    with _client() as client:
        # Resolve one full chunk; nothing stored yet -> empty placements.
        first = client.post(
            "/directory/lookup_tokens", json=_lookup_tokens_body(256)
        ).json()
        assert first["chunks"] == 1
        assert len(first["results"]) == 1
        assert first["results"][0]["placements"] == []

        # Store the exact key the resolution produced, then look up again.
        key = first["results"][0]["key"]
        _post_events(client, [_batch(entries=[{"key": key, "size_bytes": 64}])])

        second = client.post(
            "/directory/lookup_tokens", json=_lookup_tokens_body(256)
        ).json()
        [placement] = second["results"][0]["placements"]
        assert placement["instance_id"] == "node-a"
        assert placement["size_bytes"] == 64


def test_lookup_tokens_invalid_model_name_is_400():
    with _client() as client:
        resp = client.post(
            "/directory/lookup_tokens", json=_lookup_tokens_body(256, model="a@b")
        )
        assert resp.status_code == 400


# -- Stats -------------------------------------------------------------------


def test_stats_reports_counts_and_gap_flag():
    with _client() as client:
        _post_events(client, [_batch(seq=1)])
        _post_events(client, [_batch(seq=5, entries=[{"key": _key(h="bb")}])])

        data = client.get("/directory/stats").json()
        assert data["num_keys"] == 2
        assert data["num_placements"] == 2
        instance = data["instances"]["node-a"]
        assert instance["incarnation"] == 1
        assert instance["last_seq"] == 5
        assert instance["gap_detected"] is True
        assert instance["num_keys"] == 2


# -- Request validation ------------------------------------------------------


def test_tier_all_is_rejected():
    with _client() as client:
        resp = client.post("/directory/events", json={"batches": [_batch(tier="all")]})
        assert resp.status_code == 422


def test_seq_zero_is_rejected():
    with _client() as client:
        resp = client.post("/directory/events", json={"batches": [_batch(seq=0)]})
        assert resp.status_code == 422


def test_malformed_key_hex_is_rejected():
    with _client() as client:
        resp = client.post(
            "/directory/events",
            json={"batches": [_batch(entries=[{"key": _key(h="zz")}])]},
        )
        assert resp.status_code == 422


def test_malformed_content_hash_is_rejected():
    with _client() as client:
        resp = client.post(
            "/directory/events",
            json={
                "batches": [
                    _batch(entries=[{"key": _key(), "content_hash_hex": "xyz"}])
                ]
            },
        )
        assert resp.status_code == 422
