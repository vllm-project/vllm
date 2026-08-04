# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/quota`` REST API (quota writes, combined
quota+usage status, and usage-event ingestion)."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return TestClient(create_app(config))


def _key(salt: str, h: str = "aa", model: str = "m", rank: int = 0) -> dict:
    return {
        "chunk_hash_hex": h,
        "model_name": model,
        "kv_rank": rank,
        "cache_salt": salt,
    }


def _store(salt: str, nbytes: int, **kw) -> dict:
    return {"type": "store", "key": _key(salt, **kw), "bytes": nbytes}


def _lookup(salt: str, **kw) -> dict:
    return {"type": "lookup", "key": _key(salt, **kw), "bytes": 0}


def _delete(salt: str, **kw) -> dict:
    return {"type": "delete", "key": _key(salt, **kw), "bytes": 0}


_seq_counter = 0


def _events_body(events: list[dict], instance_id: str = "test-server") -> dict:
    global _seq_counter
    _seq_counter += 1
    return {"instance_id": instance_id, "seq": _seq_counter, "events": events}


# -- Quota writes ------------------------------------------------------------


def test_set_quota():
    with _client() as client:
        resp = client.put("/quota/user-a", json={"limit_gb": 2.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_salt"] == "user-a"
        assert data["limit_gb"] == 2.5
        assert data["status"] == "ok"


def test_update_quota():
    with _client() as client:
        client.put("/quota/user-a", json={"limit_gb": 1.0})
        client.put("/quota/user-a", json={"limit_gb": 5.0})
        data = client.get("/quota/user-a").json()
        assert abs(data["quota_limit_gb"] - 5.0) < 1e-6


def test_delete_quota():
    with _client() as client:
        client.put("/quota/user-a", json={"limit_gb": 1.0})
        resp = client.delete("/quota/user-a")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

        data = client.get("/quota/user-a").json()
        assert data["quota_exists"] is False


def test_delete_nonexistent_quota():
    with _client() as client:
        resp = client.delete("/quota/unknown")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"


def test_negative_limit_rejected():
    with _client() as client:
        resp = client.put("/quota/user-a", json={"limit_gb": -1.0})
        assert resp.status_code == 422


def test_missing_body_rejected():
    with _client() as client:
        resp = client.put("/quota/user-a")
        assert resp.status_code == 422


def test_zero_limit_accepted():
    with _client() as client:
        resp = client.put("/quota/user-a", json={"limit_gb": 0.0})
        assert resp.status_code == 200
        data = client.get("/quota/user-a").json()
        assert data["quota_exists"] is True
        assert data["quota_limit_gb"] == 0.0


# -- Quota config (default limit for unquota'd salts) --------------------------


def test_quota_config_defaults_to_null():
    """Boot state: no default configured — unquota'd salts are exempt."""
    with _client() as client:
        resp = client.get("/quota/config")
        assert resp.status_code == 200
        assert resp.json() == {"default_limit_gb": None}


def test_quota_config_set_and_read_back():
    with _client() as client:
        resp = client.put("/quota/config", json={"default_limit_gb": 0})
        assert resp.status_code == 200
        assert resp.json() == {"default_limit_gb": 0.0}
        assert client.get("/quota/config").json() == {"default_limit_gb": 0.0}


def test_quota_config_positive_value_round_trips():
    with _client() as client:
        client.put("/quota/config", json={"default_limit_gb": 2.5})
        data = client.get("/quota/config").json()
        assert abs(data["default_limit_gb"] - 2.5) < 1e-6


def test_quota_config_resettable_to_null():
    with _client() as client:
        client.put("/quota/config", json={"default_limit_gb": 0})
        resp = client.put("/quota/config", json={"default_limit_gb": None})
        assert resp.status_code == 200
        assert resp.json() == {"default_limit_gb": None}


def test_quota_config_negative_rejected():
    with _client() as client:
        resp = client.put("/quota/config", json={"default_limit_gb": -1.0})
        assert resp.status_code == 422


def test_quota_config_path_not_captured_as_salt():
    """``config`` is a fixed route, not a ``cache_salt`` — setting the
    default must not create a quota entry named ``config``."""
    with _client() as client:
        client.put("/quota/config", json={"default_limit_gb": 1.0})
        assert client.get("/quota/config").json() == {"default_limit_gb": 1.0}
        salts = {e["cache_salt"] for e in client.get("/quota").json()["by_cache_salt"]}
        assert "config" not in salts


def test_quota_config_arms_unquotad_eviction_flow():
    """End-to-end controller flow: unquota'd usage is exempt until the
    default flips to 0, while explicit quotas work throughout."""
    with _client() as client:
        # Usage arrives for two tenants; only user-a gets a quota.
        client.post(
            "/quota/events",
            json=_events_body(
                [
                    _store("user-a", 1000, h="01"),
                    _store("user-b", 2000, h="02"),
                ]
            ),
        )
        client.put("/quota/user-a", json={"limit_gb": 10.0})

        # Boot state: default null — user-b is exempt (nothing to assert
        # via HTTP beyond config state; eviction-plan behavior is covered
        # in test_eviction_manager.py).
        assert client.get("/quota/config").json() == {"default_limit_gb": None}

        # Controller arms allowlist enforcement.
        resp = client.put("/quota/config", json={"default_limit_gb": 0})
        assert resp.json() == {"default_limit_gb": 0.0}


# -- Usage event ingestion ---------------------------------------------------


def test_report_store_events():
    with _client() as client:
        resp = client.post(
            "/quota/events",
            json=_events_body(
                [
                    _store("user-a", 1000, h="01"),
                    _store("user-a", 500, h="02"),
                    _store("user-b", 2000, h="03"),
                ]
            ),
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] == 3

        data = client.get("/quota/user-a").json()
        assert abs(data["usage_gb"] - 1500 / 1024**3) < 1e-12

        data = client.get("/quota/user-b").json()
        assert abs(data["usage_gb"] - 2000 / 1024**3) < 1e-12


def test_report_lookup_events_accepted():
    with _client() as client:
        resp = client.post(
            "/quota/events",
            json=_events_body([_lookup("user-a")]),
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] == 1


def test_empty_events_batch():
    with _client() as client:
        resp = client.post(
            "/quota/events",
            json=_events_body([]),
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] == 0


def test_invalid_event_type_rejected():
    with _client() as client:
        resp = client.post(
            "/quota/events",
            json=_events_body([{"type": "purge", "key": _key("a"), "bytes": 100}]),
        )
        assert resp.status_code == 422


def test_delete_event_drops_key_from_tracking():
    """A DELETE event subtracts the key's bytes from per-salt usage and
    removes it from the eviction LRU. The keys' sizes come from the
    earlier STORE events the usage manager has on file."""
    with _client() as client:
        # Seed two keys for "user-a".
        client.post(
            "/quota/events",
            json=_events_body(
                [
                    _store("user-a", 1000, h="01"),
                    _store("user-a", 500, h="02"),
                ]
            ),
        )
        data = client.get("/quota/user-a").json()
        assert abs(data["usage_gb"] - 1500 / 1024**3) < 1e-12

        # Delete one of them — usage shrinks by exactly that key's
        # recorded size (1000), not the wire ``bytes=0``.
        resp = client.post(
            "/quota/events",
            json=_events_body([_delete("user-a", h="01")]),
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] == 1

        data = client.get("/quota/user-a").json()
        assert abs(data["usage_gb"] - 500 / 1024**3) < 1e-12


def test_delete_event_for_unknown_key_is_noop():
    """A DELETE for a key the coordinator never saw a STORE for is
    accepted but has no observable effect (no usage to subtract from)."""
    with _client() as client:
        resp = client.post(
            "/quota/events",
            json=_events_body([_delete("user-a", h="ff")]),
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] == 1
        data = client.get("/quota/user-a").json()
        assert data["usage_gb"] == 0.0


def test_negative_bytes_rejected():
    with _client() as client:
        resp = client.post(
            "/quota/events",
            json=_events_body([{"type": "store", "key": _key("a"), "bytes": -1}]),
        )
        assert resp.status_code == 422


# -- Combined status queries -------------------------------------------------


def test_status_single_salt():
    with _client() as client:
        client.put("/quota/user-a", json={"limit_gb": 2.5})
        client.post(
            "/quota/events",
            json=_events_body([_store("user-a", 1000)]),
        )
        data = client.get("/quota/user-a").json()
        assert data["cache_salt"] == "user-a"
        assert abs(data["quota_limit_gb"] - 2.5) < 1e-6
        assert data["quota_exists"] is True
        assert abs(data["usage_gb"] - 1000 / 1024**3) < 1e-12


def test_status_unknown_salt():
    with _client() as client:
        data = client.get("/quota/unknown").json()
        assert data["usage_gb"] == 0.0
        assert data["quota_exists"] is False
        assert data["quota_limit_gb"] == 0.0


def test_status_list():
    with _client() as client:
        client.put("/quota/a", json={"limit_gb": 1.0})
        client.post(
            "/quota/events",
            json=_events_body(
                [
                    _store("a", 100, h="01"),
                    _store("b", 200, h="02"),
                ]
            ),
        )
        data = client.get("/quota").json()
        assert abs(data["total_gb"] - 300 / 1024**3) < 1e-12
        by_salt = {e["cache_salt"]: e for e in data["by_cache_salt"]}
        assert abs(by_salt["a"]["usage_gb"] - 100 / 1024**3) < 1e-12
        assert by_salt["a"]["quota_exists"] is True
        assert abs(by_salt["b"]["usage_gb"] - 200 / 1024**3) < 1e-12
        assert by_salt["b"]["quota_exists"] is False


def test_status_list_empty():
    with _client() as client:
        data = client.get("/quota").json()
        assert data["total_gb"] == 0.0
        assert data["by_cache_salt"] == []


def test_status_list_includes_quota_only_salt():
    """A salt with a quota but no usage should appear in the list."""
    with _client() as client:
        client.put("/quota/q-only", json={"limit_gb": 5.0})
        data = client.get("/quota").json()
        by_salt = {e["cache_salt"]: e for e in data["by_cache_salt"]}
        assert "q-only" in by_salt
        assert by_salt["q-only"]["quota_exists"] is True
        assert by_salt["q-only"]["usage_gb"] == 0.0


def test_default_salt_sentinel():
    """``_default`` in path maps to the empty-string salt."""
    with _client() as client:
        client.put("/quota/_default", json={"limit_gb": 3.0})
        client.post(
            "/quota/events",
            json=_events_body([_store("", 500)]),
        )
        data = client.get("/quota/_default").json()
        assert data["cache_salt"] == ""
        assert data["quota_exists"] is True
        assert abs(data["quota_limit_gb"] - 3.0) < 1e-6
        assert abs(data["usage_gb"] - 500 / 1024**3) < 1e-12
