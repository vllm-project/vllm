# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the quota CRUD HTTP endpoints.

Uses ``fastapi.testclient.TestClient`` against the module's ``app``
with a hand-built stub engine on ``app.state.engine`` — bypasses the
lifespan so we don't spin up the full ZMQ cache server. The stub
exposes just the two attributes the endpoints touch:
``storage_manager.quota_manager`` and
``storage_manager.get_usage_bytes_by_cache_salt()``.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.multiprocess.http_server import app


@pytest.fixture
def client():
    """TestClient with a stub engine wired to app.state.

    Each test gets a fresh QuotaManager + usage map — there is no
    cross-test state leakage because the fixture reassigns both.
    """
    usage_map: dict[str, int] = {}

    stub_engine = MagicMock()
    stub_engine.storage_manager.quota_manager = QuotaManager()
    stub_engine.storage_manager.get_usage_bytes_by_cache_salt.return_value = usage_map

    app.state.engine = stub_engine
    # ``TestClient`` without the context manager skips lifespan startup/shutdown,
    # which is exactly what we want here.
    c = TestClient(app)
    yield c, stub_engine, usage_map
    # Clear the engine so later tests don't accidentally reuse the stub.
    if hasattr(app.state, "engine"):
        delattr(app.state, "engine")


@pytest.fixture
def no_engine_client():
    """Client with no engine on ``app.state`` — exercises the 503 path."""
    if hasattr(app.state, "engine"):
        delattr(app.state, "engine")
    return TestClient(app)


# ---------------------------------------------------------------------------
# PUT /quota/{salt}
# ---------------------------------------------------------------------------


class TestPutQuota:
    def test_set_quota_for_named_salt(self, client):
        c, engine, _ = client
        resp = c.put("/quota/alice", json={"limit_gb": 2.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"cache_salt": "alice", "limit_gb": 2.0, "status": "ok"}
        assert engine.storage_manager.quota_manager.get_limit_bytes("alice") == int(
            2.0 * (1024**3)
        )

    def test_set_quota_sentinel_maps_to_empty_salt(self, client):
        """`_default` in the URL must resolve to the empty-string salt
        in the registry — un-salted traffic doesn't have its own
        distinct path parameter."""
        c, engine, _ = client
        resp = c.put("/quota/_default", json={"limit_gb": 1.0})
        assert resp.status_code == 200
        assert engine.storage_manager.quota_manager.has_quota("")
        assert not engine.storage_manager.quota_manager.has_quota("_default")
        assert resp.json()["cache_salt"] == "_default"

    def test_overwrite_existing_quota(self, client):
        c, engine, _ = client
        c.put("/quota/alice", json={"limit_gb": 1.0})
        resp = c.put("/quota/alice", json={"limit_gb": 5.0})
        assert resp.status_code == 200
        assert engine.storage_manager.quota_manager.get_limit_bytes("alice") == int(
            5.0 * (1024**3)
        )

    def test_missing_limit_gb_is_400(self, client):
        c, _, _ = client
        resp = c.put("/quota/alice", json={})
        assert resp.status_code == 400
        assert "limit_gb" in resp.json()["error"]

    def test_non_numeric_limit_is_400(self, client):
        c, _, _ = client
        resp = c.put("/quota/alice", json={"limit_gb": "huge"})
        assert resp.status_code == 400
        assert "numeric" in resp.json()["error"]

    def test_negative_limit_is_400(self, client):
        c, _, _ = client
        resp = c.put("/quota/alice", json={"limit_gb": -1.0})
        assert resp.status_code == 400
        assert "non-negative" in resp.json()["error"]

    def test_nan_limit_is_400(self, client):
        """``nan`` would propagate to ``int()`` below and raise a 500;
        reject it cleanly at parse time.

        Strict JSON doesn't allow ``NaN``, but Python's ``json.loads``
        accepts it as an extension — so a client that uses the same
        library (including anyone who calls ``json.dumps(allow_nan=True)``)
        can send it. Use a raw ``content=`` body to bypass the
        strict-JSON serializer in the test client.
        """
        c, _, _ = client
        resp = c.put(
            "/quota/alice",
            content=b'{"limit_gb": NaN}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert "finite" in resp.json()["error"]

    def test_inf_limit_is_400(self, client):
        """Same rationale as NaN — ``Infinity`` is a Python json
        extension, so a malicious client can still send it."""
        c, _, _ = client
        resp = c.put(
            "/quota/alice",
            content=b'{"limit_gb": Infinity}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert "finite" in resp.json()["error"]

    def test_zero_limit_is_accepted(self, client):
        """Zero is a valid limit — it registers the salt but evaluates
        to ``0`` bytes (same effective behavior as no entry, but the
        entry shows up in list_quotas)."""
        c, engine, _ = client
        resp = c.put("/quota/alice", json={"limit_gb": 0.0})
        assert resp.status_code == 200
        qm = engine.storage_manager.quota_manager
        assert qm.has_quota("alice")
        assert qm.get_limit_bytes("alice") == 0

    def test_503_when_engine_not_initialized(self, no_engine_client):
        resp = no_engine_client.put("/quota/alice", json={"limit_gb": 1.0})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /quota/{salt}
# ---------------------------------------------------------------------------


class TestGetQuota:
    def test_reports_limit_and_usage(self, client):
        c, engine, usage_map = client
        engine.storage_manager.quota_manager.set_quota("alice", int(2.0 * (1024**3)))
        usage_map["alice"] = int(1.3 * (1024**3))
        resp = c.get("/quota/alice")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cache_salt"] == "alice"
        assert body["exists"] is True
        assert body["limit_gb"] == pytest.approx(2.0)
        assert body["current_usage_gb"] == pytest.approx(1.3)

    def test_reports_zero_for_unknown_salt(self, client):
        """Allowlist semantics — unknown salt returns ``exists=False``
        and limit/usage of 0."""
        c, _, _ = client
        resp = c.get("/quota/charlie")
        assert resp.status_code == 200
        body = resp.json()
        assert body["exists"] is False
        assert body["limit_gb"] == 0.0
        assert body["current_usage_gb"] == 0.0

    def test_sentinel_reads_empty_salt_entry(self, client):
        c, engine, usage_map = client
        engine.storage_manager.quota_manager.set_quota("", 512)
        usage_map[""] = 128
        resp = c.get("/quota/_default")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cache_salt"] == "_default"
        assert body["exists"] is True


# ---------------------------------------------------------------------------
# DELETE /quota/{salt}
# ---------------------------------------------------------------------------


class TestDeleteQuota:
    def test_delete_existing_salt(self, client):
        c, engine, _ = client
        engine.storage_manager.quota_manager.set_quota("alice", 1024)
        resp = c.delete("/quota/alice")
        assert resp.status_code == 200
        assert resp.json() == {"cache_salt": "alice", "status": "removed"}
        assert not engine.storage_manager.quota_manager.has_quota("alice")

    def test_delete_missing_salt_returns_not_found(self, client):
        c, _, _ = client
        resp = c.delete("/quota/charlie")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_delete_sentinel_removes_empty_salt_entry(self, client):
        c, engine, _ = client
        engine.storage_manager.quota_manager.set_quota("", 512)
        resp = c.delete("/quota/_default")
        assert resp.status_code == 200
        assert resp.json() == {
            "cache_salt": "_default",
            "status": "removed",
        }
        assert not engine.storage_manager.quota_manager.has_quota("")


# ---------------------------------------------------------------------------
# GET /quota
# ---------------------------------------------------------------------------


class TestListQuotas:
    def test_list_empty_registry(self, client):
        c, _, _ = client
        resp = c.get("/quota")
        assert resp.status_code == 200
        assert resp.json() == {"users": {}}

    def test_list_reports_usage_for_each_salt(self, client):
        c, engine, usage_map = client
        qm = engine.storage_manager.quota_manager
        qm.set_quota("alice", int(2.0 * (1024**3)))
        qm.set_quota("bob", int(5.0 * (1024**3)))
        qm.set_quota("", 512)
        usage_map.update({"alice": int(1.3 * (1024**3)), "bob": 0})
        resp = c.get("/quota")
        assert resp.status_code == 200
        users = resp.json()["users"]
        assert set(users) == {"alice", "bob", "_default"}
        assert users["alice"]["limit_gb"] == pytest.approx(2.0)
        assert users["alice"]["current_usage_gb"] == pytest.approx(1.3)
        assert users["bob"]["current_usage_gb"] == 0.0
        # The empty-salt entry appears under its URL sentinel.
        assert "_default" in users
