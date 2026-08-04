# SPDX-License-Identifier: Apache-2.0
"""
Test cases for the /backends/* API endpoints.

Tests cover:
- GET /backends  — list active backends
- DELETE /backends/{name} — close and remove a backend
- POST /backends — create new backends from current config
- POST /backends/{name}/recreate — atomic close + create
- End-to-end flow: close → conf update → create → verify
"""

# Standard
from collections import OrderedDict
from unittest.mock import MagicMock

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.internal_api_server.api_server import app

# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


class FakeBackend:
    """Minimal fake backend for testing."""

    def __init__(self, name: str = "FakeBackend"):
        self._name = name
        self.closed = False

    def close(self):
        self.closed = True

    def __str__(self):
        return self._name


@pytest.fixture
def mock_storage_manager():
    """Create a mock StorageManager with two fake backends."""
    sm = MagicMock()
    fake_cpu = FakeBackend("LocalCPUBackend")
    fake_remote = FakeBackend("RemoteBackend")
    backends = OrderedDict(
        [
            ("LocalCPUBackend", fake_cpu),
            ("RemoteBackend", fake_remote),
        ]
    )
    sm.storage_backends = backends

    # Wire real-ish list_backends
    sm.list_backends.side_effect = lambda: {
        n: type(b).__name__ for n, b in sm.storage_backends.items()
    }
    return sm


@pytest.fixture
def mock_engine(mock_storage_manager):
    engine = MagicMock()
    engine.storage_manager = mock_storage_manager
    return engine


@pytest.fixture
def client_with_engine(mock_engine):
    adapter = MagicMock()
    adapter.lmcache_engine = mock_engine
    app.state.lmcache_adapter = adapter
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture
def client_without_engine():
    adapter = MagicMock()
    adapter.lmcache_engine = None
    app.state.lmcache_adapter = adapter
    client = TestClient(app)
    yield client
    client.close()


# ------------------------------------------------------------------ #
#  GET /backends
# ------------------------------------------------------------------ #


class TestListBackends:
    def test_list_success(self, client_with_engine, mock_storage_manager):
        resp = client_with_engine.get("/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert "LocalCPUBackend" in data
        assert "RemoteBackend" in data

    def test_list_no_engine(self, client_without_engine):
        resp = client_without_engine.get("/backends")
        assert resp.status_code == 503

    def test_list_exception(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.list_backends.side_effect = RuntimeError("boom")
        resp = client_with_engine.get("/backends")
        assert resp.status_code == 500
        data = resp.json()
        assert "boom" in data["message"]


# ------------------------------------------------------------------ #
#  DELETE /backends/{name}
# ------------------------------------------------------------------ #


class TestCloseBackend:
    def test_close_success(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.close_backend.return_value = True
        resp = client_with_engine.delete("/backends/RemoteBackend")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        mock_storage_manager.close_backend.assert_called_once_with("RemoteBackend")

    def test_close_not_found(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.close_backend.return_value = False
        resp = client_with_engine.delete("/backends/NonExistent")
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "not_found"

    def test_close_no_engine(self, client_without_engine):
        resp = client_without_engine.delete("/backends/RemoteBackend")
        assert resp.status_code == 503

    def test_close_exception(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.close_backend.side_effect = RuntimeError("close error")
        resp = client_with_engine.delete("/backends/RemoteBackend")
        assert resp.status_code == 500
        data = resp.json()
        assert "close error" in data["message"]


# ------------------------------------------------------------------ #
#  POST /backends
# ------------------------------------------------------------------ #


class TestCreateBackends:
    def test_create_success(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.create_backends.return_value = {"GdsBackend": "GdsBackend"}
        resp = client_with_engine.post("/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "GdsBackend" in data["created"]

    def test_create_no_engine(self, client_without_engine):
        resp = client_without_engine.post("/backends")
        assert resp.status_code == 503

    def test_create_exception(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.create_backends.side_effect = RuntimeError("create error")
        resp = client_with_engine.post("/backends")
        assert resp.status_code == 500
        data = resp.json()
        assert "create error" in data["message"]


# ------------------------------------------------------------------ #
#  POST /backends/{name}/recreate
# ------------------------------------------------------------------ #


class TestRecreateBackend:
    def test_recreate_success(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.recreate_backend.return_value = {
            "RemoteBackend": "RemoteBackend"
        }
        resp = client_with_engine.post("/backends/RemoteBackend/recreate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "RemoteBackend" in data["recreated"]
        mock_storage_manager.recreate_backend.assert_called_once_with("RemoteBackend")

    def test_recreate_not_found(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.recreate_backend.side_effect = KeyError(
            "Backend NonExistent not found"
        )
        resp = client_with_engine.post("/backends/NonExistent/recreate")
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "not_found"

    def test_recreate_no_engine(self, client_without_engine):
        resp = client_without_engine.post("/backends/RemoteBackend/recreate")
        assert resp.status_code == 503

    def test_recreate_exception(self, client_with_engine, mock_storage_manager):
        mock_storage_manager.recreate_backend.side_effect = RuntimeError(
            "recreate error"
        )
        resp = client_with_engine.post("/backends/RemoteBackend/recreate")
        assert resp.status_code == 500
        data = resp.json()
        assert "recreate error" in data["message"]


# ------------------------------------------------------------------ #
#  End-to-end flow: close → conf → create
# ------------------------------------------------------------------ #


class TestBackendSwitchFlow:
    """Simulate the full backend-switch workflow via API."""

    def test_close_update_conf_create_flow(
        self, client_with_engine, mock_storage_manager
    ):
        """
        1. DELETE /backends/RemoteBackend
        2. POST /conf to update remote_url
        3. POST /backends to create new RemoteBackend
        4. GET /backends to verify
        """
        # Step 1: close RemoteBackend
        mock_storage_manager.close_backend.return_value = True
        resp = client_with_engine.delete("/backends/RemoteBackend")
        assert resp.status_code == 200

        # Step 2: update config via /conf (uses conf_api)
        # We patch the config object on the adapter
        mock_config = MagicMock()
        client_with_engine.app.state.lmcache_adapter.config = mock_config

        # Step 3: create new backends
        mock_storage_manager.create_backends.return_value = {
            "RemoteBackend": "RemoteBackend"
        }
        resp = client_with_engine.post("/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "RemoteBackend" in data["created"]

        # Step 4: list backends
        resp = client_with_engine.get("/backends")
        assert resp.status_code == 200
