# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock
import json

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.internal_api_server.api_server import app


def _make_metadata(role: str) -> MagicMock:
    """Create a mock metadata with the given role."""
    metadata = MagicMock()
    metadata.role = role
    return metadata


class TestLookupAPI:
    """Test suite for the /lookup/* API endpoints."""

    @pytest.fixture
    def mock_scheduler_manager(self):
        """Create a mock LMCacheManager for scheduler role."""
        manager = MagicMock()
        manager.lmcache_engine_metadata = _make_metadata("scheduler")
        manager.get_lookup_info.return_value = {
            "client": "HitLimitLookupClient(LMCacheBypassLookupClient)",
            "server": "None",
        }
        manager.close_lookup_client.return_value = {
            "old": "HitLimitLookupClient(LMCacheBypassLookupClient)"
        }
        manager.create_lookup_client.return_value = {"new": "LMCacheLookupClient"}
        manager.recreate_lookup_client.return_value = {
            "old": "HitLimitLookupClient(LMCacheBypassLookupClient)",
            "new": "LMCacheLookupClient",
        }
        return manager

    @pytest.fixture
    def mock_worker_manager(self):
        """Create a mock LMCacheManager for worker role."""
        manager = MagicMock()
        manager.lmcache_engine_metadata = _make_metadata("worker")
        manager.get_lookup_info.return_value = {
            "client": "None",
            "server": "LMCacheLookupServer",
        }
        manager.close_lookup_server.return_value = {"old": "LMCacheLookupServer"}
        manager.create_lookup_server.return_value = {"new": "LMCacheAsyncLookupServer"}
        manager.recreate_lookup_server.return_value = {
            "old": "LMCacheLookupServer",
            "new": "LMCacheAsyncLookupServer",
        }
        return manager

    @pytest.fixture
    def scheduler_client(self, mock_scheduler_manager):
        """Create a test client with mocked scheduler manager."""
        app.state.lmcache_adapter = mock_scheduler_manager
        client = TestClient(app)
        yield client
        client.close()

    @pytest.fixture
    def worker_client(self, mock_worker_manager):
        """Create a test client with mocked worker manager."""
        app.state.lmcache_adapter = mock_worker_manager
        client = TestClient(app)
        yield client
        client.close()

    # ==================== GET /lookup/info Tests ====================

    def test_get_lookup_info_scheduler(self, scheduler_client, mock_scheduler_manager):
        """Test get lookup info for scheduler role."""
        response = scheduler_client.get("/lookup/info")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert "LMCacheBypassLookupClient" in data["client"]
        assert data["server"] == "None"
        mock_scheduler_manager.get_lookup_info.assert_called_once()

    def test_get_lookup_info_worker(self, worker_client, mock_worker_manager):
        """Test get lookup info for worker role."""
        response = worker_client.get("/lookup/info")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["client"] == "None"
        assert data["server"] == "LMCacheLookupServer"
        mock_worker_manager.get_lookup_info.assert_called_once()

    def test_get_lookup_info_not_supported(self):
        """Test get lookup info when manager doesn't support it."""

        class SimpleAdapter:
            pass

        app.state.lmcache_adapter = SimpleAdapter()
        with TestClient(app) as client:
            response = client.get("/lookup/info")

        assert response.status_code == 503
        data = json.loads(response.text)
        assert "error" in data

    # ==================== POST /lookup/close Tests ====================

    def test_close_lookup_scheduler(self, scheduler_client, mock_scheduler_manager):
        """Test close lookup client for scheduler."""
        response = scheduler_client.post("/lookup/close")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert "LMCacheBypassLookupClient" in data["old"]
        assert data["role"] == "scheduler"
        mock_scheduler_manager.close_lookup_client.assert_called_once()

    def test_close_lookup_worker(self, worker_client, mock_worker_manager):
        """Test close lookup server for worker."""
        response = worker_client.post("/lookup/close")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["old"] == "LMCacheLookupServer"
        assert data["role"] == "worker"
        mock_worker_manager.close_lookup_server.assert_called_once()

    # ==================== POST /lookup/create Tests ====================

    def test_create_lookup_scheduler(self, scheduler_client, mock_scheduler_manager):
        """Test create lookup client for scheduler."""
        response = scheduler_client.post("/lookup/create")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["new"] == "LMCacheLookupClient"
        assert data["role"] == "scheduler"
        mock_scheduler_manager.create_lookup_client.assert_called_once_with(
            dryrun=False
        )

    def test_create_lookup_worker(self, worker_client, mock_worker_manager):
        """Test create lookup server for worker."""
        response = worker_client.post("/lookup/create")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["new"] == "LMCacheAsyncLookupServer"
        assert data["role"] == "worker"
        mock_worker_manager.create_lookup_server.assert_called_once_with(dryrun=False)

    def test_create_lookup_dryrun_scheduler(
        self, scheduler_client, mock_scheduler_manager
    ):
        """Test dryrun create lookup client for scheduler."""
        mock_scheduler_manager.create_lookup_client.return_value = {
            "new": "LMCacheLookupClient",
            "dryrun": True,
        }

        response = scheduler_client.post("/lookup/create?dryrun=true")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["new"] == "LMCacheLookupClient"
        assert data["dryrun"] is True
        mock_scheduler_manager.create_lookup_client.assert_called_once_with(dryrun=True)

    def test_create_lookup_dryrun_worker(self, worker_client, mock_worker_manager):
        """Test dryrun create lookup server for worker."""
        mock_worker_manager.create_lookup_server.return_value = {
            "new": "LMCacheAsyncLookupServer",
            "dryrun": True,
        }

        response = worker_client.post("/lookup/create?dryrun=true")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["new"] == "LMCacheAsyncLookupServer"
        assert data["dryrun"] is True
        mock_worker_manager.create_lookup_server.assert_called_once_with(dryrun=True)

    def test_create_lookup_error(self, scheduler_client, mock_scheduler_manager):
        """Test create lookup with error."""
        mock_scheduler_manager.create_lookup_client.return_value = {
            "error": "metadata not available"
        }

        response = scheduler_client.post("/lookup/create")

        assert response.status_code == 400
        data = json.loads(response.text)
        assert data["error"] == "metadata not available"

    # ==================== POST /lookup/recreate Tests ====================

    def test_recreate_lookup_scheduler(self, scheduler_client, mock_scheduler_manager):
        """Test recreate lookup client for scheduler."""
        response = scheduler_client.post("/lookup/recreate")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert "LMCacheBypassLookupClient" in data["old"]
        assert data["new"] == "LMCacheLookupClient"
        assert data["role"] == "scheduler"
        mock_scheduler_manager.recreate_lookup_client.assert_called_once()

    def test_recreate_lookup_worker(self, worker_client, mock_worker_manager):
        """Test recreate lookup server for worker."""
        response = worker_client.post("/lookup/recreate")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["old"] == "LMCacheLookupServer"
        assert data["new"] == "LMCacheAsyncLookupServer"
        assert data["role"] == "worker"
        mock_worker_manager.recreate_lookup_server.assert_called_once()

    def test_recreate_lookup_error(self, scheduler_client, mock_scheduler_manager):
        """Test recreate lookup with error."""
        mock_scheduler_manager.recreate_lookup_client.return_value = {
            "error": "only supported for scheduler role"
        }

        response = scheduler_client.post("/lookup/recreate")

        assert response.status_code == 400
        data = json.loads(response.text)
        assert "error" in data

    def test_recreate_lookup_unknown_role(self):
        """Test recreation with unknown role."""
        manager = MagicMock()
        manager.lmcache_engine_metadata = _make_metadata("unknown")
        app.state.lmcache_adapter = manager
        with TestClient(app) as client:
            response = client.post("/lookup/recreate")

        assert response.status_code == 400
        data = json.loads(response.text)
        assert data["error"] == "Unknown role"

    def test_recreate_lookup_no_metadata(self):
        """Test recreation when metadata is not available."""
        manager = MagicMock()
        manager.lmcache_engine_metadata = None
        app.state.lmcache_adapter = manager
        with TestClient(app) as client:
            response = client.post("/lookup/recreate")

        assert response.status_code == 400
        data = json.loads(response.text)
        assert data["error"] == "Unknown role"

    def test_recreate_lookup_not_supported(self):
        """Test recreation when manager doesn't support it."""

        class SimpleAdapter:
            lmcache_engine_metadata = _make_metadata("scheduler")

        app.state.lmcache_adapter = SimpleAdapter()
        with TestClient(app) as client:
            response = client.post("/lookup/recreate")

        assert response.status_code == 503
        data = json.loads(response.text)
        assert "error" in data

    # ==================== Integration Flow Tests ====================

    def test_full_flow_worker_then_scheduler(
        self, mock_worker_manager, mock_scheduler_manager
    ):
        """
        Test the complete flow:
        1. POST /lookup/recreate on worker (recreate server)
        2. POST /lookup/recreate on scheduler (recreate client)

        Note: Config should be updated via /conf API before each step.
        """
        # Step 1: Worker side - recreate server
        app.state.lmcache_adapter = mock_worker_manager
        with TestClient(app) as worker_client:
            response = worker_client.post("/lookup/recreate")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["old"] == "LMCacheLookupServer"
        assert data["new"] == "LMCacheAsyncLookupServer"
        mock_worker_manager.recreate_lookup_server.assert_called_once()

        # Step 2: Scheduler side - recreate client
        app.state.lmcache_adapter = mock_scheduler_manager
        with TestClient(app) as scheduler_client:
            response = scheduler_client.post("/lookup/recreate")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert "LMCacheBypassLookupClient" in data["old"]
        assert data["new"] == "LMCacheLookupClient"
        mock_scheduler_manager.recreate_lookup_client.assert_called_once()

    def test_step_by_step_flow(self, mock_scheduler_manager):
        """
        Test step-by-step flow: dryrun -> close -> create.
        """
        app.state.lmcache_adapter = mock_scheduler_manager
        with TestClient(app) as client:
            # Step 1: Dryrun - check what would be created
            mock_scheduler_manager.create_lookup_client.return_value = {
                "new": "LMCacheLookupClient",
                "dryrun": True,
            }
            response = client.post("/lookup/create?dryrun=true")
            assert response.status_code == 200
            data = json.loads(response.text)
            assert data["dryrun"] is True

            # Step 2: Close current client
            response = client.post("/lookup/close")
            assert response.status_code == 200
            mock_scheduler_manager.close_lookup_client.assert_called_once()

            # Step 3: Create new client
            mock_scheduler_manager.create_lookup_client.return_value = {
                "new": "LMCacheLookupClient"
            }
            response = client.post("/lookup/create")
            assert response.status_code == 200
