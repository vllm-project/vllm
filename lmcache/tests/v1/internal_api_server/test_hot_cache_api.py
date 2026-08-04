# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock
import json

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.internal_api_server.api_server import app


class TestHotCacheAPI:
    """Test suite for the /hot_cache/* API endpoints."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.set_hot_cache = MagicMock()
        engine.is_hot_cache_enabled = MagicMock(return_value=False)
        return engine

    @pytest.fixture
    def client_with_engine(self, mock_engine):
        adapter = MagicMock()
        adapter.lmcache_engine = mock_engine
        app.state.lmcache_adapter = adapter
        client = TestClient(app)
        yield client
        client.close()

    @pytest.fixture
    def client_without_engine(self):
        adapter = MagicMock()
        adapter.lmcache_engine = None
        app.state.lmcache_adapter = adapter
        client = TestClient(app)
        yield client
        client.close()

    # --- enable ---

    def test_enable_hot_cache_success(self, client_with_engine, mock_engine):
        response = client_with_engine.put("/hot_cache/enable")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["status"] == "success"
        assert data["hot_cache"] is True
        mock_engine.set_hot_cache.assert_called_once_with(True)

    def test_enable_hot_cache_no_engine(self, client_without_engine):
        response = client_without_engine.put("/hot_cache/enable")
        assert response.status_code == 503
        data = json.loads(response.text)
        assert "unavailable" in data["error"]

    def test_enable_hot_cache_exception(self, client_with_engine, mock_engine):
        mock_engine.set_hot_cache.side_effect = RuntimeError("test error")
        response = client_with_engine.put("/hot_cache/enable")
        assert response.status_code == 500
        data = json.loads(response.text)
        assert "test error" in data["message"]

    # --- disable ---

    def test_disable_hot_cache_success(self, client_with_engine, mock_engine):
        response = client_with_engine.put("/hot_cache/disable")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["status"] == "success"
        assert data["hot_cache"] is False
        mock_engine.set_hot_cache.assert_called_once_with(False)

    def test_disable_hot_cache_no_engine(self, client_without_engine):
        response = client_without_engine.put("/hot_cache/disable")
        assert response.status_code == 503

    def test_disable_hot_cache_exception(self, client_with_engine, mock_engine):
        mock_engine.set_hot_cache.side_effect = RuntimeError("boom")
        response = client_with_engine.put("/hot_cache/disable")
        assert response.status_code == 500
        data = json.loads(response.text)
        assert "boom" in data["message"]

    # --- status ---

    def test_status_hot_cache_enabled(self, client_with_engine, mock_engine):
        mock_engine.is_hot_cache_enabled.return_value = True
        response = client_with_engine.get("/hot_cache/status")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["hot_cache"] is True
        assert "enabled" in data["message"]

    def test_status_hot_cache_disabled(self, client_with_engine, mock_engine):
        mock_engine.is_hot_cache_enabled.return_value = False
        response = client_with_engine.get("/hot_cache/status")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["hot_cache"] is False
        assert "disabled" in data["message"]

    def test_status_hot_cache_no_engine(self, client_without_engine):
        response = client_without_engine.get("/hot_cache/status")
        assert response.status_code == 503

    def test_status_hot_cache_exception(self, client_with_engine, mock_engine):
        mock_engine.is_hot_cache_enabled.side_effect = RuntimeError("oops")
        response = client_with_engine.get("/hot_cache/status")
        assert response.status_code == 500
        data = json.loads(response.text)
        assert "oops" in data["message"]
