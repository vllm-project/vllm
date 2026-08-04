# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock
import json

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.internal_api_server.api_server import app


class TestCacheClearAPI:
    """Test suite for the /cache/clear API endpoint."""

    @pytest.fixture
    def mock_lmcache_adapter(self):
        """Create a mock LMCacheConnectorV1Impl adapter."""
        adapter = MagicMock()

        # Create a mock LMCache engine
        mock_engine = MagicMock(spec=LMCacheEngine)
        mock_engine.clear.return_value = 5  # Mock return value for clear operation

        adapter.lmcache_engine = mock_engine
        return adapter

    @pytest.fixture
    def client_with_adapter(self, mock_lmcache_adapter):
        """Create a test client with mocked adapter."""
        app.state.lmcache_adapter = mock_lmcache_adapter
        client = TestClient(app)
        yield client
        client.close()

    def test_cache_clear_success(self, client_with_adapter, mock_lmcache_adapter):
        """Test successful cache clear operation."""
        # Act
        response = client_with_adapter.delete("/cache/clear")

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["num_removed"] == 5

        # Verify that the clear method was called with correct parameters
        mock_lmcache_adapter.lmcache_engine.clear.assert_called_once_with(
            locations=None, request_configs=None
        )

    @pytest.mark.parametrize(
        "locations,test_description",
        [
            (["LocalCPUBackend", "LocalDiskBackend"], "multiple locations"),
            (["LocalCPUBackend"], "single location"),
        ],
    )
    def test_cache_clear_with_locations(
        self, client_with_adapter, mock_lmcache_adapter, locations, test_description
    ):
        """Test cache clear with specific locations."""
        # Act
        # curl -X DELETE "http://localhost:8000/cache/clear?locations=LocalCPUBackend&locations=LocalDiskBackend"
        response = client_with_adapter.delete(
            "/cache/clear", params={"locations": locations}
        )

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["num_removed"] == 5

        # Verify that the clear method was called with the correct locations
        mock_lmcache_adapter.lmcache_engine.clear.assert_called_once()
        call_args = mock_lmcache_adapter.lmcache_engine.clear.call_args

        # Verify the locations parameter was passed correctly
        assert call_args is not None
        assert "locations" in call_args.kwargs
        assert "request_configs" in call_args.kwargs

        # Assert that the locations parameter matches what we sent
        assert call_args.kwargs["locations"] == locations

    def test_cache_clear_engine_exception(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test cache clear when engine raises an exception."""
        # Arrange
        mock_lmcache_adapter.lmcache_engine.clear.side_effect = Exception("Cache error")

        # Act
        response = client_with_adapter.delete("/cache/clear")

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.text)
        assert response_data["error"] == "Failed to clear cache"
        assert response_data["message"] == "Cache error"

    def test_cache_clear_negative_return_value(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test cache clear when engine returns negative value (edge case)."""
        # Arrange
        mock_lmcache_adapter.lmcache_engine.clear.return_value = -1

        # Act
        response = client_with_adapter.delete("/cache/clear")

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["num_removed"] == -1

    def test_cache_clear_adapter_attribute_error(self):
        """Test cache clear when adapter doesn't have lmcache_engine attribute."""

        # Arrange
        class AdapterWithoutEngine:
            pass

        app.state.lmcache_adapter = AdapterWithoutEngine()
        with TestClient(app) as client:
            # Act
            response = client.delete("/cache/clear")
        # Assert
        assert response.status_code == 503
        response_data = json.loads(response.text)
        assert response_data["error"] == "LMCache API is unavailable"
        assert response_data["message"] == "LMCache engine not configured."
