# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import hashlib
import json

# Third Party
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.utils import compress_slot_mapping
from lmcache.v1.internal_api_server.api_server import app


class TestKVCacheCheckAPI:
    """Test suite for the /kvcache/check API endpoint."""

    @pytest.fixture(autouse=True)
    def reset_adapter(self):
        """Reset adapter before each test."""
        yield
        app.state.lmcache_adapter = None

    @pytest.fixture
    def mock_kv_caches(self) -> Dict[str, torch.Tensor]:
        """Create mock kv_caches tensors for testing."""
        return {
            "layer_0": torch.randn(4, 4, 8, 64),
            "layer_1": torch.randn(4, 4, 8, 64),
        }

    @pytest.fixture
    def mock_lmcache_adapter(self, mock_kv_caches):
        """Create a mock LMCacheConnectorV1Impl adapter."""
        adapter = MagicMock()
        adapter.kv_caches = mock_kv_caches
        adapter.kvcaches = mock_kv_caches  # API uses kvcaches (no underscore)

        # Mock lmcache_engine for record_slot tests
        mock_engine = MagicMock()
        mock_engine.kvcache_check_log_enabled = False
        adapter.lmcache_engine = mock_engine

        def compute_checksums(
            slot_indices: List[int], chunk_size: Optional[int] = None
        ) -> Optional[Dict[str, Any]]:
            if not adapter.kv_caches:
                return None

            layer_checksums: Dict[str, str] = {}
            chunk_checksums: Dict[str, List[str]] = {}

            for layer_name, kv_tensor in adapter.kv_caches.items():
                slot_tensor = torch.tensor(
                    slot_indices, dtype=torch.long, device=kv_tensor.device
                )
                kv_at_slots = kv_tensor.view(-1, *kv_tensor.shape[2:])[slot_tensor]
                tensor_bytes = kv_at_slots.detach().cpu().contiguous().numpy().tobytes()
                layer_checksums[layer_name] = hashlib.md5(tensor_bytes).hexdigest()

                if chunk_size and chunk_size > 0:
                    num_slots = len(slot_indices)
                    chunk_checksum_list: List[str] = []
                    for i in range((num_slots + chunk_size - 1) // chunk_size):
                        chunk_data = kv_at_slots[i * chunk_size : (i + 1) * chunk_size]
                        chunk_bytes = (
                            chunk_data.detach().cpu().contiguous().numpy().tobytes()
                        )
                        chunk_checksum_list.append(hashlib.md5(chunk_bytes).hexdigest())
                    chunk_checksums[layer_name] = chunk_checksum_list

            result: Dict[str, Any] = {"layer_checksums": layer_checksums}
            if chunk_size:
                result["chunk_checksums"] = chunk_checksums
                result["chunk_size"] = chunk_size
                result["num_chunks"] = (
                    (len(slot_indices) + chunk_size - 1) // chunk_size
                    if slot_indices
                    else 0
                )
            return result

        adapter.compute_kvcache_checksums = compute_checksums
        return adapter

    @pytest.fixture
    def client_with_adapter(self, mock_lmcache_adapter):
        """Create a test client with mocked adapter."""
        app.state.lmcache_adapter = mock_lmcache_adapter
        client = TestClient(app)
        yield client
        client.close()

    # ==========================================================================
    # Tests for /kvcache/check endpoint
    # ==========================================================================

    def test_kvcache_check_success_layer_only(self, client_with_adapter):
        """Test successful kvcache check with layer-level checksums."""
        response = client_with_adapter.get(
            "/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        )
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["status"] == "success"
        assert "chunk_checksums" in data

    @pytest.mark.parametrize(
        "url,expected_error",
        [
            ("/cache/kvcache/check?slot_mapping=a,b,c", "Invalid slot_mapping format"),
            (
                "/cache/kvcache/check?slot_mapping=0,1&chunk_size=0",
                "Invalid chunk_size",
            ),
            (
                "/cache/kvcache/check?slot_mapping=0,1&chunk_size=-1",
                "Invalid chunk_size",
            ),
        ],
    )
    def test_kvcache_check_invalid_params(
        self, client_with_adapter, url, expected_error
    ):
        """Test kvcache check with various invalid parameters."""
        response = client_with_adapter.get(url)
        assert response.status_code == 400
        assert json.loads(response.text)["error"] == expected_error

    @pytest.mark.parametrize(
        "slots,chunk_size,expected_chunks",
        [
            ("0,1,2,3,4,5,6,7", 4, 2),
            ("0,1,2,3,4,5,6", 3, 3),
            ("0,1,2,3", 4, 1),
            ("0,1,2,3", 1, 4),
        ],
    )
    def test_kvcache_check_with_chunk_size(
        self, client_with_adapter, slots, chunk_size, expected_chunks
    ):
        """Test kvcache check with various chunk sizes."""
        response = client_with_adapter.get(
            f"/cache/kvcache/check?slot_mapping={slots}&chunk_size={chunk_size}"
        )
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["num_chunks"] == expected_chunks

    def test_kvcache_check_consistency(self, client_with_adapter):
        """Test that checksums are consistent across calls."""
        url = "/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        data1 = json.loads(client_with_adapter.get(url).text)
        data2 = json.loads(client_with_adapter.get(url).text)
        assert data1["chunk_checksums"] == data2["chunk_checksums"]

    # ==========================================================================
    # Tests for adapter unavailable scenarios
    # ==========================================================================

    @pytest.mark.parametrize(
        "method,url",
        [
            ("get", "/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"),
            ("get", "/cache/kvcache/info"),
            ("post", "/cache/kvcache/record_slot?enabled=true"),
        ],
    )
    def test_no_adapter(self, method, url):
        """Test endpoints when adapter is not available."""
        app.state.lmcache_adapter = None
        with TestClient(app) as client:
            response = getattr(client, method)(url)
        assert response.status_code == 503
        assert json.loads(response.text)["error"] == "LMCache adapter unavailable"

    def test_kvcache_check_empty_kv_caches(self):
        """Test kvcache check when kv_caches is empty."""
        adapter = MagicMock()
        adapter.kv_caches = {}
        adapter.kvcaches = {}
        adapter.compute_kvcache_checksums = MagicMock(return_value=None)
        app.state.lmcache_adapter = adapter
        with TestClient(app) as client:
            response = client.get(
                "/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
            )
        assert response.status_code == 404

    # ==========================================================================
    # Tests for /cache/kvcache/info endpoint
    # ==========================================================================

    def test_kvcache_info_success(self, client_with_adapter):
        """Test successful kvcache info retrieval."""
        response = client_with_adapter.get("/cache/kvcache/info")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["num_layers"] == 2
        assert "layer_0" in data["layers"]

    def test_kvcache_info_empty(self):
        """Test kvcache info when kvcaches is empty."""
        adapter = MagicMock()
        adapter.kvcaches = {}
        app.state.lmcache_adapter = adapter
        with TestClient(app) as client:
            response = client.get("/cache/kvcache/info")
        assert response.status_code == 404

    # ==========================================================================
    # Tests for /cache/kvcache/record_slot endpoint
    # ==========================================================================

    @pytest.mark.parametrize(
        "enabled_param,expected_value",
        [("true", True), ("false", False), ("TRUE", True), ("False", False)],
    )
    def test_kvcache_record_slot_toggle(
        self, client_with_adapter, mock_lmcache_adapter, enabled_param, expected_value
    ):
        """Test enabling/disabling KVCache Check logging."""
        response = client_with_adapter.post(
            f"/cache/kvcache/record_slot?enabled={enabled_param}"
        )
        assert response.status_code == 200
        assert (
            mock_lmcache_adapter.lmcache_engine.kvcache_check_log_enabled
            == expected_value
        )

    def test_kvcache_record_slot_invalid(self, client_with_adapter):
        """Test with invalid enabled parameter."""
        response = client_with_adapter.post(
            "/cache/kvcache/record_slot?enabled=invalid"
        )
        assert response.status_code == 400


class TestCompressSlotMapping:
    """Test suite for the compress_slot_mapping function."""

    @pytest.mark.parametrize(
        "input_slots,expected",
        [
            ([1, 2, 3, 4, 5], [[1, 5]]),
            ([1, 2, 3, 4, 5, 9, 10, 11, 12], [[1, 5], [9, 12]]),
            # No consecutive elements, no compression
            ([1, 3, 5, 7], [1, 3, 5, 7]),
            # 5 is single, not compressed
            ([1, 2, 3, 5, 7, 8, 9], [[1, 3], 5, [7, 9]]),
            ([], []),
            ([42], [42]),  # Single element, not compressed
            # Mixed order
            ([5, 3, 1, 2, 4, 9, 10, 11, 13], [5, 3, 1, 2, 4, [9, 11], 13]),
            ([0, 1, 2, 100, 101, 102], [[0, 2], [100, 102]]),
            ([1, 2], [1, 2]),  # Two elements, not compressed
            ([1, 2, 3], [[1, 3]]),  # Three elements, compressed
        ],
    )
    def test_compress_slot_mapping(self, input_slots, expected):
        """Test compression of slot mappings."""
        assert compress_slot_mapping(input_slots) == expected
