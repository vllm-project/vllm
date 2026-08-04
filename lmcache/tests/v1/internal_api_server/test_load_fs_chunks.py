# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock
import asyncio
import json
import os
import shutil
import tempfile
import threading

# Third Party
from fastapi.testclient import TestClient
import pytest
import torch
import yaml

# First Party
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.internal_api_server.api_server import app
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from tests.v1.utils import create_test_memory_obj


class TestLoadFSChunksAPI:
    """Test suite for the /cache/load-fs-chunks API endpoint."""

    @pytest.fixture
    def temp_fs_path(self):
        """Create a temporary directory for FSConnector storage."""
        temp_dir = tempfile.mkdtemp(prefix="lmcache_test_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def async_loop(self):
        """Create an event loop for async operations."""
        loop = asyncio.new_event_loop()

        thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(loop,),
            name="test-async-loop",
        )
        thread.start()

        yield loop

        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)

    @pytest.fixture
    def local_cpu_backend(self, test_metadata):
        """Create a LocalCPUBackend for testing."""
        config = LMCacheEngineConfig.from_defaults(chunk_size=256)
        allocator = AdHocMemoryAllocator(device="cpu")
        backend = LocalCPUBackend(config, test_metadata, memory_allocator=allocator)
        yield backend
        backend.memory_allocator.close()

    @pytest.fixture
    def test_metadata(self):
        """Create test metadata."""
        return LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(28, 2, 256, 8, 128),
        )

    @pytest.fixture
    def mock_lmcache_adapter(self, local_cpu_backend, test_metadata, async_loop):
        """Create a mock LMCacheConnectorV1Impl adapter with real backend."""
        adapter = MagicMock()
        mock_engine = MagicMock(spec=LMCacheEngine)
        mock_engine.metadata = test_metadata
        mock_engine.local_cpu_backend = local_cpu_backend

        # Mock storage_manager with allocator_backend and loop
        mock_storage_manager = MagicMock()
        mock_storage_manager.allocator_backend = local_cpu_backend
        mock_storage_manager.loop = async_loop
        mock_engine.storage_manager = mock_storage_manager

        adapter.lmcache_engine = mock_engine
        return adapter

    @pytest.fixture
    def client_with_adapter(self, mock_lmcache_adapter):
        """Create a test client with mocked adapter."""
        app.state.lmcache_adapter = mock_lmcache_adapter
        client = TestClient(app)
        yield client
        client.close()

    @pytest.fixture
    def temp_config_file(self, temp_fs_path):
        """Create a temporary config file for testing."""
        config_data = {
            "chunk_size": 256,
            "local_cpu": False,
            "max_local_cpu_size": 2.0,
            "remote_url": f"fs://host:0/{temp_fs_path}",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def _create_test_key(self, key_id: int) -> CacheEngineKey:
        """Create a test CacheEngineKey."""
        return CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=key_id,
            dtype=torch.bfloat16,
        )

    def _prepare_test_data(
        self,
        temp_fs_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        test_metadata: LMCacheMetadata,
        num_chunks: int = 3,
    ):
        """Prepare test data by putting chunks into FSConnector."""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            remote_url=f"fs://host:0/{temp_fs_path}",
            remote_serde="naive",
        )

        remote_backend = RemoteBackend(
            config=config,
            metadata=test_metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        try:
            for i in range(num_chunks):
                key = self._create_test_key(i)
                memory_obj = create_test_memory_obj()
                future = remote_backend.submit_put_task(key, memory_obj)
                if future:
                    future.result(timeout=5.0)
        finally:
            remote_backend.close()
        return num_chunks

    def test_load_fs_chunks_success(
        self,
        client_with_adapter,
        mock_lmcache_adapter,
        temp_config_file,
        temp_fs_path,
        async_loop,
        local_cpu_backend,
        test_metadata,
    ):
        """Test successful load-fs-chunks operation with real FSConnector."""
        num_chunks = 3
        self._prepare_test_data(
            temp_fs_path,
            async_loop,
            local_cpu_backend,
            test_metadata,
            num_chunks=num_chunks,
        )

        response = client_with_adapter.post(
            "/cache/load-fs-chunks",
            json={"config_path": temp_config_file, "max_chunks": 2},
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["loaded_chunks"] == 2
        assert response_data["total_files"] == 3
        assert response_data["config_path"] == temp_config_file

        self._verify_hot_cache(local_cpu_backend, num_chunks, 2)

    def test_load_fs_chunks_no_engine(self, client_with_adapter, temp_config_file):
        """Test load-fs-chunks when engine is not configured."""
        app.state.lmcache_adapter = None

        response = client_with_adapter.post(
            "/cache/load-fs-chunks", json={"config_path": temp_config_file}
        )

        assert response.status_code == 503
        response_data = json.loads(response.text)
        assert response_data["error"] == "/cache/load-fs-chunks API is unavailable"
        assert response_data["message"] == "LMCache engine not configured."

    def test_load_fs_chunks_invalid_config(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test load-fs-chunks with invalid config file."""
        response = client_with_adapter.post(
            "/cache/load-fs-chunks", json={"config_path": "/nonexistent/config.json"}
        )

        assert response.status_code == 400
        response_data = json.loads(response.text)
        assert "Invalid configuration file" in response_data["detail"]

    def test_load_fs_chunks_empty_directory(
        self,
        client_with_adapter,
        mock_lmcache_adapter,
        temp_config_file,
        temp_fs_path,
    ):
        """Test load-fs-chunks with empty FSConnector directory."""
        response = client_with_adapter.post(
            "/cache/load-fs-chunks", json={"config_path": temp_config_file}
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["loaded_chunks"] == 0
        assert response_data["total_files"] == 0

    def _verify_hot_cache(
        self, local_cpu_backend: LocalCPUBackend, num_chunks: int, expected_count: int
    ):
        """Verify hot cache contains expected memory objects with ref count 1."""
        with local_cpu_backend.cpu_lock:
            assert len(local_cpu_backend.hot_cache) == expected_count

            # Check that exactly expected_count keys from total_files
            # are present in hot cache
            found_count = 0
            for i in range(num_chunks):  # total_files is num_chunks
                key = self._create_test_key(i)
                if key in local_cpu_backend.hot_cache:
                    found_count += 1
                    memory_obj = local_cpu_backend.hot_cache[key]
                    assert memory_obj.get_ref_count() == 1

            # Verify that we found exactly expected_count keys
            assert found_count == expected_count

    def test_load_fs_chunks_all_chunks(
        self,
        client_with_adapter,
        mock_lmcache_adapter,
        temp_config_file,
        temp_fs_path,
        async_loop,
        local_cpu_backend,
        test_metadata,
    ):
        """Test load-fs-chunks loading all chunks without max_chunks limit."""
        num_chunks = 3
        self._prepare_test_data(
            temp_fs_path,
            async_loop,
            local_cpu_backend,
            test_metadata,
            num_chunks=num_chunks,
        )

        response = client_with_adapter.post(
            "/cache/load-fs-chunks", json={"config_path": temp_config_file}
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["loaded_chunks"] == 3
        assert response_data["total_files"] == 3
        assert len(response_data["failed_keys"]) == 0

        self._verify_hot_cache(local_cpu_backend, num_chunks, 3)
