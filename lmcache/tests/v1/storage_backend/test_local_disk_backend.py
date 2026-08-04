# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


def create_test_config(
    disk_path: str,
    max_disk_size: float = 1.0,
    local_disk_path_sharding: str = "by_gpu",
):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        local_disk_path_sharding=local_disk_path_sharding,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def temp_disk_path():
    """Create a temporary directory for disk storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ----------------------------------------------------------------------------


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


@pytest.fixture
def local_disk_backend(temp_disk_path, async_loop, local_cpu_backend):
    """Create a LocalDiskBackend for testing."""
    config = create_test_config(temp_disk_path)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda:0",
    )


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )

        assert backend.dst_device == "cuda:0"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()

    def test_init_with_lookup_server_and_worker(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Test LocalDiskBackend initialization with lookup server and worker."""
        config = create_test_config(temp_disk_path)
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()


class TestMultiPathDiskBackend:
    """Test cases for multi-path (multi-device) LocalDiskBackend."""

    def test_init_multi_path(self, async_loop, local_cpu_backend):
        """Test initialisation with comma-separated paths."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

            # Path selected by device_id (0 % 2 = 0 -> dir_a)
            assert backend.path == dir_a
            # Both directories should exist
            assert os.path.isdir(dir_a)
            assert os.path.isdir(dir_b)
            # Block size is a plain int for the selected path
            assert isinstance(backend.os_disk_bs, int)
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_gpu_affinity_selects_path(self, async_loop, local_cpu_backend):
        """Different cuda devices select different paths via modulo."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)

            dirs_by_gpu = {}
            for device in ("cuda:0", "cuda:1"):
                backend = LocalDiskBackend(
                    config=config,
                    loop=async_loop,
                    local_cpu_backend=local_cpu_backend,
                    dst_device=device,
                )
                dirs_by_gpu[device] = backend.path

            assert dirs_by_gpu["cuda:0"] == dir_a
            assert dirs_by_gpu["cuda:1"] == dir_b
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_all_directories_created(self, async_loop, local_cpu_backend):
        """All paths in the list get their directories created."""
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            combined = ",".join(paths)
            config = create_test_config(combined)
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )
            for p in paths:
                assert os.path.isdir(p), f"{p} should exist"
        finally:
            shutil.rmtree(base, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_single_path_backward_compat(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """A single path (no commas) works exactly as before."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_default(self, temp_disk_path, async_loop, local_cpu_backend):
        """Default local_disk_path_sharding is 'by_gpu' (backend inits OK)."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_explicit_by_gpu(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Explicitly setting local_disk_path_sharding='by_gpu' works."""
        config = create_test_config(temp_disk_path, local_disk_path_sharding="by_gpu")
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_unsupported_raises(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Unsupported local_disk_path_sharding raises ValueError."""
        config = create_test_config(
            temp_disk_path, local_disk_path_sharding="round_robin"
        )
        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

    def test_cpu_dst_device_defaults_to_first_path(self, async_loop, local_cpu_backend):
        """dst_device='cpu' should fall back to device_id=0."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )
            # device_id=0 -> 0 % 2 = 0 -> dir_a
            assert backend.path == dir_a
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()


class TestParseLocalDisk:
    """Unit tests for the _parse_local_disk config parser."""

    def test_none(self):
        assert _parse_local_disk(None) is None

    def test_single_raw_path(self):
        assert _parse_local_disk("/mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri_no_trailing_slash(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache") == "/mnt/nvme0/cache"

    def test_comma_separated_raw(self):
        result = _parse_local_disk("/mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_comma_separated_file_uris(self):
        result = _parse_local_disk("file:///mnt/nvme0/,file:///mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_mixed_uri_and_raw(self):
        result = _parse_local_disk("file:///mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_whitespace_around_paths(self):
        result = _parse_local_disk("  /mnt/nvme0/ , /mnt/nvme1/  ")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_empty_string(self):
        assert _parse_local_disk("") is None


class TestGetBlockingCachePolicyUpdate:
    """Regression tests for phantom cache hit in get_blocking() (issue #3015).

    ``get_blocking()`` must call ``cache_policy.update_on_hit()`` only when
    ``load_bytes_from_disk()`` returns a valid ``MemoryObj``.  Calling it
    before confirming load success records a phantom hit that skews future
    eviction decisions.
    """

    def _inject_key(
        self,
        backend: LocalDiskBackend,
        key: CacheEngineKey,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> None:
        """Insert a key into backend.dict without writing anything to disk."""
        meta = DiskCacheMetadata(
            path="/nonexistent/path.pt",
            size=0,
            shape=shape,
            dtype=dtype,
            cached_positions=None,
            fmt=MemoryFormat.KV_2LTD,
            pin_count=0,
        )
        with backend.disk_lock:
            backend.dict[key] = meta
            backend.cache_policy.update_on_put(key)

    def test_no_phantom_hit_when_load_fails(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must NOT be called when load_bytes_from_disk returns None."""
        key = create_test_key(101)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=None
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_updates_cache_policy_on_successful_load(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must be called exactly once when the load succeeds."""
        key = create_test_key(102)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        fake_memory_obj = MagicMock(spec=MemoryObj)
        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=fake_memory_obj
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is fake_memory_obj
        mock_update.assert_called_once_with(key, local_disk_backend.dict)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_absent_returns_none_without_policy_update(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """get_blocking must return None immediately when the key is not cached."""
        key = create_test_key(103)

        with patch.object(
            local_disk_backend.cache_policy, "update_on_hit"
        ) as mock_update:
            result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()


class TestBatchedGetBlocking:
    """Tests for the concurrent batched_get_blocking() override."""

    _SHAPE = torch.Size([28, 2, 256, 8, 128])
    _DTYPE = torch.bfloat16

    def _write_key(
        self,
        backend: LocalDiskBackend,
        key: CacheEngineKey,
    ) -> bytes:
        """Write random data to disk for *key* and register it in the backend.

        Returns the raw bytes that were written so callers can verify reads.
        """
        path = backend._key_to_path(key)
        nbytes = 1
        for s in self._SHAPE:
            nbytes *= s
        nbytes *= self._DTYPE.itemsize
        data = os.urandom(nbytes)
        with open(path, "wb") as f:
            f.write(data)
        backend.insert_key(
            key,
            size=nbytes,
            shape=self._SHAPE,
            dtype=self._DTYPE,
            fmt=MemoryFormat.KV_2LTD,
        )
        return data

    def test_all_keys_missing(self, local_disk_backend: LocalDiskBackend) -> None:
        """batched_get_blocking returns [None, …] when no keys are cached."""
        keys = [create_test_key(i) for i in range(200, 204)]
        results = local_disk_backend.batched_get_blocking(keys)

        assert len(results) == len(keys)
        assert all(r is None for r in results)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_reads_match_written_data(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """Data loaded by batched_get_blocking matches what was written."""
        keys = [create_test_key(i) for i in range(210, 214)]
        expected_data = {}
        for key in keys:
            expected_data[key] = self._write_key(local_disk_backend, key)

        results = local_disk_backend.batched_get_blocking(keys)

        assert len(results) == len(keys)
        for key, mem_obj in zip(keys, results, strict=True):
            assert mem_obj is not None, f"Expected data for {key}"
            actual = bytes(mem_obj.byte_array)
            assert actual == expected_data[key]
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_mixed_hit_and_miss(self, local_disk_backend: LocalDiskBackend) -> None:
        """Handles a mix of cached and missing keys correctly."""
        present_key = create_test_key(220)
        missing_key = create_test_key(221)
        expected = self._write_key(local_disk_backend, present_key)

        results = local_disk_backend.batched_get_blocking([present_key, missing_key])

        assert len(results) == 2
        assert results[0] is not None
        assert bytes(results[0].byte_array) == expected
        assert results[1] is None
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_cache_policy_updated_only_for_hits(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit is called only for keys that were successfully loaded."""
        hit_key = create_test_key(230)
        miss_key = create_test_key(231)
        self._write_key(local_disk_backend, hit_key)

        with patch.object(
            local_disk_backend.cache_policy, "update_on_hit"
        ) as mock_update:
            results = local_disk_backend.batched_get_blocking([hit_key, miss_key])

        assert results[0] is not None
        assert results[1] is None
        mock_update.assert_called_once_with(hit_key, local_disk_backend.dict)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_single_key_delegates_to_get_blocking(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """A single-key batch falls through to get_blocking (fast path)."""
        key = create_test_key(240)
        sentinel = MagicMock(spec=MemoryObj)
        with patch.object(
            local_disk_backend, "get_blocking", return_value=sentinel
        ) as mock_get:
            results = local_disk_backend.batched_get_blocking([key])

        assert results == [sentinel]
        mock_get.assert_called_once_with(key)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_concurrent_reads_use_multiple_threads(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """Verify that reads actually fan out across threads."""
        keys = [create_test_key(i) for i in range(250, 254)]
        for key in keys:
            self._write_key(local_disk_backend, key)

        thread_ids: list[int] = []
        lock = threading.Lock()
        original_read_file = local_disk_backend.read_file

        def tracking_read_file(key, buffer, path):
            with lock:
                thread_ids.append(threading.current_thread().ident)
            return original_read_file(key, buffer, path)

        with patch.object(
            local_disk_backend, "read_file", side_effect=tracking_read_file
        ):
            results = local_disk_backend.batched_get_blocking(keys)

        assert all(r is not None for r in results)
        # With 4 keys and 4 threads, we should see more than 1 unique thread.
        assert len(set(thread_ids)) > 1, (
            f"Expected multiple threads, got {set(thread_ids)}"
        )
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_empty_keys_list(self, local_disk_backend: LocalDiskBackend) -> None:
        """An empty key list returns an empty result list."""
        results = local_disk_backend.batched_get_blocking([])
        assert results == []
        local_disk_backend.local_cpu_backend.memory_allocator.close()
