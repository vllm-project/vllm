# SPDX-License-Identifier: Apache-2.0
# Backup original state of sys.modules and google package attributes

# Standard
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch
import asyncio
import sys

_original_sys_modules: dict[str, Any] = {}
_original_google_attrs: dict[str, Any] = {}
_mocked_modules = [
    "google.api_core",
    "google.api_core.exceptions",
    "google.api_core.gapic_v1",
    "google.api_core.gapic_v1.client_info",
    "google.cloud",
    "google.cloud.bigtable",
    "google.cloud.bigtable.row_filters",
    "google.cloud.bigtable.data",
    "google.cloud.bigtable.data.row_filters",
    "google.oauth2",
    "google.oauth2.service_account",
]


class MockDeadlineExceeded(Exception):
    pass


class MockPermissionDenied(Exception):
    pass


class MockNotFound(Exception):
    pass


class MockResourceExhausted(Exception):
    pass


# Setup mock google packages before any test runs
mock_exceptions = MagicMock()
mock_exceptions.DeadlineExceeded = MockDeadlineExceeded
mock_exceptions.Timeout = MockDeadlineExceeded
mock_exceptions.PermissionDenied = MockPermissionDenied
mock_exceptions.Unauthenticated = MockPermissionDenied
mock_exceptions.NotFound = MockNotFound
mock_exceptions.ResourceExhausted = MockResourceExhausted

mock_row_filters = MagicMock()
mock_row_filters.StripValueTransformerFilter.return_value = MagicMock()

mock_data = MagicMock()
mock_data.BigtableDataClientAsync = MagicMock()


def fake_client_constructor(*args, **kwargs):
    client = mock_data.BigtableDataClientAsync.return_value
    if isinstance(client, MagicMock):
        client.get_table.return_value = client
    return client


mock_data.BigtableDataClientAsync.side_effect = fake_client_constructor


class FakeRowMutationEntry:
    def __init__(self, row_key, mutations):
        self.row_key = row_key
        self.mutations = mutations


mock_data.ReadRowsQuery = MagicMock()
mock_data.RowMutationEntry = FakeRowMutationEntry
mock_data.row_filters = mock_row_filters

mock_bigtable = MagicMock(data=mock_data, row_filters=mock_row_filters)

mock_service_account = MagicMock()
mock_oauth2 = MagicMock(service_account=mock_service_account)

mock_api_core = MagicMock(exceptions=mock_exceptions)
mock_gapic = MagicMock()
mock_cloud = MagicMock(bigtable=mock_bigtable)

# Third Party
import pytest  # noqa: E402
import torch  # noqa: E402

# First Party
from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend.connector.bigtable_connector import (  # noqa: E402
    BigtableConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402
from lmcache.v1.storage_backend.remote_backend import RemoteBackend  # noqa: E402
from tests.v1.utils import create_test_memory_obj  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def cleanup_google_mocks():
    print("\n[Fixture] cleanup_google_mocks: running setup...")
    # 1. Backup original state
    for name in _mocked_modules:
        if name in sys.modules:
            _original_sys_modules[name] = sys.modules[name]
    if "google" in sys.modules:
        google_mod = sys.modules["google"]
        for attr in ["oauth2", "cloud", "api_core"]:
            if hasattr(google_mod, attr):
                _original_google_attrs[attr] = getattr(google_mod, attr)

    # 2. Inject mocks
    sys.modules["google.api_core"] = mock_api_core
    sys.modules["google.api_core.exceptions"] = mock_exceptions
    sys.modules["google.api_core.gapic_v1"] = mock_gapic
    sys.modules["google.api_core.gapic_v1.client_info"] = MagicMock()

    sys.modules["google.cloud"] = mock_cloud
    sys.modules["google.cloud.bigtable"] = mock_bigtable
    sys.modules["google.cloud.bigtable.row_filters"] = mock_row_filters
    sys.modules["google.cloud.bigtable.data"] = mock_data
    sys.modules["google.cloud.bigtable.data.row_filters"] = mock_row_filters

    sys.modules["google.oauth2"] = mock_oauth2
    sys.modules["google.oauth2.service_account"] = mock_service_account

    if "google" in sys.modules:
        google_mod = sys.modules["google"]
        google_mod.oauth2 = mock_oauth2  # type: ignore[attr-defined]
        google_mod.cloud = mock_cloud  # type: ignore[attr-defined]
        google_mod.api_core = mock_api_core  # type: ignore[attr-defined]

    if "google.cloud" in sys.modules:
        sys.modules["google.cloud"].bigtable = mock_bigtable  # type: ignore[attr-defined]

    if "google.oauth2" in sys.modules:
        sys.modules["google.oauth2"].service_account = mock_service_account  # type: ignore[attr-defined]

    if "google.api_core" in sys.modules:
        sys.modules["google.api_core"].exceptions = mock_exceptions  # type: ignore[attr-defined]
        sys.modules["google.api_core"].gapic_v1 = mock_gapic  # type: ignore[attr-defined]

    yield
    print("\n[Fixture] cleanup_google_mocks: running teardown/cleanup...")
    # 1. Restore sys.modules
    for name in _mocked_modules:
        if name in _original_sys_modules:
            print(f"  Restoring sys.modules[{name}]")
            sys.modules[name] = _original_sys_modules[name]
        elif name in sys.modules:
            print(f"  Deleting sys.modules[{name}]")
            del sys.modules[name]

    # 2. Restore google module attributes
    if "google" in sys.modules:
        google_mod = sys.modules["google"]
        for attr in ["oauth2", "cloud", "api_core"]:
            if attr in _original_google_attrs:
                print(f"  Restoring google.{attr}")
                setattr(google_mod, attr, _original_google_attrs[attr])
            elif hasattr(google_mod, attr):
                print(f"  Deleting google.{attr}")
                delattr(google_mod, attr)


async def mock_async_gen(items):
    for item in items:
        yield item


def create_test_config(extra_overrides=None):
    extras = {
        "bigtable_project_id": "test-project",
        "bigtable_instance_id": "test-instance",
        "bigtable_table_name": "test-table",
        "bigtable_max_chunk_size_mb": 90.0,
        "bigtable_max_retries": 2,
        "bigtable_exists_cache_ttl_seconds": 10.0,
    }
    if extra_overrides:
        extras.update(extra_overrides)

    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_storage_plugins=["bigtable"],
        remote_serde="naive",
        lmcache_instance_id="test_instance",
        extra_config=extras,
    )


def create_test_metadata(kv_shape=(28, 2, 256, 8, 128), chunk_size=256):
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        chunk_size=chunk_size,
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    # Standard
    import threading

    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

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
def local_cpu_backend(memory_allocator):
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    metadata = create_test_metadata()
    return LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)


@pytest.fixture(autouse=True)
def mock_pq_executor():
    # Patch AsyncPQExecutor so it doesn't spawn real thread workers during unit tests,
    # but still executes jobs inline when submit_job is called.
    with patch(
        "lmcache.v1.storage_backend.connector.bigtable_connector.AsyncPQExecutor"
    ) as mock:
        instance = MagicMock()

        async def fake_submit_job(fn, *args, **kwargs):
            kwargs.pop("priority", None)
            return await fn(*args, **kwargs)

        instance.submit_job = AsyncMock(side_effect=fake_submit_job)
        instance.shutdown_async = AsyncMock()
        instance.shutdown = MagicMock()
        mock.return_value = instance
        yield mock


class TestBigtableConnector:
    def test_init_and_lazy_pool(self, async_loop, local_cpu_backend):
        """Verify independent lazy async client pool initialization."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(return_value=None)
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        assert isinstance(connector, BigtableConnector)
        assert connector.cfg.project_id == "test-project"
        assert connector.cfg.instance_id == "test-instance"
        assert connector.cfg.table_name == "test-table"

        # Client pool should not be initialized yet (lazy)
        assert connector._client is None

        # Trigger first operation
        key = create_test_key(1)

        assert not backend.contains(key)
        # Pool now initialized
        assert connector._client is not None
        mock_data.BigtableDataClientAsync.assert_called_once_with(
            project="test-project", client_info=ANY
        )

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_out_of_tree_plugin_init(self, async_loop, local_cpu_backend):
        """Verify standalone out-of-tree plugin initialization via
        class_name/module_path.
        """
        config = create_test_config(
            {
                "remote_storage_plugin.bigtable.module_path": "lmc_bigtable_connector",
                "remote_storage_plugin.bigtable.class_name": "BigtableConnector",
                "bigtable_project_id": "ext-project",
                "bigtable_instance_id": "ext-instance",
                "bigtable_table_name": "ext-table",
            }
        )
        metadata = create_test_metadata()

        mock_plugin_module = MagicMock()
        mock_plugin_module.BigtableConnector = BigtableConnector
        sys.modules["lmc_bigtable_connector"] = mock_plugin_module

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        assert connector.__class__.__name__ == "BigtableConnector"
        assert connector.cfg.project_id == "ext-project"
        assert connector.cfg.instance_id == "ext-instance"
        assert connector.cfg.table_name == "ext-table"

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_credentials_path_init(self, async_loop, local_cpu_backend):
        """Verify passing credentials_path initializes client with
        explicit service account.
        """
        config = create_test_config(
            {"bigtable_credentials_path": "/path/to/creds.json"}
        )
        metadata = create_test_metadata()

        mock_creds = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = (
            mock_creds
        )

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        connector._get_client()

        mock_service_account.Credentials.from_service_account_file.assert_called_once_with(
            "/path/to/creds.json"
        )
        mock_data.BigtableDataClientAsync.assert_called_with(
            project="test-project", credentials=mock_creds, client_info=ANY
        )

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_custom_row_key_template(self, async_loop, local_cpu_backend):
        """Verify substituting {model} and {hash} placeholders flexibly."""
        config = create_test_config(
            {
                "bigtable_row_key_template": "{model}#{hash}",
                "bigtable_layer_group_size": 0,
            }
        )
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(123)
        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        row_key = connector.schema.get_row_key(key)
        row_key_str = row_key.decode("utf-8")

        assert row_key_str.startswith("test_model@3@1@bfloat16#")
        assert row_key_str.endswith(key.chunk_hash_hex)

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_local_ttl_cache_hits(self, async_loop, local_cpu_backend):
        """Verify that exists calls accelerate via local TTL cache
        without hitting Bigtable.
        """
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(return_value=MagicMock())
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        assert backend.contains(key)
        assert mock_client_instance.read_row.call_count == 1

        assert backend.contains(key)
        assert mock_client_instance.read_row.call_count == 1

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_get_blocking_hit(self, async_loop, local_cpu_backend):
        config = create_test_config(extra_overrides={"bigtable_layer_group_size": 0})
        metadata = create_test_metadata(kv_shape=(1, 2, 16, 8, 128), chunk_size=16)
        local_cpu_backend.metadata = metadata

        memory_obj = create_test_memory_obj()
        expected_bytes = bytes(memory_obj.byte_array)

        mock_cell = MagicMock(value=expected_bytes, timestamp_micros=0)
        mock_row = MagicMock()
        mock_row.cells = {"cf": {b"data": [mock_cell]}}

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(return_value=mock_row)
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        res = backend.get_blocking(key)
        assert res is not None
        assert isinstance(res, MemoryObj)
        assert bytes(res.byte_array)[: len(expected_bytes)] == expected_bytes

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        assert connector.exists_cache.get(key.to_string()) is True

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_cell_size_validation_skips_large_writes(
        self, async_loop, local_cpu_backend
    ):
        """Validate chunk size at write time. Skips > 90MB rather than failing."""
        config = create_test_config(extra_overrides={"bigtable_layer_group_size": 0})
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.mutate_row = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)
        large_view = bytearray(95 * 1024 * 1024)
        mock_memory_obj = MagicMock()
        mock_memory_obj.byte_array = large_view

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        asyncio.run_coroutine_threadsafe(
            connector._put_internal(key, mock_memory_obj),
            async_loop,
        ).result()

        mock_client_instance.mutate_row.assert_not_called()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_error_handling_timeouts_as_miss(self, async_loop, local_cpu_backend):
        """TimeoutError / DeadlineExceeded treated as cache miss and fall through."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(
            side_effect=MockDeadlineExceeded("timeout")
        )
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        assert not backend.contains(key)

        res = backend.get_blocking(key)
        assert res is None

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_error_handling_auth_raises(self, async_loop, local_cpu_backend):
        """PermissionDenied / Unauthenticated propagate up."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(
            side_effect=MockPermissionDenied("auth error")
        )
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        with pytest.raises(MockPermissionDenied):
            asyncio.run_coroutine_threadsafe(connector.exists(key), async_loop).result()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_batched_get(self, async_loop, local_cpu_backend):
        """Verify batched_get retrieves multiple memory objects cleanly."""
        config = create_test_config(extra_overrides={"bigtable_layer_group_size": 0})
        metadata = create_test_metadata(kv_shape=(1, 2, 16, 8, 128), chunk_size=16)
        local_cpu_backend.metadata = metadata

        memory_obj1 = create_test_memory_obj()
        memory_obj2 = create_test_memory_obj()
        bytes1 = bytes(memory_obj1.byte_array)
        bytes2 = bytes(memory_obj2.byte_array)

        mock_client_instance = MagicMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key1 = create_test_key(1)
        key2 = create_test_key(2)

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        row_key1 = connector.schema.get_row_key(key1)
        row_key2 = connector.schema.get_row_key(key2)

        mock_cell1 = MagicMock(value=bytes1, timestamp_micros=0)
        mock_cell2 = MagicMock(value=bytes2, timestamp_micros=0)
        mock_row1 = MagicMock(row_key=row_key1)
        mock_row2 = MagicMock(row_key=row_key2)
        mock_row1.cells = {"cf": {b"data": [mock_cell1]}}
        mock_row2.cells = {"cf": {b"data": [mock_cell2]}}
        mock_client_instance.read_rows = AsyncMock(return_value=[mock_row1, mock_row2])

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.batched_get([key1, key2]),
            async_loop,
        ).result()

        assert len(res) == 2
        assert res[0] is not None and res[1] is not None
        assert bytes(res[0].byte_array)[: len(bytes1)] == bytes1
        assert bytes(res[1].byte_array)[: len(bytes2)] == bytes2

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_batched_put(self, async_loop, memory_allocator):
        """Verify batched_put packs mutations and sends bulk mutate rows cleanly."""
        config = create_test_config()
        metadata = create_test_metadata(kv_shape=(4, 2, 256, 8, 128))
        local_cpu_backend = LocalCPUBackend(
            LMCacheEngineConfig.from_legacy(chunk_size=256),
            metadata,
            memory_allocator=memory_allocator,
        )

        mock_client_instance = MagicMock()
        mock_client_instance.bulk_mutate_rows = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key1 = create_test_key(1)
        key2 = create_test_key(2)
        memory_obj1 = create_test_memory_obj(shape=metadata.get_shapes()[0])
        memory_obj2 = create_test_memory_obj(shape=metadata.get_shapes()[0])

        asyncio.run_coroutine_threadsafe(
            backend.connection.batched_put([key1, key2], [memory_obj1, memory_obj2]),
            async_loop,
        ).result()

        mock_client_instance.bulk_mutate_rows.assert_called_once()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_batched_async_contains(self, async_loop, local_cpu_backend):
        """Verify batched_async_contains returns correct match counts."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key1 = create_test_key(1)
        key2 = create_test_key(2)

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        row_key1 = connector.schema.get_row_key(key1)
        mock_row = MagicMock(row_key=row_key1)
        mock_client_instance.read_rows = AsyncMock(return_value=[mock_row])

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.batched_async_contains("test", [key1, key2]),
            async_loop,
        ).result()

        # Only key1 rowkey starts with matched row_key
        # from read_rows, so match count is 1
        assert res == 1

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_remove_sync(self, async_loop, local_cpu_backend):
        """Verify remove_sync issues DeleteAllFromRow mutation cleanly."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.mutate_row = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)
        res = backend.connection.remove_sync(key)

        assert res is True
        # Wait up to 1 second for background deletion task to execute
        # Standard
        import time

        for _ in range(100):
            if mock_client_instance.mutate_row.call_count == 1:
                break
            time.sleep(0.01)
        mock_client_instance.mutate_row.assert_called_once()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_list(self, async_loop, local_cpu_backend):
        """Verify list() returns parsed standardized LMCache key metadata formats."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_row1 = MagicMock(row_key=b"hash1#test_model@3@1@bfloat16")
        mock_row2 = MagicMock(row_key=b"hash2#test_model@3@1@bfloat16")

        mock_client_instance = MagicMock()
        mock_client_instance.read_rows = AsyncMock(return_value=[mock_row1, mock_row2])
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.list(),
            async_loop,
        ).result()

        assert len(res) == 2
        assert "test_model@3@1@hash1@bfloat16" in res
        assert "test_model@3@1@hash2@bfloat16" in res

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_ping(self, async_loop, local_cpu_backend):
        """Verify ping executes SELECT 1; health queries cleanly."""
        config = create_test_config()
        metadata = create_test_metadata()

        async def mock_query_iter():
            yield MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.execute_query = AsyncMock(return_value=mock_query_iter())
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.ping(),
            async_loop,
        ).result()

        assert res == 0
        mock_client_instance.execute_query.assert_called_once()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_get_client_credentials_success(self, async_loop, local_cpu_backend):
        config = create_test_config(
            extra_overrides={"bigtable_credentials_path": "/path/to/fake_creds.json"}
        )
        connector = BigtableConnector(async_loop, local_cpu_backend, config)

        mock_creds = MagicMock()
        with (
            patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                return_value=mock_creds,
            ) as mock_from_file,
            patch(
                "google.cloud.bigtable.data.BigtableDataClientAsync"
            ) as mock_client_cls,
        ):
            client = connector._get_client()

            mock_from_file.assert_called_once_with("/path/to/fake_creds.json")
            mock_client_cls.assert_called_once()
            assert mock_client_cls.call_args[1]["credentials"] == mock_creds
            assert mock_client_cls.call_args[1]["project"] == "test-project"
            assert client == mock_client_cls.return_value

    def test_get_client_credentials_os_error(self, async_loop, local_cpu_backend):
        config = create_test_config(
            extra_overrides={
                "bigtable_credentials_path": "/path/to/permission_denied_creds.json"
            }
        )
        connector = BigtableConnector(async_loop, local_cpu_backend, config)

        with (
            patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                side_effect=PermissionError("Permission denied"),
            ),
            patch(
                "google.cloud.bigtable.data.BigtableDataClientAsync"
            ) as mock_client_cls,
            patch(
                "lmcache.v1.storage_backend.connector.bigtable_connector.logger.warning"
            ) as mock_warning,
        ):
            client = connector._get_client()

            mock_client_cls.assert_called_once()
            assert (
                "credentials" not in mock_client_cls.call_args[1]
                or mock_client_cls.call_args[1]["credentials"] is None
            )
            assert mock_client_cls.call_args[1]["project"] == "test-project"

            mock_warning.assert_called_once()
            assert (
                "Falling back to Application Default Credentials"
                in mock_warning.call_args[0][0]
            )
            assert client == mock_client_cls.return_value

    def test_get_client_credentials_value_error(self, async_loop, local_cpu_backend):
        config = create_test_config(
            extra_overrides={
                "bigtable_credentials_path": "/path/to/corrupted_creds.json"
            }
        )
        connector = BigtableConnector(async_loop, local_cpu_backend, config)

        with (
            patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                side_effect=ValueError("Invalid JSON"),
            ),
            patch(
                "google.cloud.bigtable.data.BigtableDataClientAsync"
            ) as mock_client_cls,
            patch(
                "lmcache.v1.storage_backend.connector.bigtable_connector.logger.warning"
            ) as mock_warning,
        ):
            client = connector._get_client()

            mock_client_cls.assert_called_once()
            assert (
                "credentials" not in mock_client_cls.call_args[1]
                or mock_client_cls.call_args[1]["credentials"] is None
            )
            assert mock_client_cls.call_args[1]["project"] == "test-project"

            mock_warning.assert_called_once()
            assert (
                "Falling back to Application Default Credentials"
                in mock_warning.call_args[0][0]
            )
            assert client == mock_client_cls.return_value

    def test_get_client_credentials_auth_error(self, async_loop, local_cpu_backend):
        config = create_test_config(
            extra_overrides={"bigtable_credentials_path": "/path/to/expired_creds.json"}
        )
        connector = BigtableConnector(async_loop, local_cpu_backend, config)

        # Third Party
        import google.auth.exceptions

        with (
            patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                side_effect=google.auth.exceptions.GoogleAuthError("Auth failed"),
            ),
            patch(
                "google.cloud.bigtable.data.BigtableDataClientAsync"
            ) as mock_client_cls,
            patch(
                "lmcache.v1.storage_backend.connector.bigtable_connector.logger.warning"
            ) as mock_warning,
        ):
            client = connector._get_client()

            mock_client_cls.assert_called_once()
            assert (
                "credentials" not in mock_client_cls.call_args[1]
                or mock_client_cls.call_args[1]["credentials"] is None
            )
            assert mock_client_cls.call_args[1]["project"] == "test-project"

            mock_warning.assert_called_once()
            assert (
                "Falling back to Application Default Credentials"
                in mock_warning.call_args[0][0]
            )
            assert client == mock_client_cls.return_value

    def test_bigtable_config_defaults(self):
        """Verify that Bigtable configuration defaults are consistent."""
        # First Party
        from lmcache.v1.storage_backend.connector.bigtable_config import (
            BigtablePluginConfig,
        )

        # Test defaults directly from class instantiation
        config = BigtablePluginConfig(
            project_id="test-project",
            instance_id="test-instance",
            table_name="test-table",
        )
        assert config.max_chunk_size_mb == 90.0
        assert config.read_timeout_sec == 0.2
        assert config.write_timeout_sec == 0.5
        assert config.max_retries == 3

        # Test defaults loaded via from_extra_config (no explicit overrides)
        extra_config = {
            "bigtable_project_id": "test-project",
            "bigtable_instance_id": "test-instance",
            "bigtable_table_name": "test-table",
        }
        loaded_config = BigtablePluginConfig.from_extra_config(extra_config)
        assert loaded_config.max_chunk_size_mb == 90.0
        assert loaded_config.read_timeout_sec == 0.2
        assert loaded_config.write_timeout_sec == 0.5
        assert loaded_config.max_retries == 3

    def test_sharded_put_and_get(self, async_loop, memory_allocator):
        """Verify sharded writes and reads with layer-group sharding."""
        config = create_test_config(extra_overrides={"bigtable_layer_group_size": 10})
        metadata = create_test_metadata(kv_shape=(32, 2, 256, 8, 128))
        local_cpu_backend = LocalCPUBackend(
            LMCacheEngineConfig.from_legacy(chunk_size=256),
            metadata,
            memory_allocator=memory_allocator,
        )

        connector = BigtableConnector(async_loop, local_cpu_backend, config)

        memory_obj = create_test_memory_obj(shape=metadata.get_shapes()[0])
        expected_bytes = bytes(memory_obj.byte_array)

        # Mock SetCell constructor to return a mock with populated properties
        def fake_set_cell(family, qualifier, value, timestamp_micros=None):
            cell = MagicMock()
            cell.family = family
            cell.qualifier = (
                qualifier.encode("utf-8") if isinstance(qualifier, str) else qualifier
            )
            cell.value = value
            cell.timestamp_micros = (
                timestamp_micros if timestamp_micros is not None else 0
            )
            return cell

        mock_data.SetCell.side_effect = fake_set_cell

        mock_client_instance = MagicMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        mutations_sent = []

        async def fake_mutate_row(row_key, mutations, **kwargs):
            mutations_sent.extend(mutations)
            return MagicMock()

        mock_client_instance.mutate_row = AsyncMock(side_effect=fake_mutate_row)

        key = create_test_key(1)
        try:
            asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj),
                async_loop,
            ).result()

            assert len(mutations_sent) == 4
            qualifiers = {m.qualifier.decode("utf-8") for m in mutations_sent}
            assert qualifiers == {
                "layers_0_9",
                "layers_10_19",
                "layers_20_29",
                "layers_30_31",
            }

            mock_row = MagicMock()
            cells_dict = {}
            for m in mutations_sent:
                cells_dict[m.qualifier] = [MagicMock(value=m.value, timestamp_micros=0)]
            mock_row.cells = {"cf": cells_dict}
            mock_client_instance.read_row = AsyncMock(return_value=mock_row)

            get_res = asyncio.run_coroutine_threadsafe(
                connector.get(key),
                async_loop,
            ).result()

            assert get_res is not None
            assert bytes(get_res.byte_array) == expected_bytes
        finally:
            mock_data.SetCell.side_effect = None

    def test_layerwise_row_keys_do_not_collide(self, async_loop, memory_allocator):
        """Verify that layerwise keys generate distinct row keys and write correctly."""
        # First Party
        from lmcache.utils import LayerCacheEngineKey

        config = create_test_config()
        metadata = create_test_metadata(kv_shape=(1, 2, 256, 8, 128))
        local_cpu_backend = LocalCPUBackend(
            LMCacheEngineConfig.from_defaults(
                chunk_size=256,
                use_layerwise=True,
                remote_storage_plugins=["bigtable"],
            ),
            metadata,
            memory_allocator=memory_allocator,
        )

        connector = BigtableConnector(async_loop, local_cpu_backend, config)

        key_l0 = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=hash(42),
            dtype=torch.bfloat16,
            layer_id=0,
        )
        key_l1 = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=hash(42),
            dtype=torch.bfloat16,
            layer_id=1,
        )

        row_key_l0 = connector.schema.get_row_key(key_l0)
        row_key_l1 = connector.schema.get_row_key(key_l1)

        assert row_key_l0 != row_key_l1
        assert row_key_l0.endswith(b"@layer_0")
        assert row_key_l1.endswith(b"@layer_1")

        row_keys_written = []

        async def fake_mutate_row(row_key, mutations, **kwargs):
            row_keys_written.append(row_key)
            return MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.mutate_row = AsyncMock(side_effect=fake_mutate_row)
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        memory_obj_l0 = create_test_memory_obj(shape=metadata.get_shapes()[0])
        memory_obj_l1 = create_test_memory_obj(shape=metadata.get_shapes()[0])

        asyncio.run_coroutine_threadsafe(
            connector.put(key_l0, memory_obj_l0),
            async_loop,
        ).result()
        asyncio.run_coroutine_threadsafe(
            connector.put(key_l1, memory_obj_l1),
            async_loop,
        ).result()

        assert len(row_keys_written) == 2
        assert row_key_l0 in row_keys_written
        assert row_key_l1 in row_keys_written

    def test_cell_size_validation_skips_large_writes_sharded(
        self, async_loop, local_cpu_backend
    ):
        """Verify sharded writes skip writing if exceeding 240MB."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.mutate_row = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)
        large_view = bytearray(245 * 1024 * 1024)
        mock_memory_obj = MagicMock()
        mock_memory_obj.byte_array = large_view

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        asyncio.run_coroutine_threadsafe(
            connector._put_internal(key, mock_memory_obj),
            async_loop,
        ).result()

        mock_client_instance.mutate_row.assert_not_called()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_dual_mode_write(self, async_loop, memory_allocator):
        """Verify dual-mode write in bigtable_connector:
        - payload < 150MB combines mutations in a single mutate_row.
        - payload >= 150MB sends separate mutate_row calls concurrently.
        """
        config = create_test_config(extra_overrides={"bigtable_layer_group_size": 10})
        metadata = create_test_metadata(kv_shape=(32, 2, 256, 8, 128))
        local_cpu_backend = LocalCPUBackend(
            LMCacheEngineConfig.from_legacy(chunk_size=256),
            metadata,
            memory_allocator=memory_allocator,
        )

        connector = BigtableConnector(async_loop, local_cpu_backend, config)
        mock_client_instance = MagicMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        # 1. Test Small Write (< 150MB)
        memory_obj_small = create_test_memory_obj(shape=metadata.get_shapes()[0])

        mutate_row_calls = []

        async def fake_mutate_row(row_key, mutations, **kwargs):
            mutate_row_calls.append((row_key, mutations))
            return MagicMock()

        mock_client_instance.mutate_row = AsyncMock(side_effect=fake_mutate_row)

        key_small = create_test_key(1)
        asyncio.run_coroutine_threadsafe(
            connector.put(key_small, memory_obj_small),
            async_loop,
        ).result()

        # Should make 1 call to mutate_row with all 4 mutations
        assert len(mutate_row_calls) == 1
        row_key, mutations = mutate_row_calls[0]
        assert len(mutations) == 4  # 4 shards

        # 2. Test Large Write (>= 150MB)
        mutate_row_calls.clear()
        key_large = create_test_key(2)

        memory_obj_large = MagicMock()
        fake_blob = bytearray(100)
        memory_obj_large.byte_array = fake_blob

        # Mock the sharder to return fake shards
        connector.sharder = MagicMock()
        connector.sharder.shard.return_value = {
            "layers_0_9": b"fake_data_1",
            "layers_10_19": b"fake_data_2",
            "layers_20_29": b"fake_data_3",
            "layers_30_31": b"fake_data_4",
        }

        # Patch len in the connector module to simulate 160MB payload
        # Standard
        import builtins

        real_len = builtins.len

        def fake_len(obj):
            if obj is fake_blob:
                return 160 * 1024 * 1024
            return real_len(obj)

        with patch(
            "lmcache.v1.storage_backend.connector.bigtable_connector.len",
            side_effect=fake_len,
        ):
            asyncio.run_coroutine_threadsafe(
                connector.put(key_large, memory_obj_large),
                async_loop,
            ).result()

        # Should make 4 separate calls to mutate_row, each with 1 mutation
        assert len(mutate_row_calls) == 4
        for rk, mutations in mutate_row_calls:
            assert len(mutations) == 1

        asyncio.run_coroutine_threadsafe(connector.close(), async_loop).result()
        local_cpu_backend.memory_allocator.close()
