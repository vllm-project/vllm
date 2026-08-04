# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native Azure Blob Storage connector.

The ``azure-storage-blob`` / ``azure-identity`` SDKs are mocked at the
``sys.modules`` level so these tests run anywhere (including CPU-only / M1
machines) without the real packages installed.
"""

# Standard
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import asyncio
import sys
import threading

_original_sys_modules: dict[str, Any] = {}
_mocked_modules = [
    "azure",
    "azure.core",
    "azure.core.exceptions",
    "azure.storage",
    "azure.storage.blob",
    "azure.storage.blob.aio",
    "azure.identity",
    "azure.identity.aio",
]


class MockResourceNotFoundError(Exception):
    pass


# ---- build the mock azure package tree -------------------------------------
mock_exceptions = MagicMock()
mock_exceptions.ResourceNotFoundError = MockResourceNotFoundError

# The BlobServiceClient class mock. Both the constructor and
# from_connection_string return the same configurable service-client instance
# so tests can reach in via ``BlobServiceClient.return_value``.
mock_blob_service_cls = MagicMock(name="BlobServiceClient")
mock_blob_service_cls.from_connection_string.return_value = (
    mock_blob_service_cls.return_value
)

mock_aio = MagicMock()
mock_aio.BlobServiceClient = mock_blob_service_cls

mock_identity_aio = MagicMock()
mock_identity_aio.DefaultAzureCredential = MagicMock(name="DefaultAzureCredential")

# Third Party
import pytest  # noqa: E402
import torch  # noqa: E402

# First Party
from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def mock_azure_sdk():
    # Backup and inject mocks
    for name in _mocked_modules:
        if name in sys.modules:
            _original_sys_modules[name] = sys.modules[name]
    sys.modules["azure"] = MagicMock()
    sys.modules["azure.core"] = MagicMock()
    sys.modules["azure.core.exceptions"] = mock_exceptions
    sys.modules["azure.storage"] = MagicMock()
    sys.modules["azure.storage.blob"] = MagicMock()
    sys.modules["azure.storage.blob.aio"] = mock_aio
    sys.modules["azure.identity"] = MagicMock()
    sys.modules["azure.identity.aio"] = mock_identity_aio

    yield

    # Restore
    for name in _mocked_modules:
        if name in _original_sys_modules:
            sys.modules[name] = _original_sys_modules[name]
        elif name in sys.modules:
            del sys.modules[name]


# Imported after the SDK is mocked so the lazy imports resolve to the mocks.
@pytest.fixture(scope="module")
def AzureConnector(mock_azure_sdk):
    # First Party
    from lmcache.v1.storage_backend.connector.azure_connector import (
        AzureConnector as _AzureConnector,
    )

    return _AzureConnector


CONN_STR = (
    "DefaultEndpointsProtocol=https;AccountName=test;"
    "AccountKey=Zm9v;EndpointSuffix=core.windows.net"
)


def create_test_metadata(kv_shape=(1, 2, 16, 8, 128), chunk_size=16):
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
    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="test-azure-loop",
    )
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def local_cpu_backend(memory_allocator):
    config = LMCacheEngineConfig.from_legacy(chunk_size=16)
    metadata = create_test_metadata()
    return LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)


def run(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def make_connector(AzureConnector, async_loop, local_cpu_backend, **kwargs):
    """Build a connector wired to a fresh mock container/blob client tree."""
    container_client = MagicMock(name="container_client")
    blob_client = MagicMock(name="blob_client")
    container_client.get_blob_client.return_value = blob_client

    service = mock_blob_service_cls.return_value
    service.get_container_client.return_value = container_client
    service.close = AsyncMock()

    connector = AzureConnector(
        container="test-container",
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        connection_string=CONN_STR,
        **kwargs,
    )
    return connector, container_client, blob_client


class TestAzureConnector:
    def test_init_connection_string(
        self, AzureConnector, async_loop, local_cpu_backend
    ):
        connector, _, _ = make_connector(AzureConnector, async_loop, local_cpu_backend)
        mock_blob_service_cls.from_connection_string.assert_called_with(CONN_STR)
        assert connector.container == "test-container"

    def test_init_account_key(self, AzureConnector, async_loop, local_cpu_backend):
        mock_blob_service_cls.reset_mock()
        connector = AzureConnector(
            container="c",
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            account_url="https://acct.blob.core.windows.net",
            account_key="secret-key",
        )
        mock_blob_service_cls.assert_called_once()
        _, kwargs = mock_blob_service_cls.call_args
        assert kwargs["account_url"] == "https://acct.blob.core.windows.net"
        assert kwargs["credential"] == "secret-key"
        assert connector.container == "c"

    def test_exists_hit(self, AzureConnector, async_loop, local_cpu_backend):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        blob_client.get_blob_properties = AsyncMock(
            return_value=MagicMock(size=connector.full_chunk_size_bytes)
        )
        key = create_test_key(1)
        assert run(async_loop, connector.exists(key)) is True
        # Second call should hit the size cache (no extra SDK call).
        assert run(async_loop, connector.exists(key)) is True
        blob_client.get_blob_properties.assert_called_once()

    def test_exists_miss(self, AzureConnector, async_loop, local_cpu_backend):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        blob_client.get_blob_properties = AsyncMock(
            side_effect=MockResourceNotFoundError("nope")
        )
        key = create_test_key(2)
        assert run(async_loop, connector.exists(key)) is False

    def test_get_hit(self, AzureConnector, async_loop, local_cpu_backend):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        size = connector.full_chunk_size_bytes
        payload = bytes(range(256)) * (size // 256) + bytes(size % 256)
        blob_client.get_blob_properties = AsyncMock(return_value=MagicMock(size=size))

        async def fake_readinto(buf):
            mv = memoryview(buf).cast("B")
            mv[: len(payload)] = payload[: len(mv)]
            return len(payload)

        downloader = MagicMock()
        downloader.readinto = AsyncMock(side_effect=fake_readinto)
        blob_client.download_blob = AsyncMock(return_value=downloader)

        key = create_test_key(3)
        res = run(async_loop, connector.get(key))
        assert isinstance(res, MemoryObj)
        assert bytes(res.byte_array)[: len(payload)] == payload[: len(res.byte_array)]
        res.ref_count_down()

    def test_get_miss_returns_none(self, AzureConnector, async_loop, local_cpu_backend):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        blob_client.get_blob_properties = AsyncMock(
            side_effect=MockResourceNotFoundError("nope")
        )
        key = create_test_key(4)
        assert run(async_loop, connector.get(key)) is None

    def test_get_size_mismatch_returns_none(
        self, AzureConnector, async_loop, local_cpu_backend
    ):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        # Report a wrong size -> connector must reject and not download.
        blob_client.get_blob_properties = AsyncMock(
            return_value=MagicMock(size=connector.full_chunk_size_bytes + 8)
        )
        blob_client.download_blob = AsyncMock()
        key = create_test_key(5)
        assert run(async_loop, connector.get(key)) is None
        blob_client.download_blob.assert_not_called()

    def test_put(self, AzureConnector, async_loop, local_cpu_backend):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        blob_client.upload_blob = AsyncMock()

        # Allocate a full chunk via the backend (same path get() uses); its
        # physical size matches the connector's expected part size.
        memory_obj = local_cpu_backend.allocate(
            connector.meta_shapes,
            connector.meta_dtypes,
            connector.meta_fmt,
        )
        assert memory_obj.get_physical_size() == connector.part_size

        key = create_test_key(6)
        run(async_loop, connector.put(key, memory_obj))

        blob_client.upload_blob.assert_called_once()
        _, kwargs = blob_client.upload_blob.call_args
        assert kwargs.get("overwrite") is True
        memory_obj.ref_count_down()

    def test_put_size_mismatch_skips(
        self, AzureConnector, async_loop, local_cpu_backend
    ):
        connector, _, blob_client = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )
        blob_client.upload_blob = AsyncMock()

        memory_obj = MagicMock()
        memory_obj.get_physical_size.return_value = connector.part_size + 1

        key = create_test_key(7)
        run(async_loop, connector.put(key, memory_obj))
        blob_client.upload_blob.assert_not_called()

    def test_list(self, AzureConnector, async_loop, local_cpu_backend):
        connector, container_client, _ = make_connector(
            AzureConnector, async_loop, local_cpu_backend
        )

        async def fake_list_blobs():
            for name in ["blob_a", "blob_b"]:
                # ``name`` is reserved by MagicMock's constructor; set it after.
                blob = MagicMock()
                blob.name = name
                yield blob

        container_client.list_blobs = MagicMock(side_effect=fake_list_blobs)
        res = run(async_loop, connector.list())
        assert res == ["blob_a", "blob_b"]

    def test_close(self, AzureConnector, async_loop, local_cpu_backend):
        connector, _, _ = make_connector(AzureConnector, async_loop, local_cpu_backend)
        run(async_loop, connector.close())
        connector.service_client.close.assert_called_once()
