# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import ConnectorContext
from lmcache.v1.storage_backend.connector.hfbucket_adapter import (
    HFBucketConnectorAdapter,
)
from lmcache.v1.storage_backend.connector.hfbucket_connector import (
    HFBucketConnector,
    HFBucketConnectorConfig,
    encode_hfbucket_object_name,
    parse_hfbucket_handle,
    resolve_hfbucket_connector_config,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


@dataclass(frozen=True)
class FakeBucketEntry:
    """Small stand-in for Hugging Face bucket metadata objects."""

    path: str
    size: int
    type: str


class FakeBucketClient:
    """In-memory bucket client used to unit test the connector."""

    def __init__(self) -> None:
        self.storage: dict[str, bytes] = {}
        self.created_buckets: list[str] = []
        self.deleted_paths: list[str] = []
        self.info_paths_calls: list[tuple[str, ...]] = []
        self.download_calls: list[tuple[str, ...]] = []
        self.fail_upload_after: int | None = None
        self.raise_on_download: Exception | None = None

    def create_bucket(self, bucket_id: str) -> None:
        """Record bucket creation requests."""
        self.created_buckets.append(bucket_id)

    def bucket_info(self, bucket_id: str) -> object:
        """Return a trivial bucket info payload."""
        return {"id": bucket_id}

    def get_paths_info(self, bucket_id: str, paths: Sequence[str]) -> list[object]:
        """Return exact metadata for the requested paths."""
        del bucket_id
        self.info_paths_calls.append(tuple(paths))
        entries: list[object] = [
            FakeBucketEntry(
                path=path,
                size=len(self.storage[path]) if path in self.storage else 0,
                type="file" if path in self.storage else "missing",
            )
            for path in paths
        ]
        return entries

    def list_tree(self, bucket_id: str, prefix: str) -> list[object]:
        """List stored objects under the requested prefix."""
        del bucket_id
        prefix_with_separator = f"{prefix}/" if prefix else ""
        entries: list[object] = []
        for path, payload in sorted(self.storage.items()):
            if prefix and path != prefix and not path.startswith(prefix_with_separator):
                continue
            entries.append(FakeBucketEntry(path=path, size=len(payload), type="file"))
        return entries

    def upload_files(
        self,
        bucket_id: str,
        add: Sequence[tuple[bytes, str]],
    ) -> None:
        """Store uploaded bytes under their remote object paths."""
        del bucket_id
        for index, (payload, path) in enumerate(add):
            self.storage[path] = payload
            if self.fail_upload_after is not None and index >= self.fail_upload_after:
                raise RuntimeError("simulated partial upload failure")

    def download_files(
        self,
        bucket_id: str,
        files: Sequence[tuple[str, str]],
    ) -> None:
        """Write stored objects to the requested local paths."""
        del bucket_id
        self.download_calls.append(tuple(path for path, _ in files))
        for remote_path, local_path in files:
            payload = self.storage.get(remote_path)
            if payload is None:
                continue
            Path(local_path).write_bytes(payload)
        if self.raise_on_download is not None:
            raise self.raise_on_download

    def delete_files(
        self,
        bucket_id: str,
        delete: Sequence[str],
    ) -> None:
        """Remove stored objects."""
        del bucket_id
        for path in delete:
            self.deleted_paths.append(path)
            self.storage.pop(path, None)


def create_test_metadata() -> LMCacheMetadata:
    """Create LMCache metadata with a deterministic full chunk layout."""
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def create_test_config(
    extra_config: dict[str, object] | None = None,
    *,
    plugin_name: str = "hfbucket",
    save_unfull_chunk: bool = False,
) -> LMCacheEngineConfig:
    """Create a plugin-based config for the HFBucket connector."""
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        save_unfull_chunk=save_unfull_chunk,
        lmcache_instance_id="test-hfbucket",
        remote_storage_plugins=[plugin_name],
        extra_config=extra_config or {},
    )


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a deterministic cache key for unit tests."""
    return CacheEngineKey(
        model_name="test/model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


def create_connector(
    tmp_path: Path,
    memory_allocator,
    *,
    plugin_name: str = "hfbucket",
    extra_config: dict[str, object] | None = None,
    save_unfull_chunk: bool = False,
    bucket_client: FakeBucketClient | None = None,
) -> tuple[HFBucketConnector, FakeBucketClient, LMCacheMetadata]:
    """Create a connector with an in-memory fake bucket client."""
    metadata = create_test_metadata()
    config_dict = {
        f"remote_storage_plugin.{plugin_name}.bucket_handle": (
            "hf://buckets/test-org/test-bucket/prod"
        ),
        f"remote_storage_plugin.{plugin_name}.download_tmp_dir": str(tmp_path),
        f"remote_storage_plugin.{plugin_name}.metadata_cache_ttl_secs": 30,
    }
    if extra_config is not None:
        config_dict.update(extra_config)

    config = create_test_config(
        config_dict,
        plugin_name=plugin_name,
        save_unfull_chunk=save_unfull_chunk,
    )
    local_cpu_backend = LocalCPUBackend(
        config,
        metadata,
        memory_allocator=memory_allocator,
    )
    client = bucket_client or FakeBucketClient()
    connector = HFBucketConnector(
        local_cpu_backend=local_cpu_backend,
        config=config,
        metadata=metadata,
        connector_config=resolve_hfbucket_connector_config(config, plugin_name),
        bucket_client=client,
    )
    return connector, client, metadata


def create_full_chunk_memory_obj(
    local_cpu_backend: LocalCPUBackend,
    metadata: LMCacheMetadata,
    fill_byte: int,
) -> tuple[MemoryObj, bytes]:
    """Allocate and initialize a full chunk memory object for upload tests."""
    memory_obj = local_cpu_backend.allocate(
        metadata.get_shapes(),
        metadata.get_dtypes(),
        MemoryFormat.KV_2LTD,
    )
    assert memory_obj is not None
    byte_buffer = memoryview(memory_obj.byte_array).cast("B")
    payload = bytes([fill_byte]) * len(byte_buffer)
    byte_buffer[:] = payload
    return memory_obj, payload


def memory_obj_to_bytes(memory_obj: MemoryObj) -> bytes:
    """Convert a test memory object to raw bytes."""
    return memoryview(memory_obj.byte_array).cast("B").tobytes()


@pytest.mark.parametrize(
    ("bucket_handle", "bucket_id", "object_prefix"),
    [
        ("hf://buckets/my-org/my-bucket", "my-org/my-bucket", ""),
        (
            "hf://buckets/my-org/my-bucket/prod/checkpoints",
            "my-org/my-bucket",
            "prod/checkpoints",
        ),
    ],
)
def test_parse_hfbucket_handle(
    bucket_handle: str,
    bucket_id: str,
    object_prefix: str,
) -> None:
    """Bucket handles should split into bucket id and object prefix."""
    location = parse_hfbucket_handle(bucket_handle)
    assert location.bucket_id == bucket_id
    assert location.object_prefix == object_prefix


def test_adapter_can_parse_plugin_urls() -> None:
    """The adapter should match plugin URLs with and without instances."""
    adapter = HFBucketConnectorAdapter()
    assert adapter.can_parse("plugin://hfbucket")
    assert adapter.can_parse("plugin://hfbucket.us")
    assert not adapter.can_parse("plugin://fs")


def test_resolve_hfbucket_connector_config_uses_full_plugin_name() -> None:
    """Instance-specific plugin config should be resolved from the full name."""
    config = create_test_config(
        {
            "remote_storage_plugin.hfbucket.us.bucket_handle": (
                "hf://buckets/test-org/us-bucket/prod"
            ),
            "remote_storage_plugin.hfbucket.us.token_env": "HF_US_TOKEN",
            "remote_storage_plugin.hfbucket.us.create_bucket_if_missing": True,
            "remote_storage_plugin.hfbucket.us.download_tmp_dir": "/tmp/hf-us",
            "remote_storage_plugin.hfbucket.us.metadata_cache_ttl_secs": 12,
        },
        plugin_name="hfbucket.us",
    )

    connector_config = resolve_hfbucket_connector_config(config, "hfbucket.us")

    assert connector_config.plugin_name == "hfbucket.us"
    assert connector_config.bucket_location.bucket_id == "test-org/us-bucket"
    assert connector_config.bucket_location.object_prefix == "prod"
    assert connector_config.token_env == "HF_US_TOKEN"
    assert connector_config.create_bucket_if_missing is True
    assert connector_config.download_tmp_dir == Path("/tmp/hf-us")
    assert connector_config.metadata_cache_ttl_secs == 12.0


def test_adapter_create_connector_uses_plugin_scoped_config(monkeypatch) -> None:
    """Adapter connector creation should pass through the resolved plugin config."""

    created: dict[str, object] = {}

    class DummyConnector:
        """Capture adapter constructor arguments without invoking the real client."""

        def __init__(
            self,
            local_cpu_backend: object,
            config: LMCacheEngineConfig,
            metadata: LMCacheMetadata,
            connector_config: HFBucketConnectorConfig,
        ) -> None:
            created["local_cpu_backend"] = local_cpu_backend
            created["config"] = config
            created["metadata"] = metadata
            created["connector_config"] = connector_config

    config = create_test_config(
        {
            "remote_storage_plugin.hfbucket.prod.bucket_handle": (
                "hf://buckets/test-org/test-bucket/prod"
            ),
        },
        plugin_name="hfbucket.prod",
    )
    metadata = create_test_metadata()
    loop = asyncio.new_event_loop()
    adapter = HFBucketConnectorAdapter()

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connector.hfbucket_adapter.HFBucketConnector",
        DummyConnector,
    )

    connector = adapter.create_connector(
        ConnectorContext(
            url="plugin://hfbucket.prod",
            loop=loop,
            local_cpu_backend=None,
            config=config,
            metadata=metadata,
            plugin_name="hfbucket.prod",
        )
    )

    assert isinstance(connector, DummyConnector)
    connector_config = created["connector_config"]
    assert isinstance(connector_config, HFBucketConnectorConfig)
    assert connector_config.plugin_name == "hfbucket.prod"
    assert connector_config.bucket_location.object_prefix == "prod"


def test_put_get_exists_list_and_remove_roundtrip(
    tmp_path: Path,
    memory_allocator,
) -> None:
    """Single-object operations should round-trip against the fake client."""
    connector, fake_client, metadata = create_connector(tmp_path, memory_allocator)
    key = create_test_key(1)
    memory_obj, payload = create_full_chunk_memory_obj(
        connector.local_cpu_backend,
        metadata,
        fill_byte=17,
    )

    try:
        asyncio.run(connector.put(key, memory_obj))
        assert fake_client.created_buckets == []

        assert connector.exists_sync(key) is True
        assert asyncio.run(connector.exists(key)) is True

        results = asyncio.run(connector.list())
        assert results == [key.to_string()]

        loaded = asyncio.run(connector.get(key))
        assert loaded is not None
        try:
            assert memory_obj_to_bytes(loaded) == payload
        finally:
            loaded.ref_count_down()

        assert connector.remove_sync(key) is True
        assert connector.exists_sync(key) is False
    finally:
        memory_obj.ref_count_down()
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_batched_put_and_batched_get_preserve_order(
    tmp_path: Path,
    memory_allocator,
) -> None:
    """Batched operations should upload once and preserve result order."""
    connector, fake_client, metadata = create_connector(
        tmp_path,
        memory_allocator,
        extra_config={
            "remote_storage_plugin.hfbucket.create_bucket_if_missing": True,
        },
    )
    keys = [create_test_key(10), create_test_key(11)]
    memory_objs_and_payloads = [
        create_full_chunk_memory_obj(
            connector.local_cpu_backend, metadata, fill_byte=33
        ),
        create_full_chunk_memory_obj(
            connector.local_cpu_backend, metadata, fill_byte=44
        ),
    ]
    memory_objs = [item[0] for item in memory_objs_and_payloads]
    payloads = [item[1] for item in memory_objs_and_payloads]

    try:
        asyncio.run(connector.batched_put(keys, memory_objs))
        assert fake_client.created_buckets == ["test-org/test-bucket"]

        missing_key = create_test_key(12)
        results = asyncio.run(connector.batched_get([keys[0], missing_key, keys[1]]))
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
        try:
            assert memory_obj_to_bytes(results[0]) == payloads[0]
            assert memory_obj_to_bytes(results[2]) == payloads[1]
        finally:
            for result in results:
                if result is not None:
                    result.ref_count_down()
    finally:
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_batched_contains_returns_prefix_hits(tmp_path: Path, memory_allocator) -> None:
    """Prefix hit counting should stop at the first missing or invalid object."""
    connector, fake_client, metadata = create_connector(tmp_path, memory_allocator)
    keys = [create_test_key(20), create_test_key(21), create_test_key(22)]
    try:
        for key, fill_byte in zip(keys[:2], [51, 52], strict=True):
            memory_obj, _ = create_full_chunk_memory_obj(
                connector.local_cpu_backend,
                metadata,
                fill_byte=fill_byte,
            )
            try:
                asyncio.run(connector.put(key, memory_obj))
            finally:
                memory_obj.ref_count_down()

        assert connector.batched_contains(keys) == 2
        assert asyncio.run(connector.batched_async_contains("lookup", keys)) == 2
    finally:
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_rejects_save_unfull_chunk_and_save_chunk_meta(
    tmp_path: Path,
    memory_allocator,
) -> None:
    """Constructor validation should reject unsupported chunk persistence modes."""
    with pytest.raises(ValueError, match="save_unfull_chunk must be False"):
        create_connector(
            tmp_path,
            memory_allocator,
            save_unfull_chunk=True,
        )

    with pytest.raises(ValueError, match="save_chunk_meta must be False"):
        create_connector(
            tmp_path,
            memory_allocator,
            extra_config={"save_chunk_meta": True},
        )


def test_put_rejects_partial_chunk_upload(tmp_path: Path, memory_allocator) -> None:
    """Uploads should reject buffers that are smaller than a full chunk."""
    connector, _, metadata = create_connector(tmp_path, memory_allocator)
    key = create_test_key(30)
    try:
        partial_obj = connector.local_cpu_backend.allocate(
            metadata.get_shapes(num_tokens=128),
            metadata.get_dtypes(),
            MemoryFormat.KV_2LTD,
        )
        assert partial_obj is not None
        with pytest.raises(ValueError, match="Partial/unfull chunks are not supported"):
            asyncio.run(connector.put(key, partial_obj))
    finally:
        partial_obj.ref_count_down()
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_get_rejects_download_size_mismatch(tmp_path: Path, memory_allocator) -> None:
    """Downloads should return None when bucket objects are not full chunks."""
    connector, fake_client, _ = create_connector(tmp_path, memory_allocator)
    key = create_test_key(40)
    fake_client.storage[f"prod/{encode_hfbucket_object_name(key.to_string())}"] = (
        b"x" * 13
    )

    try:
        assert connector.exists_sync(key) is False
        loaded = asyncio.run(connector.get(key))
        assert loaded is None
    finally:
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_close_cleans_temp_dir(tmp_path: Path, memory_allocator) -> None:
    """Connector close should clean up its per-connector temp directory."""
    connector, _, _ = create_connector(tmp_path, memory_allocator)
    download_root_entries = list(tmp_path.iterdir())
    assert download_root_entries, (
        "Expected connector to create a temp session directory"
    )

    asyncio.run(connector.close())

    assert list(tmp_path.iterdir()) == []
    connector.local_cpu_backend.memory_allocator.close()
