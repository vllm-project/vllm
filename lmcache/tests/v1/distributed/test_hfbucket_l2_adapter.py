# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the HFBucket MP L2 adapter."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters import hfbucket_l2_adapter as hfmod
from lmcache.v1.distributed.l2_adapters.hfbucket_l2_adapter import (
    HFBucketL2Adapter,
    HFBucketL2AdapterConfig,
    _object_key_to_bucket_path,
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd
from lmcache.v1.storage_backend.connector.hfbucket_connector import (
    parse_hfbucket_handle,
)

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

_TEST_BUCKET_HANDLE = "hf://buckets/test-org/test-bucket/prod"
_TEST_BUCKET_LOCATION = parse_hfbucket_handle(_TEST_BUCKET_HANDLE)


@dataclass(frozen=True)
class _FakePathInfo:
    path: str
    type: str
    size: int


class _FakeBucketClient:
    """In-memory HFBucket client used by adapter unit tests."""

    def __init__(self) -> None:
        self.storage: dict[str, bytes] = {}
        self.created_buckets: list[str] = []
        self.deleted_paths: list[str] = []
        self.fail_upload_after: int | None = None
        self._lock = threading.Lock()

    def create_bucket(self, bucket_id: str) -> None:
        with self._lock:
            self.created_buckets.append(bucket_id)

    def bucket_info(self, bucket_id: str) -> object:
        return {"bucket_id": bucket_id}

    def get_paths_info(
        self,
        bucket_id: str,
        paths: Sequence[str],
    ) -> list[object]:
        del bucket_id
        with self._lock:
            return [
                _FakePathInfo(path=path, type="file", size=len(self.storage[path]))
                for path in paths
                if path in self.storage
            ]

    def list_tree(self, bucket_id: str, prefix: str) -> list[object]:
        del bucket_id
        with self._lock:
            return [
                _FakePathInfo(path=path, type="file", size=len(data))
                for path, data in self.storage.items()
                if not prefix or path.startswith(prefix)
            ]

    def upload_files(
        self,
        bucket_id: str,
        add: Sequence[tuple[bytes, str]],
    ) -> None:
        del bucket_id
        with self._lock:
            for index, (data, path) in enumerate(add, start=1):
                self.storage[path] = bytes(data)
                if (
                    self.fail_upload_after is not None
                    and index >= self.fail_upload_after
                ):
                    raise RuntimeError("injected partial upload failure")

    def download_files(
        self,
        bucket_id: str,
        files: Sequence[tuple[str, str]],
    ) -> None:
        del bucket_id
        with self._lock:
            items = [
                (remote, local, self.storage.get(remote)) for remote, local in files
            ]

        for _remote, local, data in items:
            if data is None:
                continue
            Path(local).write_bytes(data)

    def delete_files(
        self,
        bucket_id: str,
        delete: Sequence[str],
    ) -> None:
        del bucket_id
        with self._lock:
            for path in delete:
                self.deleted_paths.append(path)
                self.storage.pop(path, None)

    def contains(self, path: str) -> bool:
        with self._lock:
            return path in self.storage


def create_object_key(chunk_id: int, model_name: str = "test/model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> MemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def bucket_path_for_key(key: ObjectKey) -> str:
    return _object_key_to_bucket_path(key, _TEST_BUCKET_LOCATION)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


@pytest.fixture
def fake_client() -> _FakeBucketClient:
    return _FakeBucketClient()


@pytest.fixture
def adapter(tmp_path: Path, fake_client: _FakeBucketClient):
    cfg = HFBucketL2AdapterConfig(
        bucket_handle=_TEST_BUCKET_HANDLE,
        download_tmp_dir=str(tmp_path),
        metadata_cache_ttl_secs=30,
        num_workers=2,
        max_capacity_gb=0.001,
    )
    adapter = HFBucketL2Adapter(cfg, bucket_client=fake_client)
    yield adapter
    adapter.close()


class _RecordingListener(L2AdapterListener):
    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys, sizes):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys):
        self.deleted.append(list(keys))


class TestObjectKeySerialization:
    def test_format(self) -> None:
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        assert _object_key_to_string(key) == "llama@000000ff@0@00010203"

    def test_object_group_id_embedded(self) -> None:
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            object_group_id=5,
        )
        assert _object_key_to_string(key) == "llama@000000ff@5@00010203"

    def test_cache_salt_appended(self) -> None:
        base_key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        salted = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            cache_salt="user-42",
        )
        assert _object_key_to_string(base_key) == "llama@000000ff@0@00010203"
        assert _object_key_to_string(salted) == "llama@000000ff@0@00010203@user-42"
        assert _object_key_to_string(base_key) != _object_key_to_string(salted)

    def test_bucket_path_uses_prefix_and_encoding(self) -> None:
        cfg = HFBucketL2AdapterConfig(bucket_handle=_TEST_BUCKET_HANDLE)
        key = create_object_key(1)
        path = _object_key_to_bucket_path(key, cfg.bucket_location)
        assert path.startswith("prod/")
        assert "/" not in path.removeprefix("prod/")


class TestEventFdInterface:
    def test_three_distinct_fds(self, adapter: HFBucketL2Adapter) -> None:
        a = adapter.get_store_event_fd()
        b = adapter.get_lookup_and_lock_event_fd()
        c = adapter.get_load_event_fd()
        assert a >= 0 and b >= 0 and c >= 0
        assert len({a, b, c}) == 3


class TestStoreLookupLoad:
    def test_roundtrip_single_key(self, adapter: HFBucketL2Adapter) -> None:
        key = create_object_key(1)
        obj = create_memory_obj(fill_value=3.14)

        tid = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[tid].is_successful()

        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is True

        dst = create_memory_obj(fill_value=0.0)
        tid = adapter.submit_load_task([key], [dst])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is True
        assert torch.allclose(dst.tensor, torch.full((16,), 3.14))

    def test_partial_hits(self, adapter: HFBucketL2Adapter) -> None:
        stored = [create_object_key(0), create_object_key(2)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(2)]
        adapter.submit_store_task(stored, objs)
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        keys = [create_object_key(i) for i in range(4)]
        tid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None
        assert bm.test(0) is True
        assert bm.test(1) is False
        assert bm.test(2) is True
        assert bm.test(3) is False

    def test_load_miss_returns_zero_bit(self, adapter: HFBucketL2Adapter) -> None:
        key = create_object_key(99)
        dst = create_memory_obj()
        tid = adapter.submit_load_task([key], [dst])
        wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is False

    def test_load_size_mismatch_returns_zero_bit(
        self,
        adapter: HFBucketL2Adapter,
        fake_client: _FakeBucketClient,
    ) -> None:
        key = create_object_key(7)
        object_path = bucket_path_for_key(key)
        fake_client.storage[object_path] = b"too-small"

        dst = create_memory_obj()
        tid = adapter.submit_load_task([key], [dst])
        wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is False

    def test_query_lookup_returns_none_after_pop(
        self,
        adapter: HFBucketL2Adapter,
    ) -> None:
        key = create_object_key(1)
        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(tid) is not None
        assert adapter.query_lookup_and_lock_result(tid) is None

    def test_partial_store_failure_accounts_written_keys(
        self,
        adapter: HFBucketL2Adapter,
        fake_client: _FakeBucketClient,
    ) -> None:
        fake_client.fail_upload_after = 1
        keys = [create_object_key(0), create_object_key(1)]
        objs = [create_memory_obj(), create_memory_obj()]

        tid = adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert not adapter.pop_completed_store_tasks()[tid].is_successful()

        assert fake_client.contains(bucket_path_for_key(keys[0]))
        assert not fake_client.contains(bucket_path_for_key(keys[1]))
        assert adapter.get_usage().total_bytes_used == 64


class TestEviction:
    def _store(self, adapter: HFBucketL2Adapter, key: ObjectKey) -> None:
        adapter.submit_store_task([key], [create_memory_obj()])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

    def _lookup(self, adapter: HFBucketL2Adapter, key: ObjectKey):
        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        return adapter.query_lookup_and_lock_result(tid)

    def test_delete_removes_key(
        self,
        adapter: HFBucketL2Adapter,
        fake_client: _FakeBucketClient,
    ) -> None:
        key = create_object_key(1)
        self._store(adapter, key)
        object_path = bucket_path_for_key(key)
        assert fake_client.contains(object_path)

        adapter.delete([key])
        assert not fake_client.contains(object_path)

    def test_lock_blocks_delete(
        self,
        adapter: HFBucketL2Adapter,
        fake_client: _FakeBucketClient,
    ) -> None:
        key = create_object_key(1)
        self._store(adapter, key)
        bm = self._lookup(adapter, key)
        assert bm is not None and bm.test(0) is True

        deletes_before = len(fake_client.deleted_paths)
        adapter.delete([key])
        assert len(fake_client.deleted_paths) == deletes_before

        adapter.submit_unlock([key])
        adapter.delete([key])
        object_path = bucket_path_for_key(key)
        assert not fake_client.contains(object_path)

    def test_refcount_unlock(
        self,
        adapter: HFBucketL2Adapter,
        fake_client: _FakeBucketClient,
    ) -> None:
        key = create_object_key(1)
        self._store(adapter, key)
        self._lookup(adapter, key)
        self._lookup(adapter, key)

        adapter.submit_unlock([key])
        adapter.delete([key])
        object_path = bucket_path_for_key(key)
        assert fake_client.contains(object_path)

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not fake_client.contains(object_path)

    def test_delete_on_unknown_key(self, adapter: HFBucketL2Adapter) -> None:
        adapter.delete([create_object_key(42)])


class TestGetUsage:
    def test_disabled_returns_minus_one(self, tmp_path: Path) -> None:
        cfg = HFBucketL2AdapterConfig(
            bucket_handle="hf://buckets/test-org/test-bucket",
            download_tmp_dir=str(tmp_path),
            max_capacity_gb=0.0,
        )
        adapter = HFBucketL2Adapter(cfg, bucket_client=_FakeBucketClient())
        try:
            usage = adapter.get_usage()
            # 0/0 is defined as -1.0 to indicate disabled
            assert usage.usage_fraction == -1.0
            assert usage.total_bytes_used == 0
            assert usage.total_capacity_bytes == 0
        finally:
            adapter.close()

    def test_usage_grows_on_store_and_shrinks_on_delete(
        self,
        adapter: HFBucketL2Adapter,
    ) -> None:
        keys = [create_object_key(i) for i in range(4)]
        objs = [create_memory_obj() for _ in range(4)]

        adapter.submit_store_task(keys, objs)
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        total = 4 * 64
        capacity = int(0.001 * 1024**3)
        usage = adapter.get_usage()
        assert usage.total_bytes_used == total
        assert usage.total_capacity_bytes == capacity
        assert usage.usage_fraction == pytest.approx(total / capacity)

        adapter.delete(keys)
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 0
        assert usage.usage_fraction == 0.0


class TestListener:
    def test_stored_accessed_and_deleted_fire(
        self,
        adapter: HFBucketL2Adapter,
    ) -> None:
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        adapter.submit_store_task([key], [create_memory_obj()])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()
        time.sleep(0.05)
        assert any(key in batch for batch in listener.stored)

        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(tid)
        time.sleep(0.05)
        assert any(key in batch for batch in listener.accessed)
        accessed_count = len(listener.accessed)

        dst = create_memory_obj(fill_value=0.0)
        tid = adapter.submit_load_task([key], [dst])
        wait_for_event_fd(adapter.get_load_event_fd())
        adapter.query_load_result(tid)
        time.sleep(0.05)
        assert len(listener.accessed) == accessed_count

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert any(key in batch for batch in listener.deleted)


class TestConfig:
    def test_from_dict_requires_bucket_handle(self) -> None:
        with pytest.raises(ValueError):
            HFBucketL2AdapterConfig.from_dict({"type": "hfbucket"})

    def test_from_dict_parses_all_fields(self) -> None:
        cfg = HFBucketL2AdapterConfig.from_dict(
            {
                "type": "hfbucket",
                "bucket_handle": _TEST_BUCKET_HANDLE,
                "token_env": "HF_TEST_TOKEN",
                "token": "direct-token",
                "create_bucket_if_missing": True,
                "download_tmp_dir": "/tmp/hf",
                "metadata_cache_ttl_secs": 12.5,
                "num_workers": 8,
                "max_capacity_gb": 2.5,
            }
        )
        assert cfg.bucket_handle == _TEST_BUCKET_HANDLE
        assert cfg.bucket_location.bucket_id == "test-org/test-bucket"
        assert cfg.bucket_location.object_prefix == "prod"
        assert cfg.token_env == "HF_TEST_TOKEN"
        assert cfg.token == "direct-token"
        assert cfg.create_bucket_if_missing is True
        assert cfg.download_tmp_dir == Path("/tmp/hf")
        assert cfg.metadata_cache_ttl_secs == 12.5
        assert cfg.num_workers == 8
        assert cfg.max_capacity_gb == 2.5

    # strict boolean parsing
    def test_from_dict_rejects_string_boolean(self) -> None:
        with pytest.raises(ValueError, match="create_bucket_if_missing"):
            HFBucketL2AdapterConfig.from_dict(
                {
                    "type": "hfbucket",
                    "bucket_handle": _TEST_BUCKET_HANDLE,
                    "create_bucket_if_missing": "false",
                }
            )

    def test_help_nonempty(self) -> None:
        assert isinstance(HFBucketL2AdapterConfig.help(), str)
        assert "bucket_handle" in HFBucketL2AdapterConfig.help()


class TestFactoryRegistration:
    def test_create_l2_adapter_registers_hfbucket(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        monkeypatch.setattr(
            hfmod,
            "HFBucketClient",
            lambda token=None: _FakeBucketClient(),
        )
        cfg = HFBucketL2AdapterConfig.from_dict(
            {
                "type": "hfbucket",
                "bucket_handle": _TEST_BUCKET_HANDLE,
                "download_tmp_dir": str(tmp_path),
                "num_workers": 1,
            }
        )
        adapter = create_l2_adapter(cfg)
        try:
            assert isinstance(adapter, HFBucketL2Adapter)
        finally:
            adapter.close()


class TestCleanup:
    def test_close_cleans_temp_dir(
        self,
        tmp_path: Path,
        fake_client: _FakeBucketClient,
    ) -> None:
        cfg = HFBucketL2AdapterConfig(
            bucket_handle=_TEST_BUCKET_HANDLE,
            download_tmp_dir=str(tmp_path),
        )
        adapter = HFBucketL2Adapter(cfg, bucket_client=fake_client)
        assert list(tmp_path.iterdir())

        adapter.close()

        assert list(tmp_path.iterdir()) == []
