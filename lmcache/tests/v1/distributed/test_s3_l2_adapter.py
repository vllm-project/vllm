# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for S3L2Adapter.

The real ``awscrt.s3.S3Request`` is replaced with an in-memory fake that
routes PUT/GET/HEAD/DELETE against a shared dict.  No network required.
"""

# Standard
from concurrent.futures import Future as ConcurrentFuture
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters import s3_l2_adapter as s3mod
from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
    S3L2Adapter,
    S3L2AdapterConfig,
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# =============================================================================
# Fake awscrt.s3.S3Request
# =============================================================================


class _FakeBackend:
    """In-memory backing store shared by all fake S3Requests in a test."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        # When non-None, every request raises with this message
        # (used to simulate network failures for the circuit-breaker test).
        self._inject_error: str | None = None
        self._put_count = 0
        self._get_count = 0
        self._delete_count = 0
        self._head_count = 0

    def reset(self):
        with self._lock:
            self._data.clear()
            self._inject_error = None
            self._put_count = self._get_count = 0
            self._delete_count = self._head_count = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, data: bytes):
        with self._lock:
            self._data[key] = data

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def set_error(self, msg: str | None):
        with self._lock:
            self._inject_error = msg

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "put": self._put_count,
                "get": self._get_count,
                "delete": self._delete_count,
                "head": self._head_count,
            }


_BACKEND = _FakeBackend()


def _path_to_key(path: str) -> str:
    # Adapter uses url-encoded key in path; "@" is %-encoded. Reverse by
    # unquoting.  Simpler: path always starts with "/", strip it, unquote.
    # Standard
    from urllib.parse import unquote

    return unquote(path.lstrip("/"))


def _build_list_objects_v2_response(
    backend: "_FakeBackend",
    prefix: str | None,
    max_keys: int,
    continuation_token: str | None,
) -> bytes:
    """Render a minimal ListObjectsV2 XML response from the fake backend.

    The adapter's parser only consumes ``<Contents>/<Key>`` +
    ``<Contents>/<Size>`` and ``<NextContinuationToken>``, so other
    fields can be omitted. Continuation tokens here are just the
    "start-after-this-key" cursor — opaque to the adapter, simple
    enough for tests.
    """
    with backend._lock:
        all_keys = sorted(backend._data.keys())
        sizes = {k: len(backend._data[k]) for k in all_keys}

    if prefix:
        all_keys = [k for k in all_keys if k.startswith(prefix)]
    if continuation_token is not None:
        # In our fake, the token IS the last key returned on the
        # previous page. Start strictly after it.
        all_keys = [k for k in all_keys if k > continuation_token]

    page = all_keys[:max_keys]
    truncated = len(all_keys) > max_keys
    next_token = page[-1] if (truncated and page) else None

    body = ["<?xml version='1.0' encoding='UTF-8'?>"]
    body.append("<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>")
    body.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    for k in page:
        body.append(f"<Contents><Key>{k}</Key><Size>{sizes[k]}</Size></Contents>")
    if next_token is not None:
        body.append(f"<NextContinuationToken>{next_token}</NextContinuationToken>")
    body.append("</ListBucketResult>")
    return "".join(body).encode("utf-8")


def _parse_list_query(path: str) -> tuple[str | None, int, str | None]:
    """Pull ``prefix`` / ``max-keys`` / ``continuation-token`` out of the
    ListObjectsV2 query string."""
    # Standard
    from urllib.parse import parse_qs, urlsplit

    qs = parse_qs(urlsplit(path).query)
    prefix = qs.get("prefix", [None])[0]
    max_keys = int(qs.get("max-keys", ["1000"])[0])
    cont = qs.get("continuation-token", [None])[0]
    return prefix, max_keys, cont


class _FakeS3Request:
    """In-process substitute for awscrt.s3.S3Request.

    Dispatches by HTTP method + operation_name against ``_BACKEND`` and
    fires the on_headers / on_body / on_done callbacks that the real CRT
    would invoke, then completes the finished_future.
    """

    def __init__(
        self,
        *,
        client,
        type,  # noqa: A002
        request,
        credential_provider,
        region,
        on_body=None,
        on_done=None,
        on_headers=None,
        operation_name=None,
        **kwargs,
    ):
        self.finished_future: ConcurrentFuture = ConcurrentFuture()

        # Extract method + path from the HttpRequest
        method = request.method
        path = request.path

        # ListObjectsV2 is dispatched as GET on the bucket root with a
        # ``?list-type=2&...`` query string — intercept before the
        # per-key GET branch below so it doesn't try to look up an
        # object named "".
        if (
            method == "GET"
            and operation_name == "ListObjectsV2"
            and path.startswith("/?")
        ):
            prefix, max_keys, cont = _parse_list_query(path)
            xml = _build_list_objects_v2_response(_BACKEND, prefix, max_keys, cont)
            try:
                if on_headers is not None:
                    on_headers(200, [("content-type", "application/xml")])
                if on_body is not None:
                    on_body(xml, 0)
                if on_done is not None:
                    on_done(error=None, status_code=200)
                self.finished_future.set_result(None)
            except Exception as e:
                if not self.finished_future.done():
                    self.finished_future.set_exception(e)
            return
        key_str = _path_to_key(path)

        err_inj = _BACKEND._inject_error

        try:
            if err_inj:
                raise RuntimeError(err_inj)

            if method == "PUT":
                data = b""
                body = request.body_stream
                # body is a MemoryViewStream — read it fully
                body.seek(0)
                data = bytes(body.read())
                _BACKEND.put(key_str, data)
                with _BACKEND._lock:
                    _BACKEND._put_count += 1
                if on_headers is not None:
                    on_headers(200, [])
                if on_done is not None:
                    on_done(error=None, status_code=200)
                self.finished_future.set_result(None)

            elif method == "GET":
                with _BACKEND._lock:
                    _BACKEND._get_count += 1
                data = _BACKEND.get(key_str)
                if data is None:
                    if on_done is not None:
                        try:
                            on_done(error=None, status_code=404)
                        except Exception:
                            pass
                    self.finished_future.set_exception(
                        RuntimeError(f"S3 GET 404 for {key_str}")
                    )
                    return
                if on_body is not None:
                    on_body(data, 0)
                if on_done is not None:
                    on_done(error=None, status_code=200)
                self.finished_future.set_result(None)

            elif method == "HEAD":
                with _BACKEND._lock:
                    _BACKEND._head_count += 1
                data = _BACKEND.get(key_str)
                if data is None:
                    if on_headers is not None:
                        on_headers(404, [])
                    self.finished_future.set_exception(
                        RuntimeError(f"S3 HEAD 404 for {key_str}")
                    )
                    return
                if on_headers is not None:
                    on_headers(200, [("content-length", str(len(data)))])
                self.finished_future.set_result(None)

            elif method == "DELETE":
                with _BACKEND._lock:
                    _BACKEND._delete_count += 1
                _BACKEND.delete(key_str)
                if on_headers is not None:
                    on_headers(204, [])
                if on_done is not None:
                    on_done(error=None, status_code=204)
                self.finished_future.set_result(None)

            else:
                self.finished_future.set_exception(
                    RuntimeError(f"unexpected method {method}")
                )

        except Exception as e:
            try:
                if on_done is not None:
                    on_done(error=str(e), status_code=None)
            except Exception:
                pass
            if not self.finished_future.done():
                self.finished_future.set_exception(e)


class _FakeHttpRequest:
    """Plain-Python HttpRequest that preserves body_stream as-is."""

    def __init__(self, method, path, headers, body_stream=None):
        self.method = method
        self.path = path
        self.headers = headers
        self.body_stream = body_stream


@pytest.fixture(autouse=True)
def patch_s3_request(monkeypatch):
    """Replace awscrt.s3.S3Request in the adapter with the in-memory fake.

    Also patch HttpRequest so body_stream is preserved as a plain Python
    object (real HttpRequest wraps it in an opaque awscrt InputStream).

    Stubs the credentials provider so tests don't depend on boto3 or any
    AWS credentials being present in the environment; mocked requests
    never reach a real signer.
    """
    _BACKEND.reset()
    monkeypatch.setattr(s3mod.s3, "S3Request", _FakeS3Request)
    monkeypatch.setattr(s3mod, "HttpRequest", _FakeHttpRequest)
    monkeypatch.setattr(
        s3mod,
        "_make_credentials_provider",
        lambda _config: s3mod.auth.AwsCredentialsProvider.new_static(
            "test-key", "test-secret"
        ),
    )
    yield


# =============================================================================
# Helpers
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> TensorMemoryObj:
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
def adapter():
    config = S3L2AdapterConfig(
        s3_endpoint="s3://test-bucket",
        s3_region="us-east-1",
        s3_prefer_http2=False,  # skip TLS ALPN setup for faster init
        # Keep awscrt EventLoopGroup tiny so CI runners with tight FD
        # ulimits (each epoll event loop costs a few fds) don't exhaust
        # file descriptors across ~15 adapter fixtures in this module.
        # Production callers should leave this at the default 64.
        s3_num_io_threads=1,
        max_capacity_gb=0.001,  # 1 MB
    )
    a = S3L2Adapter(config)
    yield a
    a.close()


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys, sizes):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys):
        self.deleted.append(list(keys))


# =============================================================================
# Key serialization
# =============================================================================


class TestObjectKeySerialization:
    def test_format(self):
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        assert _object_key_to_string(key) == "llama@000000ff@0@00010203"

    def test_object_group_id_embedded(self):
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            object_group_id=5,
        )
        assert _object_key_to_string(key) == "llama@000000ff@5@00010203"

    def test_cache_salt_appended(self):
        """A non-empty cache_salt must be included in the S3 object name so
        two users with the same model/rank/chunk map to distinct objects.
        """
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


# =============================================================================
# Event fd interface
# =============================================================================


class TestEventFdInterface:
    def test_three_distinct_fds(self, adapter):
        a = adapter.get_store_event_fd()
        b = adapter.get_lookup_and_lock_event_fd()
        c = adapter.get_load_event_fd()
        assert a >= 0 and b >= 0 and c >= 0
        assert len({a, b, c}) == 3


# =============================================================================
# Round-trip
# =============================================================================


class TestStoreLookupLoad:
    def test_roundtrip_single_key(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj(fill_value=3.14)

        # Store
        tid = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert completed[tid].is_successful()

        # Lookup
        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is True

        # Load into a fresh buffer and verify the bytes match.
        dst = create_memory_obj(fill_value=0.0)
        tid = adapter.submit_load_task([key], [dst])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is True
        assert torch.allclose(dst.tensor, torch.full((16,), 3.14))

    def test_partial_hits(self, adapter):
        # Store keys 0, 2
        stored = [create_object_key(0), create_object_key(2)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(2)]
        adapter.submit_store_task(stored, objs)
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        # Lookup 0, 1, 2, 3 — expect bitmap 1010
        keys = [create_object_key(i) for i in range(4)]
        tid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None
        assert bm.test(0) is True
        assert bm.test(1) is False
        assert bm.test(2) is True
        assert bm.test(3) is False

    def test_load_miss_returns_zero_bit(self, adapter):
        key = create_object_key(99)
        dst = create_memory_obj()
        tid = adapter.submit_load_task([key], [dst])
        wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None
        assert bm.test(0) is False

    def test_query_lookup_returns_none_after_pop(self, adapter):
        key = create_object_key(1)
        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(tid) is not None
        assert adapter.query_lookup_and_lock_result(tid) is None


# =============================================================================
# Eviction (delete + locking)
# =============================================================================


class TestEviction:
    def _store(self, adapter, key, obj):
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

    def _lookup(self, adapter, key):
        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        return adapter.query_lookup_and_lock_result(tid)

    def test_delete_removes_key(self, adapter):
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj())
        assert _BACKEND.contains(_object_key_to_string(key))
        adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_lock_blocks_delete(self, adapter):
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj())
        bm = self._lookup(adapter, key)  # bumps refcount
        assert bm.test(0) is True

        deletes_before = _BACKEND.counts()["delete"]
        adapter.delete([key])
        assert _BACKEND.counts()["delete"] == deletes_before
        # Still there.
        assert _BACKEND.contains(_object_key_to_string(key))

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_refcount_unlock(self, adapter):
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj())
        self._lookup(adapter, key)
        self._lookup(adapter, key)  # refcount now 2

        adapter.submit_unlock([key])  # refcount 1, still locked
        adapter.delete([key])
        assert _BACKEND.contains(_object_key_to_string(key))

        adapter.submit_unlock([key])  # refcount 0
        adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_delete_on_unknown_key(self, adapter):
        # Should not raise.
        adapter.delete([create_object_key(42)])


# =============================================================================
# get_usage
# =============================================================================


class TestGetUsage:
    def test_disabled_returns_minus_one(self):
        cfg = S3L2AdapterConfig(
            s3_endpoint="s3://b",
            s3_region="r",
            s3_prefer_http2=False,
            s3_num_io_threads=1,
            max_capacity_gb=0.0,
        )
        a = S3L2Adapter(cfg)
        try:
            usage = a.get_usage()
            assert usage.usage_fraction == -1.0
            assert usage.total_bytes_used == 0
            assert usage.total_capacity_bytes == 0
        finally:
            a.close()

    def test_usage_grows_on_store_and_shrinks_on_delete(self, adapter):
        # adapter max_capacity_gb = 0.001 = 1 MB
        # each obj is 16 floats = 64 bytes
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


# =============================================================================
# Circuit breaker
# =============================================================================


class TestCircuitBreaker:
    def test_trips_after_three_connection_errors(self, adapter):
        _BACKEND.set_error("CONNECTION_REFUSED: mock")

        keys = [create_object_key(i) for i in range(3)]
        for k in keys:
            obj = create_memory_obj()
            adapter.submit_store_task([k], [obj])
            wait_for_event_fd(adapter.get_store_event_fd(), timeout=2.0)
            adapter.pop_completed_store_tasks()

        status = adapter.report_status()
        assert status["connection_disabled"] is True
        assert status["is_healthy"] is False

        # Subsequent submit short-circuits.
        _BACKEND.set_error(None)  # even if we un-inject, the breaker is tripped
        put_before = _BACKEND.counts()["put"]
        disabled_tid = adapter.submit_store_task(
            [create_object_key(42)], [create_memory_obj()]
        )
        wait_for_event_fd(adapter.get_store_event_fd(), timeout=2.0)
        completed = adapter.pop_completed_store_tasks()
        assert not completed[disabled_tid].is_successful()
        assert _BACKEND.counts()["put"] == put_before  # never reached the backend

        # Lookup and load also short-circuit to all-zero bitmaps.
        tid = adapter.submit_lookup_and_lock_task([create_object_key(1)], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd(), timeout=2.0)
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is False

        tid = adapter.submit_load_task([create_object_key(1)], [create_memory_obj()])
        wait_for_event_fd(adapter.get_load_event_fd(), timeout=2.0)
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is False


# =============================================================================
# Listener notifications
# =============================================================================


class TestListener:
    def test_stored_and_deleted_fire(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        adapter.submit_store_task([key], [create_memory_obj()])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()
        # Give the coroutine a moment to fire notify; it fires inline but
        # on the loop thread, so yield briefly.
        time.sleep(0.05)
        assert any(key in batch for batch in listener.stored)

        adapter.delete([key])
        assert any(key in batch for batch in listener.deleted)

    def test_accessed_fires_on_hit(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        adapter.submit_store_task([key], [create_memory_obj()])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(tid)
        time.sleep(0.05)
        assert any(key in batch for batch in listener.accessed)


# =============================================================================
# Config
# =============================================================================


class TestConfig:
    def test_from_dict_requires_endpoint_and_region(self):
        with pytest.raises(ValueError):
            S3L2AdapterConfig.from_dict({"s3_region": "us-east-1"})
        with pytest.raises(ValueError):
            S3L2AdapterConfig.from_dict({"s3_endpoint": "s3://b"})

    def test_from_dict_parses_all_fields(self):
        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "s3://my-bucket",
                "s3_region": "us-west-2",
                "s3_num_io_threads": 32,
                "s3_prefer_http2": False,
                "s3_enable_s3express": True,
                "disable_tls": True,
                "aws_access_key_id": "id",
                "aws_secret_access_key": "secret",
                "max_capacity_gb": 2.5,
            }
        )
        assert cfg.s3_endpoint == "s3://my-bucket"
        assert cfg.s3_region == "us-west-2"
        assert cfg.s3_num_io_threads == 32
        assert cfg.s3_prefer_http2 is False
        assert cfg.s3_enable_s3express is True
        assert cfg.disable_tls is True
        assert cfg.aws_access_key_id == "id"
        assert cfg.aws_secret_access_key == "secret"
        assert cfg.max_capacity_gb == 2.5

    def test_help_nonempty(self):
        assert isinstance(S3L2AdapterConfig.help(), str)
        assert "s3_endpoint" in S3L2AdapterConfig.help()


# =============================================================================
# Factory registration
# =============================================================================


class TestFactoryRegistration:
    def test_create_l2_adapter_registers_s3(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "s3://fac-test",
                "s3_region": "us-east-1",
                "s3_prefer_http2": False,
                "s3_num_io_threads": 1,
            }
        )
        adp = create_l2_adapter(cfg)
        try:
            assert isinstance(adp, S3L2Adapter)
        finally:
            adp.close()


# =============================================================================
# list_l2_keys
# =============================================================================
#
# These tests populate the fake S3 backend directly via ``_BACKEND.put``,
# bypassing the adapter's store pipeline. The listing path is what's
# under test: ListObjectsV2 dispatch, XML parse, prefix filter, and
# cursor pagination.


def _put_object(key: ObjectKey, size: int = 0) -> None:
    """Place a key in the fake S3 backend with deterministic content.

    Mirrors the production PUT pipeline: ``_object_key_to_string`` →
    ``_format_safe_path`` (URL-encoding only — no path-flattening) →
    URL-unquote in the fake → backend storage. So the fake's stored
    name is byte-identical to the literal S3 object name a real PUT
    would land.
    """
    # First Party
    from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
        _object_key_to_string,
    )

    _BACKEND.put(_object_key_to_string(key), b"\x00" * size)


class TestS3L2AdapterListKeys:
    def test_empty_bucket_returns_empty_page(self, adapter):
        page = adapter.list_l2_keys()
        assert page.entries == ()
        assert page.next_page_token is None

    def test_lists_all_keys_when_unfiltered(self, adapter):
        for i in range(3):
            _put_object(create_object_key(i, model_name="llama"), size=100 + i)

        page = adapter.list_l2_keys(page_size=10)

        assert len(page.entries) == 3
        assert {e.size_bytes for e in page.entries} == {100, 101, 102}
        assert page.next_page_token is None

    def test_model_name_pushed_down_as_prefix(self, adapter):
        llama = ObjectKey(chunk_hash=b"\x00" * 4, model_name="llama", kv_rank=0)
        mistral = ObjectKey(chunk_hash=b"\x00" * 4, model_name="mistral", kv_rank=0)
        _put_object(llama, size=1)
        _put_object(mistral, size=1)

        page = adapter.list_l2_keys(model_name="llama", page_size=10)
        assert [e.key for e in page.entries] == [llama.to_encoded_object_key()]

    def test_model_name_with_slash_round_trips(self, adapter):
        # Real HF model ids contain ``/``. The adapter stores the
        # literal object name on S3 (no path-flattening) and the
        # listing decodes it back unchanged.
        original = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="meta-llama/Llama-3.1-8B",
            kv_rank=7,
        )
        encoded = original.to_encoded_object_key()
        _put_object(original, size=128)

        # The listing returns the original model_name (slash intact).
        page = adapter.list_l2_keys(page_size=10)
        assert len(page.entries) == 1
        assert page.entries[0].key == encoded
        assert page.entries[0].key.model_name == "meta-llama/Llama-3.1-8B"

        # And the model_name filter accepts the caller-supplied form
        # verbatim — no encoding magic to remember.
        page = adapter.list_l2_keys(model_name="meta-llama/Llama-3.1-8B", page_size=10)
        assert [e.key for e in page.entries] == [encoded]

    def test_pagination_cursor_walks_full_set(self, adapter):
        # 5 keys, page_size=2 → expect 3 calls: (0,1), (2,3), (4,).
        for i in range(5):
            _put_object(create_object_key(i, model_name="llama"), size=1)

        collected = []
        cursor = None
        while True:
            page = adapter.list_l2_keys(page_size=2, cursor=cursor)
            collected.extend(page.entries)
            cursor = page.next_page_token
            if cursor is None:
                break
        assert len(collected) == 5

    def test_page_size_must_be_positive(self, adapter):
        with pytest.raises(ValueError):
            adapter.list_l2_keys(page_size=0)

    def test_page_size_clamped_to_s3_max(self, adapter):
        # Ask for far more than S3's MaxKeys ceiling. The adapter
        # clamps internally and the fake honors whatever it receives,
        # so 5 keys come back without error.
        for i in range(5):
            _put_object(create_object_key(i, model_name="m"), size=1)
        page = adapter.list_l2_keys(page_size=999_999)
        assert len(page.entries) == 5
        assert page.next_page_token is None

    def test_circuit_breaker_blocks_listing(self, adapter):
        # When the connection is disabled, listing must surface a
        # clear error instead of issuing a doomed request.
        with adapter._lock:
            adapter._connection_disabled = True
        with pytest.raises(RuntimeError) as exc:
            adapter.list_l2_keys()
        assert "disabled" in str(exc.value)

    def test_unparsable_objects_silently_skipped(self, adapter):
        # An object with a name that isn't this adapter's format
        # (e.g. left behind by another tool) must not poison the
        # response — it's just dropped from the listing.
        _put_object(create_object_key(0, model_name="m"), size=1)
        _BACKEND.put("not-our-format-key", b"junk")

        page = adapter.list_l2_keys(page_size=10)

        assert len(page.entries) == 1
        assert (
            page.entries[0].key
            == create_object_key(0, model_name="m").to_encoded_object_key()
        )
