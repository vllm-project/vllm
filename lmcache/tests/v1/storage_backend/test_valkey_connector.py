# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ValkeyConnector.

These tests verify the Valkey connector implementation, including:
- Basic operations (exists, get, set)
- Batch operations (batch_get, batch_put, batch_exists)
- Partial misses and non-blocking prefix truncation
- Cluster mode vs standalone config passthrough
- Error handling
- Worker scaling

The worker pool is mocked to avoid requiring glide_sync or a real Valkey server.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import patch
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.valkey.worker_pool import GET_MISS, ValkeyWorkerPool

# Local
from ...conftest import MockSyncGlideClient
from ..utils import (
    check_mem_obj_equal,
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)


class MockThreadWorkerPool:
    """In-memory mock of ValkeyWorkerPool that uses MockSyncGlideClient.

    Runs all operations in-process so tests don't need glide_sync or a
    real Valkey server.  Captures constructor kwargs so tests can verify
    config passthrough (cluster_mode, database_id, tls_enable, etc.).
    """

    # Class-level record of the last __init__ kwargs for config assertions.
    last_init_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        MockThreadWorkerPool.last_init_kwargs = {
            "args": args,
            "kwargs": kwargs,
        }
        self.num_workers = kwargs.get("num_workers", 8)
        if len(args) > 2 and isinstance(args[2], int):
            self.num_workers = args[2]
        self._ttl_seconds = kwargs.get("ttl_seconds", None)
        self._client = MockSyncGlideClient()
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def _do_set(self, key_str: str, data: bytes) -> None:
        """SET a key, mirroring the real pool's TTL handling."""
        # ("EX", ttl) is a stand-in for glide_sync.ExpirySet, which isn't
        # importable in the test env.
        expiry = ("EX", self._ttl_seconds) if self._ttl_seconds is not None else None
        self._client.set(key_str.encode(), bytes(data), expiry=expiry)

    def _do_get_into(self, key_str: str, buf: memoryview) -> int:
        """GET a key into a buffer.

        Mirrors ``ValkeyWorkerPool._do_get_into``: returns the number of
        payload bytes for the key, or ``-1`` (GET_MISS) when absent.
        """
        data = self._client.get(key_str.encode())
        if data is None:
            return -1
        flat = buf.cast("B") if buf.format != "B" else buf
        n = len(data)
        flat[: min(n, len(flat))] = data[: min(n, len(flat))]
        return n

    def _do_exists(self, key_str: str) -> bool:
        """Check if a key exists."""
        return bool(self._client.exists([key_str.encode()]))

    def _do_delete(self, key_str: str) -> bool:
        """Delete a key; return True iff a value was removed."""
        return int(self._client.delete([key_str.encode()])) > 0

    def submit_set(self, key_str: str, data: bytes) -> Future:
        """Submit a SET operation."""
        return self._executor.submit(self._do_set, key_str, data)

    def submit_get_into(self, key_str: str, buf: memoryview) -> Future:
        """Submit a GET-into-buffer operation."""
        return self._executor.submit(self._do_get_into, key_str, buf)

    def submit_exists(self, key_str: str) -> Future:
        """Submit an EXISTS check."""
        return self._executor.submit(self._do_exists, key_str)

    def submit_delete(self, key_str: str) -> Future:
        """Submit a DELETE operation."""
        return self._executor.submit(self._do_delete, key_str)

    @property
    def has_buffer_get(self) -> bool:
        """Mock always exposes buffer GET."""
        return True

    def close(self) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=True)


def _get_metadata():
    """Helper to create test metadata."""
    kv_shape = (32, 2, 256, 8, 128)
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        use_mla=False,
    )


def _create_local_cpu_backend(memory_allocator, config=None):
    """Helper to create a local CPU backend for testing."""
    if config is None:
        config = LMCacheEngineConfig.from_defaults(
            extra_config={"valkey_num_workers": 4}
        )
    metadata = _get_metadata()
    return LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )


def _create_test_memory_obj(local_backend, seed=42):
    """Allocate a test memory object with deterministic random data.

    Args:
        local_backend: The local CPU backend for allocation.
        seed: Random seed for reproducible data.

    Returns:
        A MemoryObj with ref_count incremented and random data filled.
    """
    mem_obj_shape = torch.Size([2, 32, 256, 1024])
    dtype = torch.bfloat16
    memory_obj = local_backend.allocate(mem_obj_shape, dtype)
    memory_obj.ref_count_up()
    torch.manual_seed(seed)
    test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
    memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
    return memory_obj


@pytest.fixture(autouse=True)
def mock_thread_worker_pool():
    """Replace ValkeyWorkerPool with in-memory mock so tests never need
    glide_sync or a real Valkey server."""
    MockSyncGlideClient.reset_store()
    MockThreadWorkerPool.last_init_kwargs = {}
    with patch(
        "lmcache.v1.storage_backend.connector.valkey_connector.ValkeyWorkerPool",
        MockThreadWorkerPool,
    ):
        yield


@pytest.fixture
def valkey_url():
    """URL for testing."""
    return "valkey://mock.local:0"


@pytest.fixture
def valkey_config():
    """Config for ValkeyConnector testing."""
    return LMCacheEngineConfig.from_defaults(extra_config={"valkey_num_workers": 4})


@pytest.fixture
def local_backend():
    """Create a local CPU backend for testing."""
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    backend = _create_local_cpu_backend(memory_allocator)
    yield backend
    backend.close()


def test_valkey_basic_operations(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test basic operations: exists, put, get."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        random_key = dumb_cache_engine_key()

        # Key doesn't exist initially
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert not future.result(), "Key should not exist initially"

        memory_obj = _create_test_memory_obj(local_backend, seed=42)

        # Put data
        future = asyncio.run_coroutine_threadsafe(
            connector.put(random_key, memory_obj), async_loop
        )
        future.result()

        # Key exists after put
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert future.result(), "Key should exist after put"

        # Get and verify data
        future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
        retrieved = future.result()
        check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_batch_operations(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test batch operations: batched_put, batched_get, batched_async_contains."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        num_keys = 10
        keys = [dumb_cache_engine_key(i) for i in range(num_keys)]

        # Batch exists — all should be False initially
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        assert future.result() == 0, "No keys should exist initially"

        memory_objs = [
            _create_test_memory_obj(local_backend, seed=42 + i) for i in range(num_keys)
        ]

        # Batch put
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, memory_objs), async_loop
        )
        future.result()

        # Batch exists — all should be True now
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        assert future.result() == num_keys, "All keys should exist after batch_put"

        # Batch get and verify
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys), async_loop
        )
        retrieved_objs = future.result()

        assert len(retrieved_objs) == num_keys
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_nonexistent_key(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test exists and get on a non-existent key."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        nonexistent_key = dumb_cache_engine_key()

        future = asyncio.run_coroutine_threadsafe(
            connector.exists(nonexistent_key), async_loop
        )
        assert not future.result()

        future = asyncio.run_coroutine_threadsafe(
            connector.get(nonexistent_key), async_loop
        )
        assert future.result() is None, "get() should return None for missing key"

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_sequential_operations(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test multiple sequential put/get cycles."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        for i in range(5):
            key = dumb_cache_engine_key(i)
            memory_obj = _create_test_memory_obj(local_backend, seed=1000 + i)

            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result()

            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            retrieved = future.result()
            check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_concurrent_operations(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test concurrent put/get operations."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        num_concurrent = 5
        keys = [dumb_cache_engine_key(i) for i in range(num_concurrent)]
        memory_objs = [
            _create_test_memory_obj(local_backend, seed=2000 + i)
            for i in range(num_concurrent)
        ]

        put_futures = [
            asyncio.run_coroutine_threadsafe(
                connector.put(keys[i], memory_objs[i]), async_loop
            )
            for i in range(num_concurrent)
        ]
        for f in put_futures:
            f.result()

        get_futures = [
            asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            for key in keys
        ]
        retrieved_objs = [f.result() for f in get_futures]
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_exists_sync(valkey_url, local_backend, valkey_config, autorelease_v1):
    """Test synchronous exists method."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        key = dumb_cache_engine_key()
        assert not connector.exists_sync(key)

        memory_obj = _create_test_memory_obj(local_backend)

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        assert connector.exists_sync(key)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_batched_contains_prefix(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test that batched_contains returns prefix count correctly."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        keys = [dumb_cache_engine_key(i) for i in range(5)]

        # Put only first 3 keys
        for i in range(3):
            memory_obj = _create_test_memory_obj(local_backend, seed=i)
            future = asyncio.run_coroutine_threadsafe(
                connector.put(keys[i], memory_obj), async_loop
            )
            future.result()

        count = connector.batched_contains(keys)
        assert count == 3, f"Expected 3 consecutive keys, got {count}"

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_different_chunk_sizes(autorelease_v1):
    """Test that the connector works with different chunk sizes."""
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    config = LMCacheEngineConfig.from_defaults(extra_config={"valkey_num_workers": 4})

    kv_shape = (32, 2, 512, 8, 128)
    dtype = torch.bfloat16
    metadata = LMCacheMetadata(
        model_name="test-model-large",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=kv_shape,
        use_mla=False,
        chunk_size=512,
    )
    local_backend = LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )

    try:
        connector = autorelease_v1(
            CreateConnector(
                "valkey://mock.local:0",
                async_loop,
                local_backend,
                config,
            )
        )

        key = dumb_cache_engine_key()
        mem_obj_shape = torch.Size([2, 32, 512, 1024])
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(100)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved = future.result()
        check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


def test_valkey_pipelined_batch_exceeds_arena(
    valkey_url, valkey_config, autorelease_v1
):
    """Test batched put/get when batch size > num_workers.

    With num_workers=4, a batch of 12 keys forces the connector to handle
    more concurrent operations than workers, verifying that all data is
    correctly processed.

    Uses a dedicated 2 GB allocator so the 12 put objects + 12 get objects
    (~768 MB total at 32 MB each) fit without blocking.
    """
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(2 * 1024 * 1024 * 1024)
    local_backend = _create_local_cpu_backend(memory_allocator, valkey_config)

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        num_keys = 12
        keys = [dumb_cache_engine_key(i) for i in range(num_keys)]
        memory_objs = [
            _create_test_memory_obj(local_backend, seed=5000 + i)
            for i in range(num_keys)
        ]

        # Batch put all 12
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, memory_objs), async_loop
        )
        future.result()

        # All 12 should exist
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        assert future.result() == num_keys

        # Batch get and verify every item matches
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys), async_loop
        )
        retrieved_objs = future.result()

        assert len(retrieved_objs) == num_keys
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


@pytest.mark.parametrize("num_workers", [1, 4, 8])
def test_valkey_worker_scaling(num_workers, autorelease_v1):
    """Test ValkeyConnector with different numbers of worker threads."""
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={"valkey_num_workers": num_workers}
    )
    metadata = _get_metadata()
    local_backend = LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )

    try:
        connector = autorelease_v1(
            CreateConnector(
                "valkey://mock.local:0",
                async_loop,
                local_backend,
                config,
            )
        )

        key = dumb_cache_engine_key()
        memory_obj = _create_test_memory_obj(local_backend, seed=3000)

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved = future.result()
        check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


# ── New tests: partial misses, non-blocking prefix, config passthrough ──


def test_valkey_batched_get_partial_misses(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test batched_get when some keys exist and some don't.

    Put keys 0-4, request keys 0-9.  The result should have 5 MemoryObjs
    followed by 5 Nones.
    """
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        num_present = 5
        num_total = 10
        keys = [dumb_cache_engine_key(i) for i in range(num_total)]
        put_objs = [
            _create_test_memory_obj(local_backend, seed=7000 + i)
            for i in range(num_present)
        ]

        # Put only first 5
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys[:num_present], put_objs), async_loop
        )
        future.result()

        # Batch get all 10
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys), async_loop
        )
        results = future.result()

        assert len(results) == num_total
        # First 5 should be valid MemoryObjs matching what we put
        for i in range(num_present):
            assert results[i] is not None, f"Key {i} should exist"
        check_mem_obj_equal(results[:num_present], put_objs)
        # Last 5 should be None
        for i in range(num_present, num_total):
            assert results[i] is None, f"Key {i} should be None"

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_batched_get_non_blocking_all_present(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test batched_get_non_blocking when all keys are present."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        num_keys = 5
        keys = [dumb_cache_engine_key(i) for i in range(num_keys)]
        put_objs = [
            _create_test_memory_obj(local_backend, seed=8000 + i)
            for i in range(num_keys)
        ]

        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, put_objs), async_loop
        )
        future.result()

        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get_non_blocking("lookup", keys), async_loop
        )
        prefix = future.result()

        assert len(prefix) == num_keys
        check_mem_obj_equal(prefix, put_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_batched_get_non_blocking_prefix_truncation(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test batched_get_non_blocking returns only the consecutive prefix.

    Put keys 0-2, request keys 0-4.  Should return only the first 3
    objects; keys 3-4 are missing so the prefix stops there.
    """
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        num_present = 3
        num_total = 5
        keys = [dumb_cache_engine_key(i) for i in range(num_total)]
        put_objs = [
            _create_test_memory_obj(local_backend, seed=9000 + i)
            for i in range(num_present)
        ]

        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys[:num_present], put_objs), async_loop
        )
        future.result()

        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get_non_blocking("lookup", keys), async_loop
        )
        prefix = future.result()

        assert len(prefix) == num_present
        check_mem_obj_equal(prefix, put_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_batched_get_non_blocking_first_missing(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """Test batched_get_non_blocking returns empty when first key is missing."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        keys = [dumb_cache_engine_key(i) for i in range(3)]
        # Don't put anything

        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get_non_blocking("lookup", keys), async_loop
        )
        prefix = future.result()

        assert len(prefix) == 0

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_standalone_mode_config(local_backend, autorelease_v1):
    """Test that standalone mode (default) passes correct config to pool."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_database": 3,
        }
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://standalone.local:6379",
                async_loop,
                local_backend,
                config,
            )
        )

        init_info = MockThreadWorkerPool.last_init_kwargs
        # ValkeyWorkerPool is constructed with keyword args:
        # addresses=[(host, port)], num_workers=..., etc.
        assert init_info["kwargs"]["addresses"] == [("standalone.local", 6379)]
        assert init_info["kwargs"]["num_workers"] == 2
        assert init_info["kwargs"].get("cluster_mode") is False
        assert init_info["kwargs"].get("database_id") == 3

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_cluster_mode_config(local_backend, autorelease_v1):
    """Test that cluster mode passes cluster_mode=True and ignores database_id."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 4,
            "valkey_mode": "cluster",
            "valkey_database": 5,  # should be ignored in cluster mode
        }
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://cluster.local:7000",
                async_loop,
                local_backend,
                config,
            )
        )

        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"]["addresses"] == [("cluster.local", 7000)]
        assert init_info["kwargs"].get("cluster_mode") is True
        # database_id should be None (ignored in cluster mode)
        assert init_info["kwargs"].get("database_id") is None

    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize(
    "raw_value,expected",
    [("true", True), ("True", True), ("1", True), ("false", False), ("0", False)],
)
def test_valkey_tls_enable_string_coercion(
    local_backend, autorelease_v1, raw_value, expected
):
    """tls_enable from free-form extra_config may be a string; it must be
    coerced (a bare bool("false") would wrongly enable TLS)."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={"valkey_num_workers": 2, "tls_enable": raw_value}
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://tlsstr.local:6380",
                async_loop,
                local_backend,
                config,
            )
        )
        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"].get("tls_enable") is expected
    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_tls_config(local_backend, autorelease_v1):
    """Test that tls_enable is passed through to the pool."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "tls_enable": True,
        }
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://tls.local:6380",
                async_loop,
                local_backend,
                config,
            )
        )

        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"].get("tls_enable") is True

    finally:
        close_asyncio_loop(async_loop, async_thread)


# ── TTL feature flag (volatile-* eviction support) ──────────────────────


def test_valkey_ttl_sec_disabled_by_default(local_backend, autorelease_v1):
    """By default (no valkey_enable_ttl) no TTL is passed to the pool."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(extra_config={"valkey_num_workers": 2})

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
        )
        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"].get("ttl_seconds") is None

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_ttl_sec_enabled_passthrough(local_backend, autorelease_v1):
    """valkey_enable_ttl + valkey_ttl_sec flow through to the pool as ttl_seconds."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
            "valkey_ttl_sec": 3600,
        }
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
        )
        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"].get("ttl_seconds") == 3600

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_ttl_sec_enabled_without_value_uses_default(
    local_backend, autorelease_v1
):
    """Enabling the flag without valkey_ttl_sec falls back to DEFAULT_TTL_SECS."""
    # First Party
    from lmcache.v1.storage_backend.connector.valkey_connector import (
        DEFAULT_TTL_SECS,
    )

    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
        }
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
        )
        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"].get("ttl_seconds") == DEFAULT_TTL_SECS

    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize("raw_ttl", [0, -1, -100, "-100", "0.5", "-0.5"])
def test_valkey_ttl_sec_invalid_value_raises(raw_ttl, local_backend, autorelease_v1):
    """A non-positive valkey_ttl_sec is rejected when the flag is enabled.

    Covers zero, negative ints/strings, and sub-second values that truncate
    to zero.
    """
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
            "valkey_ttl_sec": raw_ttl,
        }
    )

    try:
        with pytest.raises(ValueError, match="valkey_ttl_sec must be a positive"):
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize(
    "raw_ttl,expected",
    [("1800", 1800), ("1800.0", 1800), (1800.0, 1800), (1800, 1800)],
)
def test_valkey_ttl_sec_numeric_coercion(
    raw_ttl, expected, local_backend, autorelease_v1
):
    """valkey_ttl_sec may arrive as an int/float or an int/float-like string
    from free-form extra_config; all coerce to a positive int TTL."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
            "valkey_ttl_sec": raw_ttl,
        }
    )

    try:
        autorelease_v1(
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
        )
        init_info = MockThreadWorkerPool.last_init_kwargs
        assert init_info["kwargs"].get("ttl_seconds") == expected

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_ttl_sec_non_numeric_raises(local_backend, autorelease_v1):
    """A non-numeric valkey_ttl_sec is rejected with a clear ValueError."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
            "valkey_ttl_sec": "not-a-number",
        }
    )

    try:
        with pytest.raises(ValueError, match="valkey_ttl_sec must be a positive"):
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize("raw_ttl", [True, False])
def test_valkey_ttl_sec_bool_raises(raw_ttl, local_backend, autorelease_v1):
    """A boolean valkey_ttl_sec is rejected (bool is an int subclass, so it
    would otherwise silently coerce to a 1-second TTL)."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
            "valkey_ttl_sec": raw_ttl,
        }
    )

    try:
        with pytest.raises(ValueError, match="valkey_ttl_sec must be a positive"):
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_ttl_sec_applied_on_put(local_backend, autorelease_v1):
    """When TTL is enabled, SET carries an expiry; otherwise it does not."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "valkey_enable_ttl": True,
            "valkey_ttl_sec": 1800,
        }
    )

    try:
        connector = autorelease_v1(
            CreateConnector(
                "valkey://ttl.local:6379", async_loop, local_backend, config
            )
        )

        key = dumb_cache_engine_key()
        memory_obj = _create_test_memory_obj(local_backend, seed=11)
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        # The mock client records the expiry passed to the last SET.
        assert MockSyncGlideClient.last_expiry == ("EX", 1800)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_no_ttl_on_put_when_disabled(
    valkey_url, local_backend, valkey_config, autorelease_v1
):
    """With TTL disabled, SET passes no expiry (legacy behavior)."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(valkey_url, async_loop, local_backend, valkey_config)
        )

        key = dumb_cache_engine_key()
        memory_obj = _create_test_memory_obj(local_backend, seed=12)
        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        assert MockSyncGlideClient.last_expiry is None

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_rejects_save_unfull_chunk(local_backend, autorelease_v1):
    """Valkey uses single-key fixed-size storage with no per-chunk metadata,
    so partial/unfull chunks are not allowed (same as RESP)."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={"valkey_num_workers": 2},
    )
    # Force the partial-chunk flag on after construction so we exercise the
    # adapter's guard rather than any config auto-adjust logic.
    config.save_unfull_chunk = True

    try:
        with pytest.raises(ValueError, match="save_unfull_chunk must be False"):
            autorelease_v1(
                CreateConnector(
                    "valkey://nopartial.local:6379",
                    async_loop,
                    local_backend,
                    config,
                )
            )
    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_valkey_rejects_save_chunk_meta(local_backend, autorelease_v1):
    """Valkey does not persist per-chunk metadata, so save_chunk_meta must be
    False (same as RESP)."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "save_chunk_meta": True,
        },
    )

    try:
        with pytest.raises(ValueError, match="save_chunk_meta must be False"):
            autorelease_v1(
                CreateConnector(
                    "valkey://nometa.local:6379",
                    async_loop,
                    local_backend,
                    config,
                )
            )
    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize(
    "raw_value,should_reject",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("false", False),
        ("0", False),
    ],
)
def test_valkey_save_chunk_meta_string_coercion(
    local_backend, autorelease_v1, raw_value, should_reject
):
    """save_chunk_meta from free-form extra_config may be a string; it must be
    coerced (a bare truthiness check would treat "false" as True)."""
    async_loop, async_thread = init_asyncio_loop()

    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "valkey_num_workers": 2,
            "save_chunk_meta": raw_value,
        },
    )

    try:
        if should_reject:
            with pytest.raises(ValueError, match="save_chunk_meta must be False"):
                autorelease_v1(
                    CreateConnector(
                        "valkey://strmeta.local:6379",
                        async_loop,
                        local_backend,
                        config,
                    )
                )
        else:
            autorelease_v1(
                CreateConnector(
                    "valkey://strmeta.local:6379",
                    async_loop,
                    local_backend,
                    config,
                )
            )
    finally:
        close_asyncio_loop(async_loop, async_thread)


class _FakeGlideClient:
    """Minimal glide-client stand-in for ``ValkeyWorkerPool._do_get_into``.

    ``payload`` of ``None`` models an absent key. The buffer-GET path writes
    what fits into the caller-provided buffer and returns the byte count,
    mirroring glide's native buffer protocol.
    """

    def __init__(self, payload):
        self._payload = payload

    def get(self, key, buffer=None):
        if self._payload is None:
            return None
        if buffer is not None:
            n = min(len(self._payload), len(buffer))
            buffer[:n] = self._payload[:n]
            return n
        return self._payload


def _fake_pool(payload, has_buffer_get):
    """Build a ``SimpleNamespace`` usable as ``self`` for the unbound
    ``ValkeyWorkerPool._do_get_into`` (no real client / server needed)."""
    # Standard
    from types import SimpleNamespace

    return SimpleNamespace(
        _get_client=lambda: _FakeGlideClient(payload),
        has_buffer_get=has_buffer_get,
        _get_scratch=lambda size: bytearray(size),
    )


@pytest.mark.parametrize("has_buffer_get", [True, False])
def test_worker_pool_get_into_exact_read_is_hit(has_buffer_get):
    """An exact-size read returns the byte count and fills the buffer."""
    expected = 8
    buf = memoryview(bytearray(expected))
    pool = _fake_pool(b"\xbb" * expected, has_buffer_get=has_buffer_get)
    assert ValkeyWorkerPool._do_get_into(pool, "k", buf) == expected
    assert bytes(buf) == b"\xbb" * expected


@pytest.mark.parametrize("has_buffer_get", [True, False])
def test_worker_pool_get_into_absent_is_miss(has_buffer_get):
    """An absent key returns GET_MISS."""
    buf = memoryview(bytearray(8))
    pool = _fake_pool(None, has_buffer_get=has_buffer_get)
    assert ValkeyWorkerPool._do_get_into(pool, "k", buf) == GET_MISS


@pytest.mark.parametrize("has_buffer_get", [True, False])
def test_worker_pool_get_into_short_read_is_miss(has_buffer_get):
    """A short read (fewer bytes than the buffer) is rejected as a miss so
    stale tail bytes are never surfaced to the caller."""
    buf = memoryview(bytearray(8))
    pool = _fake_pool(b"\xaa" * 4, has_buffer_get=has_buffer_get)
    assert ValkeyWorkerPool._do_get_into(pool, "k", buf) == GET_MISS


def test_worker_pool_get_into_oversized_read_is_miss():
    """An oversized read (more bytes than the buffer) is rejected as a miss
    (non-buffer path; the buffer path is bounded by the buffer length)."""
    buf = memoryview(bytearray(8))
    pool = _fake_pool(b"\xcc" * 12, has_buffer_get=False)
    assert ValkeyWorkerPool._do_get_into(pool, "k", buf) == GET_MISS
