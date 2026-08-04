# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import asyncio
import mmap
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryObj
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from tests.v1.utils import (
    check_method_signatures,
    get_abstract_methods,
    get_methods_implemented_in_class,
)

maru = pytest.importorskip("maru", reason="maru package not installed")
maru_lmcache = pytest.importorskip(
    "maru_lmcache", reason="maru_lmcache package not installed"
)

# Third Party
from maru_handler.memory import AllocHandle  # noqa: E402
from maru_handler.memory.types import MappedRegion, MemoryInfo  # noqa: E402
from maru_lmcache.adapter import CxlMemoryAdapter  # noqa: E402

# First Party
from lmcache.v1.storage_backend.maru_backend import MaruBackend  # noqa: E402

# =========================================================================
# Constants
# =========================================================================

TEST_CHUNK_SIZE = 1024
TEST_DTYPE = torch.float32
TEST_SHAPE = torch.Size([256])  # 256 * 4B = 1024 bytes = chunk_size


# =========================================================================
# Helpers
# =========================================================================


def _make_mock_handler(pool_size=4096, chunk_size=TEST_CHUNK_SIZE):
    """Create a mock MaruHandler with mmap-backed regions."""
    handler = MagicMock()
    handler._connected = True

    region_id = 100
    page_count = pool_size // chunk_size

    mmap_obj = mmap.mmap(-1, pool_size)
    mapped_region = MappedRegion(
        region_id=region_id,
        handle=MagicMock(region_id=region_id, length=pool_size),
        size=pool_size,
        _mmap_obj=mmap_obj,
    )

    handler.get_buffer_view.side_effect = lambda rid, offset, size: (
        mapped_region.get_buffer_view(offset, size) if rid == region_id else None
    )
    handler.get_region_page_count.side_effect = lambda rid: (
        page_count if rid == region_id else None
    )
    handler.get_owned_region_ids.return_value = [region_id]
    handler.get_chunk_size.return_value = chunk_size

    def mock_set_on_region_added(callback):
        if callback is not None:
            callback(region_id, page_count)

    handler.set_on_region_added.side_effect = mock_set_on_region_added

    page_counter = [0]

    def mock_alloc(size):
        idx = page_counter[0]
        page_counter[0] += 1
        buf = mapped_region.get_buffer_view(idx * chunk_size, size)
        return AllocHandle(buf=buf, _region_id=region_id, _page_index=idx, _size=size)

    handler.alloc.side_effect = mock_alloc
    handler.free = MagicMock()
    handler.connect.return_value = True
    handler.close.return_value = None
    handler.store.return_value = True
    handler.batch_store.return_value = None
    handler.retrieve.return_value = None
    handler.batch_retrieve.return_value = []
    handler.exists.return_value = False
    handler.batch_exists.return_value = []
    handler.delete.return_value = True
    handler.pin.return_value = True
    handler.unpin.return_value = True
    handler.batch_pin.return_value = []
    handler.batch_unpin.return_value = None

    return handler


def _make_cache_key(chunk_hash: int = 12345) -> CacheEngineKey:
    """Create a CacheEngineKey for testing."""
    return CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float32,
    )


def _make_memory_obj(adapter: CxlMemoryAdapter) -> TensorMemoryObj:
    """Allocate a TensorMemoryObj from the adapter."""
    obj = adapter.allocate(TEST_SHAPE, TEST_DTYPE)
    assert obj is not None
    return obj


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def _init_pin_monitor():
    """Initialize PinMonitor singleton required by TensorMemoryObj.pin()."""
    PinMonitor._instance = None
    PinMonitor.GetOrCreate(LMCacheEngineConfig.from_defaults())
    yield
    PinMonitor._instance = None


@pytest.fixture
def async_loop():
    """Provide an asyncio event loop running in a background thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture
def mock_handler():
    return _make_mock_handler()


@pytest.fixture
def adapter(mock_handler):
    return CxlMemoryAdapter(
        handler=mock_handler,
        shapes=[TEST_SHAPE],
        dtypes=[TEST_DTYPE],
        fmt=MemoryFormat.KV_2LTD,
        chunk_size=TEST_CHUNK_SIZE,
    )


@pytest.fixture
def backend(mock_handler, adapter, async_loop):
    """Create a MaruBackend with mocked internals."""
    # Local

    with patch.object(MaruBackend, "initialize_allocator", return_value=adapter):
        backend = MaruBackend.__new__(MaruBackend)
        backend.dst_device = "cpu"
        backend.config = MagicMock()
        backend.config.maru_pool_size = 4.0
        backend.loop = async_loop
        backend.memory_allocator = adapter
        backend._handler = mock_handler

        backend._full_chunk_size_bytes = TEST_CHUNK_SIZE
        backend._single_token_size = TEST_CHUNK_SIZE // 256  # 4 bytes per token
        backend._mla_worker_id_as0_mode = False

        backend.put_lock = threading.Lock()
        backend.put_tasks = set()
    return backend


def _run_async(loop, coro):
    """Submit a coroutine to a running event loop and wait for result."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=5)


# =========================================================================
# Tests — Init & Interface Compliance
# =========================================================================


class TestMaruBackendInit:
    def test_str(self, backend):
        assert str(backend) == "MaruBackend"

    def test_get_allocator_backend_returns_self(self, backend):
        assert backend.get_allocator_backend() is backend

    def test_get_memory_allocator_returns_adapter(self, backend, adapter):
        assert backend.get_memory_allocator() is adapter


class TestMaruBackendPoolSizeGbToBytes:
    """Test _pool_size_gb_to_bytes static method."""

    def test_4gb(self):
        assert MaruBackend._pool_size_gb_to_bytes(4.0) == 4 * 1024**3

    def test_half_gb(self):
        assert MaruBackend._pool_size_gb_to_bytes(0.5) == 512 * 1024**2

    def test_1gb(self):
        assert MaruBackend._pool_size_gb_to_bytes(1.0) == 1024**3

    def test_zero(self):
        assert MaruBackend._pool_size_gb_to_bytes(0.0) == 0


class TestMaruBackendInterfaceCompliance:
    """Verify MaruBackend implements all required interface methods."""

    def test_implements_all_abstract_methods(self):
        abstract = get_abstract_methods(AllocatorBackendInterface)
        implemented = get_methods_implemented_in_class(
            MaruBackend, AllocatorBackendInterface
        )
        missing = abstract - implemented
        assert not missing, f"Missing abstract methods: {missing}"

    def test_method_signatures_match(self):
        # Known: batched_submit_put_task uses 'memory_objs' instead of 'objs'
        # TODO: Rename to 'objs' for full compliance
        known_param_renames = {"batched_submit_put_task"}

        mismatches = check_method_signatures(AllocatorBackendInterface, MaruBackend)
        unexpected = [m for m in mismatches if m["method"] not in known_param_renames]
        assert not unexpected, f"Signature mismatches: {unexpected}"


# =========================================================================
# Tests — Allocate
# =========================================================================


class TestMaruBackendAllocate:
    def test_allocate_returns_memory_obj(self, backend):
        obj = backend.allocate(TEST_SHAPE, TEST_DTYPE)
        assert obj is not None
        assert obj.tensor is not None
        assert obj.metadata.dtype == TEST_DTYPE

    def test_batched_allocate_returns_list(self, backend):
        objs = backend.batched_allocate(TEST_SHAPE, TEST_DTYPE, batch_size=3)
        assert objs is not None
        assert len(objs) == 3
        for obj in objs:
            assert obj.tensor is not None


# =========================================================================
# Tests — Put (async)
# =========================================================================


class TestMaruBackendPut:
    def test_submit_put_task_returns_future(self, backend, adapter):
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()

        future = backend.submit_put_task(key, obj)
        assert future is not None
        future.result(timeout=5)

        backend._handler.store.assert_called_once()

    def test_submit_put_task_tracks_in_flight(self, backend, adapter):
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()

        assert not backend.exists_in_put_tasks(key)

        future = backend.submit_put_task(key, obj)
        future.result(timeout=5)

        # After completion, key should be removed from put_tasks
        assert not backend.exists_in_put_tasks(key)

    def test_exists_in_put_tasks_true_during_store(self, backend, adapter):
        """Verify exists_in_put_tasks returns True while store is in progress."""
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()

        store_entered = threading.Event()
        store_proceed = threading.Event()

        def blocking_store(*args, **kwargs):
            store_entered.set()
            store_proceed.wait(timeout=5)
            return True

        backend._handler.store.side_effect = blocking_store

        future = backend.submit_put_task(key, obj)

        # Wait until store is actually running
        assert store_entered.wait(timeout=5)
        assert backend.exists_in_put_tasks(key)

        # Let store complete
        store_proceed.set()
        future.result(timeout=5)
        assert not backend.exists_in_put_tasks(key)

    def test_batched_submit_put_task(self, backend, adapter):
        keys = [_make_cache_key(i) for i in range(3)]
        objs = [_make_memory_obj(adapter) for _ in range(3)]
        for obj in objs:
            obj.parent_allocator = None

        backend._handler.batch_store.return_value = [True, True, True]

        futures = backend.batched_submit_put_task(keys, objs)
        assert futures is not None

        for future in futures:
            future.result(timeout=5)

        backend._handler.batch_store.assert_called_once()

    def test_submit_put_calls_callback(self, backend, adapter):
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()
        callback_called = []

        def callback(k):
            callback_called.append(k)

        future = backend.submit_put_task(key, obj, on_complete_callback=callback)
        future.result(timeout=5)

        assert len(callback_called) == 1
        assert callback_called[0] == key

    def test_batched_submit_put_calls_callback_per_key(self, backend, adapter):
        keys = [_make_cache_key(i) for i in range(3)]
        objs = [_make_memory_obj(adapter) for _ in range(3)]
        for obj in objs:
            obj.parent_allocator = None

        backend._handler.batch_store.return_value = [True, True, True]
        callback_keys = []

        def callback(k):
            callback_keys.append(k)

        futures = backend.batched_submit_put_task(
            keys, objs, on_complete_callback=callback
        )
        for future in futures:
            future.result(timeout=5)

        assert set(callback_keys) == set(keys)

    def test_submit_put_task_skips_in_mla_mode(self, backend, adapter):
        """In MLA worker_id_as0 mode, submit_put_task should skip store."""
        backend._mla_worker_id_as0_mode = True
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()

        future = backend.submit_put_task(key, obj)
        assert future.result(timeout=5) is None
        backend._handler.store.assert_not_called()

    def test_submit_put_task_refcount_down_on_failure(self, backend, adapter):
        """On store failure, ref_count should return to pre-submit level."""
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()
        initial_ref = obj.get_ref_count()

        backend._handler.store.side_effect = RuntimeError("store failed")

        future = backend.submit_put_task(key, obj)
        with pytest.raises(RuntimeError):
            future.result(timeout=5)

        assert obj.get_ref_count() == initial_ref
        assert not backend.exists_in_put_tasks(key)

    def test_batched_submit_put_task_refcount_down_on_failure(self, backend, adapter):
        """On batch_store failure, ref_count should return to pre-submit level."""
        keys = [_make_cache_key(i) for i in range(3)]
        objs = [_make_memory_obj(adapter) for _ in range(3)]
        for obj in objs:
            obj.parent_allocator = None
        initial_refs = [obj.get_ref_count() for obj in objs]

        backend._handler.batch_store.side_effect = RuntimeError("batch failed")

        futures = backend.batched_submit_put_task(keys, objs)
        for future in futures:
            with pytest.raises(RuntimeError):
                future.result(timeout=5)

        for obj, initial_ref in zip(objs, initial_refs, strict=False):
            assert obj.get_ref_count() == initial_ref
        for key in keys:
            assert not backend.exists_in_put_tasks(key)

    def test_batched_submit_put_task_skips_in_mla_mode(self, backend, adapter):
        """In MLA worker_id_as0 mode, batched_submit_put_task should skip."""
        backend._mla_worker_id_as0_mode = True
        keys = [_make_cache_key(i) for i in range(3)]
        objs = [_make_memory_obj(adapter) for _ in range(3)]
        for obj in objs:
            obj.parent_allocator = None

        result = backend.batched_submit_put_task(keys, objs)
        assert result is None
        backend._handler.batch_store.assert_not_called()


# =========================================================================
# Tests — Get (sync)
# =========================================================================


class TestMaruBackendGet:
    def test_get_blocking_hit(self, backend, adapter):
        key = _make_cache_key()

        data_size = TEST_CHUNK_SIZE
        data = bytearray(data_size)
        mock_info = MemoryInfo(
            view=memoryview(data),
            region_id=100,
            page_index=0,
        )
        backend._handler.retrieve.return_value = mock_info

        result = backend.get_blocking(key)
        assert result is not None
        backend._handler.retrieve.assert_called_once()

    def test_get_blocking_miss(self, backend):
        key = _make_cache_key()
        backend._handler.retrieve.return_value = None

        result = backend.get_blocking(key)
        assert result is None

    def test_get_blocking_ref_count_increases(self, backend, adapter):
        """After get_blocking, the returned MemoryObj should have ref_count
        incremented."""
        # Pre-allocate so pool has page 0
        _make_memory_obj(adapter)

        key = _make_cache_key()
        mock_info = MemoryInfo(
            view=memoryview(bytearray(TEST_CHUNK_SIZE)),
            region_id=100,
            page_index=0,
        )
        backend._handler.retrieve.return_value = mock_info

        result = backend.get_blocking(key)
        assert result is not None
        # Pool objects start with ref_count=1, get_blocking calls ref_count_up
        assert result.get_ref_count() >= 2

    def test_batched_get_blocking(self, backend, adapter):
        """batched_get_blocking returns list of MemoryObj via batch_retrieve."""
        objs = [_make_memory_obj(adapter) for _ in range(2)]
        keys = [_make_cache_key(i) for i in range(2)]

        infos = []
        for obj in objs:
            rid, pid = CxlMemoryAdapter.decode_address(obj.metadata.address)
            infos.append(
                MemoryInfo(
                    view=memoryview(bytearray(TEST_CHUNK_SIZE)),
                    region_id=rid,
                    page_index=pid,
                )
            )
        backend._handler.batch_retrieve.return_value = infos

        results = backend.batched_get_blocking(keys)
        assert len(results) == 2
        for r in results:
            assert r is not None

    def test_batched_get_blocking_with_miss(self, backend, adapter):
        """batched_get_blocking returns None for missing keys."""
        obj = _make_memory_obj(adapter)
        keys = [_make_cache_key(i) for i in range(2)]

        rid, pid = CxlMemoryAdapter.decode_address(obj.metadata.address)
        info = MemoryInfo(
            view=memoryview(bytearray(TEST_CHUNK_SIZE)),
            region_id=rid,
            page_index=pid,
        )
        backend._handler.batch_retrieve.return_value = [info, None]

        results = backend.batched_get_blocking(keys)
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None


# =========================================================================
# Tests — Contains
# =========================================================================


class TestMaruBackendContains:
    def test_contains_true(self, backend):
        key = _make_cache_key()
        backend._handler.exists.return_value = True

        assert backend.contains(key) is True
        backend._handler.exists.assert_called_once_with(key.to_string())

    def test_contains_false(self, backend):
        key = _make_cache_key()
        backend._handler.exists.return_value = False

        assert backend.contains(key) is False

    def test_batched_contains_all_hit(self, backend):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_exists.return_value = [True, True, True]

        result = backend.batched_contains(keys)
        assert result == 3

    def test_batched_contains_partial_prefix(self, backend):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_exists.return_value = [True, True, False]

        result = backend.batched_contains(keys)
        assert result == 2

    def test_batched_contains_first_miss(self, backend):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_exists.return_value = [False, True, True]

        result = backend.batched_contains(keys)
        assert result == 0

    def test_contains_with_pin(self, backend):
        key = _make_cache_key()
        backend._handler.pin.return_value = True

        assert backend.contains(key, pin=True) is True
        backend._handler.pin.assert_called_once_with(key.to_string())
        backend._handler.exists.assert_not_called()

    def test_contains_with_pin_false(self, backend):
        key = _make_cache_key()
        backend._handler.pin.return_value = False

        assert backend.contains(key, pin=True) is False

    def test_batched_contains_with_pin(self, backend):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_pin.return_value = [True, True, True]

        result = backend.batched_contains(keys, pin=True)
        assert result == 3
        backend._handler.batch_pin.assert_called_once_with(
            [k.to_string() for k in keys]
        )
        backend._handler.batch_exists.assert_not_called()

    def test_batched_contains_with_pin_partial(self, backend):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_pin.return_value = [True, False, True]

        result = backend.batched_contains(keys, pin=True)
        assert result == 1

    def test_batched_contains_empty(self, backend):
        backend._handler.batch_exists.return_value = []
        assert backend.batched_contains([]) == 0


# =========================================================================
# Tests — Async Lookup
# =========================================================================


class TestMaruBackendAsyncLookup:
    def test_batched_async_contains_all_hit(self, backend, async_loop):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_exists.return_value = [True, True, True]

        result = _run_async(
            async_loop, backend.batched_async_contains("lookup-1", keys)
        )
        assert result == 3

    def test_batched_async_contains_partial_prefix(self, backend, async_loop):
        keys = [_make_cache_key(i) for i in range(3)]
        backend._handler.batch_exists.return_value = [True, False, True]

        result = _run_async(
            async_loop, backend.batched_async_contains("lookup-2", keys)
        )
        assert result == 1

    def test_batched_async_contains_empty(self, backend, async_loop):
        backend._handler.batch_exists.return_value = []
        result = _run_async(async_loop, backend.batched_async_contains("lookup-3", []))
        assert result == 0

    def test_batched_get_non_blocking_all_hit(self, backend, adapter, async_loop):
        keys = [_make_cache_key(i) for i in range(2)]

        objs = [_make_memory_obj(adapter) for _ in range(2)]
        infos = []
        for obj in objs:
            rid, pid = CxlMemoryAdapter.decode_address(obj.metadata.address)
            infos.append(
                MemoryInfo(
                    view=memoryview(bytearray(TEST_CHUNK_SIZE)),
                    region_id=rid,
                    page_index=pid,
                )
            )
        backend._handler.batch_retrieve.return_value = infos

        results = _run_async(
            async_loop, backend.batched_get_non_blocking("lookup-4", keys)
        )
        assert len(results) == 2
        for obj in results:
            assert obj is not None

    def test_batched_get_non_blocking_prefix_stop_on_miss(
        self, backend, adapter, async_loop
    ):
        """Second key is a miss -> only first returned (prefix semantics)."""
        keys = [_make_cache_key(i) for i in range(3)]

        obj = _make_memory_obj(adapter)
        rid, pid = CxlMemoryAdapter.decode_address(obj.metadata.address)
        info = MemoryInfo(
            view=memoryview(bytearray(TEST_CHUNK_SIZE)),
            region_id=rid,
            page_index=pid,
        )
        # hit, miss, hit -> should return only [hit]
        backend._handler.batch_retrieve.return_value = [info, None, info]

        results = _run_async(
            async_loop, backend.batched_get_non_blocking("lookup-5", keys)
        )
        assert len(results) == 1

    def test_batched_get_non_blocking_empty(self, backend, async_loop):
        backend._handler.batch_retrieve.return_value = []
        results = _run_async(
            async_loop, backend.batched_get_non_blocking("lookup-6", [])
        )
        assert results == []


# =========================================================================
# Tests — Pin / Unpin / Remove
# =========================================================================


class TestMaruBackendPinRemove:
    def test_pin_delegates_to_handler(self, backend):
        key = _make_cache_key()
        backend._handler.pin.return_value = True

        assert backend.pin(key) is True
        backend._handler.pin.assert_called_once_with(key.to_string())

    def test_pin_returns_false_on_failure(self, backend):
        key = _make_cache_key()
        backend._handler.pin.return_value = False

        assert backend.pin(key) is False

    def test_unpin_delegates_to_handler(self, backend):
        key = _make_cache_key()
        backend._handler.unpin.return_value = True

        assert backend.unpin(key) is True
        backend._handler.unpin.assert_called_once_with(key.to_string())

    def test_unpin_returns_false_on_failure(self, backend):
        key = _make_cache_key()
        backend._handler.unpin.return_value = False

        assert backend.unpin(key) is False

    def test_batched_unpin(self, backend):
        keys = [_make_cache_key(i) for i in range(3)]

        backend.batched_unpin(keys)
        backend._handler.batch_unpin.assert_called_once_with(
            [k.to_string() for k in keys]
        )

    def test_batched_unpin_empty(self, backend):
        backend.batched_unpin([])
        backend._handler.batch_unpin.assert_not_called()

    def test_remove_existing_key(self, backend):
        key = _make_cache_key()
        backend._handler.delete.return_value = True

        result = backend.remove(key)
        assert result is True
        backend._handler.delete.assert_called_once_with(key.to_string())

    def test_remove_nonexistent_key(self, backend):
        key = _make_cache_key()
        backend._handler.delete.return_value = False

        result = backend.remove(key)
        assert result is False


# =========================================================================
# Tests — Lifecycle
# =========================================================================


class TestMaruBackendLifecycle:
    def test_close_calls_handler_and_allocator(self, backend):
        backend.memory_allocator = MagicMock()
        backend.close()
        backend.memory_allocator.close.assert_called_once()
        backend._handler.close.assert_called_once()

    def test_close_drains_pending_put_tasks(self, backend, adapter):
        """close() should wait for in-flight put tasks to complete."""
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None
        key = _make_cache_key()

        # Submit a real put task that will complete via the event loop
        future = backend.submit_put_task(key, obj)
        future.result(timeout=5)

        # After drain, close should succeed
        backend.close()
        backend._handler.close.assert_called_once()


# =========================================================================
# Tests — Store Handle Roundtrip
# =========================================================================


class TestMaruBackendStoreHandle:
    def test_store_handle_roundtrip(self, backend, adapter):
        """AllocHandle from create_store_handle should match original."""
        obj = _make_memory_obj(adapter)
        obj.parent_allocator = None

        handle = adapter.create_store_handle(obj)
        assert handle.region_id == 100
        assert handle.page_index == 0
        assert handle._size == obj.metadata.phy_size
