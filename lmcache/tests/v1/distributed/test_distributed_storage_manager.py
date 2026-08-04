# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for StorageManager.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

try:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
except ImportError:
    # Skip tests if L1Manager cannot be imported
    pytest.skip(
        "Skipping because StorageManager cannot be imported", allow_module_level=True
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_memory_config():
    """Create a basic L1MemoryManagerConfig for testing."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,  # 4KB
    )


@pytest.fixture
def small_memory_config():
    """Create a small L1MemoryManagerConfig to test memory exhaustion."""
    return L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,  # 64MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,
    )


@pytest.fixture
def basic_l1_config(basic_memory_config):
    """Create a basic L1ManagerConfig for testing."""
    return L1ManagerConfig(
        memory_config=basic_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def small_l1_config(small_memory_config):
    """Create a small L1ManagerConfig to test memory exhaustion."""
    return L1ManagerConfig(
        memory_config=small_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def basic_storage_manager_config(basic_l1_config):
    """Create a basic StorageManagerConfig for testing"""
    return StorageManagerConfig(
        l1_manager_config=basic_l1_config,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
    )


@pytest.fixture
def small_storage_manager_config(small_l1_config):
    """Create a small StorageManagerConfig to test memory exhaustion."""
    return StorageManagerConfig(
        l1_manager_config=small_l1_config,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
    )


@pytest.fixture
def basic_layout():
    """Create a basic MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def large_layout():
    """Create a large MemoryLayoutDesc that will exhaust small memory.

    Each allocation is 8MB (2M elements * 4 bytes).
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size([2048, 1024])],  # 2M elements * 4 bytes = 8MB
        dtypes=[torch.float32],
    )


def make_object_key(chunk_hash: int, model_name: str = "test_model", kv_rank: int = 0):
    """Helper to create ObjectKey instances."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model_name, kv_rank=kv_rank)


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until a predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(
    sm: StorageManager,
    handle,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_status until it returns a non-None value.

    Returns the contiguous prefix-hit count (``count_leading_ones``) of the
    found bitmap, matching the dense semantics these tests assert on.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result.count_leading_ones()
        time.sleep(poll_interval)
    return None


def wait_for_sparse_found(
    sm: StorageManager,
    handle,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> set[int] | None:
    """Poll query_prefetch_status; return the found-key index set.

    For SPARSE prefetches the result bitmap is gap-tolerant, so callers read
    the full set via ``get_indices_list`` rather than ``count_leading_ones``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return set(result.get_indices_list())
        time.sleep(poll_interval)
    return None


# =============================================================================
# Tests
# =============================================================================


class TestStorageManagerBasic:
    """Tests for basic functionality of StorageManager."""

    def test_basic_reserve_write(self, basic_storage_manager_config, basic_layout):
        """Test basic reserve and write functionality."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_key = make_object_key(chunk_hash=12345)

        # Reserve space for the object
        ret = storage_manager.reserve_write([object_key], basic_layout, mode="new")
        assert object_key in ret
        assert ret[object_key] is not None

        # Should not have any error
        storage_manager.finish_write([object_key])

        storage_manager.close()

    def test_reserve_write_multiple_keys(
        self, basic_storage_manager_config, basic_layout
    ):
        """Test reserve_write with multiple keys."""
        storage_manager = StorageManager(basic_storage_manager_config)

        keys = [make_object_key(i) for i in range(5)]

        ret = storage_manager.reserve_write(keys, basic_layout, mode="new")

        # All keys should be allocated
        assert len(ret) == len(keys)
        for key in keys:
            assert key in ret
            assert ret[key] is not None

        storage_manager.close()

    def test_reserve_write_oom(self, small_storage_manager_config, large_layout):
        """Test reserve_write raises L1Error on out-of-memory."""
        storage_manager = StorageManager(small_storage_manager_config)

        keys = [make_object_key(i) for i in range(20)]

        ret = storage_manager.reserve_write(keys, large_layout, mode="new")

        # At least some of the keys could be allocated
        assert len(ret) < len(keys)

        # If some keys were allocated, they should not be None
        for key, obj in ret.items():
            assert obj is not None

        storage_manager.close()

    def test_basic_prefetch(self, basic_storage_manager_config, basic_layout):
        """Test basic prefetch functionality."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(5)]

        # Write keys into storage manager
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="new")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch all the objects
        handle = storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(object_keys, basic_layout)
        )

        hit_count = storage_manager.query_prefetch_status(handle).count_leading_ones()
        assert hit_count is not None
        assert hit_count == len(object_keys)

        storage_manager.close()

    def test_prefetch_partial_prefix_hits(
        self, basic_storage_manager_config, basic_layout
    ):
        """Test prefetch with partial hits."""
        # 5 keys: 0, 1, 3, 4 are written, 2 is missing
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(5)]

        # Write only some keys into storage manager
        keys_to_write = [object_keys[0], object_keys[1], object_keys[3], object_keys[4]]
        ret = storage_manager.reserve_write(keys_to_write, basic_layout, mode="new")
        for key in keys_to_write:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch all the objects
        handle = storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(object_keys, basic_layout)
        )

        hit_count = storage_manager.query_prefetch_status(handle).count_leading_ones()
        assert hit_count is not None
        assert hit_count == 2  # Only 2 keys were written

        # The last 2 keys should be "writable"
        ret = storage_manager.reserve_write(
            object_keys[3:], basic_layout, mode="update"
        )
        for key in object_keys[3:]:
            assert key in ret
            assert ret[key] is not None

        storage_manager.close()

    def test_read_prefetched_basic(self, basic_storage_manager_config, basic_layout):
        """Test reading prefetched objects."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(3)]

        # Write keys into storage manager
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="new")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch all the objects
        handle = storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(object_keys, basic_layout)
        )

        hit_count = storage_manager.query_prefetch_status(handle).count_leading_ones()
        assert hit_count is not None
        assert hit_count == len(object_keys)

        # Read the prefetched objects
        with storage_manager.read_prefetched_results(object_keys) as retrieved_objects:
            assert retrieved_objects is not None
            assert len(retrieved_objects) == len(object_keys)

        # Finish reading
        storage_manager.finish_read_prefetched(object_keys)

        # Now the objects should be writable again
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="update")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None

        storage_manager.close()

    def test_read_prefetched_not_found(
        self, basic_storage_manager_config, basic_layout
    ):
        """Test reading prefetched objects that were not found."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(5)]

        # Write all objects into storage manager
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="new")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch objects except the first one
        handle = storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(object_keys[1:], basic_layout)
        )
        hit_count = storage_manager.query_prefetch_status(handle).count_leading_ones()
        assert hit_count is not None
        assert hit_count == len(object_keys) - 1

        # Attempt to read all the objects, should get None
        with storage_manager.read_prefetched_results(object_keys) as retrieved_objects:
            assert retrieved_objects is None

        # Remaining 4 objects should still be writable (i.e., no dangling read locks)
        ret = storage_manager.reserve_write(
            object_keys[1:], basic_layout, mode="update"
        )
        for key in object_keys[1:]:
            assert key in ret
            assert ret[key] is not None
        storage_manager.close()


# =============================================================================
# Tests for multi-reader (count / num_readers) support
# =============================================================================


class TestStorageManagerMultiReader:
    """Tests for the num_readers / count parameters.

    These parameters allow multiple workers (e.g. MLA with
    TP > 1) to each hold an independent read lock on the same
    prefetched object.
    """

    def test_prefetch_with_extra_count(
        self, basic_storage_manager_config, basic_layout
    ):
        """submit_prefetch_task(extra_count=N-1) acquires N locks."""
        sm = StorageManager(basic_storage_manager_config)
        keys = [make_object_key(i) for i in range(3)]

        # Write keys
        ret = sm.reserve_write(keys, basic_layout, mode="new")
        assert len(ret) == len(keys)
        sm.finish_write(list(ret.keys()))

        extra_count = 2  # total = 1 + 2 = 3 locks
        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(keys, basic_layout, extra_count=extra_count)
        )
        hit = sm.query_prefetch_status(handle).count_leading_ones()
        assert hit == len(keys)

        # Release with matching extra_count
        sm.finish_read_prefetched(keys, extra_count=extra_count)

        # All locks released -> objects writable again
        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == len(keys)

        sm.close()

    def test_finish_read_prefetched_partial_extra_count(
        self, basic_storage_manager_config, basic_layout
    ):
        """Partial extra_count release leaves locks held."""
        sm = StorageManager(basic_storage_manager_config)
        keys = [make_object_key(i) for i in range(2)]

        ret = sm.reserve_write(keys, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        extra_count = 3  # total = 1 + 3 = 4 locks
        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(keys, basic_layout, extra_count=extra_count)
        )
        hit = sm.query_prefetch_status(handle).count_leading_ones()
        assert hit == len(keys)

        # Release 2 of 4 (1 + extra_count=1)
        sm.finish_read_prefetched(keys, extra_count=1)

        # Objects should NOT be writable (2 locks remain)
        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == 0

        # Release remaining 2 (1 + extra_count=1)
        sm.finish_read_prefetched(keys, extra_count=1)

        # Now writable
        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == len(keys)

        sm.close()

    def test_prefetch_skipped_keys_released_with_extra(
        self, basic_storage_manager_config, basic_layout
    ):
        """Non-prefix L1 hits are released with correct extra.

        Keys {0,1,3,4} exist, key 2 is missing.  The prefix
        hits are {0,1}; keys {3,4} must have their N locks
        released to avoid dangling locks.
        """
        sm = StorageManager(basic_storage_manager_config)
        all_keys = [make_object_key(i) for i in range(5)]
        existing = [all_keys[i] for i in [0, 1, 3, 4]]

        ret = sm.reserve_write(existing, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        extra_count = 1  # total = 1 + 1 = 2 locks
        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(all_keys, basic_layout, extra_count=extra_count)
        )
        hit = sm.query_prefetch_status(handle).count_leading_ones()
        # Only prefix {0,1} count as hits
        assert hit is not None
        assert hit == 2

        # Finish the prefix hits
        sm.finish_read_prefetched(all_keys[:2], extra_count=extra_count)

        # Keys {3,4} should be writable (skipped locks released)
        ret = sm.reserve_write(
            [all_keys[3], all_keys[4]],
            basic_layout,
            mode="update",
        )
        assert len(ret) == 2

        sm.close()

    def test_extra_count_default_is_zero(
        self, basic_storage_manager_config, basic_layout
    ):
        """Default extra_count=0 behaves same as before."""
        sm = StorageManager(basic_storage_manager_config)
        keys = [make_object_key(i) for i in range(3)]

        ret = sm.reserve_write(keys, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, basic_layout))
        hit = sm.query_prefetch_status(handle).count_leading_ones()
        assert hit == len(keys)

        # Single finish is enough
        sm.finish_read_prefetched(keys)

        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == len(keys)

        sm.close()


# =============================================================================
# L2 Prefetch Integration Tests
# =============================================================================


@pytest.fixture
def l2_storage_manager_config(basic_l1_config):
    """Create a StorageManagerConfig with one MockL2Adapter."""
    return StorageManagerConfig(
        l1_manager_config=basic_l1_config,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[
                MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0),
            ],
        ),
    )


class TestStorageManagerL2Prefetch:
    """Tests for prefetching from L2 through StorageManager."""

    def _write_keys_and_wait_for_l2(
        self,
        sm: StorageManager,
        keys: list[ObjectKey],
        layout: MemoryLayoutDesc,
    ) -> None:
        """Write keys to L1 via StorageManager and wait for L2 store."""
        ret = sm.reserve_write(keys, layout, mode="new")
        assert len(ret) == len(keys)
        sm.finish_write(list(ret.keys()))

        # Wait for StoreController to propagate all keys to L2
        adapter = sm._l2_adapters[0]
        ok = wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in keys),  # type: ignore
            timeout=10.0,
        )
        assert ok, "Keys should be stored in L2 by StoreController"

    def test_prefetch_from_l2(self, l2_storage_manager_config, basic_layout):
        """Write to L1 → store to L2 → clear L1 → prefetch from L2."""
        sm = StorageManager(l2_storage_manager_config)
        keys = [make_object_key(i) for i in range(5)]

        self._write_keys_and_wait_for_l2(sm, keys, basic_layout)

        # Brief sleep to let StoreController release read locks
        # after L2 store completion, then clear L1
        time.sleep(0.05)
        sm.clear()
        used, _ = sm._l1_manager.get_memory_usage()
        assert used == 0, f"L1 should be empty after clear, but {used} bytes used"

        # Prefetch — L1 has 0 hits, L2 should have all 5
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, basic_layout))
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 5, f"Expected 5 hits from L2, got {hit_count}"

        # Verify keys are read-locked in L1 after prefetch
        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == len(keys)

        sm.finish_read_prefetched(keys)
        sm.close()

    def test_prefetch_mixed_l1_l2(self, l2_storage_manager_config, basic_layout):
        """Some keys in L1, rest in L2 → combined prefix hits."""
        sm = StorageManager(l2_storage_manager_config)

        # Use distinct hash ranges for L1-only vs L2 keys
        all_keys = [make_object_key(i) for i in range(5)]
        l2_only_keys = all_keys[2:]  # keys 2, 3, 4 only in L2

        # Write all keys to L1 (StoreController will push to L2)
        self._write_keys_and_wait_for_l2(sm, all_keys, basic_layout)

        # Delete only keys 2, 3, 4 from L1 so they must come from L2
        sm._l1_manager.delete(l2_only_keys)

        # Prefetch all 5 keys: first 2 from L1, next 3 from L2
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(all_keys, basic_layout))
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 5, (
            f"Expected 5 combined hits (2 L1 + 3 L2), got {hit_count}"
        )

        # Clean up read locks
        sm.finish_read_prefetched(all_keys)
        sm.close()

    def test_prefetch_nothing_in_l2(self, l2_storage_manager_config, basic_layout):
        """Prefetch keys not in L2 → returns 0 L2 hits."""
        sm = StorageManager(l2_storage_manager_config)

        # Don't write anything — keys exist nowhere
        keys = [make_object_key(i) for i in range(3)]

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, basic_layout))
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 0, f"Expected 0 hits, got {hit_count}"

        sm.close()

    def test_warm_skip_l2_is_noop(self, l2_storage_manager_config, basic_layout):
        """``mode=WARM`` + ``skip_l2=True`` submits no controller request.

        Regression: the WARM branch must honor ``skip_l2`` (it previously
        ignored it, issuing L2 lookup/load work and mutating L1). The returned
        :class:`PrefetchHandle` is the public signal — no request id to track
        and no L2-sourced indices; since the controller is the only path that
        loads from L2, this also means L1 is left untouched.
        """
        sm = StorageManager(l2_storage_manager_config)
        keys = [make_object_key(i) for i in range(3)]

        # Put the keys in L2 so a non-skip warm would have something to load,
        # then clear L1 so a load would be the only way they could reappear.
        self._write_keys_and_wait_for_l2(sm, keys, basic_layout)
        sm.clear()

        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(keys, basic_layout, mode=PrefetchMode.WARM),
            skip_l2=True,
        )

        # skip_l2 honored: the controller was never asked.
        assert handle.prefetch_request_id == -1
        assert handle.l2_orig_indices == ()
        assert handle.total_requested_keys == len(keys)

        sm.close()

    def test_sparse_skip_l2_locks_only_l1(
        self, l2_storage_manager_config, basic_layout
    ):
        """``policy=SPARSE`` + ``skip_l2=True`` submits no controller request.

        Only L1-resident keys are locked and reported found; keys present only
        in L2 are reported as misses.
        """
        sm = StorageManager(l2_storage_manager_config)
        all_keys = [make_object_key(i) for i in range(3)]

        # All keys reach L2; delete key 1 from L1 so it is L2-only.
        self._write_keys_and_wait_for_l2(sm, all_keys, basic_layout)
        time.sleep(0.05)
        deleted, skipped = sm.delete_l1_keys([all_keys[1]])
        assert (deleted, skipped) == (1, 0)

        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(all_keys, basic_layout, policy=TrimPolicy.SPARSE),
            skip_l2=True,
        )

        # skip_l2 honored: the controller was never asked.
        assert handle.prefetch_request_id == -1
        assert handle.l2_orig_indices == ()
        assert handle.l1_found_indices == (0, 2)

        found = sm.query_prefetch_status(handle)
        assert found is not None
        assert found.get_indices_list() == [0, 2]

        sm.finish_read_prefetched([all_keys[0], all_keys[2]])
        sm.close()

    def test_prefetch_l2_partial_prefix(self, l2_storage_manager_config, basic_layout):
        """L2 has keys {0,1,3,4} but not 2 → L2 returns prefix of 2."""
        sm = StorageManager(l2_storage_manager_config)

        all_keys = [make_object_key(i) for i in range(5)]
        # Write only keys 0, 1, 3, 4 (skip key 2)
        keys_to_write = [all_keys[i] for i in [0, 1, 3, 4]]
        self._write_keys_and_wait_for_l2(sm, keys_to_write, basic_layout)

        # Brief sleep to let StoreController release read locks
        # after L2 store completion, then clear L1
        time.sleep(0.05)
        sm.clear()
        used, _ = sm._l1_manager.get_memory_usage()
        assert used == 0, f"L1 should be empty after clear, but {used} bytes used"

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(all_keys, basic_layout))
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 2, (
            f"Expected 2 prefix hits from L2 (gap at index 2), got {hit_count}"
        )

        # Only prefix keys {0, 1} should be readable
        with sm.read_prefetched_results(all_keys[:2]) as objs:
            assert objs is not None
            assert len(objs) == 2

        sm.finish_read_prefetched(all_keys[:2])
        sm.close()

    def test_prefetch_l1_prefix_plus_l2_continuation(
        self, l2_storage_manager_config, basic_layout
    ):
        """L1 has keys {0,1}, L2 has {2,3,4} → combined prefix of 5."""
        sm = StorageManager(l2_storage_manager_config)

        all_keys = [make_object_key(i) for i in range(5)]

        # Write all keys → StoreController stores all to L2
        self._write_keys_and_wait_for_l2(sm, all_keys, basic_layout)

        # Delete keys {2,3,4} from L1 only, keeping them in L2
        sm._l1_manager.delete(all_keys[2:])

        # Prefetch: L1 prefix hits = 2 (keys 0,1), L2 loads {2,3,4} → total = 5
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(all_keys, basic_layout))
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None
        assert hit_count == 5, f"Expected 5 total hits (2 L1 + 3 L2), got {hit_count}"

        sm.finish_read_prefetched(all_keys)
        sm.close()


# =============================================================================
# Tests for LM-291 failure event production
# =============================================================================


@pytest.fixture
def captured_events():
    """Enable the global event bus and capture all L1/L2 failure events.

    Replaces the process-wide ``_global_bus`` with a fresh enabled bus,
    subscribes a callback for the three failure event types, yields the
    captured-events list, then resets the bus to disabled on teardown so
    later tests don't see leftover state.
    """
    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=10_000))
    events: list[Event] = []

    def _capture(event: Event) -> None:
        events.append(event)

    for et in (
        EventType.L1_ALLOCATION_FAILED,
        EventType.L1_READ_FAILED,
        EventType.L2_PREFETCH_FAILED,
    ):
        bus.subscribe(et, _capture)
    bus.start()
    try:
        yield events
    finally:
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


def _events_of_type(events: list, event_type: EventType) -> list:
    return [e for e in events if e.event_type == event_type]


class TestFailureEventProduction:
    """Verifies LM-291 health-monitoring events are published at the
    right producer call sites with the expected metadata."""

    def test_reserve_write_oom_emits_l1_allocation_failed(
        self, small_storage_manager_config, large_layout, captured_events
    ):
        """OOM during user store must publish L1_ALLOCATION_FAILED with
        during=l1_store and the OOM keys."""
        sm = StorageManager(small_storage_manager_config)
        try:
            keys = [make_object_key(i) for i in range(20)]
            sm.reserve_write(keys, large_layout, mode="new")

            # Allow drain thread to deliver the event.
            assert wait_for_condition(
                lambda: (
                    len(
                        _events_of_type(captured_events, EventType.L1_ALLOCATION_FAILED)
                    )
                    >= 1
                ),
                timeout=2.0,
            )

            alloc_events = _events_of_type(
                captured_events, EventType.L1_ALLOCATION_FAILED
            )
            assert len(alloc_events) == 1
            meta = alloc_events[0].metadata
            assert meta["during"] == "l1_store"
            assert len(meta["keys"]) > 0
            # All emitted keys must be from the OOM subset of the request.
            assert set(meta["keys"]).issubset(set(keys))
        finally:
            sm.close()

    def test_unsafe_read_missing_key_emits_l1_read_failed(
        self, basic_storage_manager_config, basic_layout, captured_events
    ):
        """Deleting a key between reserve_read and unsafe_read must publish
        L1_READ_FAILED with during=l1_retrieve, reason=not_found."""
        sm = StorageManager(basic_storage_manager_config)
        try:
            keys = [make_object_key(i) for i in range(3)]

            # Write + finish_write so the keys are readable.
            ret = sm.reserve_write(keys, basic_layout, mode="new")
            assert len(ret) == len(keys)
            sm.finish_write(list(ret.keys()))

            # Prefetch to acquire read locks on all keys.
            handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, basic_layout))
            assert wait_for_prefetch_status(sm, handle) == len(keys)

            # Force a mid-read race by removing the key from L1Manager's
            # internal state dict, bypassing the lock check that
            # ``L1Manager.delete`` enforces. This simulates the exact TOCTOU
            # anomaly the metric is designed to catch: reserve_read acquired
            # a lock, but the key vanished before unsafe_read.
            del sm._l1_manager._objects[keys[1]]

            # Now attempt to read — unsafe_read should find the middle key
            # missing, emitting L1_READ_FAILED(during=l1_retrieve,
            # reason=not_found).
            with sm.read_prefetched_results(keys) as objs:
                assert objs is None  # all_good=False because middle key is gone

            assert wait_for_condition(
                lambda: (
                    len(_events_of_type(captured_events, EventType.L1_READ_FAILED)) >= 1
                ),
                timeout=2.0,
            )

            read_events = _events_of_type(captured_events, EventType.L1_READ_FAILED)
            assert len(read_events) == 1
            meta = read_events[0].metadata
            assert meta["during"] == "l1_retrieve"
            assert meta["reason"] == "not_found"
            assert keys[1] in meta["keys"]
        finally:
            sm.close()


class TestStorageManagerSparsePrefetch:
    """SPARSE prefetch: retain a read lock on every found key, not just the
    leading contiguous prefix."""

    def test_sparse_keeps_all_found_not_just_prefix(
        self, basic_storage_manager_config, basic_layout
    ):
        """Sparse L1 prefetch retains + read-locks every found key, including
        those past a gap (unlike the contiguous-prefix default)."""
        sm = StorageManager(basic_storage_manager_config)
        all_keys = [make_object_key(i) for i in range(5)]
        # Write {0,1,3,4}; key 2 is the gap.
        existing = [all_keys[i] for i in (0, 1, 3, 4)]
        ret = sm.reserve_write(existing, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(all_keys, basic_layout, policy=TrimPolicy.SPARSE)
        )
        found = wait_for_sparse_found(sm, handle, timeout=10.0)

        # Sparse: all four found indices, NOT just the prefix {0, 1}.
        assert found == {0, 1, 3, 4}

        # Every found key is read-locked (none write-reservable).
        locked = sm.reserve_write(existing, basic_layout, mode="update")
        assert len(locked) == 0

        # Releasing the full found set frees them.
        sm.finish_read_prefetched(existing)
        freed = sm.reserve_write(existing, basic_layout, mode="update")
        assert len(freed) == len(existing)

        sm.close()

    def test_sparse_from_l2_loads_all_found(
        self, l2_storage_manager_config, basic_layout
    ):
        """Sparse prefetch from L2 loads every found key (controller skips the
        prefix-only trim), not just the prefix before a gap."""
        sm = StorageManager(l2_storage_manager_config)
        all_keys = [make_object_key(i) for i in range(5)]
        # L2 has {0,1,3,4}; gap at 2.
        existing = [all_keys[i] for i in (0, 1, 3, 4)]
        wret = sm.reserve_write(existing, basic_layout, mode="new")
        sm.finish_write(list(wret.keys()))
        adapter = sm._l2_adapters[0]
        assert wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in existing),  # type: ignore
            timeout=10.0,
        )
        time.sleep(0.05)
        sm.clear()
        used, _ = sm._l1_manager.get_memory_usage()
        assert used == 0

        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(all_keys, basic_layout, policy=TrimPolicy.SPARSE)
        )
        found = wait_for_sparse_found(sm, handle, timeout=10.0)

        # Sparse from L2: all found {0,1,3,4}, NOT the contiguous prefix {0,1}.
        assert found == {0, 1, 3, 4}

        sm.finish_read_prefetched(existing)
        sm.close()


class TestStorageManagerDelete:
    """Tests for StorageManager.delete_l1_keys()."""

    def test_delete_counts_deleted_and_missing(
        self, basic_storage_manager_config, basic_layout
    ):
        """delete_l1_keys reports deleted keys; missing keys are a no-op."""
        storage_manager = StorageManager(basic_storage_manager_config)

        resident = make_object_key(1)
        missing = make_object_key(2)
        storage_manager.reserve_write([resident], basic_layout, mode="new")
        storage_manager.finish_write([resident])

        assert storage_manager.delete_l1_keys([resident, missing]) == (1, 0)
        assert storage_manager._l1_manager.get_object_state(resident) is None

        storage_manager.close()

    def test_delete_non_force_skips_locked_force_removes(
        self, basic_storage_manager_config, basic_layout
    ):
        """A locked key is skipped by non-force delete and removed by force."""
        storage_manager = StorageManager(basic_storage_manager_config)

        # Reserve write without finishing -> the key is write-locked.
        key = make_object_key(1)
        storage_manager.reserve_write([key], basic_layout, mode="new")

        # Non-force delete refuses the locked key; it survives.
        assert storage_manager.delete_l1_keys([key]) == (0, 1)
        assert storage_manager._l1_manager.get_object_state(key) is not None

        # Force delete removes it regardless of the lock.
        assert storage_manager.delete_l1_keys([key], force=True) == (1, 0)
        assert storage_manager._l1_manager.get_object_state(key) is None

        storage_manager.close()
