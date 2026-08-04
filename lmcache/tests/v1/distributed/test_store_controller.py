# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for StoreController.

Tests verify the end-to-end flow: L1 write -> listener notification ->
L2 store submission -> completion handling -> L1 read lock release.

Uses a real L1Manager and MockL2Adapter (with debug methods) to exercise
the full integration without mocking internals.
"""

# Standard
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.eviction_policy.noop import (
    NoOpEvictionPolicy,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    BufferOnlyStorePolicy,
    DefaultStorePolicy,
    StorePolicy,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """
    Poll until a predicate returns True or timeout.

    Args:
        predicate: Callable returning bool.
        timeout: Max wait time in seconds.
        poll_interval: Time between polls in seconds.

    Returns:
        True if predicate was satisfied, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def write_keys_to_l1(
    l1_manager: L1Manager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> list[ObjectKey]:
    """
    Write keys to L1 (reserve_write + finish_write).

    Args:
        l1_manager: The L1 manager to write to.
        keys: Object keys to write.
        layout: Memory layout description.

    Returns:
        List of keys that were successfully written.
    """
    results = l1_manager.reserve_write(
        keys=keys,
        is_temporary=[False] * len(keys),
        layout_desc=layout,
        mode="new",
    )
    written = [k for k, (e, m) in results.items() if m is not None]
    if written:
        l1_manager.finish_write(written)
    return written


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    """Create an L1Manager with a reasonable memory config."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


def make_adapter() -> MockL2Adapter:
    """Create a MockL2Adapter with fast bandwidth."""
    config = MockL2AdapterConfig(
        max_size_gb=0.01,
        mock_bandwidth_gb=10.0,
    )
    return MockL2Adapter(config)


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


# =============================================================================
# StoreListener Tests
# =============================================================================


class TestStoreListener:
    """Test StoreListener as a standalone component."""

    def test_pop_pending_keys_returns_empty_initially(self):
        """A fresh listener should have no pending keys."""
        listener = StoreListener()
        assert listener.pop_pending_keys() == []
        listener.close()

    def test_on_l1_keys_write_finished_enqueues_keys(self):
        """Keys passed to on_l1_keys_write_finished should be retrievable via pop."""
        listener = StoreListener()
        keys = [make_object_key(i) for i in range(3)]

        listener.on_l1_keys_write_finished(keys)
        popped = listener.pop_pending_keys()

        assert popped == keys
        listener.close()

    def test_pop_pending_keys_clears_queue(self):
        """pop_pending_keys should drain the queue."""
        listener = StoreListener()
        listener.on_l1_keys_write_finished([make_object_key(0)])
        listener.pop_pending_keys()

        assert listener.pop_pending_keys() == []
        listener.close()

    def test_on_l1_keys_write_finished_signals_eventfd(self):
        """The eventfd should become readable after on_l1_keys_write_finished."""
        listener = StoreListener()
        efd = listener.get_event_fd()
        poller = select.poll()
        poller.register(efd, select.POLLIN)

        listener.on_l1_keys_write_finished([make_object_key(0)])

        events = poller.poll(1000)
        assert len(events) > 0
        listener.close()

    def test_multiple_writes_accumulate(self):
        """Multiple on_l1_keys_write_finished calls should accumulate keys."""
        listener = StoreListener()
        listener.on_l1_keys_write_finished([make_object_key(0)])
        listener.on_l1_keys_write_finished([make_object_key(1), make_object_key(2)])

        popped = listener.pop_pending_keys()
        assert len(popped) == 3
        listener.close()

    def test_finish_write_and_reserve_read_does_not_enqueue(self):
        """on_l1_keys_finish_write_and_reserve_read should not enqueue keys."""
        listener = StoreListener()
        keys = [make_object_key(i) for i in range(3)]

        listener.on_l1_keys_finish_write_and_reserve_read(keys)

        assert listener.pop_pending_keys() == []
        assert listener.pending_count() == 0
        listener.close()


# =============================================================================
# StoreController Integration Tests
# =============================================================================


class TestStoreControllerLifecycle:
    """Test StoreController start/stop behavior."""

    def test_start_stop(self, l1_manager):
        """StoreController should start and stop cleanly."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        ctrl.stop()
        adapter.close()

    def test_start_stop_no_adapters(self, l1_manager):
        """Controller should start and stop cleanly with no adapters."""
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        ctrl.stop()


class TestStoreControllerSingleAdapter:
    """Test StoreController with one MockL2Adapter."""

    def test_l1_write_triggers_l2_store(self, l1_manager):
        """Writing to L1 should cause the object to appear in L2."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for the object to appear in L2
        ok = wait_for_condition(
            lambda: adapter.debug_get_stored_object_count() == 1,
            timeout=5.0,
        )
        assert ok, "Object should be stored in L2 after L1 write"
        assert adapter.debug_has_key(keys[0])

        ctrl.stop()
        adapter.close()

    def test_multiple_keys_stored_to_l2(self, l1_manager):
        """Multiple L1 writes should result in all keys in L2."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(
            lambda: adapter.debug_get_stored_object_count() == 5,
            timeout=5.0,
        )
        assert ok, "All 5 objects should be stored in L2"
        for key in keys:
            assert adapter.debug_has_key(key)

        ctrl.stop()
        adapter.close()

    def test_read_lock_released_after_store(self, l1_manager):
        """After L2 store completes, L1 read locks should be released."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for L2 store to complete
        ok = wait_for_condition(
            lambda: adapter.debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok

        # Verify read lock is released: the key should be updatable
        ok = wait_for_condition(
            lambda: (
                l1_manager.reserve_write(
                    keys=keys,
                    is_temporary=[False],
                    layout_desc=layout,
                    mode="update",
                )[keys[0]][1]
                is not None
            ),
            timeout=5.0,
        )
        assert ok, "Key should be updatable after store controller releases read lock"

        ctrl.stop()
        adapter.close()

    def test_multiple_concurrent_requests_same_adapter(self, l1_manager):
        """Each L1-write batch produces an independent in-flight store request.

        When two batches target the same adapter, the adapter's single
        eventfd may batch-signal both completions. All in-flight requests
        should be advanced and their L1 read locks released.
        """
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        batch_a = [make_object_key(i) for i in range(3)]
        batch_b = [make_object_key(i) for i in range(3, 6)]

        write_keys_to_l1(l1_manager, batch_a, layout)
        write_keys_to_l1(l1_manager, batch_b, layout)

        all_keys = batch_a + batch_b
        ok = wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in all_keys),
            timeout=5.0,
        )
        assert ok, "Both batches should reach L2"

        # Every key must be updatable: read locks from both requests released.
        # ``reserve_write`` acquires a write lock on success, so release it
        # via ``finish_write`` after each successful check — otherwise the
        # second poll would find already-locked keys and time out.
        def all_keys_updatable() -> bool:
            for k in all_keys:
                result = l1_manager.reserve_write(
                    keys=[k],
                    is_temporary=[False],
                    layout_desc=layout,
                    mode="update",
                )
                if result[k][1] is None:
                    return False
                l1_manager.finish_write([k])
            return True

        ok = wait_for_condition(all_keys_updatable, timeout=5.0)
        assert ok, "Read locks must be released for every concurrent request"

        ctrl.stop()
        adapter.close()


class TestStoreControllerMultipleAdapters:
    """Test StoreController with multiple L2 adapters."""

    def test_stores_to_all_adapters(self, l1_manager):
        """DefaultStorePolicy sends keys to all adapters."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for both adapters to have the object
        ok = wait_for_condition(
            lambda: all(a.debug_has_key(keys[0]) for a in adapters),
            timeout=5.0,
        )
        assert ok, "Object should be stored in both L2 adapters"

        ctrl.stop()
        for a in adapters:
            a.close()

    def test_read_lock_released_after_all_adapters_complete(self, l1_manager):
        """Read locks should be released after all adapter stores complete."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for stores to both adapters, then check updatability
        ok = wait_for_condition(
            lambda: all(a.debug_has_key(keys[0]) for a in adapters),
            timeout=5.0,
        )
        assert ok

        ok = wait_for_condition(
            lambda: (
                l1_manager.reserve_write(
                    keys=keys,
                    is_temporary=[False],
                    layout_desc=layout,
                    mode="update",
                )[keys[0]][1]
                is not None
            ),
            timeout=5.0,
        )
        assert ok, "Key should be updatable after all adapter stores complete"

        ctrl.stop()
        for a in adapters:
            a.close()


class TestStoreControllerNoAdapters:
    """Test StoreController with no L2 adapters."""

    def test_write_with_no_adapters_no_read_lock(self, l1_manager):
        """With no adapters, no read locks should be held after L1 write."""
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Give the controller time to process the listener event
        time.sleep(0.3)

        # Key should be updatable (no read locks)
        result = l1_manager.reserve_write(
            keys=keys,
            is_temporary=[False],
            layout_desc=layout,
            mode="update",
        )
        assert result[keys[0]][1] is not None

        ctrl.stop()


class TestStoreControllerCustomPolicy:
    """Test StoreController with custom StorePolicy implementations."""

    def test_policy_that_skips_all_adapters(self, l1_manager):
        """A policy returning empty targets should not trigger L2 stores."""

        class SkipAllPolicy(StorePolicy):
            """Never store to L2."""

            def select_store_targets(self, keys, adapters):
                return {}

            def select_l1_deletions(self, keys):
                return []

        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=SkipAllPolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Give the controller time to process
        time.sleep(0.5)

        assert adapter.debug_get_stored_object_count() == 0, (
            "No objects should be stored when policy skips all adapters"
        )

        ctrl.stop()
        adapter.close()

    def test_policy_that_deletes_from_l1(self, l1_manager):
        """A policy that requests L1 deletion should result in deleted keys."""

        class DeleteAfterStorePolicy(StorePolicy):
            """Delete all keys from L1 after storing to L2."""

            def select_store_targets(self, keys, adapters):
                return {ad.index: list(keys) for ad in adapters}

            def select_l1_deletions(self, keys):
                return list(keys)

        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DeleteAfterStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for L2 store and L1 deletion
        ok = wait_for_condition(
            lambda: adapter.debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok, "Object should be stored in L2"

        # After deletion, reserve_write with mode="new" should succeed
        ok = wait_for_condition(
            lambda: (
                l1_manager.reserve_write(
                    keys=keys,
                    is_temporary=[False],
                    layout_desc=layout,
                    mode="new",
                )[keys[0]][1]
                is not None
            ),
            timeout=5.0,
        )
        assert ok, "Key should be re-creatable after L1 deletion by policy"

        ctrl.stop()
        adapter.close()

    def test_policy_selects_subset_of_adapters(self, l1_manager):
        """A policy that targets only some adapters should skip others."""

        class FirstAdapterOnlyPolicy(StorePolicy):
            """Only store to adapter index 0."""

            def select_store_targets(self, keys, adapters):
                for ad in adapters:
                    if ad.index == 0:
                        return {0: list(keys)}
                return {}

            def select_l1_deletions(self, keys):
                return []

        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=FirstAdapterOnlyPolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(
            lambda: adapters[0].debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok, "Object should be stored in first adapter"

        # Give some time and verify second adapter is empty
        time.sleep(0.5)
        assert adapters[1].debug_get_stored_object_count() == 0, (
            "Second adapter should have no objects"
        )

        ctrl.stop()
        for a in adapters:
            a.close()


class TestBufferOnlyMode:
    """Test buffer-only mode: BufferOnlyStorePolicy + NoOpEvictionPolicy."""

    def test_l1_cleaned_after_l2_store(self, l1_manager):
        """L1 data should be deleted after successful L2 store."""
        adapter = make_adapter()

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=BufferOnlyStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for L2 store
        ok = wait_for_condition(
            lambda: adapter.debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok, "Object should be stored in L2"

        # L1 should be cleaned: check key is gone
        ok = wait_for_condition(
            lambda: l1_manager.get_object_state(keys[0]) is None,
            timeout=5.0,
        )
        assert ok, "Key should be deleted from L1 in buffer-only mode"

        ctrl.stop()
        adapter.close()

    def test_noop_eviction_never_evicts(self):
        """NoOpEvictionPolicy should never return eviction actions."""
        noop = NoOpEvictionPolicy()
        noop.on_keys_created([make_object_key(i) for i in range(10)])
        assert noop.get_eviction_actions(1.0) == []

    def test_buffer_only_multiple_keys(self, l1_manager):
        """Multiple keys should all flow to L2 and be removed
        from L1."""
        adapter = make_adapter()

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=BufferOnlyStorePolicy(),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(
            lambda: adapter.debug_get_stored_object_count() == 5,
            timeout=5.0,
        )
        assert ok, "All 5 objects should be stored in L2"

        # All keys should be gone from L1
        ok = wait_for_condition(
            lambda: all(l1_manager.get_object_state(k) is None for k in keys),
            timeout=5.0,
        )
        assert ok, "All keys should be deleted from L1 after buffer-only cleanup"

        ctrl.stop()
        adapter.close()
