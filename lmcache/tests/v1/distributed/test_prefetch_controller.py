# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PrefetchController.

Tests verify the end-to-end prefetch flow: submit request → lookup in L2 →
compute prefix-trimmed load plan → reserve L1 buffers → load from L2 →
transition to read-locked → report prefix hits.

Uses a real L1Manager and MockL2Adapter (with debug methods) to exercise
the full integration without mocking internals.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
    build_trim_mask,
    merge_bitmaps,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
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
    """Poll until a predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_result(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_result until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None:
            return result.count_leading_ones()
        time.sleep(poll_interval)
    return None


def wait_for_lookup_result(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_lookup_result until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_lookup_result(req_id)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def wait_for_prefetch_result_bitmap(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
):
    """Poll query_prefetch_result, returning the raw retained Bitmap.

    Unlike :func:`wait_for_prefetch_result`, this keeps the full bitmap so a
    caller can inspect non-contiguous retained sets (e.g. SEGMENTED_PREFIX).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def make_adapter() -> MockL2Adapter:
    """Create a MockL2Adapter with fast bandwidth."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return MockL2Adapter(config)


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def store_keys_in_l2(
    adapter: MockL2Adapter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> None:
    """Store test data directly in L2 adapter and wait for completion."""
    if not keys:
        return
    objs = []
    for _ in keys:
        tensor = torch.randn(layout.shapes[0], dtype=layout.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        obj = TensorMemoryObj(raw_data=tensor, metadata=metadata, parent_allocator=None)
        objs.append(obj)
    adapter.submit_store_task(keys, objs)  # type: ignore
    ok = wait_for_condition(
        lambda: all(adapter.debug_has_key(k) for k in keys),
        timeout=5.0,
    )
    assert ok, "Failed to store test data in L2 adapter"


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


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestPrefetchControllerLifecycle:
    """Test PrefetchController start/stop behavior."""

    def test_start_stop(self, l1_manager):
        """Controller should start and stop cleanly."""
        adapter = make_adapter()
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        ctrl.stop()
        adapter.close()

    def test_start_stop_no_adapters(self, l1_manager):
        """Controller should start and stop cleanly with no adapters."""
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        ctrl.stop()


# =============================================================================
# Single Adapter Prefetch
# =============================================================================


class TestSingleAdapterPrefetch:
    """Test PrefetchController with one MockL2Adapter."""

    def test_full_prefix_hit(self, l1_manager):
        """All keys in L2 → all loaded, prefix hits = total keys."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]

        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 5, f"Expected 5 prefix hits, got {result}"

        # Verify prefix keys are read-locked in L1
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Cleanup: release read locks
        l1_manager.finish_read(keys)

        ctrl.stop()
        adapter.close()

    def test_prefix_with_gap(self, l1_manager):
        """L2 has keys {0,1,3,4} but not 2 → only prefix {0,1} loaded."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store only keys 0, 1, 3, 4 (gap at index 2)
        stored_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(all_keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2, f"Expected 2 prefix hits (gap at index 2), got {result}"

        # Verify prefix keys {0,1} are read-locked in L1
        prefix_keys = all_keys[:2]
        read_results = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Verify keys beyond prefix {2,3,4} are NOT in L1
        non_prefix_keys = all_keys[2:]
        read_results = l1_manager.reserve_read(non_prefix_keys)
        for key in non_prefix_keys:
            assert read_results[key][0] == L1Error.KEY_NOT_EXIST

        l1_manager.finish_read(prefix_keys)
        ctrl.stop()
        adapter.close()

    def test_segmented_prefix_with_gap(self, l1_manager):
        """SEGMENTED_PREFIX retains the post-gap keys: L2 {0,1,3,4} -> {0,1,3,4}.

        The PREFIX counterpart (test_prefix_with_gap) truncates the same gap to
        {0,1}; SEGMENTED_PREFIX keeps the post-gap chunks L1-resident so only
        the hole (index 2) needs recomputing.
        """
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store only keys 0, 1, 3, 4 (gap at index 2).
        store_keys_in_l2(adapter, [all_keys[i] for i in [0, 1, 3, 4]], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(all_keys, layout, policy=TrimPolicy.SEGMENTED_PREFIX)
        )
        retained = wait_for_prefetch_result_bitmap(ctrl, req_id)
        assert retained is not None
        assert retained.get_indices_list() == [0, 1, 3, 4], (
            "SEGMENTED_PREFIX should retain post-gap keys, got "
            f"{retained.get_indices_list()}"
        )

        # Retained keys {0,1,3,4} are read-locked in L1; the gap (2) is absent.
        retained_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        read = l1_manager.unsafe_read(retained_keys)
        for key in retained_keys:
            assert read[key][0] == L1Error.SUCCESS
        gap_read = l1_manager.reserve_read([all_keys[2]])
        assert gap_read[all_keys[2]][0] == L1Error.KEY_NOT_EXIST

        l1_manager.finish_read(retained_keys)
        ctrl.stop()
        adapter.close()

    def test_fault_inject_load_gap_segmented_vs_prefix(self, l1_manager):
        """fault_inject (load fails at index 2) drives the segmented path.

        Lookup reports all 5 keys present; the *load* of index 2 fails (the L2
        retrieve error the adapter simulates). SEGMENTED_PREFIX retains the
        post-gap keys {0,1,3,4}; PREFIX truncates at the hole to {0,1}. Distinct
        key ranges per policy keep the shared L1 from serving the second pass.
        """
        layout = make_layout()
        for base, trim, expected in (
            (0, TrimPolicy.SEGMENTED_PREFIX, [0, 1, 3, 4]),
            (10, TrimPolicy.PREFIX, [0, 1]),
        ):
            keys = [make_object_key(base + i) for i in range(5)]
            inner = make_adapter()
            store_keys_in_l2(inner, keys, layout)  # all 5 present at lookup
            # Drop task-position 2 at load -> a mid-prefix L2 retrieve failure.
            fault = FaultInjectL2Adapter(inner, rate=0.0, seed=0, gap_indices=(2,))

            ctrl = PrefetchController(
                l1_manager=l1_manager,
                l2_adapters=[fault],
                adapter_descriptors=[make_descriptor(0)],
                policy=DefaultPrefetchPolicy(),
            )
            ctrl.start()
            req_id = ctrl.submit_prefetch_request(
                PrefetchRequestSpec(keys, layout, policy=trim)
            )
            retained = wait_for_prefetch_result_bitmap(ctrl, req_id)
            assert retained is not None
            assert retained.get_indices_list() == expected, (
                f"{trim.name}: expected {expected}, got {retained.get_indices_list()}"
            )

            held = [keys[i] for i in retained.get_indices_list()]
            if held:
                l1_manager.finish_read(held)
            ctrl.stop()
            fault.close()

    def test_key0_missing(self, l1_manager):
        """L2 has keys {1,2,3} but not 0 → prefix = 0, nothing loaded."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(4)]
        # Store keys 1, 2, 3 but not 0
        stored_keys = all_keys[1:]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(all_keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0, f"Expected 0 prefix hits (key 0 missing), got {result}"

        # Verify no keys are in L1
        read_results = l1_manager.reserve_read(all_keys)
        for key in all_keys:
            assert read_results[key][0] == L1Error.KEY_NOT_EXIST

        ctrl.stop()
        adapter.close()


# =============================================================================
# Multi Adapter Prefetch
# =============================================================================


class TestMultiAdapterPrefetch:
    """Test PrefetchController with multiple MockL2Adapters."""

    def test_disjoint_adapters(self, l1_manager):
        """Adapter 0 has {0,1}, adapter 1 has {2,3} → full prefix of 4."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]

        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[2:], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 4, f"Expected 4 prefix hits, got {result}"

        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()

    def test_overlap_first_wins(self, l1_manager):
        """Both adapters have key 1. Adapter 0 (lower index) loads it."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        # Adapter 0 has keys {0, 1}, adapter 1 has keys {1, 2}
        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[1:], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 3, f"Expected 3 prefix hits, got {result}"

        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()


# =============================================================================
# No Hits
# =============================================================================


class TestNoHits:
    """Test PrefetchController when no keys are found in L2."""

    def test_no_keys_in_l2(self, l1_manager):
        """Prefetch keys not in any L2 → 0 prefix hits."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        # Don't store anything in L2

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0, f"Expected 0 prefix hits, got {result}"

        ctrl.stop()
        adapter.close()

    def test_no_adapters(self, l1_manager):
        """No adapters → 0 prefix hits immediately."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0, f"Expected 0 prefix hits, got {result}"

        ctrl.stop()


# =============================================================================
# Query Result
# =============================================================================


class TestQueryResult:
    """Test query_prefetch_result semantics."""

    def test_query_returns_int_then_none(self, l1_manager):
        """Result is consumed on first query; second query returns None."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2
        # Second query should return None (already consumed)
        assert ctrl.query_prefetch_result(req_id) is None

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_query_before_completion_returns_none(self, l1_manager):
        """Querying a nonexistent request ID returns None."""
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        assert ctrl.query_prefetch_result(999) is None

        ctrl.stop()


# =============================================================================
# L2 Lock Release
# =============================================================================


class TestPrefetchL2LockRelease:
    """Test that L2 locks are properly released after prefetch."""

    def test_locks_released_after_full_hit(self, l1_manager):
        """All L2 locks should be released after a successful prefetch."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Wait for L2 unlock operations to be processed
        ok = wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0,
            timeout=5.0,
        )
        assert ok, "L2 locks should be released after prefetch completion"

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_locks_released_after_prefix_trim(self, l1_manager):
        """L2 locks released for both prefix and non-prefix keys."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store keys 0, 1, 3, 4 (gap at index 2)
        stored_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(all_keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2

        # All L2 locks should be released (both prefix and trimmed keys)
        ok = wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0,
            timeout=5.0,
        )
        assert ok, "All L2 locks should be released after prefix-trimmed prefetch"

        l1_manager.finish_read(all_keys[:2])
        ctrl.stop()
        adapter.close()

    def test_locks_released_after_no_hits(self, l1_manager):
        """L2 locks released even when nothing is found."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        # Don't store anything

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 0

        ok = wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0,
            timeout=5.0,
        )
        assert ok, "L2 locks should be 0 when nothing was found"

        ctrl.stop()
        adapter.close()

    def test_multi_adapter_locks_released(self, l1_manager):
        """Both adapters' L2 locks released after overlapping prefetch."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        # Adapter 0 has {0, 1}, adapter 1 has {1, 2}
        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[1:], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Both adapters should have all locks released
        for i, adapter in enumerate(adapters):
            ok = wait_for_condition(
                lambda a=adapter: a.debug_get_locked_key_count() == 0,
                timeout=5.0,
            )
            assert ok, f"Adapter {i} should have all L2 locks released"

        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()


# =============================================================================
# Max In-Flight
# =============================================================================


class TestMaxInFlight:
    """Test PrefetchController max in-flight request limiting."""

    def test_queuing_beyond_max_in_flight(self, l1_manager):
        """Submit more requests than max_in_flight → all eventually complete."""
        adapter = make_adapter()
        layout = make_layout()

        # Store keys for 4 separate requests (2 keys each)
        all_keys = [make_object_key(i) for i in range(8)]
        store_keys_in_l2(adapter, all_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
            max_in_flight=2,
        )
        ctrl.start()

        # Submit 4 requests (max_in_flight=2, so 2 queued)
        req_ids = []
        for i in range(4):
            batch_keys = all_keys[i * 2 : (i + 1) * 2]
            req_id = ctrl.submit_prefetch_request(
                PrefetchRequestSpec(batch_keys, layout)
            )
            req_ids.append(req_id)

        # All 4 requests should eventually complete
        results = []
        for req_id in req_ids:
            result = wait_for_prefetch_result(ctrl, req_id, timeout=10.0)
            assert result is not None, f"Request {req_id} should complete"
            results.append(result)

        assert results == [2, 2, 2, 2]

        # Release read locks for all keys
        l1_manager.finish_read(all_keys)

        ctrl.stop()
        adapter.close()


# =============================================================================
# Multiple Sequential Requests
# =============================================================================


class TestMultipleRequests:
    """Test multiple sequential prefetch requests."""

    def test_two_sequential_requests(self, l1_manager):
        """Two back-to-back requests should both complete correctly."""
        adapter = make_adapter()
        layout = make_layout()
        keys1 = [make_object_key(i) for i in range(3)]
        keys2 = [make_object_key(i) for i in range(10, 14)]

        store_keys_in_l2(adapter, keys1, layout)
        store_keys_in_l2(adapter, keys2, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req1 = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys1, layout))
        result1 = wait_for_prefetch_result(ctrl, req1)
        assert result1 == 3

        req2 = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys2, layout))
        result2 = wait_for_prefetch_result(ctrl, req2)
        assert result2 == 4

        l1_manager.finish_read(keys1)
        l1_manager.finish_read(keys2)
        ctrl.stop()
        adapter.close()


# =============================================================================
# extra_count Path
# =============================================================================


class TestExtraCountPrefetch:
    """Test that extra_count is correctly propagated through the prefetch path.

    When extra_count=N is passed to submit_prefetch_request, the controller
    must acquire 1 + N read locks per key (one for the prefetch controller
    itself, plus N for additional TP workers).  Each consumer must call
    finish_read (or finish_read_prefetched) once to release its lock.

    These tests verify:
    1. Keys remain accessible after the first finish_read when extra_count > 0.
    2. Keys are evictable only after ALL 1 + N locks are released.
    3. extra_count=0 (default) behaves identically to the original single-lock
       path.
    4. Prefix trimming still works correctly with extra_count > 0.
    5. Non-prefix loaded keys have all extra locks released by _finalize_load.
    """

    def test_extra_count_zero_default_behavior(self, l1_manager):
        """extra_count=0 (default): single read lock, key freed after one
        finish_read."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(keys, layout, extra_count=0)
        )
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Keys should be readable immediately after prefetch
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Release the single read lock — keys should become unlocked
        finish_results = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results[key] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_extra_count_one_requires_two_finish_reads(self, l1_manager):
        """extra_count=1: two read locks acquired; key stays locked after
        first finish_read and is released after second."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        # extra_count=1 → 2 read locks per key
        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(keys, layout, extra_count=1)
        )
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Keys must be readable right after prefetch
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Release lock #1 (the "prefetch controller" lock, extra_count=0)
        finish_results = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results[key] == L1Error.SUCCESS

        # Keys must STILL be readable — lock #2 (the TP worker lock) is held
        read_results2 = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results2[key][0] == L1Error.SUCCESS, (
                f"Key {key} should still be read-locked after first finish_read"
            )

        # Release lock #2 (the TP worker lock, extra_count=0)
        finish_results2 = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results2[key] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_extra_count_three_requires_four_finish_reads(self, l1_manager):
        """extra_count=3 (TP=4): four read locks; key stays locked until all
        four are released."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        # extra_count=3 → 4 read locks per key
        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(keys, layout, extra_count=3)
        )
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2

        # Release locks one by one; key must remain readable until the last
        for release_idx in range(3):
            finish_results = l1_manager.finish_read(keys, extra_count=0)
            for key in keys:
                assert finish_results[key] == L1Error.SUCCESS

            # Still readable — remaining locks are held
            read_results = l1_manager.unsafe_read(keys)
            for key in keys:
                assert read_results[key][0] == L1Error.SUCCESS, (
                    f"Key {key} should still be locked after {release_idx + 1} "
                    f"finish_read calls"
                )

        # Release the final lock
        finish_results = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results[key] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_extra_count_with_prefix_trim(self, l1_manager):
        """extra_count=1 with a gap in L2: only prefix keys get 2 locks;
        non-prefix keys are never loaded."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store keys 0, 1, 3, 4 — gap at index 2
        stored_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(all_keys, layout, extra_count=1)
        )
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2, f"Expected 2 prefix hits (gap at index 2), got {result}"

        prefix_keys = all_keys[:2]

        # Prefix keys must be readable
        read_results = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Release lock #1 — prefix keys still held by lock #2
        l1_manager.finish_read(prefix_keys, extra_count=0)

        read_results2 = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results2[key][0] == L1Error.SUCCESS, (
                f"Prefix key {key} should still be locked after first finish_read"
            )

        # Release lock #2
        l1_manager.finish_read(prefix_keys, extra_count=0)

        # Non-prefix keys must NOT be in L1
        non_prefix_keys = all_keys[2:]
        reserve_results = l1_manager.reserve_read(non_prefix_keys)
        for key in non_prefix_keys:
            assert reserve_results[key][0] == L1Error.KEY_NOT_EXIST, (
                f"Non-prefix key {key} should not be in L1"
            )

        ctrl.stop()
        adapter.close()

    def test_extra_count_non_prefix_loaded_keys_fully_released(self, l1_manager):
        """Keys loaded beyond the prefix (due to partial load failure) must
        have ALL extra locks released by _finalize_load so they can be evicted.

        We simulate this by storing keys {0, 1, 2} in L2 but making key 1
        fail to reserve in L1 (by pre-occupying it), creating a gap so that
        key 2 is loaded but lies beyond the prefix.  _finalize_load must
        release 1 + extra_count locks for key 2.
        """
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        # Pre-occupy key[1] in L1 with a write lock so reserve_write fails for it.
        # This forces a gap: key 0 is prefix (hit), key 1 fails reservation
        # (gap), key 2 is loaded but beyond the prefix.
        pre_write_results = l1_manager.reserve_write(
            keys=[keys[1]],
            is_temporary=[False],
            layout_desc=layout,
            mode="new",
        )
        assert pre_write_results[keys[1]][0] == L1Error.SUCCESS

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(keys, layout, extra_count=1)
        )
        result = wait_for_prefetch_result(ctrl, req_id)
        # Only key 0 is in the prefix (key 1 reservation failed → gap)
        assert result == 1, f"Expected 1 prefix hit, got {result}"

        # key[0] should be in L1 with 2 read locks (1 + extra_count=1)
        read_results = l1_manager.unsafe_read([keys[0]])
        assert read_results[keys[0]][0] == L1Error.SUCCESS

        # key[2] was loaded but is beyond the prefix; _finalize_load must have
        # released all 1 + extra_count=2 locks, so it should be gone from L1
        # (it's a temporary object and its lock count should be 0).
        reserve_results = l1_manager.reserve_read([keys[2]])
        assert reserve_results[keys[2]][0] == L1Error.KEY_NOT_EXIST, (
            "Non-prefix loaded key[2] should have all locks released and be "
            "evicted from L1"
        )

        # Clean up: release key[0]'s 2 locks and key[1]'s write lock
        l1_manager.finish_read([keys[0]], extra_count=0)
        l1_manager.finish_read([keys[0]], extra_count=0)
        l1_manager.finish_write([keys[1]])
        l1_manager.delete([keys[1]])

        ctrl.stop()
        adapter.close()


# =============================================================================
# Query Lookup Result
# =============================================================================


class TestQueryLookupResult:
    """Test query_lookup_result semantics."""

    def test_lookup_result_available_before_prefetch_completes(self, l1_manager):
        """Lookup result is available while load is still in progress."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))

        # Lookup result should be available before or at the same time as
        # the full prefetch result.
        lookup_hits = wait_for_lookup_result(ctrl, req_id)
        assert lookup_hits is not None
        assert lookup_hits == 3

        # Wait for full prefetch to complete and clean up
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_lookup_result_with_prefix_gap(self, l1_manager):
        """Lookup result reflects prefix-only hits (gap breaks prefix)."""
        adapter = make_adapter()
        layout = make_layout()
        # Store keys 0 and 2 (gap at 1)
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, [keys[0], keys[2]], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        lookup_hits = wait_for_lookup_result(ctrl, req_id)
        assert lookup_hits is not None
        # Only key 0 is in the prefix (gap at key 1 breaks it)
        assert lookup_hits == 1

        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 1

        l1_manager.finish_read([keys[0]])
        ctrl.stop()
        adapter.close()

    def test_lookup_result_zero_hits(self, l1_manager):
        """Lookup result is 0 when no keys are found in L2."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        # Don't store anything in L2

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        lookup_hits = wait_for_lookup_result(ctrl, req_id)
        assert lookup_hits is not None
        assert lookup_hits == 0

        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 0

        ctrl.stop()
        adapter.close()

    def test_lookup_result_not_popped_by_query(self, l1_manager):
        """query_lookup_result does not consume the result (idempotent reads)."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        lookup_hits = wait_for_lookup_result(ctrl, req_id)
        assert lookup_hits == 2

        # Second query should still return the same value (not consumed)
        assert ctrl.query_lookup_result(req_id) == 2

        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_lookup_result_cleaned_up_by_prefetch_result(self, l1_manager):
        """query_prefetch_result cleans up the lookup result entry."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        lookup_hits = wait_for_lookup_result(ctrl, req_id)
        assert lookup_hits == 2

        # Consume the prefetch result (should also clean up lookup entry)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2

        # Lookup result should now be gone
        assert ctrl.query_lookup_result(req_id) is None

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_lookup_result_nonexistent_request(self, l1_manager):
        """Querying a nonexistent request ID returns None."""
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        assert ctrl.query_lookup_result(999) is None

        ctrl.stop()


class TestBuildTrimMask:
    """build_trim_mask picks the retained subset per policy: PREFIX trims at
    the first gap; SEGMENTED_PREFIX and SPARSE keep every set bit (gaps and
    all). The retained bitmap is consumed unchanged at the controller's load
    sites, so testing the mask directly covers the policy semantics."""

    @staticmethod
    def _bm(n, idxs):
        bm = Bitmap(n)
        for i in idxs:
            bm.set(i)
        return bm

    def test_prefix_trims_at_first_gap(self):
        found = self._bm(5, [0, 1, 3, 4])  # gap at index 2
        assert build_trim_mask(found, 5, TrimPolicy.PREFIX).get_indices_list() == [
            0,
            1,
        ]

    def test_segmented_prefix_keeps_gaps(self):
        # Models an L2 hit whose L1 load failed mid-prefix (e.g. OOM at index
        # 2): the keys that did load are kept, not trimmed to the first gap.
        found = self._bm(5, [0, 1, 3, 4])
        assert build_trim_mask(
            found, 5, TrimPolicy.SEGMENTED_PREFIX
        ).get_indices_list() == [0, 1, 3, 4]

    def test_sparse_keeps_all_found(self):
        found = self._bm(5, [0, 2, 4])
        assert build_trim_mask(found, 5, TrimPolicy.SPARSE).get_indices_list() == [
            0,
            2,
            4,
        ]


class TestMergeBitmaps:
    """merge_bitmaps always returns a num_keys-sized bitmap."""

    def test_empty_input_returns_sized_bitmap(self):
        """Empty input -> num_keys-sized all-zeros bitmap (not Bitmap(0)), so a
        downstream ``&`` with a same-sized mask never hits a size mismatch."""
        merged = merge_bitmaps([], 5)
        assert merged.popcount() == 0
        mask = Bitmap(5)
        mask.set(2)
        assert (merged & mask).popcount() == 0  # would raise on size mismatch

    def test_empty_generator_returns_sized_bitmap(self):
        """A generator is truthy even when empty; the result is still size-5."""
        merged = merge_bitmaps((b for b in []), 5)
        assert merged.popcount() == 0
        assert (merged & Bitmap(5)).popcount() == 0

    def test_union_of_bitmaps(self):
        """Non-empty inputs are OR-merged into one num_keys-sized bitmap."""
        a, b = Bitmap(5), Bitmap(5)
        a.set(0)
        b.set(3)
        assert merge_bitmaps([a, b], 5).get_indices_list() == [0, 3]


class TestWaitPrefetchResult:
    """Test the blocking wait_prefetch_result interface."""

    def test_wait_blocks_until_result_ready(self, l1_manager):
        """wait_prefetch_result blocks until the background result is published,
        returns True, and does not consume the result."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        # Blocks until the background thread publishes the result.
        assert ctrl.wait_prefetch_result(req_id, timeout=10.0) is True
        # wait_prefetch_result must not consume the result.
        result = ctrl.query_prefetch_result(req_id)
        assert result is not None
        assert result.count_leading_ones() == 5

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_wait_times_out_for_unknown_request(self, l1_manager):
        """wait_prefetch_result returns False, after genuinely waiting, when no
        result arrives within the timeout."""
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        start = time.monotonic()
        assert ctrl.wait_prefetch_result(999999, timeout=0.2) is False
        assert time.monotonic() - start >= 0.2

        ctrl.stop()


# =============================================================================
# Retention Policy
# =============================================================================


class TestPrefetchMode:
    """``mode=WARM`` (the warm path) loads keys **permanent** and
    **without a read lock**, vs ``LOOKUP`` which read-locks temporary
    objects that vanish on release.

    Both tests use ``DefaultPrefetchPolicy`` so the only difference is the
    per-request ``mode`` argument.
    """

    def test_warm_loads_unlocked_and_permanent(self, l1_manager):
        """WARM loads keys permanent and with NO read lock: immediately ready
        (reserve_read SUCCEEDS), holding no lock (unsafe_read NOT_READABLE), and
        not deleted on a reserve_read/finish_read cycle (permanent, not temp)."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(keys, layout, mode=PrefetchMode.WARM)
        )
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # No warming lock: unsafe_read (which needs an active read lock) reports
        # NOT_READABLE even though the keys are present and ready.
        unsafe = l1_manager.unsafe_read(keys)
        for key in keys:
            assert unsafe[key][0] == L1Error.KEY_NOT_READABLE

        # They are ready and re-lookupable: reserve_read SUCCEEDS...
        read_results = l1_manager.reserve_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # ...and releasing that probe lock does NOT delete them (permanent).
        l1_manager.finish_read(keys)
        again = l1_manager.reserve_read(keys)
        for key in keys:
            assert again[key][0] == L1Error.SUCCESS

        l1_manager.finish_read(keys)
        l1_manager.delete(keys)
        ctrl.stop()
        adapter.close()

    def test_warm_sparse_skips_write_locked_prefix_and_loads_later_keys(
        self, l1_manager
    ):
        """SPARSE WARM does not let a write-locked earlier key suppress later
        keys that can be reserved and loaded."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        existing = l1_manager.reserve_write(
            [keys[0]], is_temporary=[False], layout_desc=layout, mode="new"
        )
        assert existing[keys[0]][0] == L1Error.SUCCESS

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            PrefetchRequestSpec(
                keys,
                layout,
                policy=TrimPolicy.SPARSE,
                mode=PrefetchMode.WARM,
            )
        )
        result = wait_for_prefetch_result_bitmap(ctrl, req_id)
        assert result is not None
        assert result.get_indices_list() == [1, 2]

        l1_manager.finish_write([keys[0]])
        read_results = l1_manager.reserve_read(keys[1:])
        for key in keys[1:]:
            assert read_results[key][0] == L1Error.SUCCESS

        l1_manager.finish_read(keys[1:])
        l1_manager.delete(keys)
        ctrl.stop()
        adapter.close()

    def test_default_deletes_keys_after_finish_read(self, l1_manager):
        """LOOKUP defers to ``DefaultPrefetchPolicy`` (temporary), so
        the keys are deleted from L1 once the read-lock is released."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Releasing the read-lock deletes the temporary objects, so a
        # subsequent lookup (reserve_read) misses with KEY_NOT_EXIST.
        l1_manager.finish_read(keys)
        read_results = l1_manager.reserve_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.KEY_NOT_EXIST

        ctrl.stop()
        adapter.close()
