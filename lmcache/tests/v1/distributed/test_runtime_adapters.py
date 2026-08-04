# SPDX-License-Identifier: Apache-2.0
"""
Tests for runtime add/remove of L2 adapters.

Covers both the controller-level primitives (StoreController /
PrefetchController add_adapter / request_remove_adapter with graceful
drain) and the StorageManager orchestration (add_l2_adapter /
delete_l2_adapter), including stable-id semantics and the
active/draining observability surface.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)
from lmcache.v1.distributed.storage_manager import StorageManager
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
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(predicate, timeout: float = 5.0, poll: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def make_mock_config() -> MockL2AdapterConfig:
    return MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)


def make_adapter() -> MockL2Adapter:
    return MockL2Adapter(make_mock_config())


def make_descriptor(index: int) -> AdapterDescriptor:
    return AdapterDescriptor(index=index, config=make_mock_config())


def adapter_by_id(sm: StorageManager, adapter_id: int) -> MockL2Adapter:
    """Fetch an active adapter by stable id via the public ``l2_adapters()``
    API (no private-member access)."""
    for desc, adapter in sm.l2_adapters():
        if desc.index == adapter_id:
            return adapter  # type: ignore[return-value]
    raise AssertionError(f"no active L2 adapter with id {adapter_id}")


def active_adapter_ids(sm: StorageManager) -> set[int]:
    """Set of active adapter ids exposed by the public ``l2_adapters()`` API."""
    return {desc.index for desc, _ in sm.l2_adapters()}


def write_keys_to_l1(
    l1_manager: L1Manager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> None:
    results = l1_manager.reserve_write(
        keys=keys,
        is_temporary=[False] * len(keys),
        layout_desc=layout,
        mode="new",
    )
    written = [k for k, (e, m) in results.items() if m is not None]
    if written:
        l1_manager.finish_write(written)


@pytest.fixture
def l1_manager():
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


@pytest.fixture
def empty_storage_manager_config():
    """A StorageManagerConfig with no L2 adapters configured."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[]),
    )


# =============================================================================
# StoreController-level tests
# =============================================================================


class TestStoreControllerRuntimeAdapters:
    def test_add_adapter_routes_new_stores(self, l1_manager):
        """A key written after add_adapter is stored to the new adapter."""
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            adapter = make_adapter()
            ctrl.add_adapter(0, adapter, make_descriptor(0))

            assert ctrl.report_status()["num_active_adapters"] == 1
            assert ctrl.report_status()["num_draining_adapters"] == 0

            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            write_keys_to_l1(l1_manager, keys, layout)

            ok = wait_for_condition(
                lambda: adapter.debug_get_stored_object_count() == 3
            )
            assert ok, "Keys should be stored to the runtime-added adapter"
        finally:
            ctrl.stop()
            adapter.close()

    def test_remove_adapter_drains_and_detaches(self, l1_manager):
        """request_remove_adapter drains in-flight work then detaches."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            write_keys_to_l1(l1_manager, keys, layout)
            assert wait_for_condition(
                lambda: adapter.debug_get_stored_object_count() == 3
            )

            done = ctrl.request_remove_adapter(0)
            assert done.wait(timeout=5.0), "Adapter should drain and detach"

            status = ctrl.report_status()
            assert status["num_l2_adapters"] == 0
            assert status["num_active_adapters"] == 0
            assert status["num_draining_adapters"] == 0
        finally:
            ctrl.stop()
            adapter.close()

    def test_remove_stops_new_routing(self, l1_manager):
        """After removal, new writes are not routed to the removed adapter."""
        keep = make_adapter()
        drop = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[keep, drop],
            adapter_descriptors=[make_descriptor(0), make_descriptor(1)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            done = ctrl.request_remove_adapter(1)
            assert done.wait(timeout=5.0)

            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            write_keys_to_l1(l1_manager, keys, layout)

            assert wait_for_condition(lambda: keep.debug_get_stored_object_count() == 3)
            # The dropped adapter must not receive the new keys.
            time.sleep(0.2)
            assert drop.debug_get_stored_object_count() == 0
        finally:
            ctrl.stop()
            keep.close()
            drop.close()

    def test_double_remove_is_safe(self, l1_manager):
        """Removing an already-removed adapter signals immediately."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
            # Second removal of the same (now-detached) id must not hang.
            assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
        finally:
            ctrl.stop()
            adapter.close()


# =============================================================================
# StorageManager-level tests
# =============================================================================


class TestStorageManagerRuntimeAdapters:
    def test_add_to_empty_manager(self, empty_storage_manager_config):
        """add_l2_adapter wires a new adapter into all controllers."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            assert sm.report_status()["num_l2_adapters"] == 0

            adapter_id = sm.add_l2_adapter(make_mock_config())
            assert adapter_id == 0

            status = sm.report_status()
            assert status["num_l2_adapters"] == 1
            assert status["store_controller"]["num_active_adapters"] == 1
            assert status["prefetch_controller"]["num_active_adapters"] == 1
        finally:
            sm.close()

    def test_add_then_store(self, empty_storage_manager_config):
        """Keys written after add_l2_adapter land in the new adapter."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            adapter_id = sm.add_l2_adapter(make_mock_config())
            adapter = adapter_by_id(sm, adapter_id)

            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            ret = sm.reserve_write(keys, layout, mode="new")
            sm.finish_write(list(ret.keys()))

            assert wait_for_condition(
                lambda: all(adapter.debug_has_key(k) for k in keys),  # type: ignore
                timeout=10.0,
            )
        finally:
            sm.close()

    def test_delete_detaches_everywhere(self, empty_storage_manager_config):
        """delete_l2_adapter drains and removes the adapter from all controllers."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            adapter_id = sm.add_l2_adapter(make_mock_config())
            assert sm.report_status()["num_l2_adapters"] == 1

            sm.delete_l2_adapter(adapter_id, timeout=10.0)

            status = sm.report_status()
            assert status["num_l2_adapters"] == 0
            assert status["store_controller"]["num_l2_adapters"] == 0
            assert status["prefetch_controller"]["num_l2_adapters"] == 0
        finally:
            sm.close()

    def test_delete_unknown_id_raises(self, empty_storage_manager_config):
        sm = StorageManager(empty_storage_manager_config)
        try:
            with pytest.raises(ValueError):
                sm.delete_l2_adapter(999)
        finally:
            sm.close()

    def test_stable_ids_across_add_delete(self, empty_storage_manager_config):
        """Ids are monotonic and never reused; delete does not shift others."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            id0 = sm.add_l2_adapter(make_mock_config())
            id1 = sm.add_l2_adapter(make_mock_config())
            assert (id0, id1) == (0, 1)

            sm.delete_l2_adapter(id0, timeout=10.0)

            id2 = sm.add_l2_adapter(make_mock_config())
            assert id2 == 2, "Ids must not be reused after a delete"

            # The surviving adapter keeps its original id.
            assert active_adapter_ids(sm) == {1, 2}
            assert sm.report_status()["num_l2_adapters"] == 2
        finally:
            sm.close()

    def test_add_after_delete_routes_to_new_adapter(self, empty_storage_manager_config):
        """A fresh adapter added after a delete receives new stores."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            first = sm.add_l2_adapter(make_mock_config())
            sm.delete_l2_adapter(first, timeout=10.0)
            second = sm.add_l2_adapter(make_mock_config())
            adapter = adapter_by_id(sm, second)

            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            ret = sm.reserve_write(keys, layout, mode="new")
            sm.finish_write(list(ret.keys()))

            assert wait_for_condition(
                lambda: all(adapter.debug_has_key(k) for k in keys),  # type: ignore
                timeout=10.0,
            )
        finally:
            sm.close()

    def test_l2_adapters_reflects_runtime_changes(self, empty_storage_manager_config):
        """l2_adapters() tracks runtime add/delete in stable-id order."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            assert sm.l2_adapters() == []

            id0 = sm.add_l2_adapter(make_mock_config())
            id1 = sm.add_l2_adapter(make_mock_config())
            pairs = sm.l2_adapters()
            assert [d.index for d, _ in pairs] == [id0, id1]
            a0, a1 = pairs[0][1], pairs[1][1]
            assert a0 is not a1

            # Deleting one leaves the survivor (same instance); order is by id.
            sm.delete_l2_adapter(id0, timeout=10.0)
            pairs = sm.l2_adapters()
            assert len(pairs) == 1
            assert pairs[0][0].index == id1
            assert pairs[0][1] is a1
        finally:
            sm.close()
