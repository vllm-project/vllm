# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the four L1/L2 state metrics.

Drives real controllers through real state changes and reads back values
via a process-shared ``InMemoryMetricReader``.  Assertions are on deltas
so prior tests that exercised the meters don't poison the baseline.
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
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

# Importing this sets the process-wide MeterProvider with an
# InMemoryMetricReader; controllers bind their instruments to it.
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

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


def make_adapter(bandwidth_gb: float = 10.0) -> MockL2Adapter:
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=bandwidth_gb)
    return MockL2Adapter(config)


def make_descriptor(index: int) -> AdapterDescriptor:
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def wait_for_condition(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def write_keys_to_l1(
    l1_manager: L1Manager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> list[ObjectKey]:
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


def store_keys_in_l2(
    adapter: MockL2Adapter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> None:
    """Push test data into L2 directly so subsequent prefetches hit."""
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
    adapter.submit_store_task(keys, objs)  # type: ignore[arg-type]
    assert wait_for_condition(
        lambda: all(adapter.debug_has_key(k) for k in keys),
    ), "Failed to seed L2 with test data"


def _read_points_by_attrs() -> dict[str, dict[tuple, float]]:
    """``{metric: {sorted_attrs_tuple: value}}`` from the shared reader."""
    data = _reader.get_metrics_data()
    result: dict[str, dict[tuple, float]] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    key = tuple(sorted(dict(dp.attributes).items()))
                    result.setdefault(metric.name, {})[key] = dp.value
    return result


def _value_for(metric: str, attrs: dict | None = None) -> float:
    """Look up the current value of ``metric`` for the exact attribute set."""
    snapshot = _read_points_by_attrs().get(metric, {})
    key = tuple(sorted((attrs or {}).items()))
    return snapshot.get(key, 0)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=True,
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
# L1 memory usage gauge
# =============================================================================


class TestL1MemoryUsageGauge:
    """``lmcache_mp.l1_memory_usage_bytes`` reports current L1 usage."""

    def test_gauge_reports_zero_initially(self, l1_manager):
        # A fresh L1 with no writes should report 0 used bytes.
        before = _value_for("lmcache_mp.l1_memory_usage_bytes")
        assert before == 0

    def test_gauge_grows_after_writes(self, l1_manager):
        before = _value_for("lmcache_mp.l1_memory_usage_bytes")

        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        write_keys_to_l1(l1_manager, keys, layout)

        after = _value_for("lmcache_mp.l1_memory_usage_bytes")
        assert after > before, "Gauge should reflect bytes written to L1"


# =============================================================================
# StoreController in-flight counter
# =============================================================================


class TestNumInflightL2Stores:
    """``lmcache_mp.num_inflight_l2_stores`` per (l2_name, adapter_index)."""

    def test_counter_balances_to_zero_after_store_cycle(self, l1_manager):
        adapter = make_adapter()
        adapter_index = 0
        attrs = {"l2_name": "mock", "adapter_index": adapter_index}

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(adapter_index)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        before = _value_for("lmcache_mp.num_inflight_l2_stores", attrs)

        layout = make_layout()
        keys = [make_object_key(100 + i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Every store must complete and decrement back to the baseline.
        assert wait_for_condition(lambda: adapter.debug_get_stored_object_count() == 2)
        assert wait_for_condition(
            lambda: _value_for("lmcache_mp.num_inflight_l2_stores", attrs) == before
        )

        ctrl.stop()
        adapter.close()

    def test_counter_uses_per_adapter_attribution(self, l1_manager):
        # Two adapters of the same backend type get distinct datapoints
        # via adapter_index; their counters return to zero independently.
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(0), make_descriptor(1)]
        attrs0 = {"l2_name": "mock", "adapter_index": 0}
        attrs1 = {"l2_name": "mock", "adapter_index": 1}

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,  # type: ignore[arg-type]
            adapter_descriptors=descriptors,
            policy=DefaultStorePolicy(),
        )
        ctrl.start()

        before0 = _value_for("lmcache_mp.num_inflight_l2_stores", attrs0)
        before1 = _value_for("lmcache_mp.num_inflight_l2_stores", attrs1)

        layout = make_layout()
        keys = [make_object_key(200 + i) for i in range(3)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Both adapters should have stored every key (DefaultStorePolicy
        # writes to all) and counters should return to baseline.
        assert wait_for_condition(
            lambda: adapters[0].debug_get_stored_object_count() == 3
            and adapters[1].debug_get_stored_object_count() == 3
        )
        assert wait_for_condition(
            lambda: (
                _value_for("lmcache_mp.num_inflight_l2_stores", attrs0) == before0
                and _value_for("lmcache_mp.num_inflight_l2_stores", attrs1) == before1
            )
        )

        ctrl.stop()
        for a in adapters:
            a.close()


# =============================================================================
# PrefetchController in-flight counters
# =============================================================================


class TestNumInflightL2Loads:
    """``num_inflight_l2_loads`` and ``inflight_load_memory_usage_bytes``
    return to baseline after a successful prefetch cycle."""

    def test_load_counters_balance_to_zero(self, l1_manager):
        adapter = make_adapter()
        adapter_index = 0
        attrs = {"l2_name": "mock", "adapter_index": adapter_index}
        layout = make_layout()
        keys = [make_object_key(300 + i) for i in range(4)]

        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(adapter_index)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        before_loads = _value_for("lmcache_mp.num_inflight_l2_loads", attrs)
        before_bytes = _value_for("lmcache_mp.inflight_load_memory_usage_bytes", attrs)

        req_id = ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))

        # Wait for the request to fully resolve, then the counters should
        # come back to where they started.
        assert wait_for_condition(
            lambda: ctrl.query_prefetch_result(req_id) is not None
        )
        assert wait_for_condition(
            lambda: (
                _value_for("lmcache_mp.num_inflight_l2_loads", attrs) == before_loads
                and _value_for("lmcache_mp.inflight_load_memory_usage_bytes", attrs)
                == before_bytes
            )
        )

        # Release any read locks so teardown is clean.
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_no_leak_after_shutdown_with_inflight_loads(self, l1_manager):
        # Slow adapter so the load is still in flight when stop() runs;
        # the gauge callback reads live state, so once cleanup clears the
        # in-flight dict the gauges naturally report 0.  We assert via
        # the metric that the load is observably in-flight before calling
        # stop(), otherwise the test could pass via normal completion.
        slow_gb = 0.001  # 1 MB/s — a 200 KB key takes ~200 ms.
        adapter = make_adapter(bandwidth_gb=slow_gb)
        adapter_index = 0
        attrs = {"l2_name": "mock", "adapter_index": adapter_index}
        layout = make_layout()
        keys = [make_object_key(400 + i) for i in range(4)]

        # Seed L2 (also slow at this bandwidth, so allow plenty of time).
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(adapter_index)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        before_loads = _value_for("lmcache_mp.num_inflight_l2_loads", attrs)
        before_bytes = _value_for("lmcache_mp.inflight_load_memory_usage_bytes", attrs)

        ctrl.submit_prefetch_request(PrefetchRequestSpec(keys, layout))

        # Wait until the load is actually in flight on this adapter; only
        # then is ``_cleanup_in_flight_requests`` the path that brings the
        # counters back down.  Without this wait, the test could pass via
        # the normal completion path instead.
        assert wait_for_condition(
            lambda: (
                _value_for("lmcache_mp.num_inflight_l2_loads", attrs) > before_loads
            ),
            timeout=10.0,
        ), "Load never entered the in-flight state"
        assert (
            _value_for("lmcache_mp.inflight_load_memory_usage_bytes", attrs)
            > before_bytes
        )

        # Stop while the load is still in flight; cleanup must decrement.
        ctrl.stop()

        assert _value_for("lmcache_mp.num_inflight_l2_loads", attrs) == before_loads
        assert (
            _value_for("lmcache_mp.inflight_load_memory_usage_bytes", attrs)
            == before_bytes
        )

        adapter.close()
