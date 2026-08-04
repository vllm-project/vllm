# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for multi-model support in the StoreController.

A single ``finish_write`` with mixed-model keys must result in per-model
``submit_store_task`` calls so each submission sees uniform ``(shape, dtype)``.
"""

# Standard
from unittest.mock import patch
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
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int, model_name: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def wait_for_condition(predicate, timeout: float = 10.0, poll_interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def make_storage_manager(l1_size_mb: int = 256) -> StorageManager:
    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=l1_size_mb * 1024 * 1024,
                use_lazy=True,
                init_size_in_bytes=min(l1_size_mb, 64) * 1024 * 1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=10.0)],
        ),
    )
    return StorageManager(cfg)


# =============================================================================
# Tests
# =============================================================================


class TestStoreControllerMultimodel:
    """Mixed-model keys in one ``finish_write`` must be split per model
    before reaching ``submit_store_task``."""

    def test_each_submit_store_task_has_uniform_shape(self):
        layout_a = MemoryLayoutDesc(
            shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16]
        )
        layout_b = MemoryLayoutDesc(
            shapes=[torch.Size([50, 2, 256])], dtypes=[torch.bfloat16]
        )
        keys_a = [make_object_key(i, "model_a") for i in range(3)]
        keys_b = [make_object_key(100 + i, "model_b") for i in range(3)]

        # Record the (shape, dtype) set observed per submit_store_task call.
        submit_shape_groups = []
        original_submit = MockL2Adapter.submit_store_task

        def recording_submit(self, keys, objects):
            shape_set = frozenset(
                (tuple(obj.get_shapes()), tuple(obj.get_dtypes())) for obj in objects
            )
            submit_shape_groups.append(shape_set)
            return original_submit(self, keys, objects)

        with patch.object(MockL2Adapter, "submit_store_task", recording_submit):
            sm = make_storage_manager()
            adapter = sm._l2_adapters[0]

            ret_a = sm.reserve_write(keys_a, layout_a, mode="new")
            ret_b = sm.reserve_write(keys_b, layout_b, mode="new")
            for i, k in enumerate(keys_a):
                ret_a[k].tensor.fill_(float(i + 1))
            for i, k in enumerate(keys_b):
                ret_b[k].tensor.fill_(float(100 + i))

            # Single finish_write forces a mixed-model batch in the listener.
            sm.finish_write(keys_a + keys_b)

            # Wait until every key has landed in L2 — avoids a race on the
            # in_flight counter (which is briefly 0 between pop and submit).
            ok = wait_for_condition(
                lambda: all(
                    adapter.debug_has_key(k)  # type: ignore[attr-defined]
                    for k in keys_a + keys_b
                ),
                timeout=10.0,
            )
            assert ok, "Not all keys were stored in L2 within timeout"

            assert submit_shape_groups, "No submit_store_task calls were recorded"
            for i, shapes in enumerate(submit_shape_groups):
                assert len(shapes) == 1, (
                    f"submit_store_task #{i} received keys with mixed "
                    f"(shape, dtype): {shapes}. Mixed-model batches must be "
                    f"grouped by model before reaching the adapter."
                )

            distinct_shapes = {s for group in submit_shape_groups for s in group}
            assert len(distinct_shapes) == 2, (
                f"Expected both models' shapes to appear across submits, "
                f"got {len(distinct_shapes)}: {distinct_shapes}"
            )

            sm.close()
