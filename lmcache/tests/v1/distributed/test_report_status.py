# SPDX-License-Identifier: Apache-2.0
"""
Tests for the report_status() interface across the StorageManager subtree.
"""

# Standard

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
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2AdapterConfig,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

try:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
except ImportError:
    pytest.skip(
        "Skipping because StorageManager cannot be imported", allow_module_level=True
    )

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_memory_config():
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,
        align_bytes=0x1000,
    )


@pytest.fixture
def basic_l1_config(basic_memory_config):
    return L1ManagerConfig(
        memory_config=basic_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def basic_layout():
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def storage_manager_no_l2(basic_l1_config):
    """StorageManager without L2 adapters."""
    config = StorageManagerConfig(
        l1_manager_config=basic_l1_config,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    sm = StorageManager(config)
    yield sm
    sm.close()


@pytest.fixture
def storage_manager_with_l2(basic_l1_config):
    """StorageManager with one mock L2 adapter."""
    config = StorageManagerConfig(
        l1_manager_config=basic_l1_config,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[
                MockL2AdapterConfig(
                    max_size_gb=0.1,
                    mock_bandwidth_gb=10.0,
                ),
            ]
        ),
    )
    sm = StorageManager(config)
    yield sm
    sm.close()


def make_object_key(chunk_hash: int, model_name: str = "test_model", kv_rank: int = 0):
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model_name, kv_rank=kv_rank)


# =============================================================================
# Tests
# =============================================================================


class TestStorageManagerReportStatus:
    """Tests for StorageManager.report_status() shape and values."""

    def test_report_status_shape_no_l2(self, storage_manager_no_l2):
        status = storage_manager_no_l2.report_status()

        # Top-level keys
        assert "is_healthy" in status
        assert "l1_manager" in status
        assert "store_controller" in status
        assert "prefetch_controller" in status
        assert "l1_eviction_controller" in status
        assert "l2_adapters" in status
        assert "num_l2_adapters" in status

        assert status["is_healthy"] is True
        assert status["num_l2_adapters"] == 0
        assert status["l2_adapters"] == []

    def test_report_status_shape_with_l2(self, storage_manager_with_l2):
        status = storage_manager_with_l2.report_status()

        assert status["is_healthy"] is True
        assert status["num_l2_adapters"] == 1
        assert len(status["l2_adapters"]) == 1

        adapter_status = status["l2_adapters"][0]
        assert adapter_status["is_healthy"] is True
        assert adapter_status["type"] == "MockL2Adapter"
        assert "stored_object_count" in adapter_status
        assert "max_capacity_bytes" in adapter_status

    def test_l1_manager_status_shape(self, storage_manager_no_l2):
        l1 = storage_manager_no_l2.report_status()["l1_manager"]

        assert l1["is_healthy"] is True
        assert l1["total_object_count"] == 0
        assert l1["write_locked_count"] == 0
        assert l1["read_locked_count"] == 0
        assert l1["temporary_count"] == 0
        assert l1["memory_used_bytes"] == 0
        assert l1["memory_total_bytes"] > 0
        assert l1["memory_usage_ratio"] == 0.0
        assert "write_ttl_seconds" in l1
        assert "read_ttl_seconds" in l1

    def test_store_controller_status_shape(self, storage_manager_no_l2):
        sc = storage_manager_no_l2.report_status()["store_controller"]

        assert sc["is_healthy"] is True
        assert sc["thread_alive"] is True
        assert "pending_keys_count" in sc
        assert "in_flight_task_count" in sc
        assert "num_l2_adapters" in sc

    def test_prefetch_controller_status_shape(self, storage_manager_no_l2):
        pc = storage_manager_no_l2.report_status()["prefetch_controller"]

        assert pc["is_healthy"] is True
        assert pc["thread_alive"] is True
        assert "max_in_flight" in pc
        assert "submission_queue_size" in pc
        assert "pending_queue_size" in pc
        assert "in_flight_request_count" in pc
        assert "lookup_phase_count" in pc
        assert "load_phase_count" in pc
        assert "completed_results_count" in pc
        assert "num_l2_adapters" in pc

    def test_eviction_controller_status_shape(self, storage_manager_no_l2):
        ec = storage_manager_no_l2.report_status()["l1_eviction_controller"]

        assert ec["is_healthy"] is True
        assert ec["thread_alive"] is True
        assert "eviction_policy" in ec
        assert "trigger_watermark" in ec
        assert "eviction_ratio" in ec

    def test_l1_status_reflects_writes(self, storage_manager_no_l2, basic_layout):
        """After writing objects, L1 status should reflect them."""
        keys = [make_object_key(i) for i in range(3)]
        reserved = storage_manager_no_l2.reserve_write(keys, basic_layout, "new")
        assert len(reserved) == 3

        l1 = storage_manager_no_l2.report_status()["l1_manager"]
        assert l1["total_object_count"] == 3
        assert l1["write_locked_count"] == 3
        assert l1["memory_used_bytes"] > 0

        # Finish writes
        storage_manager_no_l2.finish_write(keys)
        l1 = storage_manager_no_l2.report_status()["l1_manager"]
        assert l1["write_locked_count"] == 0

    def test_health_propagation(self, storage_manager_no_l2):
        """Top-level is_healthy should be True when all children are healthy."""
        status = storage_manager_no_l2.report_status()
        assert status["is_healthy"] is True

        # Verify all child components are healthy
        assert status["l1_manager"]["is_healthy"] is True
        assert status["store_controller"]["is_healthy"] is True
        assert status["prefetch_controller"]["is_healthy"] is True
        assert status["l1_eviction_controller"]["is_healthy"] is True
