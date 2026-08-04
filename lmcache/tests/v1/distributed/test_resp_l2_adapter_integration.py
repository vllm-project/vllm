# SPDX-License-Identifier: Apache-2.0
"""
Integration test for the RESP L2 adapter in MP mode.

Requires a running Redis server. Skipped if Redis or the C++ extension
is unavailable.
"""

# Standard
import os
import select
import subprocess

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6399"))


def _redis_available() -> bool:
    """Check if Redis is reachable."""
    try:
        result = subprocess.run(
            ["redis-cli", "-h", REDIS_HOST, "-p", str(REDIS_PORT), "ping"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() == "PONG"
    except Exception:
        return False


def _native_client_available() -> bool:
    """Check if the C++ Redis extension can be imported."""
    try:
        # First Party
        from lmcache.lmcache_redis import LMCacheRedisClient  # noqa: F401

        return True
    except ImportError:
        return False


requires_redis = pytest.mark.skipif(
    not _redis_available(),
    reason=f"Redis not available at {REDIS_HOST}:{REDIS_PORT}",
)
requires_native = pytest.mark.skipif(
    not _native_client_available(),
    reason="C++ Redis extension (lmcache_redis) not available",
)


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
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


def wait_for_event_fd(event_fd: int, timeout: float = 10.0) -> bool:
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


def flush_redis():
    """Flush the test Redis database."""
    subprocess.run(
        ["redis-cli", "-h", REDIS_HOST, "-p", str(REDIS_PORT), "FLUSHDB"],
        capture_output=True,
        timeout=3,
    )


@requires_redis
@requires_native
class TestRESPL2AdapterIntegration:
    """Integration tests using real Redis + real C++ RESP connector."""

    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        flush_redis()

        # First Party
        from lmcache.lmcache_redis import LMCacheRedisClient
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            NativeConnectorL2Adapter,
        )

        native_client = LMCacheRedisClient(REDIS_HOST, REDIS_PORT, 4)
        self.adapter = NativeConnectorL2Adapter(native_client)
        yield
        self.adapter.close()
        flush_redis()

    def test_event_fds_are_distinct(self):
        fds = {
            self.adapter.get_store_event_fd(),
            self.adapter.get_lookup_and_lock_event_fd(),
            self.adapter.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_store_and_lookup(self):
        """Store objects to Redis, then verify lookup finds them."""
        keys = [create_object_key(i) for i in range(5)]
        objs = [create_memory_obj(size=64, fill_value=float(i)) for i in range(5)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd)
        completed = self.adapter.pop_completed_store_tasks()
        assert completed[store_tid].is_successful()

        # Lookup all — should find everything
        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap is not None
        for i in range(5):
            assert bitmap.test(i) is True, f"Key {i} not found in lookup"

        # Unlock
        self.adapter.submit_unlock(keys)

    def test_lookup_nonexistent_keys(self):
        """Lookup for keys not in Redis should return all zeros."""
        keys = [create_object_key(i + 1000) for i in range(3)]
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap is not None
        for i in range(3):
            assert bitmap.test(i) is False

    def test_full_store_lookup_load_workflow(self):
        """End-to-end: store → lookup → load, verify data integrity."""
        key = create_object_key(42)
        store_obj = create_memory_obj(size=512, fill_value=3.14)
        load_obj = create_memory_obj(size=512, fill_value=0.0)

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()
        load_fd = self.adapter.get_load_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Lookup
        lookup_tid = self.adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is True

        # Load
        load_tid = self.adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd)
        bitmap = self.adapter.query_load_result(load_tid)
        assert bitmap.test(0) is True

        # Verify data integrity
        assert torch.allclose(load_obj.tensor, store_obj.tensor), (
            "Loaded data does not match stored data"
        )

        # Unlock
        self.adapter.submit_unlock([key])

    def test_batch_store_lookup_load(self):
        """Batch workflow with multiple objects."""
        n = 10
        keys = [create_object_key(i) for i in range(n)]
        store_objs = [
            create_memory_obj(size=128, fill_value=float(i * 7)) for i in range(n)
        ]
        load_objs = [create_memory_obj(size=128, fill_value=0.0) for _ in range(n)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()
        load_fd = self.adapter.get_load_event_fd()

        # Store all
        store_tid = self.adapter.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Lookup all
        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(n):
            assert bitmap.test(i) is True

        # Load all
        load_tid = self.adapter.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd)
        bitmap = self.adapter.query_load_result(load_tid)
        for i in range(n):
            assert bitmap.test(i) is True
            assert torch.allclose(load_objs[i].tensor, store_objs[i].tensor), (
                f"Data mismatch for key {i}"
            )

        self.adapter.submit_unlock(keys)

    def test_mixed_lookup_existing_and_missing(self):
        """Lookup a mix of stored and non-stored keys."""
        stored_keys = [create_object_key(i) for i in range(3)]
        stored_objs = [create_memory_obj(fill_value=float(i)) for i in range(3)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store first 3
        self.adapter.submit_store_task(stored_keys, stored_objs)
        assert wait_for_event_fd(store_fd)
        self.adapter.pop_completed_store_tasks()

        # Lookup 5 keys (3 stored + 2 missing)
        all_keys = stored_keys + [create_object_key(100), create_object_key(101)]
        lookup_tid = self.adapter.submit_lookup_and_lock_task(all_keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)

        for i in range(3):
            assert bitmap.test(i) is True, f"Stored key {i} should be found"
        assert bitmap.test(3) is False, "Missing key should not be found"
        assert bitmap.test(4) is False, "Missing key should not be found"

        self.adapter.submit_unlock(stored_keys)

    def test_factory_creates_adapter(self):
        """Verify the factory can create a RESP L2 adapter from config."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        config = RESPL2AdapterConfig(host=REDIS_HOST, port=REDIS_PORT, num_workers=2)
        adapter = create_l2_adapter(config)
        try:
            # Should have valid event fds
            assert adapter.get_store_event_fd() >= 0
            assert adapter.get_lookup_and_lock_event_fd() >= 0
            assert adapter.get_load_event_fd() >= 0
        finally:
            adapter.close()
