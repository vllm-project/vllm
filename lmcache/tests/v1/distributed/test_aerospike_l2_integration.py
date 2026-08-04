# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the Aerospike L2 adapter (native connector).

Requires Aerospike CE and BUILD_AEROSPIKE=1 extension. Skipped otherwise.
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.factory import create_l2_adapter_from_registry
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

AEROSPIKE_HOST = os.environ.get("AEROSPIKE_TEST_HOST", "127.0.0.1")
AEROSPIKE_PORT = int(os.environ.get("AEROSPIKE_TEST_PORT", "3000"))
AEROSPIKE_NAMESPACE = os.environ.get("AEROSPIKE_TEST_NAMESPACE", "lmcache")
RUN_AEROSPIKE_IT = os.environ.get("RUN_AEROSPIKE_INTEGRATION") == "1"


def _aerospike_available() -> bool:
    if not RUN_AEROSPIKE_IT:
        return False
    try:
        # Third Party
        import aerospike

        client = aerospike.client(
            {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
        ).connect()
        info = client.info_random_node(f"namespace/{AEROSPIKE_NAMESPACE}")
        client.close()
        return "nsup-period" in info
    except Exception:
        return False


def _native_extension_available() -> bool:
    try:
        # First Party
        from lmcache.lmcache_aerospike import LMCacheAerospikeClient  # noqa: F401

        return True
    except ImportError:
        return False


requires_aerospike = pytest.mark.skipif(
    not _aerospike_available(),
    reason=(
        f"Aerospike not available at {AEROSPIKE_HOST}:{AEROSPIKE_PORT} "
        "(set RUN_AEROSPIKE_INTEGRATION=1)"
    ),
)
requires_native = pytest.mark.skipif(
    not _native_extension_available(),
    reason="lmcache.lmcache_aerospike extension not built",
)


def _wait_fd(fd: int, timeout: float = 30.0) -> None:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    events = poller.poll(timeout * 1000)
    assert events, "timed out waiting for eventfd"
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass


def _make_tensor_obj(size: int, fill: float) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _object_key(suffix: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(suffix),
        model_name="aerospike-it",
        kv_rank=0,
    )


def _adapter_config():
    # First Party
    from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
        AerospikeL2AdapterConfig,
    )

    return AerospikeL2AdapterConfig(
        hosts=f"{AEROSPIKE_HOST}:{AEROSPIKE_PORT}",
        namespace=AEROSPIKE_NAMESPACE,
        set_name="kv_chunks_aerospike_it",
        num_workers=2,
    )


@requires_aerospike
@requires_native
class TestAerospikeL2Integration:
    def test_store_lookup_load_roundtrip(self):
        adapter = create_l2_adapter_from_registry(_adapter_config())
        try:
            key = _object_key(9001)
            store_obj = _make_tensor_obj(64, 42.0)
            load_obj = _make_tensor_obj(64, 0.0)

            tid = adapter.submit_store_task([key], [store_obj])
            _wait_fd(adapter.get_store_event_fd())
            done = adapter.pop_completed_store_tasks()
            assert done[tid].is_successful()

            lookup_tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
            _wait_fd(adapter.get_lookup_and_lock_event_fd())
            lookup_bm = adapter.query_lookup_and_lock_result(lookup_tid)
            assert lookup_bm is not None
            assert lookup_bm.test(0)

            load_tid = adapter.submit_load_task([key], [load_obj])
            _wait_fd(adapter.get_load_event_fd())
            load_bm = adapter.query_load_result(load_tid)
            assert load_bm is not None
            assert load_bm.test(0)
            assert torch.all(load_obj.tensor == 42.0)

            adapter.submit_unlock([key])
        finally:
            adapter.close()
