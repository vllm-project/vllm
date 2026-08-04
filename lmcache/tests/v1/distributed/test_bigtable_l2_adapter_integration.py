# SPDX-License-Identifier: Apache-2.0
"""
Integration test for the Bigtable L2 adapter in MP mode.

Requires a running Bigtable emulator (provided by gcloud CLI).
"""

# Standard
import select
import time

# Third Party
from google.cloud.bigtable import Client
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter import (
    BigtableL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

TEST_PROJECT = "test-project"
TEST_INSTANCE = "test-instance"
TEST_TABLE = "test-table"


def create_object_key(chunk_id: int, model_name: str = "test-model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_id.to_bytes(4, byteorder="big"),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.full((size,), fill_value, dtype=torch.uint8, device="cpu")
    metadata = MemoryObjMetadata(
        shape=raw_data.shape,
        dtype=raw_data.dtype,
        address=0,
        phy_size=size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
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


@pytest.mark.integration
class TestBigtableL2AdapterIntegration:
    """Integration tests using the real Bigtable L2 Adapter against the emulator."""

    @pytest.fixture(autouse=True)
    def setup_emulator_table(self, bigtable_emulator):
        """Prepare instance, table, and column family in the emulator."""
        # Initialize sync admin client using the emulator host
        client = Client(project=TEST_PROJECT, admin=True)
        instance = client.instance(TEST_INSTANCE)

        table = instance.table(TEST_TABLE)
        try:
            if table.exists():
                table.delete()
        except Exception:
            pass

        table.create()
        cf = table.column_family("cf")
        cf.create()

        # Initialize L2 Adapter
        cfg = BigtableL2AdapterConfig(
            project_id=TEST_PROJECT,
            instance_id=TEST_INSTANCE,
            table_name=TEST_TABLE,
            family_name="cf",
            column_name="data",
        )
        self.adapter = create_l2_adapter(cfg)

        yield

        # Teardown
        self.adapter.close()
        try:
            if table.exists():
                table.delete()
        except Exception:
            pass

    def test_store_and_load_integration_saves_and_loads_successfully_on_emulator(self):
        key1 = create_object_key(1)
        key2 = create_object_key(2)
        obj1 = create_memory_obj(512, 10.0)
        obj2 = create_memory_obj(1024, 20.0)

        # 1. Store
        store_tid = self.adapter.submit_store_task([key1, key2], [obj1, obj2])
        assert wait_for_event_fd(self.adapter.get_store_event_fd())

        stores = self.adapter.pop_completed_store_tasks()
        assert store_tid in stores
        assert stores[store_tid].is_successful()
        assert stores[store_tid].bytes_transferred() == 512 + 1024

        # 2. Lookup
        lookup_tid = self.adapter.submit_lookup_and_lock_task(
            [key1, key2], _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(self.adapter.get_lookup_and_lock_event_fd())
        lookup_res = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert lookup_res is not None
        assert lookup_res.test(0)
        assert lookup_res.test(1)

        # 3. Load
        dest1 = create_memory_obj(512, 0.0)
        dest2 = create_memory_obj(1024, 0.0)
        load_tid = self.adapter.submit_load_task([key1, key2], [dest1, dest2])
        assert wait_for_event_fd(self.adapter.get_load_event_fd())

        load_res = self.adapter.query_load_result(load_tid)
        assert load_res is not None
        assert load_res.test(0)
        assert load_res.test(1)

        # Verify exact byte contents
        assert (dest1.tensor == 10).all()
        assert (dest2.tensor == 20).all()

        # Unlock keys
        self.adapter.submit_unlock([key1, key2])

    def test_delete_integration_removes_unlocked_keys_from_emulator(self):
        key = create_object_key(3)
        obj = create_memory_obj(128, 99.0)

        # Store
        self.adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(self.adapter.get_store_event_fd())
        self.adapter.pop_completed_store_tasks()

        # Listener
        class TestListener(L2AdapterListener):
            def __init__(self):
                self.deleted = []

            def on_l2_keys_stored(self, keys):
                pass

            def on_l2_keys_accessed(self, keys):
                pass

            def on_l2_keys_deleted(self, keys):
                self.deleted.extend(keys)

        listener = TestListener()
        self.adapter.register_listener(listener)

        # Delete
        self.adapter.delete([key])
        time.sleep(0.2)

        # Assert listener notified
        assert key in listener.deleted

        # Verify key no longer exists on emulator
        lookup_tid = self.adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(self.adapter.get_lookup_and_lock_event_fd())
        lookup_res = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert lookup_res is not None
        assert not lookup_res.test(0)

    def test_adapter_writes_sharded_row_keys_to_emulator(self):
        # Initialize a custom adapter with sharding configuration
        cfg = BigtableL2AdapterConfig(
            project_id=TEST_PROJECT,
            instance_id=TEST_INSTANCE,
            table_name=TEST_TABLE,
            family_name="cf",
            column_name="data",
            num_layers=4,
            layer_group_size=2,
            kv_size=2,
        )
        custom_adapter = create_l2_adapter(cfg)

        key = create_object_key(99)
        # 4 layers * 2 kv components = 8 elements. Let's create an 8-byte payload.
        # K0, K1, K2, K3, V0, V1, V2, V3
        # If size is 8, the payload tensor is 8 bytes.
        obj = create_memory_obj(size=8, fill_value=0)
        # Initialize the elements to specific values
        obj.tensor[0] = 0x01  # K0
        obj.tensor[1] = 0x02  # K1
        obj.tensor[2] = 0x03  # K2
        obj.tensor[3] = 0x04  # K3
        obj.tensor[4] = 0x05  # V0
        obj.tensor[5] = 0x06  # V1
        obj.tensor[6] = 0x07  # V2
        obj.tensor[7] = 0x08  # V3

        # Store
        custom_adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(custom_adapter.get_store_event_fd())
        custom_adapter.pop_completed_store_tasks()

        # Connect directly to emulator using raw Bigtable Client to inspect row
        client = Client(project=TEST_PROJECT)
        instance = client.instance(TEST_INSTANCE)
        table = instance.table(TEST_TABLE)

        row_key = custom_adapter._key_encoder.encode_row_key(key)
        # Assert key structure contains model name and details
        # For chunk_id = 99, hex is 00000063
        # hash_prefix is "0000"
        expected_key = b"0000@test-model@lg2@00000000@0@00000063"
        assert row_key == expected_key

        row = table.read_row(row_key)
        assert row is not None

        # Verify that it is sharded across column qualifiers layers_0_1 and layers_2_3
        # layers_0_1 should contain: K0K1V0V1 -> \x01\x02\x05\x06
        # layers_2_3 should contain: K2K3V2V3 -> \x03\x04\x07\x08
        cells = row.cells["cf"]
        assert b"layers_0_1" in cells
        assert b"layers_2_3" in cells

        assert cells[b"layers_0_1"][0].value == b"\x01\x02\x05\x06"
        assert cells[b"layers_2_3"][0].value == b"\x03\x04\x07\x08"

        custom_adapter.close()
