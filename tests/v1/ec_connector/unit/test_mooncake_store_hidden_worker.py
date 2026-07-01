# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import sys
import struct
import types

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_LAYOUT_VERSION,
    HiddenKeyMetadata,
    HiddenPoolKey,
    HiddenSaveRequest,
    HiddenTensorDatabase,
    LoadSpec,
    MMMeta,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.keys import (
    make_hidden_data_key,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.store_client import (
    MooncakeHiddenStoreClient,
    _get_hidden_state_object_data_type,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.worker import (
    HiddenStoreSendingThread,
    HiddenStoreWorker,
)

TENSOR_METADATA_SIZE = 304
TENSOR_OBJECT_MAGIC = 0x4D4F4F4E
TENSOR_OBJECT_VERSION = 1
TORCH_DTYPE_TO_MOONCAKE_DTYPE = {
    torch.float32: 0,
    torch.float16: 11,
    torch.bfloat16: 12,
}


class FakeStore:
    def __init__(self):
        self.objects = {}
        self.registered = []
        self.pub_tensors = []
        self.range_gets = []

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def register_buffer(self, addr, size):
        self.registered.append((addr, size))
        return 0

    def unregister_buffer(self, addr):
        return 0

    def pub_tensor(self, key, tensor, replicate_config=None):
        self.pub_tensors.append((key, tensor, replicate_config))
        self.objects[key] = _serialize_tensor_object(tensor)
        return 0

    def put_tensor(self, key, tensor):
        return self.pub_tensor(key, tensor)

    def get_into_ranges(
        self,
        buffer_ptrs,
        all_keys,
        all_dst_offsets,
        all_src_offsets,
        all_sizes,
    ):
        self.range_gets.append(
            (buffer_ptrs, all_keys, all_dst_offsets, all_src_offsets, all_sizes)
        )
        results = []
        for buffer_ptr, keys, dst_offsets, src_offsets, sizes in zip(
            buffer_ptrs,
            all_keys,
            all_dst_offsets,
            all_src_offsets,
            all_sizes,
            strict=True,
        ):
            key_results = []
            for key, key_dst_offsets, key_src_offsets, key_sizes in zip(
                keys,
                dst_offsets,
                src_offsets,
                sizes,
                strict=True,
            ):
                payload = self.objects.get(key)
                fragment_results = []
                for dst_offset, src_offset, size in zip(
                    key_dst_offsets,
                    key_src_offsets,
                    key_sizes,
                    strict=True,
                ):
                    if payload is None or src_offset + size > len(payload):
                        fragment_results.append(-1)
                        continue
                    ctypes.memmove(
                        buffer_ptr + dst_offset,
                        payload[src_offset : src_offset + size],
                        size,
                    )
                    fragment_results.append(size)
                key_results.append(fragment_results)
            results.append(key_results)
        return results


class FakeReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.nof_replica_num = 0
        self.with_soft_pin = False
        self.with_hard_pin = False
        self.preferred_segments = []
        self.preferred_nof_segments = []
        self.preferred_segment = ""
        self.prefer_alloc_in_same_node = False
        self.data_type = None
        self.group_ids = None


class FakeReplicateConfigWithoutGroups:
    def __init__(self):
        self.replica_num = 1


class FakeObjectDataTypeWithHidden:
    HIDDEN_STATE = 10
    TENSOR = 2


class FakeObjectDataTypeOnlyTensor:
    TENSOR = 2


class FakeObjectDataTypeNoTensor:
    UNKNOWN = 0


def make_pool_key(identifier: str = "image-hash") -> HiddenPoolKey:
    return HiddenPoolKey(
        key_metadata=HiddenKeyMetadata(
            model_name="qwen",
            mm_encoder_config_hash="encoder-config-a",
            hidden_parallel_key="tp:1@pp:1@pcp:1@dcp:1@mm_tp:weights@storage:replicated",
            layout=HIDDEN_LAYOUT_VERSION,
        ),
        identifier=identifier,
    )


def test_hidden_tensor_database_prepares_data_key_addrs_and_sizes():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)

    key, addrs, sizes = HiddenTensorDatabase().prepare_value(pool_key, tensor)

    assert key == make_hidden_data_key(pool_key)
    assert addrs == [tensor.data_ptr()]
    assert sizes == [tensor.numel() * tensor.element_size()]


def test_store_client_checks_single_tensor_object_exists():
    pool_key = make_pool_key()
    store = FakeStore()
    client = MooncakeHiddenStoreClient(store)

    assert not client.exists(pool_key)

    store.objects[make_hidden_data_key(pool_key)] = b"tensor-object"
    assert client.exists(pool_key)


def test_worker_lookup_checks_existence_without_reading_tensor_metadata():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key.key_metadata,
    )
    worker.save_tensor(pool_key, tensor, now_ms=1234)

    assert worker.lookup(pool_key.identifier)
    assert not worker.lookup("missing-image-hash")
    assert store.range_gets == []


def test_worker_save_stores_hidden_as_single_tensor_object():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(
            store,
            replicate_config=FakeReplicateConfig(),
        ),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor, now_ms=1234)

    assert store.pub_tensors[0][0] == make_hidden_data_key(pool_key)
    assert store.pub_tensors[0][2] is not None
    assert make_hidden_data_key(pool_key) in store.objects


def test_worker_save_marks_hidden_state_data_type(monkeypatch):
    fake_mooncake = types.ModuleType("mooncake")
    fake_store = types.ModuleType("mooncake.store")
    fake_store.ObjectDataType = FakeObjectDataTypeWithHidden
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store)

    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    replicate_config = FakeReplicateConfig()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(
            store,
            replicate_config=replicate_config,
        ),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor, now_ms=1234)

    used_config = store.pub_tensors[0][2]
    assert used_config is not replicate_config
    assert int(used_config.data_type) == 10


def test_hidden_state_data_type_falls_back_to_tensor(monkeypatch):
    fake_mooncake = types.ModuleType("mooncake")
    fake_store = types.ModuleType("mooncake.store")
    fake_store.ObjectDataType = FakeObjectDataTypeOnlyTensor
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store)

    assert _get_hidden_state_object_data_type() == FakeObjectDataTypeOnlyTensor.TENSOR


def test_hidden_state_data_type_missing_type_returns_none(monkeypatch):
    fake_mooncake = types.ModuleType("mooncake")
    fake_store = types.ModuleType("mooncake.store")
    fake_store.ObjectDataType = FakeObjectDataTypeNoTensor
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store)

    assert _get_hidden_state_object_data_type() is None


def test_worker_save_does_not_require_mooncake_object_group_support():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(
            store,
            replicate_config=FakeReplicateConfigWithoutGroups(),
        ),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor, now_ms=1234)

    assert store.pub_tensors[0][0] == make_hidden_data_key(pool_key)


def test_worker_save_skips_existing_tensor_object():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor, now_ms=1234)
    worker.save_tensor(pool_key, tensor, now_ms=1235)

    assert len(store.pub_tensors) == 1


def test_sending_thread_stores_hidden_tensor_asynchronously():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        producer_engine_id="encoder-1",
    )
    sending_thread = HiddenStoreSendingThread(worker)
    sending_thread.start()

    sending_thread.add_request(
        HiddenSaveRequest(pool_key=pool_key, tensor=tensor, now_ms=1234)
    )
    sending_thread.request_queue.join()

    assert store.pub_tensors[0][0] == make_hidden_data_key(pool_key)
    assert sending_thread.get_and_clear_finished_identifiers() == {pool_key.identifier}
    sending_thread.close()


def test_worker_load_gets_tensor_data_into_encoder_cache_before_returning():
    pool_key = make_pool_key()
    stored = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key.key_metadata,
    )
    worker.save_tensor(pool_key, stored, now_ms=1234)

    encoder_cache = {}
    worker.load(
        [MMMeta(identifier=pool_key.identifier, load_spec=LoadSpec(can_load=True))],
        encoder_cache,
        device="cpu",
    )

    assert pool_key.identifier in encoder_cache
    assert tuple(encoder_cache[pool_key.identifier].shape) == tuple(stored.shape)
    assert str(encoder_cache[pool_key.identifier].dtype) == str(stored.dtype)
    assert store.range_gets[0][1] == [[make_hidden_data_key(pool_key)]]
    assert store.range_gets[0][3] == [[[0]]]
    assert store.range_gets[0][4] == [[[TENSOR_METADATA_SIZE]]]
    assert store.range_gets[-1][1] == [[make_hidden_data_key(pool_key)]]
    assert store.range_gets[-1][3] == [[[TENSOR_METADATA_SIZE]]]


def _serialize_tensor_object(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    nbytes = tensor.numel() * tensor.element_size()
    header = struct.pack(
        "<IHHiiIIQQ",
        TENSOR_OBJECT_MAGIC,
        TENSOR_OBJECT_VERSION,
        TENSOR_METADATA_SIZE,
        TORCH_DTYPE_TO_MOONCAKE_DTYPE[tensor.dtype],
        tensor.dim(),
        0,
        0,
        TENSOR_METADATA_SIZE,
        nbytes,
    )
    global_shape = _pack_shape(tuple(tensor.shape))
    local_shape = _pack_shape(tuple(tensor.shape))
    axes = b"\0" * (32 * 4)
    metadata = header + global_shape + local_shape + struct.pack("<II", 0, 0) + axes
    assert len(metadata) == TENSOR_METADATA_SIZE
    return metadata + tensor.view(torch.uint8).numpy().tobytes()


def _pack_shape(shape: tuple[int, ...]) -> bytes:
    dims = list(shape) + [-1] * (8 - len(shape))
    return struct.pack("<8q", *dims)
