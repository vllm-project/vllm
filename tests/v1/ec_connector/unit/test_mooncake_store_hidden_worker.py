# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import sys
import struct
import types
from concurrent.futures import Future

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_TENSOR_LAYOUT,
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
    HiddenStoreError,
    HiddenStoreLoadError,
    HiddenStoreSaveError,
    MooncakeHiddenStoreClient,
    _get_hidden_state_object_data_type,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.worker import (
    HiddenLookupClient,
    HiddenLookupServer,
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
        self.batch_is_exist_calls = []
        self.registered = []
        self.unregistered = []
        self.pub_tensors = []
        self.range_gets = []
        self.fail_register_addrs = set()
        self.raise_on_batch_put = False
        self.batch_put_results = [0]

    def batch_is_exist(self, keys):
        self.batch_is_exist_calls.append(list(keys))
        return [1 if key in self.objects else 0 for key in keys]

    def register_buffer(self, addr, size):
        if addr in self.fail_register_addrs:
            return -1
        self.registered.append((addr, size))
        return 0

    def unregister_buffer(self, addr):
        self.unregistered.append(addr)
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


class FakeClosableStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


class FakeStoreWithTeardown(FakeStore):
    def __init__(self):
        super().__init__()
        self.teardown_called = False

    def teardown(self):
        self.teardown_called = True


class FakeSocket:
    def __init__(self):
        self.closed = False
        self.linger = None

    def close(self, linger=0):
        self.closed = True
        self.linger = linger


class FakeContext:
    def __init__(self):
        self.destroy_called = False
        self.term_called = False

    def destroy(self, linger=0):
        self.destroy_called = True

    def term(self):
        self.term_called = True


class FakeThread:
    def __init__(self):
        self.join_called = False
        self.timeout = None

    def join(self, timeout=None):
        self.join_called = True
        self.timeout = timeout

    def is_alive(self):
        return False


class FakeBufferStore(FakeStore):
    def batch_put_from_multi_buffers(
        self,
        keys,
        buffer_ptrs,
        buffer_sizes,
        replicate_config=None,
    ):
        if self.raise_on_batch_put:
            raise RuntimeError("batch put failed")
        self.objects[keys[0]] = b"tensor-object"
        return self.batch_put_results


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
            cache_prefix="",
            kind="encoder_output",
            model_name="qwen",
            encoder="encoder-config-a",
            storage="replicated_object",
            parallel="tp:1@pp:1@pcp:1@dcp:1@mm_tp:weights",
            tensor_layout=HIDDEN_TENSOR_LAYOUT,
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
    worker.save_tensor(pool_key, tensor)

    assert worker.lookup(pool_key.identifier)
    assert not worker.lookup("missing-image-hash")
    assert store.range_gets == []


def test_worker_batch_lookup_checks_existence_in_one_store_call():
    pool_key_a = make_pool_key("image-a")
    pool_key_b = make_pool_key("image-b")
    store = FakeBufferStore()
    store.objects[make_hidden_data_key(pool_key_a)] = b"tensor-object"
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key_a.key_metadata,
    )

    results = worker.lookup_batch(["image-a", "image-b"])

    assert results == {"image-a": True, "image-b": False}
    assert store.batch_is_exist_calls == [
        [make_hidden_data_key(pool_key_a), make_hidden_data_key(pool_key_b)]
    ]
    assert store.range_gets == []


def test_lookup_client_discard_removes_identifier_future_mapping():
    client = HiddenLookupClient.__new__(HiddenLookupClient)
    future: Future[dict[str, bool]] = Future()
    client.futures = {
        "image-a": future,
        "image-b": future,
    }

    client.discard("image-a")

    assert "image-a" not in client.futures
    assert client.futures == {"image-b": future}
    assert not future.cancelled()

    client.discard("image-b")

    assert client.futures == {}
    assert future.cancelled()


def test_worker_lookup_records_minimal_operation_stats():
    pool_key = make_pool_key()
    store = FakeBufferStore()
    store.objects[make_hidden_data_key(pool_key)] = b"tensor-object"
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key.key_metadata,
    )

    assert worker.lookup_batch(["image-hash", "missing-image-hash"]) == {
        "image-hash": True,
        "missing-image-hash": False,
    }

    stats = worker.get_operation_stats()
    records = stats.data["lookup_exists"]
    assert len(records) == 1
    assert records[0]["num_keys"] == 2
    assert records[0]["num_bytes"] == 0
    assert records[0]["status"] == "miss"
    assert records[0]["num_failed_keys"] == 1
    assert worker.get_operation_stats() is None


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

    worker.save_tensor(pool_key, tensor)

    assert store.pub_tensors[0][0] == make_hidden_data_key(pool_key)
    assert store.pub_tensors[0][2] is not None
    assert make_hidden_data_key(pool_key) in store.objects


def test_worker_save_rejects_dtype_that_load_cannot_decode():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float64)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )

    try:
        worker.save_tensor(pool_key, tensor)
    except HiddenStoreSaveError as exc:
        assert "unsupported hidden tensor dtype" in str(exc)
    else:
        raise AssertionError("unsupported hidden dtype should fail before store put")

    assert store.pub_tensors == []


def test_buffer_put_unregisters_payload_and_metadata_buffers():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeBufferStore()
    client = MooncakeHiddenStoreClient(store, replicate_config=FakeReplicateConfig())

    client.put_tensor(pool_key, tensor)

    payload_addr = tensor.data_ptr()
    metadata_addr = next(
        addr for addr, size in store.registered if size == TENSOR_METADATA_SIZE
    )
    assert payload_addr in store.unregistered
    assert metadata_addr in store.unregistered
    assert store.unregistered[-2:] == [metadata_addr, payload_addr]


def test_buffer_put_unregisters_payload_and_metadata_when_put_raises():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeBufferStore()
    store.raise_on_batch_put = True
    client = MooncakeHiddenStoreClient(store, replicate_config=FakeReplicateConfig())

    try:
        client.put_tensor(pool_key, tensor)
    except RuntimeError as exc:
        assert "batch put failed" in str(exc)
    else:
        raise AssertionError("batch put exception should propagate")

    payload_addr = tensor.data_ptr()
    metadata_addr = next(
        addr for addr, size in store.registered if size == TENSOR_METADATA_SIZE
    )
    assert payload_addr in store.unregistered
    assert metadata_addr in store.unregistered


def test_buffer_put_unregisters_payload_when_metadata_registration_fails():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeBufferStore()
    original_register = store.register_buffer

    def register_buffer(addr, size):
        if size == TENSOR_METADATA_SIZE:
            store.fail_register_addrs.add(addr)
        return original_register(addr, size)

    store.register_buffer = register_buffer
    client = MooncakeHiddenStoreClient(store, replicate_config=FakeReplicateConfig())

    try:
        client.put_tensor(pool_key, tensor)
    except HiddenStoreError:
        pass
    else:
        raise AssertionError("metadata registration failure should raise")

    assert tensor.data_ptr() in store.unregistered


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

    worker.save_tensor(pool_key, tensor)

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

    worker.save_tensor(pool_key, tensor)

    assert store.pub_tensors[0][0] == make_hidden_data_key(pool_key)


def test_worker_save_skips_existing_tensor_object():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor)
    worker.save_tensor(pool_key, tensor)

    assert len(store.pub_tensors) == 1


def test_worker_save_records_exists_and_put_operation_stats():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor)

    stats = worker.get_operation_stats()
    assert stats.data["save_exists"][0]["status"] == "miss"
    assert stats.data["save_exists"][0]["num_keys"] == 1
    assert stats.data["save_put"][0]["status"] == "ok"
    assert stats.data["save_put"][0]["num_keys"] == 1
    assert stats.data["save_put"][0]["num_bytes"] == (
        tensor.numel() * tensor.element_size()
    )


def test_worker_save_existing_records_only_save_exists():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    store.objects[make_hidden_data_key(pool_key)] = b"tensor-object"
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.save_tensor(pool_key, tensor)

    stats = worker.get_operation_stats()
    assert stats.data["save_exists"][0]["status"] == "ok"
    assert "save_put" not in stats.data


def test_sending_thread_stores_hidden_tensor_asynchronously():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )
    sending_thread = HiddenStoreSendingThread(worker)
    sending_thread.start()

    sending_thread.add_request(
        HiddenSaveRequest(pool_key=pool_key, tensor=tensor)
    )
    sending_thread.request_queue.join()

    assert store.pub_tensors[0][0] == make_hidden_data_key(pool_key)
    assert sending_thread.get_and_clear_finished_identifiers() == {pool_key.identifier}
    sending_thread.close()


def test_sending_thread_records_failed_identifier_without_finishing():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeBufferStore()
    store.raise_on_batch_put = True
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )
    sending_thread = HiddenStoreSendingThread(worker)
    sending_thread.start()

    sending_thread.add_request(
        HiddenSaveRequest(pool_key=pool_key, tensor=tensor)
    )
    sending_thread.request_queue.join()

    assert sending_thread.get_and_clear_finished_identifiers() == set()
    assert sending_thread.get_and_clear_failed_identifiers() == {pool_key.identifier}
    assert pool_key.identifier in sending_thread.failure_reasons
    assert worker.get_operation_stats().data["save_put"][0]["status"] == "error"
    sending_thread.close()


def test_worker_drains_failed_sending_reasons():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeBufferStore()
    store.raise_on_batch_put = True
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )
    worker.start_sending_thread()

    assert worker.sending_thread is not None
    worker.enqueue_save(HiddenSaveRequest(pool_key=pool_key, tensor=tensor))
    worker.sending_thread.request_queue.join()

    failed = worker.get_failed_sending()

    assert set(failed) == {pool_key.identifier}
    assert "batch put failed" in failed[pool_key.identifier]
    assert worker.get_failed_sending() == {}
    worker.shutdown()


def test_sending_thread_close_joins_worker_thread():
    pool_key = make_pool_key()
    tensor = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )
    sending_thread = HiddenStoreSendingThread(worker)
    sending_thread.start()
    sending_thread.add_request(HiddenSaveRequest(pool_key=pool_key, tensor=tensor))
    sending_thread.request_queue.join()

    sending_thread.close()

    assert not sending_thread.is_alive()


def test_worker_shutdown_closes_store_client():
    store = FakeClosableStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
    )

    worker.shutdown()

    assert store.closed


def test_store_client_close_uses_fallback_close_method():
    store = FakeStoreWithTeardown()
    client = MooncakeHiddenStoreClient(store)

    client.close()

    assert store.teardown_called


def test_lookup_server_close_joins_thread_and_closes_context(tmp_path):
    socket = FakeSocket()
    ctx = FakeContext()
    thread = FakeThread()
    ipc_path = tmp_path / "hidden_lookup.ipc"
    ipc_path.write_text("socket")
    server = HiddenLookupServer.__new__(HiddenLookupServer)
    server.running = True
    server.socket = socket
    server.ctx = ctx
    server.thread = thread
    server._ipc_path = str(ipc_path)

    server.close()

    assert not server.running
    assert socket.closed
    assert socket.linger == 0
    assert thread.join_called
    assert ctx.destroy_called or ctx.term_called
    assert not ipc_path.exists()


def test_lookup_client_close_shuts_down_executor_socket_and_context():
    socket = FakeSocket()
    ctx = FakeContext()
    executor = types.SimpleNamespace(
        shutdown_called=False,
        shutdown=lambda wait=False, cancel_futures=True: setattr(
            executor, "shutdown_called", True
        ),
    )
    client = HiddenLookupClient.__new__(HiddenLookupClient)
    client.executor = executor
    client.futures = {"image-hash": Future()}
    client.socket = socket
    client.ctx = ctx

    client.close()

    assert executor.shutdown_called
    assert client.futures == {}
    assert socket.closed
    assert socket.linger == 0
    assert ctx.destroy_called or ctx.term_called


def test_worker_load_gets_tensor_data_into_encoder_cache_before_returning():
    pool_key = make_pool_key()
    stored = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key.key_metadata,
    )
    worker.save_tensor(pool_key, stored)

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

    stats = worker.get_operation_stats()
    assert stats.data["load_get"][0]["status"] == "ok"
    assert stats.data["load_get"][0]["num_keys"] == 1
    assert stats.data["load_get"][0]["num_bytes"] == (
        stored.numel() * stored.element_size()
    )


def test_worker_load_records_error_without_writing_encoder_cache():
    pool_key = make_pool_key()
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key.key_metadata,
    )
    encoder_cache = {}

    try:
        worker.load(
            [MMMeta(identifier=pool_key.identifier, load_spec=LoadSpec(can_load=True))],
            encoder_cache,
            device="cpu",
        )
    except HiddenStoreLoadError:
        pass
    else:
        raise AssertionError("missing hidden tensor should fail fast")

    assert pool_key.identifier not in encoder_cache
    stats = worker.get_operation_stats()
    assert stats.data["load_get"][0]["status"] == "error"
    assert stats.data["load_get"][0]["num_failed_keys"] == 1


def test_get_tensor_payload_unregisters_target_buffer_after_success():
    pool_key = make_pool_key()
    stored = torch.zeros((2, 4), dtype=torch.float16)
    store = FakeStore()
    worker = HiddenStoreWorker(
        store_client=MooncakeHiddenStoreClient(store),
        tensor_database=HiddenTensorDatabase(),
        key_metadata=pool_key.key_metadata,
    )
    worker.save_tensor(pool_key, stored)
    target = torch.empty_like(stored)

    worker.store_client.get_tensor_payload(
        pool_key,
        target.data_ptr(),
        target.numel() * target.element_size(),
        TENSOR_METADATA_SIZE,
    )

    assert target.data_ptr() in store.unregistered


def test_get_tensor_payload_unregisters_target_buffer_after_load_error():
    pool_key = make_pool_key()
    target = torch.empty((2, 4), dtype=torch.float16)
    store = FakeStore()
    client = MooncakeHiddenStoreClient(store)

    try:
        client.get_tensor_payload(
            pool_key,
            target.data_ptr(),
            target.numel() * target.element_size(),
            TENSOR_METADATA_SIZE,
        )
    except HiddenStoreLoadError:
        pass
    else:
        raise AssertionError("missing payload should raise")

    assert target.data_ptr() in store.unregistered


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
