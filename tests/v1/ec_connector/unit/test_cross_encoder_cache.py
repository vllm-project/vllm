# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import gc
import struct
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.config import MultiModalConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import MooncakeECConfig
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakePushSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
    backend as store_backend,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
    store_client,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
    build_embedding_namespace,
    make_embedding_key,
)

pytestmark = pytest.mark.cpu_test
KEY = (
    "test@embedding@model:model@revision:revision"
    "@encoder:encoder@dtype:torch.float32@protocol:v3"
)
SPEC = TensorSpec((2, 2), "torch.float32", 16)


def test_embedding_key_combines_namespace_and_identifier():
    config = create_ec_vllm_config(ec_role="ec_producer")
    config.ec_transfer_config.ec_connector_extra_config.update(
        embedding_cache_prefix="team@cache", embedding_model_identity="shared/model"
    )
    config.model_config.revision = "rev:1"
    config.model_config.dtype = torch.float32
    config.model_config.multimodal_config = None
    assert make_embedding_key(build_embedding_namespace(config), "image@1") == (
        "team@cache@embedding@model:shared/model@revision:rev:1"
        "@encoder:encoder:default@dtype:torch.float32@protocol:v3@id:image@1"
    )


def test_shared_reuse_requires_content_stable_identifiers():
    """Encoder-only cache-off rendering uses process-local counter identifiers."""
    config = create_ec_vllm_config(ec_role="ec_producer")
    config.ec_transfer_config.ec_connector = "ECMooncakeConnector"
    config.ec_transfer_config.ec_connector_extra_config["cross_encoder_cache"] = True
    config.model_config.multimodal_config = MultiModalConfig(mm_processor_cache_gb=0)
    config.use_v2_model_runner = True
    config.lora_config = None
    with pytest.raises(ValueError, match="mm_processor_cache_gb"):
        MooncakeECConfig.from_vllm_config(config)


class MemoryStore:
    """Only the native boundary is faked; codec, resolver and publisher are real."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.registered: dict[int, int] = {}

    def close(self):
        return 0

    def batch_is_exist(self, keys):
        return [int(k in self.objects) for k in keys]

    def register_buffer(self, addr, size):
        self.registered[addr] = size
        return 0

    def unregister_buffer(self, addr):
        del self.registered[addr]
        return 0

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config):
        [key], [ptrs], [lengths] = keys, pointers, sizes
        self.objects[key] = b"".join(
            ctypes.string_at(p, n) for p, n in zip(ptrs, lengths, strict=True)
        )
        return [0]

    def batch_get_into(self, keys, pointers, sizes):
        results = []
        for key, ptr, size in zip(keys, pointers, sizes, strict=True):
            assert any(
                base <= ptr and ptr + size <= base + capacity
                for base, capacity in self.registered.items()
            )
            obj = self.objects.get(key)
            if obj is None:
                results.append(-704)
            elif len(obj) > size:
                results.append(-600)
            else:
                ctypes.memmove(ptr, obj, len(obj))
                results.append(len(obj))
        return results


@pytest.fixture
def backend():
    instance = store_backend.MooncakeEmbeddingStoreBackend(
        store_client.MooncakeEmbeddingStoreClient(
            MemoryStore(), read_buffer_bytes=1024
        ),
        KEY,
        max_pending_items=32,
        max_pending_bytes=2 * 1024**3,
    )
    yield instance
    # Some tests deliberately poison I/O; always stop the Python executor.
    instance._executor.shutdown(wait=True)
    if not instance.store_client._poisoned and (
        instance.store_client._read_buffer is not None
        or instance.store_client._put_header is not None
    ):
        instance.store_client.close()


def test_tensor_round_trip(backend):
    client, native = backend.store_client, backend.store_client.store
    native.register_buffer = MagicMock(wraps=native.register_buffer)
    header_ptr = None
    for index, dtype in enumerate((torch.float16, torch.bfloat16, torch.float32)):
        tensor = torch.arange(6, dtype=dtype).reshape(index + 1, -1)
        key = make_embedding_key(KEY, str(dtype))
        client.put_tensor(key, tensor)
        pointer = ctypes.addressof(client._put_header)
        header_ptr = header_ptr or pointer
        assert pointer == header_ptr
        expected = TensorSpec(tuple(tensor.shape), str(dtype), tensor.nbytes)
        with torch.device("meta"):
            loaded = client.load_tensors({key: expected}, "cpu")[key]
        assert client._read_buffer.device.type == "cpu"
        assert loaded.dtype == dtype and torch.equal(loaded, tensor)
        assert len(native.objects[key]) == 24 + tensor.nbytes
    assert (
        sum(c.args[0] == header_ptr for c in native.register_buffer.call_args_list) == 1
    )
    assert len(native.registered) == 2


@pytest.mark.parametrize("capacity", [80, 120])
def test_batch_reads_reuse_registration_without_aliasing_outputs(backend, capacity):
    client, native = backend.store_client, backend.store_client.store
    client._read_buffer_bytes = capacity
    tensors = {key: torch.full((2, 2), float(i)) for i, key in enumerate("abc")}
    for key, tensor in tensors.items():
        client.put_tensor(key, tensor)
    native.register_buffer = MagicMock(wraps=native.register_buffer)
    native.unregister_buffer = MagicMock(wraps=native.unregister_buffer)
    native.batch_get_into = MagicMock(wraps=native.batch_get_into)
    expected = dict.fromkeys(tensors, SPEC)
    first = client.load_tensors(expected, "cpu")
    for key, tensor in tensors.items():
        native.objects[key] = native.objects[key][:24] + struct.pack(
            "<4f", *[tensor[0, 0].item() + 10] * 4
        )
    second = client.load_tensors(expected, "cpu")
    assert all(torch.equal(first[key], tensor) for key, tensor in tensors.items())
    assert all(torch.equal(second[key], tensor + 10) for key, tensor in tensors.items())
    assert native.batch_get_into.call_count == (4 if capacity == 80 else 2)
    native.register_buffer.assert_called_once()
    native.unregister_buffer.assert_not_called()
    client.close()
    assert native.unregister_buffer.call_count == 2
    assert not native.registered


@pytest.mark.parametrize("mode", ["miss", "oversized"])
def test_unloadable_inputs_do_not_allocate_staging(backend, mode):
    client, native = backend.store_client, backend.store_client.store
    client._read_buffer_bytes = 39
    if mode == "oversized":
        client.put_tensor(make_embedding_key(KEY, "a"), torch.ones(2, 2))
    native.register_buffer = MagicMock(wraps=native.register_buffer)
    assert not backend.resolve_inputs({"a": SPEC}, {}, "cpu")
    assert client._read_buffer is None
    if mode == "miss":
        assert client._put_header is None
        client.close()
    native.register_buffer.assert_not_called()


@pytest.mark.parametrize(
    "results", [[40, -800], [40], [40, True], [[40], [40]], None, [40, 41]]
)
def test_unsafe_batch_result_retains_whole_staging(backend, results):
    client, native = backend.store_client, backend.store_client.store
    native.batch_get_into = MagicMock(return_value=results)
    native.unregister_buffer = MagicMock(wraps=native.unregister_buffer)
    for _ in range(2):
        with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
            client.load_tensors(dict.fromkeys("ab", SPEC), "cpu")
    native.batch_get_into.assert_called_once()
    with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
        client.close()
    native.unregister_buffer.assert_not_called()
    assert client._read_buffer is client._unsafe_owners[0]


@pytest.mark.parametrize("failure", ["allocation", "copy", "record", "completion"])
def test_copy_failure_preserves_staging_lifetime(backend, failure):
    """Emulate CUDA allocation/event boundaries without requiring a GPU."""
    client = backend.store_client
    for key in "ab":
        client.put_tensor(key, torch.ones(2, 2))
    allocate, copy = torch.empty, torch.Tensor.copy_
    targets: list[torch.Tensor] = []
    copies: list[torch.Tensor] = []
    ready = MagicMock()
    if failure == "record":
        ready.record.side_effect = RuntimeError("record failure")
    elif failure == "completion":
        ready.synchronize.side_effect = RuntimeError("copy failure")

    def allocate_on_cpu(*args, **kwargs):
        is_target = kwargs.get("device") == torch.device("cuda")
        if is_target and targets and failure == "allocation":
            raise torch.OutOfMemoryError("allocation failure")
        kwargs.update(device="cpu", pin_memory=False)
        tensor = allocate(*args, **kwargs)
        if is_target:
            targets.append(tensor)
        return tensor

    def record_copy(target, source, **kwargs):
        if copies and failure == "copy":
            raise RuntimeError("copy failure")
        copies.append(target)
        return copy(target, source, **kwargs)

    with (
        patch.object(torch, "empty", side_effect=allocate_on_cpu),
        patch.object(torch, "Event", return_value=ready),
        patch.object(torch.accelerator, "current_stream"),
        patch.object(torch.Tensor, "copy_", new=record_copy),
        pytest.raises(RuntimeError, match="allocation failure|unconfirmed"),
    ):
        client.load_tensors(dict.fromkeys("ab", SPEC), "cuda")
    if failure == "allocation":
        assert not copies and not client._poisoned
    else:
        assert len(copies) == (1 if failure == "copy" else 2)
        assert client._poisoned
        assert any(owner is client._read_buffer for owner in client._unsafe_owners)
        assert any(owner is ready for owner in client._unsafe_owners)
        assert all(
            any(owner is target for owner in client._unsafe_owners)
            for target in targets
        )
        with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
            client.close()


def test_non_matrix_output_is_rejected_before_registration(backend):
    client = backend.store_client
    client.store.register_buffer = MagicMock()
    with pytest.raises(store_client.EmbeddingStoreOperationError, match="2D"):
        client.put_tensor(make_embedding_key(KEY, "a"), torch.ones(2, 2, 2))
    client.store.register_buffer.assert_not_called()


def make_worker(backend):
    from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
        ECMooncakeWorker,
    )

    worker = object.__new__(ECMooncakeWorker)
    worker.is_producer = True
    worker.is_consumer = False
    worker._output_store = backend
    worker._buffer_device = "cpu"
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.reserve.return_value = (MagicMock(), True)
    worker._producer_pushes.poll.return_value = []
    worker._producer_pushes.cancel_requests.return_value = []
    worker._producer_pushes.pending = False
    worker._control_executor = MagicMock()
    worker._consumer_memory = MagicMock()
    worker._consumer_memory.drain_reclaimed.return_value = set()
    worker._completed_loads = set()
    worker._failed_loads = set()
    worker._failed_saves = set()
    worker._push_ready = threading.Event()
    worker._flush_pending_pushes = MagicMock()
    worker._bind_push_source = MagicMock()
    connector = object.__new__(ECMooncakeConnector)
    connector._worker = worker
    connector._role = ECConnectorRole.WORKER
    connector._is_producer = True
    connector._is_consumer = False
    connector._connector_metadata = None
    return connector


@pytest.mark.parametrize(
    "mode",
    [
        "miss",
        "legacy",
        "hit",
        "partial",
        "get-failure",
        "short-data",
        "lease-expired",
        "registration-failure",
    ],
)
def test_reuse_pipeline(backend, mode):
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )
    from vllm.v1.worker.gpu.ec_connector import ActiveECConnector
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    native = backend.store_client.store
    values = {"a": 1.0, "b": 2.0, "c": 3.0}
    if mode == "legacy":
        old_namespace = KEY.replace("@protocol:v3", "@protocol:v2")
        for key in "abc":
            native.objects[make_embedding_key(old_namespace, key)] = b"old format"
    if mode == "partial":
        hits = {"a", "c"}
    elif mode not in ("miss", "legacy"):
        hits = set("abc")
    else:
        hits = set()
    for key in hits:
        backend.store_client.put_tensor(
            make_embedding_key(KEY, key), torch.full((2, 2), values[key])
        )
    if mode == "short-data":
        key = make_embedding_key(KEY, "b")
        native.objects[key] = native.objects[key][:-1]
        hits.remove("b")
    if mode in ("get-failure", "lease-expired"):
        read = native.batch_get_into

        def failed_read(keys, pointers, sizes):
            results = read(keys, pointers, sizes)
            results[keys.index(make_embedding_key(KEY, "b"))] = (
                -707 if mode == "lease-expired" else -704
            )
            return results

        native.batch_get_into = failed_read
        hits.remove("b")
    if mode == "registration-failure":
        native.register_buffer = MagicMock(return_value=-500)
        hits.clear()
    connector = make_worker(backend)
    metadata = ECMooncakeConnectorMetadata(
        store_candidates=dict.fromkeys("abc", SPEC),
        pushes=[
            ECMooncakePushSpec(
                mm_hash=k,
                shape=(2, 2),
                dtype="float32",
                nbytes=16,
                consumer_zmq="ipc:///tmp/test",
                transfer_id=k,
                request_id="req",
            )
            for k in "abc"
        ],
    )
    cache = EncoderCache()
    cache.mm_features["req"] = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem({}),
            modality="image",
            identifier=k,
            mm_position=PlaceholderRange(offset=i * 2, length=2),
        )
        for i, k in enumerate("abc")
    ]
    runner = EncoderRunner(
        model=None,
        max_num_tokens=8,
        hidden_size=2,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    runner.enable_timing = False
    missing = [k for k in "abc" if k not in hits]
    runner.execute_mm_encoder = MagicMock(
        return_value=[torch.full((2, 2), values[k] + 10) for k in missing]
    )
    state = SimpleNamespace(encoder_runner=runner, encoder_cache=cache)
    with patch(
        "vllm.v1.worker.gpu.ec_connector.get_ec_transfer", return_value=connector
    ):
        adapter = ActiveECConnector(SimpleNamespace(), cache.encoder_outputs)
    with adapter.maybe_get_output(
        SimpleNamespace(ec_connector_metadata=metadata, finished_req_ids=set())
    ) as output:
        assert runner.prepare_mm_inputs({"req": [0, 1, 2]})[0] == missing
        ModelState.execute_mm_encoder(state, {"req": [0, 1, 2]})
    assert runner.execute_mm_encoder.call_count == bool(missing)
    assert output.ec_connector_worker_meta.pending_saves is False
    assert all(
        torch.equal(
            cache.encoder_outputs[k],
            torch.full((2, 2), values[k] + (0 if k in hits else 10)),
        )
        for k in "abc"
    )
    assert {
        c.args[1] for c in connector._worker._bind_push_source.call_args_list
    } == set("abc")
    backend.shutdown()
    assert not native.registered
    if mode in ("miss", "legacy"):
        other = store_backend.MooncakeEmbeddingStoreBackend(
            store_client.MooncakeEmbeddingStoreClient(native, read_buffer_bytes=1024),
            KEY,
            max_pending_items=32,
            max_pending_bytes=2 * 1024**3,
        )
        try:
            loaded: dict[str, torch.Tensor] = {}
            assert other.resolve_inputs(
                dict.fromkeys("abc", SPEC),
                loaded,
                "cpu",
            ) == set("abc")
            assert all(torch.equal(loaded[k], cache.encoder_outputs[k]) for k in "abc")
        finally:
            other.shutdown()


@pytest.mark.parametrize(
    "mismatch",
    ["revision", "shape", "dtype", "magic", "short-header", "short-data", "long-data"],
)
def test_incompatible_output_is_a_miss(backend, mismatch):
    key = make_embedding_key(KEY, "a")
    dtype = torch.float16 if mismatch == "dtype" else torch.float32
    backend.store_client.put_tensor(key, torch.ones(2, 2, dtype=dtype))
    if mismatch == "revision":
        backend.namespace = KEY.replace("@revision:revision", "@revision:other")
    objects = backend.store_client.store.objects
    if mismatch == "magic":
        objects[key] = b"\0" * 4 + objects[key][4:]
    elif mismatch == "short-header":
        objects[key] = objects[key][:23]
    elif mismatch == "short-data":
        objects[key] = objects[key][:-1]
    elif mismatch == "long-data":
        objects[key] += b"extra"
    expected = replace(SPEC, shape=(4, 1)) if mismatch == "shape" else SPEC
    outputs: dict[str, torch.Tensor] = {}
    assert not backend.resolve_inputs({"a": expected}, outputs, "cpu")
    assert not outputs
    tensor = torch.ones(2, 2)
    backend.store_client.put_tensor(
        make_embedding_key(backend.namespace, "valid"), tensor
    )
    assert backend.resolve_inputs({"valid": SPEC}, outputs, "cpu") == {"valid"}
    assert torch.equal(outputs["valid"], tensor)
    assert len(backend.store_client.store.registered) == 2


def test_partial_registration_rejection_releases_publication(backend):
    """A rejected header registration must not prevent a later publication."""
    native = backend.store_client.store
    register = native.register_buffer
    put = native.batch_put_from_multi_buffers
    native.register_buffer = lambda addr, size: (
        -500 if size == 24 else register(addr, size)
    )
    native.batch_put_from_multi_buffers = MagicMock()
    tensor = torch.ones(2, 2)
    owner = weakref.ref(tensor)
    assert backend._enqueue_save("a", tensor)
    storage_owner = weakref.ref(tensor.untyped_storage())
    future = backend._pending["a"][0]
    with pytest.raises(store_client.EmbeddingStoreOperationError, match="register"):
        future.result(timeout=5)
    del tensor
    gc.collect()
    assert owner() is None and not native.registered
    native.batch_put_from_multi_buffers.assert_not_called()
    backend.resolve_inputs({}, {}, "cpu")
    assert backend._pending_bytes == 0
    gc.collect()
    assert storage_owner() is None  # The failed Future is still alive.
    assert backend.store_client._put_header is None
    native.register_buffer = register
    native.batch_put_from_multi_buffers = put
    backend.store_client.put_tensor("retry", torch.ones(2, 2))
    assert len(native.registered) == 1


@pytest.mark.parametrize(
    "operation",
    ["read-unregister", "header-unregister", "put", "rejected-put", "unregister"],
)
def test_native_io_owners(backend, operation):
    """Uncertain I/O retains owners; completed rejection frees them even via Future."""
    native, client = backend.store_client.store, backend.store_client
    put = native.batch_put_from_multi_buffers
    backend.max_pending_bytes = SPEC.nbytes
    tensor = torch.ones(2, 2)
    owner = weakref.ref(tensor)
    rejected = operation == "rejected-put"
    if operation in ("read-unregister", "header-unregister"):
        key = make_embedding_key(KEY, "a")
        client.put_tensor(key, tensor)
        if operation == "read-unregister":
            client.load_tensors({key: SPEC}, "cpu")
        native.unregister_buffer = MagicMock(return_value=-500)
        with pytest.raises(store_client.EmbeddingStoreError, match="unregister"):
            client.close()
        owner = weakref.ref(
            client._read_buffer
            if operation == "read-unregister"
            else client._put_header
        )
    else:
        if operation == "unregister":
            native.unregister_buffer = MagicMock(return_value=-500)
        else:
            native.batch_put_from_multi_buffers = MagicMock(
                return_value=[-1700 if rejected else -800]
            )
        assert backend._enqueue_save("a", tensor)
        future = backend._pending["a"][0]
        with pytest.raises(store_client.EmbeddingStoreError):
            future.result(timeout=5)
    del tensor
    gc.collect()
    if rejected:
        assert owner() is None  # A live Future must not retain the tensor.
        tensor = torch.ones(4, 4)  # Too large for admission after reaping.
        caller_owner = weakref.ref(tensor)
        backend._step_candidates = {"next"}
        backend.save_output("next", tensor)
        del tensor
        gc.collect()
        assert caller_owner() is None
        assert backend._pending_bytes == 0 and len(native.registered) == 1
        header = client._put_header
        native.batch_put_from_multi_buffers = put
        client.put_tensor("retry", torch.ones(1, 4))
        assert client._put_header is header
    else:
        assert owner() is not None and native.registered
        header = bytes(client._put_header)
        with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
            client.put_tensor("retry", torch.ones(1, 4))
        assert bytes(client._put_header) == header
        with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
            client.close()


def test_publication_budget_charges_and_releases_backing_storage(backend):
    tensor = torch.arange(16, dtype=torch.float32)[:4].view(2, 2)
    owner = weakref.ref(tensor)
    future: Future[None] = Future()
    backend.max_pending_items = 1
    backend.max_pending_bytes = tensor.nbytes
    with patch.object(backend._executor, "submit", return_value=future):
        assert not backend._enqueue_save("a", tensor)
        backend.max_pending_bytes = 64
        assert backend._enqueue_save("a", tensor)
        storage_owner = weakref.ref(tensor.untyped_storage())
        assert not backend._enqueue_save("b", tensor)
    del tensor
    gc.collect()
    backend.reap()  # The final step-end poll precedes publication completion.
    assert owner() is not None
    backend._save(make_embedding_key(KEY, "a"), backend._pending["a"][1])
    future.set_result(None)
    assert owner() is None and storage_owner() is not None

    def load_after_reaping(*args):
        assert not backend._pending and backend._pending_bytes == 0
        assert storage_owner() is None
        return {}

    with patch.object(backend.store_client, "load_tensors", load_after_reaping):
        assert not backend.resolve_inputs({"b": SPEC}, {}, "cpu")


@pytest.mark.parametrize("rejected", [False, True])
def test_shared_storage_budget_is_released_after_last_publication(backend, rejected):
    batch = torch.arange(16, dtype=torch.float32).view(4, 4)
    first: Future[None] = Future()
    last: Future[None] = Future()
    separate: Future[None] = Future()
    backend.max_pending_items = 2
    backend.max_pending_bytes = batch.nbytes
    with patch.object(backend._executor, "submit", side_effect=[first, last, separate]):
        assert backend._enqueue_save("a", batch[:2])
        assert backend._enqueue_save("b", batch[2:])
        assert backend._pending_bytes == batch.nbytes
        assert not backend._enqueue_save("a", batch[:2])
        assert not backend._enqueue_save("extra", batch[:1])
        if rejected:
            first.set_exception(store_client.EmbeddingStoreOperationError("rejected"))
        else:
            first.set_result(None)
        backend.reap()
        assert backend._pending_bytes == batch.nbytes
        assert not backend._enqueue_save("c", torch.ones(2, 2))
        last.set_result(None)
        backend.reap()
        assert backend._pending_bytes == 0
        assert backend._enqueue_save("c", torch.ones(2, 2))
        assert backend._pending_bytes == 16
        separate.set_result(None)
        backend.reap()
        assert backend._pending_bytes == 0


@pytest.mark.parametrize("shared_storage", [False, True])
def test_completion_race_preserves_failure_and_reclaims_other_items(
    backend, shared_storage
):
    class RacingFuture(Future[None]):
        def done(self):
            observed = super().done()
            if not observed:
                self.set_exception(store_client.EmbeddingStoreError("uncertain"))
            return observed

    racing = RacingFuture()
    other: Future[None] = Future()
    tensor = torch.ones(2, 2)
    other_tensor = tensor[:1] if shared_storage else torch.ones(2, 2)
    with patch.object(backend._executor, "submit", side_effect=[racing, other]):
        assert backend._enqueue_save("race", tensor)
        assert backend._enqueue_save("other", other_tensor)
    backend.reap()
    assert backend._pending_bytes == (16 if shared_storage else 32)
    other.set_result(None)
    with pytest.raises(store_client.EmbeddingStoreError):
        backend.reap()
    assert backend._pending_bytes == 16
    with pytest.raises(store_client.EmbeddingStoreError):
        backend._enqueue_save("later", torch.ones(2, 2))


def test_readiness_and_shutdown_do_not_block_p2p_completion(backend):
    ready, waiting, entered = [threading.Event() for _ in range(3)]

    original = backend._executor.shutdown

    def closing(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)

    native = backend.store_client.store

    def close_store():
        assert ready.is_set() and not native.registered
        assert backend._pending_bytes == 0
        return 0

    native.close = MagicMock(side_effect=close_store)
    put = native.batch_put_from_multi_buffers

    def waiting_put(*args):
        waiting.set()
        assert ready.wait(5)
        assert backend.store_client._put_header is not None and native.registered
        return put(*args)

    native.batch_put_from_multi_buffers = waiting_put
    connector = make_worker(backend)
    connector.bind_connector_metadata(
        ECMooncakeConnectorMetadata(store_candidates={"a": SPEC})
    )
    cache: dict[str, torch.Tensor] = {}
    with (
        patch.object(
            store_backend, "_record_tensor_ready_event", return_value=MagicMock()
        ),
        patch.object(backend._executor, "shutdown", side_effect=closing),
        ThreadPoolExecutor(max_workers=1) as closer,
    ):
        connector.start_save_caches(encoder_cache=cache)
        cache["a"] = torch.arange(4, dtype=torch.float32).view(2, 2)
        connector.save_caches(cache, "a")
        assert connector.build_connector_worker_meta().pending_saves is False
        assert waiting.wait(5) and not backend.store_client.store.objects
        close = closer.submit(backend.shutdown)
        try:
            assert entered.wait(5) and not close.done()
            assert not backend._enqueue_save("late", torch.ones(2, 2))
        finally:
            ready.set()
        close.result(timeout=5)
    native.close.assert_called_once_with()
    assert not backend.store_client.store.registered and backend._pending_bytes == 0

    assert backend.store_client.store.objects[make_embedding_key(KEY, "a")][
        24:
    ] == struct.pack("<4f", 0, 1, 2, 3)
