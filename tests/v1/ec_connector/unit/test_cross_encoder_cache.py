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
    EmbeddingKeyMetadata,
    EmbeddingPoolKey,
    TensorSpec,
)

pytestmark = pytest.mark.cpu_test
KEY = EmbeddingKeyMetadata(
    "test", "model", "revision", "encoder", "torch.float32", "v2"
)
SPEC = TensorSpec((2, 2), "torch.float32", 16)


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

    def get_into_ranges(self, pointers, keys, dst_offsets, src_offsets, sizes):
        [ptr], [[key]], [[[dst]]], [[[src]]], [[[size]]] = (
            pointers,
            keys,
            dst_offsets,
            src_offsets,
            sizes,
        )
        obj = self.objects.get(key)
        if obj is None:
            return [[[-704]]]
        ctypes.memmove(ptr + dst, obj[src : src + size], size)
        return [[[size]]]


@pytest.fixture
def backend():
    instance = store_backend.MooncakeEmbeddingStoreBackend(
        store_client.MooncakeEmbeddingStoreClient(MemoryStore()),
        KEY,
        max_pending_items=32,
        max_pending_bytes=2 * 1024**3,
    )
    yield instance
    # Some tests deliberately poison I/O; always stop the Python executor.
    instance._executor.shutdown(wait=True)


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
        "hit",
        "partial",
        "get-failure",
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
    if mode == "partial":
        hits = {"a", "c"}
    elif mode in ("hit", "get-failure", "lease-expired", "registration-failure"):
        hits = set("abc")
    else:
        hits = set()
    for key in hits:
        backend.store_client.put_tensor(EmbeddingPoolKey(KEY, key), torch.ones(2, 2))
    if mode in ("get-failure", "lease-expired"):
        read = native.get_into_ranges

        def failed_read(pointers, keys, dst, src, sizes):
            if keys == [[EmbeddingPoolKey(KEY, "b").to_string()]] and src == [[[304]]]:
                if mode == "lease-expired":
                    # Native ranged GET checks the lease after the transfer.
                    read(pointers, keys, dst, src, sizes)
                    return [[[-707]]]
                return [[[-704]]]
            return read(pointers, keys, dst, src, sizes)

        native.get_into_ranges = failed_read
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
        return_value=[torch.full((2, 2), 7.0) for _ in missing]
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
            cache.encoder_outputs[k], torch.full((2, 2), 1.0 if k in hits else 7.0)
        )
        for k in "abc"
    )
    assert {
        c.args[1] for c in connector._worker._bind_push_source.call_args_list
    } == set("abc")
    backend.shutdown()
    assert not native.registered
    if mode == "miss":
        other = store_backend.MooncakeEmbeddingStoreBackend(
            store_client.MooncakeEmbeddingStoreClient(native),
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


@pytest.mark.parametrize("mismatch", ["revision", "shape"])
def test_incompatible_output_is_a_miss(backend, mismatch):
    backend.store_client.put_tensor(EmbeddingPoolKey(KEY, "a"), torch.ones(2, 2))
    if mismatch == "revision":
        backend.key_metadata = replace(KEY, model_revision="other")
    expected = replace(SPEC, shape=(4, 1)) if mismatch == "shape" else SPEC
    outputs: dict[str, torch.Tensor] = {}
    assert not backend.resolve_inputs({"a": expected}, outputs, "cpu")
    assert not outputs and not backend.store_client.store.registered


def test_partial_registration_rejection_releases_publication(backend):
    """Rejecting the header must undo the prior tensor registration."""
    native = backend.store_client.store
    register = native.register_buffer
    native.register_buffer = lambda addr, size: (
        -500 if size == 304 else register(addr, size)
    )
    native.batch_put_from_multi_buffers = MagicMock()
    tensor = torch.ones(2, 2)
    owner = weakref.ref(tensor)
    assert backend._enqueue_save("a", tensor)
    future = backend._pending["a"][0]
    with pytest.raises(store_client.EmbeddingStoreOperationError, match="register"):
        future.result(timeout=5)
    del tensor
    gc.collect()
    assert owner() is None and not native.registered
    native.batch_put_from_multi_buffers.assert_not_called()
    backend.reap()
    assert backend._pending_bytes == 0


@pytest.mark.parametrize("operation", ["get", "put", "rejected-put", "unregister"])
def test_native_io_owners(backend, operation):
    """Uncertain I/O retains owners; completed rejection frees them even via Future."""
    native, client = backend.store_client.store, backend.store_client
    tensor = torch.ones(2, 2)
    owner = weakref.ref(tensor)
    rejected = operation == "rejected-put"
    if operation == "get":
        native.get_into_ranges = MagicMock(return_value=[[[-800]]])
        with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
            client.get_tensor_payload(
                EmbeddingPoolKey(KEY, "a"), tensor.data_ptr(), 16, 304, owner=tensor
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
        backend.reap()
        assert backend._pending_bytes == 0 and not native.registered
    else:
        assert owner() is not None and native.registered
        with pytest.raises(store_client.EmbeddingStoreError, match="unconfirmed"):
            client.close()


def test_publication_budget_charges_and_releases_backing_storage(backend):
    tensor = torch.arange(16, dtype=torch.float32)[:4]
    owner = weakref.ref(tensor)
    future: Future[None] = Future()
    backend.max_pending_items = 1
    backend.max_pending_bytes = tensor.nbytes
    with patch.object(backend._executor, "submit", return_value=future):
        assert not backend._enqueue_save("a", tensor)
        backend.max_pending_bytes = 64
        assert backend._enqueue_save("a", tensor)
        backend.max_pending_bytes = 128
        assert not backend._enqueue_save("b", tensor)
    del tensor
    gc.collect()
    assert owner() is not None
    future.set_result(None)
    backend.reap()
    gc.collect()
    assert owner() is None and backend._pending_bytes == 0


def test_completion_race_preserves_failure_and_reclaims_other_items(backend):
    class RacingFuture(Future[None]):
        def done(self):
            observed = super().done()
            if not observed:
                self.set_exception(store_client.EmbeddingStoreError("uncertain"))
            return observed

    racing = RacingFuture()
    other: Future[None] = Future()
    with patch.object(backend._executor, "submit", side_effect=[racing, other]):
        assert backend._enqueue_save("race", torch.ones(2, 2))
        assert backend._enqueue_save("other", torch.ones(2, 2))
    assert not backend._drain_completed()
    assert backend._pending_bytes == 32
    other.set_result(None)
    with pytest.raises(store_client.EmbeddingStoreError):
        backend.reap()
    assert backend._pending_bytes == 16
    with pytest.raises(store_client.EmbeddingStoreError):
        backend._enqueue_save("later", torch.ones(2, 2))


def test_readiness_and_shutdown_do_not_block_p2p_completion(backend):
    ready, waiting, entered = [threading.Event() for _ in range(3)]

    class ReadyEvent:
        def synchronize(self):
            waiting.set()
            assert ready.wait(5)

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
    connector = make_worker(backend)
    connector.bind_connector_metadata(
        ECMooncakeConnectorMetadata(store_candidates={"a": SPEC})
    )
    cache: dict[str, torch.Tensor] = {}
    with (
        patch.object(
            store_backend, "_record_tensor_ready_event", return_value=ReadyEvent()
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

    assert backend.store_client.store.objects[EmbeddingPoolKey(KEY, "a").to_string()][
        304:
    ] == struct.pack("<4f", 0, 1, 2, 3)
