# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactRequestMetadata,
    ArtifactRequestOutput,
    ArtifactSchedulerConnector,
    PackedBlockHashes,
)
from vllm.distributed.artifact_connector.routed_experts import (
    RoutedExpertsArtifactBuffer,
    get_routing_shape_and_dtype,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_key,
)
from vllm.distributed.artifact_connector.shm import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
    ArtifactStoreError,
    BackgroundArtifactStore,
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.artifact_connector.worker import (
    ArtifactWorkerConnector,
    _WorkerRequestState,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test

_SHAPE = (3, 2)
_DTYPE = np.dtype("uint8")
_BLOCK_SIZE = 4


class _BlockingArtifactStore:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.objects: dict[str, bytes] = {}
        self.closed = False

    def put(self, objects: list[ArtifactObject]) -> None:
        self.started.set()
        self.release.wait()
        self.objects.update((obj.key, obj.payload) for obj in objects)

    def get(self, keys: list[str]) -> list[bytes]:
        return [self.objects[key] for key in keys]

    def close(self) -> None:
        self.closed = True


def test_background_store_put_is_async_and_ordered():
    underlying = _BlockingArtifactStore()
    store = BackgroundArtifactStore(underlying, max_pending_batches=2)

    store.put([ArtifactObject("key", b"value")])
    assert underlying.started.wait(timeout=1)
    assert "key" not in underlying.objects

    underlying.release.set()
    assert store.get(["key"]) == [b"value"]
    store.close()
    assert underlying.closed


def test_background_store_surfaces_publication_failure():
    underlying = Mock()
    underlying.put.side_effect = ArtifactCapacityError("full")
    store = BackgroundArtifactStore(underlying, max_pending_batches=2)

    store.put([ArtifactObject("key", b"value")])
    with pytest.raises(ArtifactStoreError, match="publication failed"):
        store.get(["key"])
    with pytest.raises(ArtifactStoreError, match="publication failed"):
        store.close()


def test_background_store_close_does_not_race_blocked_put():
    underlying = _BlockingArtifactStore()
    store = BackgroundArtifactStore(underlying, max_pending_batches=1)
    store.put([ArtifactObject("first", b"1")])
    assert underlying.started.wait(timeout=1)
    store.put([ArtifactObject("second", b"2")])

    producer = threading.Thread(
        target=store.put,
        args=([ArtifactObject("third", b"3")],),
    )
    producer.start()
    closer = threading.Thread(target=store.close)
    closer.start()
    underlying.release.set()

    producer.join(timeout=2)
    closer.join(timeout=2)
    assert not producer.is_alive()
    assert not closer.is_alive()
    assert underlying.objects == {"first": b"1", "second": b"2", "third": b"3"}
    assert underlying.closed


def _make_vllm_config(
    tmp_path,
    *,
    enable_prefix_caching: bool = True,
    max_shm_bytes: int | None = 1 << 20,
):
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=3),
        get_num_experts_per_tok=lambda: 2,
        get_num_experts=lambda: 256,
        get_total_num_hidden_layers=lambda: 3,
        max_model_len=4096,
    )
    return SimpleNamespace(
        artifact_config=SimpleNamespace(
            enabled=True,
            enable_return_routed_experts=True,
            shm_dir=str(tmp_path),
            max_shm_bytes=max_shm_bytes,
            shm_ttl_seconds=60,
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0, rank=0),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        model_config=model_config,
        instance_id="instance",
    )


def test_routed_experts_shape_uses_model_arch_config(tmp_path):
    config = _make_vllm_config(tmp_path)

    assert get_routing_shape_and_dtype(config) == ((3, 2), "uint8")


def _make_connector(tmp_path, *, enable_prefix_caching: bool = True):
    return ArtifactSchedulerConnector(
        _make_vllm_config(
            tmp_path,
            enable_prefix_caching=enable_prefix_caching,
        ),
        block_size=_BLOCK_SIZE,
    )


def _make_worker(tmp_path, max_num_seqs: int) -> ArtifactWorkerConnector:
    store = BackgroundArtifactStore(_make_store(tmp_path), max_pending_batches=2)
    worker = object.__new__(ArtifactWorkerConnector)
    worker._store = store
    worker._buffer = RoutedExpertsArtifactBuffer(
        _DTYPE, _SHAPE, _BLOCK_SIZE, max_num_seqs, max_num_seqs * _BLOCK_SIZE
    )
    worker._requests = {}
    worker._generation = -1
    worker._shape_per_token = _SHAPE
    worker._dtype = _DTYPE
    worker._lock = threading.Lock()
    return worker


def _process_output(worker, metadata, rows, request_ids, num_rejected):
    worker.begin_step(metadata)
    return worker.process_output(metadata, rows, request_ids, num_rejected)


def test_worker_rejects_mismatched_capture_shape(tmp_path):
    worker = _make_worker(tmp_path, 1)
    metadata = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 1, 0, True, [])],
        {},
    )

    with pytest.raises(RuntimeError, match="capture profile changed"):
        _process_output(
            worker,
            metadata,
            np.zeros((1, _SHAPE[0], _SHAPE[1] + 1), dtype=_DTYPE),
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_materialize_rejects_invalid_object_size(tmp_path):
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = array.tobytes()

    store = _make_store(tmp_path)
    store.put([ArtifactObject("key", payload[:-1])])
    with pytest.raises(ArtifactCorruptionError, match="size"):
        materialize_routed_experts(
            store,
            ["key"],
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        )
    store.close()


def test_logical_buffer_handles_overlap_and_release():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,), 4, 2, 8)
    assert (
        buffer.capture("request", 4, np.arange(4, 7, dtype=np.int32).reshape(-1, 1))
        == []
    )
    completed = buffer.capture(
        "request", 6, np.array([[60], [70], [80]], dtype=np.uint8)
    )

    np.testing.assert_array_equal(
        completed[0][1].ravel(),
        [4, 5, 60, 70],
    )
    buffer.release_block(completed[0][1])
    np.testing.assert_array_equal(buffer.read("request", 8, 9).ravel(), [80])


def test_logical_buffer_captures_one_row_per_decode_step():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, 3, 2)

    completed = []
    for step in range(_BLOCK_SIZE):
        completed += buffer.capture("request", step, logical[step : step + 1])

    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0][1], logical)


def _make_store(tmp_path, *, max_bytes: int = 1 << 20, instance="instance"):
    return LocalSharedMemoryArtifactStore(
        str(tmp_path),
        instance,
        0,
        max_bytes=max_bytes,
        ttl_seconds=60,
    )


def test_publish_routed_experts_publishes_full_blocks(tmp_path):
    store = _make_store(tmp_path)
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8)
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    hashes = [b"a" * 32, b"b" * 32]
    keys = [routed_experts_key(block_hash, "0") for block_hash in hashes]
    blocks = buffer.capture("request", 0, logical)
    publish_routed_experts(
        store,
        batches=[(keys, blocks)],
        block_size=_BLOCK_SIZE,
    )

    np.testing.assert_array_equal(
        materialize_routed_experts(
            store,
            keys,
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        ),
        logical,
    )
    store.close()


def test_worker_data_plane_publishes_blocks_and_reuses_prefix(tmp_path):
    worker = _make_worker(tmp_path, 2)

    hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    logical = np.arange(10 * 3 * 2, dtype=np.uint8).reshape(10, 3, 2)
    first = ArtifactConnectorMetadata(
        generation=0,
        block_size=_BLOCK_SIZE,
        requests=[ArtifactRequestMetadata("first", 0, 8, 0, True, hashes)],
        finished_requests={},
    )
    output = _process_output(worker, first, logical[:8], ["first"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output.requests["first"].rows, logical[:8])

    second = ArtifactConnectorMetadata(
        generation=0,
        block_size=_BLOCK_SIZE,
        requests=[ArtifactRequestMetadata("second", 8, 2, 0, True, hashes)],
        finished_requests={"first": hashes},
    )
    output = _process_output(worker, second, logical[8:], ["second"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output.requests["second"].rows, logical)
    worker.close()


def test_worker_publishes_entire_batch_before_materializing_prefix(tmp_path):
    worker = _make_worker(tmp_path, 2)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    metadata = ArtifactConnectorMetadata(
        generation=0,
        block_size=_BLOCK_SIZE,
        requests=[
            ArtifactRequestMetadata("consumer", 4, 4, 0, True, hashes),
            ArtifactRequestMetadata("producer", 0, 4, 0, True, hashes),
        ],
        finished_requests={},
    )

    output = _process_output(
        worker,
        metadata,
        np.concatenate((logical[4:], logical[:4])),
        ["consumer", "producer"],
        np.array([0, 0]),
    )

    assert output is not None
    np.testing.assert_array_equal(output.requests["consumer"].rows, logical)
    np.testing.assert_array_equal(output.requests["producer"].rows, logical[:4])
    worker.close()


def test_worker_defers_full_block_until_kv_hash_arrives(tmp_path):
    worker = _make_worker(tmp_path, 1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 4, 0, True, [])],
        {},
    )

    output = _process_output(worker, first, logical[:4], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output.requests["request"].rows, logical[:4])
    assert worker._requests[("request", 0)].pending_blocks[0][0] == 0

    block_hash = b"a" * 32
    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 4, 1, 4, True, [block_hash])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output.requests["request"].rows, logical[4:])
    assert not worker._requests[("request", 0)].pending_blocks
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            [routed_experts_key(block_hash, "0")],
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        ),
        logical[:4],
    )
    worker.close()


def test_worker_publishes_full_block_before_restarted_epoch(tmp_path):
    worker = _make_worker(tmp_path, 1)
    block_hash = b"a" * 32
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 4, 0, False, [])],
        {},
    )
    output = _process_output(worker, first, logical[:4], ["request"], np.array([0]))
    assert not output.requests

    restarted = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 4, 1, 0, True, [block_hash], epoch=1)],
        {("request", 0): PackedBlockHashes(block_hash, len(block_hash))},
    )
    output = _process_output(worker, restarted, logical[4:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output.requests["request"].rows, logical)
    worker.close()


def test_worker_does_not_rematerialize_emitted_rows(tmp_path):
    worker = _make_worker(tmp_path, 1)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 4, 0, True, hashes)],
        {},
    )
    output = _process_output(worker, first, logical[:4], ["request"], np.array([0]))
    assert output is not None

    worker._materialize = Mock(wraps=worker._materialize)
    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        # Async scheduling may build this before the scheduler consumes first.
        [ArtifactRequestMetadata("request", 4, 4, 0, True, hashes)],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output.requests["request"].rows, logical[4:])
    worker._materialize.assert_not_called()
    worker.close()


def test_worker_keeps_only_chunked_prefill_tail(tmp_path):
    worker = _make_worker(tmp_path, 1)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 6, 0, False, hashes)],
        {},
    )
    output = _process_output(worker, first, logical[:6], ["request"], np.array([0]))
    assert output is not None and not output.requests
    assert worker._buffer._rows.shape[1] == _BLOCK_SIZE

    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 6, 2, 0, True, hashes)],
        {},
    )
    output = _process_output(worker, second, logical[6:], ["request"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output.requests["request"].rows, logical)
    worker.close()


def test_worker_emits_mid_block_chunked_prefill(tmp_path):
    worker = _make_worker(tmp_path, 1)
    logical = np.arange(3 * 3 * 2, dtype=np.uint8).reshape(3, 3, 2)

    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 2, 0, False, [])],
        {},
    )
    output = _process_output(worker, first, logical[:2], ["request"], np.array([0]))
    assert not output.requests

    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 2, 1, 0, True, [])],
        {},
    )
    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output.requests["request"].rows, logical)
    worker.close()


def test_worker_invalidates_prefix_sharing_inflight_request(tmp_path):
    worker = _make_worker(tmp_path, 2)
    rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [
            ArtifactRequestMetadata(
                "first", 0, 4, 0, True, [b"a" * 32], kv_block_ids=[7]
            )
        ],
        {},
    )
    shared = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [
            ArtifactRequestMetadata(
                "shared", 0, 4, 0, True, [b"a" * 32], kv_block_ids=[7]
            )
        ],
        {},
    )
    worker.begin_step(first)
    worker.begin_step(shared)

    failed = worker.process_output(
        first, rows, ["first"], np.array([0]), invalid_block_ids={7}
    )
    later = worker.process_output(shared, rows, ["shared"], np.array([0]))

    assert failed.invalid_requests == {("first", 0)}
    assert later.invalid_requests == {("shared", 0)}
    assert not later.requests
    worker.close()


def test_worker_emits_published_block_and_mid_block_tail(tmp_path):
    worker = _make_worker(tmp_path, 1)
    logical = np.arange(7 * 3 * 2, dtype=np.uint8).reshape(7, 3, 2)

    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 6, 0, False, [b"a" * 32])],
        {},
    )
    output = _process_output(worker, first, logical[:6], ["request"], np.array([0]))
    assert not output.requests

    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 6, 1, 0, True, [b"a" * 32])],
        {},
    )
    output = _process_output(worker, second, logical[6:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output.requests["request"].rows, logical)
    worker.close()


def test_worker_output_does_not_alias_released_tail_buffer(tmp_path):
    worker = _make_worker(tmp_path, 1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 2, 0, False, [b"a" * 32])],
        {},
    )
    _process_output(worker, first, logical[:2], ["request"], np.array([0]))
    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 2, 2, 0, True, [b"a" * 32])],
        {},
    )

    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))
    emitted = output.requests["request"].rows
    expected = logical.copy()

    replacement = np.full_like(logical, 99)
    reuse = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("reuse", 0, 4, 0, False, [b"b" * 32])],
        {},
    )
    _process_output(worker, reuse, replacement, ["reuse"], np.array([0]))

    np.testing.assert_array_equal(emitted, expected)
    worker.close()


def test_worker_does_not_publish_artifact_for_failed_kv_load(tmp_path):
    worker = _make_worker(tmp_path, 1)
    block_hash = b"a" * 32
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    metadata = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [
            ArtifactRequestMetadata(
                "request",
                0,
                4,
                0,
                True,
                [block_hash],
                kv_block_ids=[7],
            )
        ],
        {},
    )

    worker.begin_step(metadata)
    output = worker.process_output(
        metadata,
        logical,
        ["request"],
        np.array([0]),
        invalid_block_ids={7},
    )

    assert not output.requests
    assert output.invalid_requests == {("request", 0)}

    # A later async frame from the same request epoch is invalid too, even if
    # the KV connector reported the failed load only on the first frame.
    output = worker.process_output(
        metadata,
        logical,
        ["request"],
        np.array([0]),
    )
    assert not output.requests
    assert worker._store is not None
    with pytest.raises(ArtifactNotFoundError):
        worker._store.get([routed_experts_key(block_hash, "0")])
    worker.close()


def test_worker_unions_invalid_blocks_across_tp(monkeypatch):
    worker = object.__new__(ArtifactWorkerConnector)
    worker._sync_invalid_blocks = True
    cpu_group = object()
    tp_group = SimpleNamespace(world_size=2, cpu_group=cpu_group)
    monkeypatch.setattr(
        "vllm.distributed.artifact_connector.worker.get_tp_group",
        lambda: tp_group,
    )

    def all_gather_object(gathered, local, *, group):
        assert local == {7}
        assert group is cpu_group
        gathered[:] = [{7}, {9}]

    monkeypatch.setattr(
        "vllm.distributed.artifact_connector.worker.torch.distributed.all_gather_object",
        all_gather_object,
    )

    assert worker.sync_invalid_block_ids({7}) == {7, 9}


def test_worker_excludes_rejected_speculative_rows(tmp_path):
    worker = _make_worker(tmp_path, 1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    metadata = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 5, 0, True, [b"a" * 32])],
        {},
    )

    output = _process_output(worker, metadata, logical, ["request"], np.array([1]))

    np.testing.assert_array_equal(output.requests["request"].rows, logical[:4])
    worker.close()


def test_worker_retains_tail_until_inflight_output_finishes(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    assert worker._buffer is not None
    rows = np.zeros((2, *_SHAPE), dtype=_DTYPE)
    worker._buffer.capture(("request", 0), 0, rows)
    worker._requests[("request", 0)] = _WorkerRequestState(pending_outputs=1)

    cleanup = ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {("request", 0): []})
    worker.begin_step(cleanup)
    np.testing.assert_array_equal(worker._buffer.read(("request", 0), 0, 2), rows)

    inflight = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 2, 1, 0, True, [b"a" * 32])],
        {},
    )
    worker.output_finished(inflight)
    with pytest.raises(RuntimeError, match="missing request"):
        worker._buffer.read(("request", 0), 0, 2)
    worker.close()


def test_worker_discards_inflight_output_from_old_generation(tmp_path):
    worker = _make_worker(tmp_path, 1)
    old = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("old", 0, 1, 0, True, [b"a" * 32])],
        {},
    )
    worker.begin_step(old)
    worker._requests[("old", 0)].pending_outputs = 1

    new = ArtifactConnectorMetadata(
        1,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("new", 0, 1, 0, True, [b"b" * 32])],
        {},
    )
    worker.begin_step(new)

    output = worker.process_output(
        old,
        np.zeros((1, *_SHAPE), dtype=_DTYPE),
        ["old"],
        np.array([0]),
    )
    worker.output_finished(old)

    assert not output.requests
    assert worker._generation == 1
    assert worker._requests[("new", 0)].block_hashes == [b"b" * 32]
    assert not worker._requests[("new", 0)].pending_outputs
    worker.close()


def test_worker_merges_block_hash_deltas(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker.begin_step(
        ArtifactConnectorMetadata(
            0,
            _BLOCK_SIZE,
            [ArtifactRequestMetadata("request", 0, 1, 0, True, [b"a" * 32])],
            {},
        )
    )
    worker.begin_step(
        ArtifactConnectorMetadata(
            0,
            _BLOCK_SIZE,
            [
                ArtifactRequestMetadata(
                    "request", 1, 1, 0, True, [b"b" * 32], block_hash_start=1
                )
            ],
            {},
        )
    )

    assert worker._requests[("request", 0)].block_hashes == [
        b"a" * 32,
        b"b" * 32,
    ]
    worker.close()


def test_worker_discards_finished_block_without_kv_hash(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    worker._requests[("request", 0)] = _WorkerRequestState(
        pending_blocks=[(0, np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE))]
    )

    worker.begin_step(
        ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {("request", 0): []})
    )
    assert ("request", 0) not in worker._requests
    worker.close()


def test_worker_discards_uncommitted_blocks_for_aborted_request(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    worker._requests[("request", 0)] = _WorkerRequestState(
        pending_blocks=[(0, np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE))]
    )
    assert worker._buffer is not None
    worker._buffer.capture(("request", 0), _BLOCK_SIZE, np.zeros((1, *_SHAPE)))

    worker.begin_step(
        ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {("request", 0): None})
    )

    assert ("request", 0) not in worker._requests
    with pytest.raises(RuntimeError, match="missing request"):
        worker._buffer.read(("request", 0), _BLOCK_SIZE, _BLOCK_SIZE + 1)
    worker.close()


def test_worker_commits_pending_block_with_finished_request_hash(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    worker._requests[("request", 0)] = _WorkerRequestState(pending_outputs=1)

    block_hash = b"a" * 32
    worker.begin_step(
        ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {("request", 0): [block_hash]})
    )
    inflight = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, _BLOCK_SIZE, 0, True, [])],
        {},
    )
    _process_output(worker, inflight, logical, ["request"], np.array([0]))
    worker.output_finished(inflight)

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            [routed_experts_key(block_hash, "0")],
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        ),
        logical,
    )
    worker.close()


def test_worker_fails_when_cached_artifact_is_missing(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    metadata = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [
            ArtifactRequestMetadata(
                "request",
                8,
                1,
                0,
                True,
                [b"a" * 32, b"b" * 32],
            )
        ],
        {},
    )

    with pytest.raises(ArtifactNotFoundError):
        _process_output(
            worker,
            metadata,
            np.zeros((1, *_SHAPE), dtype=_DTYPE),
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_store_rejects_oversized_batch_without_partial_write(tmp_path):
    store = _make_store(tmp_path, max_bytes=5)
    store.put([ArtifactObject("retained", b"r")])

    with pytest.raises(ArtifactCapacityError):
        store.put([ArtifactObject("first", b"111"), ArtifactObject("second", b"222")])

    assert store.get(["retained"]) == [b"r"]
    with pytest.raises(ArtifactNotFoundError):
        store.get(["first"])
    store.close()


def test_store_lru_and_immutable_put(tmp_path):
    store = _make_store(tmp_path, max_bytes=8)
    store.put([ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")])
    assert store.get(["first"]) == [b"1111"]

    store.put([ArtifactObject("first", b"xxxx"), ArtifactObject("third", b"3333")])

    assert store.get(["first", "third"]) == [b"1111", b"3333"]
    with pytest.raises(ArtifactNotFoundError, match="Increase artifact_config"):
        store.get(["second"])
    store.close()


def test_store_compacts_in_offset_order(tmp_path):
    store = _make_store(tmp_path, max_bytes=12)
    store.put(
        [
            ArtifactObject("first", b"1111"),
            ArtifactObject("second", b"2222"),
            ArtifactObject("third", b"3333"),
        ]
    )
    assert store.get(["second"]) == [b"2222"]
    store.put([ArtifactObject("fourth", b"4444")])

    store._compact()

    assert store.get(["second", "third", "fourth"]) == [
        b"2222",
        b"3333",
        b"4444",
    ]
    store.close()


def test_ttl_collects_only_inactive_store(tmp_path):
    stale = LocalSharedMemoryArtifactStore(
        str(tmp_path), "stale", 0, max_bytes=100, ttl_seconds=1
    )
    stale.put([ArtifactObject("key", b"value")])
    stale_root = stale.root
    stale.close()
    old_time = time.time() - 5
    for path in [
        stale.arena_path,
        stale_root / ".writer.lock",
        stale_root,
    ]:
        os.utime(path, (old_time, old_time))

    live = LocalSharedMemoryArtifactStore(
        str(tmp_path), "live", 0, max_bytes=100, ttl_seconds=1
    )
    assert not stale_root.exists()
    live.close()


def test_writer_lock_retries_after_inode_replacement(tmp_path, monkeypatch):
    store = object.__new__(LocalSharedMemoryArtifactStore)
    store.root = tmp_path / "store"
    real_flock = fcntl.flock
    first_call = True

    def replace_on_first_flock(fd, operation):
        nonlocal first_call
        real_flock(fd, operation)
        if first_call:
            first_call = False
            (store.root / ".writer.lock").unlink()
            (store.root / ".writer.lock").touch(mode=0o600)

    monkeypatch.setattr(fcntl, "flock", replace_on_first_flock)
    fd = store._acquire_writer_lock()
    try:
        opened = os.fstat(fd)
        current = (store.root / ".writer.lock").stat()
        assert (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
    finally:
        os.close(fd)


def test_store_reports_live_writer_collision(tmp_path):
    store = _make_store(tmp_path)
    try:
        with pytest.raises(ArtifactStoreError, match="live writer"):
            _make_store(tmp_path)
    finally:
        store.close()


def _scheduler_request(
    request_id: str,
    block_hashes: list[bytes],
    *,
    num_tokens: int = 10,
    prompt_start: int = 0,
):
    request = Mock()
    request.request_id = request_id
    request.block_hashes = block_hashes
    request.num_tokens = num_tokens
    request.num_computed_tokens = num_tokens
    request.num_in_flight_tokens = 0
    request.num_prompt_tokens = num_tokens
    request.sampling_params = SimpleNamespace(routed_experts_prompt_start=prompt_start)
    return request


def _step_output(
    request_ids: list[str], token_starts: list[int], token_counts: list[int]
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=request_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in request_ids],
        all_token_ids={},
        new_block_ids=[None for _ in request_ids],
        num_computed_tokens=token_starts,
        num_output_tokens=[0 for _ in request_ids],
    )
    output.num_scheduled_tokens = dict(zip(request_ids, token_counts, strict=True))
    output.total_num_scheduled_tokens = sum(token_counts)
    return output


def test_scheduler_connector_builds_worker_metadata_and_forwards_output(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(request)
    scheduler_output = _step_output([request.request_id], [0], [4])

    metadata = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert metadata.generation == 0
    assert metadata.block_size == _BLOCK_SIZE
    assert metadata.requests[0].token_start == 0
    assert list(metadata.requests[0].block_hashes) == [b"a" * 32]

    routing = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    output = ArtifactConnectorOutput({"request": ArtifactRequestOutput(0, routing)})
    np.testing.assert_array_equal(connector.take_output(request, True, output), routing)


def test_scheduler_connector_sends_final_block_hashes(tmp_path):
    connector = _make_connector(tmp_path)
    block_hashes = [b"a" * 32]
    request = _scheduler_request("request", block_hashes)
    connector.request_started(request)
    connector.request_finished(request)
    scheduler_output = _step_output([], [], [])
    scheduler_output.finished_req_ids = {request.request_id}

    metadata = connector.build_connector_meta(scheduler_output, {})

    assert list(metadata.finished_requests[(request.request_id, 0)]) == block_hashes


def test_scheduler_connector_restarts_preempted_request_epoch(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(request)
    first = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    assert first.requests[0].epoch == 0

    preempted = _step_output([], [], [])
    preempted.preempted_req_ids = {request.request_id}
    cleanup = connector.build_connector_meta(
        preempted,
        {request.request_id: request},
    )
    assert list(cleanup.finished_requests[(request.request_id, 0)]) == [b"a" * 32]

    resumed = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    assert resumed.requests[0].epoch == 1
    assert resumed.requests[0].block_hash_start == 0
    assert list(resumed.requests[0].block_hashes) == [b"a" * 32]


def test_scheduler_skips_invalid_inflight_epoch(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(request)
    connector.request_restarted(request)
    request.num_computed_tokens = 4

    output = ArtifactConnectorOutput({}, {(request.request_id, 0)})

    assert connector.take_output(request, True, output) is None


def test_scheduler_consumes_ordered_stale_artifact_outputs(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [], num_tokens=1)
    connector.request_started(request)
    first_rows = np.arange(2 * 3 * 2, dtype=np.uint8).reshape(2, 3, 2)
    second_rows = first_rows + 20
    first = ArtifactConnectorOutput({"request": ArtifactRequestOutput(0, first_rows)})
    second = ArtifactConnectorOutput({"request": ArtifactRequestOutput(1, second_rows)})

    np.testing.assert_array_equal(
        connector.take_output(request, True, first, is_stale=True), first_rows
    )
    np.testing.assert_array_equal(
        connector.take_output(request, True, second, is_stale=True), second_rows[1:]
    )


def test_scheduler_rejects_stale_artifact_token_gap(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [], num_tokens=1)
    connector.request_started(request)
    output = ArtifactConnectorOutput(
        {"request": ArtifactRequestOutput(1, np.zeros((1, 3, 2), dtype=np.uint8))}
    )

    with pytest.raises(RuntimeError, match="token gap"):
        connector.take_output(request, True, output, is_stale=True)


def test_scheduler_restart_truncates_invalid_hash_suffix(tmp_path):
    connector = _make_connector(tmp_path)
    hashes = [b"a" * 32, b"b" * 32]
    request = _scheduler_request("request", hashes, num_tokens=8)
    connector.request_started(request)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [8]),
        {request.request_id: request},
    )

    connector.request_restarted(request, num_valid_tokens=_BLOCK_SIZE)
    cleanup = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert list(cleanup.finished_requests[(request.request_id, 0)]) == hashes[:1]


def test_scheduler_connector_sends_only_new_block_hashes(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=8)
    connector.request_started(request)
    scheduler_output = _step_output([request.request_id], [0], [4])

    first = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )
    request.block_hashes.append(b"b" * 32)
    second = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert list(first.requests[0].block_hashes) == [b"a" * 32]
    assert first.requests[0].block_hash_start == 0
    assert list(second.requests[0].block_hashes) == [b"b" * 32]
    assert second.requests[0].block_hash_start == 1


def test_scheduler_connector_sends_kv_block_ids_once_per_epoch(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=8)
    connector.request_started(request)
    scheduler_output = _step_output([request.request_id], [0], [4])

    assert connector.needs_kv_block_ids(request.request_id)
    first = connector.build_connector_meta(
        scheduler_output,
        {request.request_id: request},
        {request.request_id: [3, 4]},
    )
    assert list(first.requests[0].kv_block_ids) == [3, 4]
    assert not connector.needs_kv_block_ids(request.request_id)

    second = connector.build_connector_meta(
        scheduler_output,
        {request.request_id: request},
        {},
    )
    assert not second.requests[0].kv_block_ids

    connector.request_restarted(request)
    assert connector.needs_kv_block_ids(request.request_id)


def test_scheduler_connector_uses_fixed_size_synthetic_hashes(tmp_path):
    connector = _make_connector(tmp_path, enable_prefix_caching=False)
    request = _scheduler_request("request", [], num_tokens=48)
    connector.request_started(request)

    first = connector.build_connector_meta(
        _step_output([request.request_id], [0], [44]),
        {request.request_id: request},
    )
    second = connector.build_connector_meta(
        _step_output([request.request_id], [44], [4]),
        {request.request_id: request},
    )

    assert first.requests[0].block_hash_start == 0
    assert len(first.requests[0].block_hashes) == 11
    assert {len(block_hash) for block_hash in first.requests[0].block_hashes} == {32}
    assert second.requests[0].block_hash_start == 11
    assert len(second.requests[0].block_hashes) == 1
    assert len(second.requests[0].block_hashes[0]) == 32


def test_scheduler_connector_reset_preserves_emit_cursor(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(request)
    routing = np.zeros((3, *_SHAPE), dtype=_DTYPE)
    request.num_computed_tokens = 3
    connector.take_output(
        request,
        True,
        ArtifactConnectorOutput({"request": ArtifactRequestOutput(0, routing)}),
    )

    connector.reset()
    connector.request_started(request)
    scheduler_output = _step_output([request.request_id], [3], [1])
    metadata = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert metadata.generation == 1
    assert metadata.requests[0].emit_start == 3


@pytest.mark.parametrize("reset_successful", [False, True])
def test_scheduler_resets_artifacts_only_after_successful_kv_reset(reset_successful):
    scheduler = object.__new__(Scheduler)
    scheduler.running = []
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache.return_value = reset_successful
    scheduler.artifact_connector = Mock()
    scheduler.connector = None

    assert scheduler.reset_prefix_cache() is reset_successful
    if reset_successful:
        scheduler.artifact_connector.reset.assert_called_once_with()
    else:
        scheduler.artifact_connector.reset.assert_not_called()


def test_scheduler_requires_explicit_remote_kv_reset():
    scheduler = object.__new__(Scheduler)
    scheduler.running = []
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache.return_value = True
    scheduler.artifact_connector = Mock()
    scheduler.connector = Mock()
    scheduler.connector.reset_cache.return_value = True
    scheduler.log_stats = False

    assert not scheduler.reset_prefix_cache()
    scheduler.kv_cache_manager.reset_prefix_cache.assert_not_called()
    scheduler.connector.reset_cache.assert_not_called()
    scheduler.artifact_connector.reset.assert_not_called()


def test_scheduler_resets_remote_kv_before_artifacts():
    scheduler = object.__new__(Scheduler)
    scheduler.running = []
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache.return_value = True
    scheduler.artifact_connector = Mock()
    scheduler.connector = Mock()
    scheduler.connector.reset_cache.return_value = True
    scheduler.log_stats = False

    assert scheduler.reset_prefix_cache(reset_connector=True)
    scheduler.connector.reset_cache.assert_called_once_with()
    scheduler.artifact_connector.reset.assert_called_once_with()


def test_scheduler_preserves_artifacts_if_remote_kv_reset_fails():
    scheduler = object.__new__(Scheduler)
    scheduler.running = []
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache.return_value = True
    scheduler.artifact_connector = Mock()
    scheduler.connector = Mock()
    scheduler.connector.reset_cache.return_value = False
    scheduler.log_stats = False

    assert not scheduler.reset_prefix_cache(reset_connector=True)
    scheduler.artifact_connector.reset.assert_not_called()


def test_scheduler_fails_closed_if_connector_reset_is_unsupported():
    scheduler = object.__new__(Scheduler)
    scheduler.running = []
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache.return_value = True
    scheduler.artifact_connector = Mock()
    scheduler.connector = Mock()
    scheduler.connector.reset_cache.return_value = None
    scheduler.log_stats = False

    assert not scheduler.reset_prefix_cache(reset_connector=True)
    scheduler.artifact_connector.reset.assert_not_called()


def test_scheduler_explicitly_resets_connector_after_local_reset_failure():
    scheduler = object.__new__(Scheduler)
    scheduler.running = []
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache.return_value = False
    scheduler.artifact_connector = None
    scheduler.connector = Mock()
    scheduler.connector.reset_cache.return_value = True
    scheduler.log_stats = False

    assert not scheduler.reset_prefix_cache(reset_connector=True)
    scheduler.connector.reset_cache.assert_called_once_with()
