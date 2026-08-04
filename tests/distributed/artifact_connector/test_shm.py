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

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactRequestMetadata,
    ArtifactRequestOutput,
    ArtifactSchedulerConnector,
)
from vllm.distributed.artifact_connector.routed_experts import (
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_key,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
    ArtifactStoreError,
    BackgroundArtifactStore,
)
from vllm.distributed.artifact_connector.worker import ArtifactWorkerConnector
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


def _make_vllm_config(
    tmp_path,
    *,
    enable_prefix_caching: bool = True,
    max_shm_bytes: int | None = 1 << 20,
):
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=3),
        get_num_experts_per_token=lambda: 2,
        get_num_experts=lambda: 256,
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
    worker._pending_blocks = {}
    worker._emit_cursors = {}
    worker._generation = -1
    worker._shape_per_token = _SHAPE
    worker._dtype = _DTYPE
    worker._pending_requests = {}
    worker._finished_requests = {}
    worker._pending_lock = threading.Lock()
    return worker


def _process_output(worker, metadata, rows, request_ids, num_rejected):
    worker.begin_step(metadata)
    return worker.process_output(metadata, rows, request_ids, num_rejected)


def test_materialize_rejects_invalid_object_size():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = array.tobytes()

    store = Mock()
    store.get.return_value = [payload[:-1]]
    with pytest.raises(ArtifactCorruptionError, match="size"):
        materialize_routed_experts(
            store,
            ["key"],
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        )


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
    blocks = buffer.capture("request", 0, logical)
    publish_routed_experts(
        store,
        artifact_namespace="0",
        batches=[(hashes, blocks)],
        block_size=_BLOCK_SIZE,
    )

    keys = [routed_experts_key(block_hash, "0") for block_hash in hashes]
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
    assert worker._pending_blocks["request"][0][0] == 0

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
    assert "request" not in worker._pending_blocks
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


def test_worker_retains_tail_until_inflight_output_finishes(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    assert worker._buffer is not None
    rows = np.zeros((2, *_SHAPE), dtype=_DTYPE)
    worker._buffer.capture("request", 0, rows)
    worker._pending_requests["request"] = 1

    cleanup = ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {"request": []})
    worker.begin_step(cleanup)
    np.testing.assert_array_equal(worker._buffer.read("request", 0, 2), rows)

    inflight = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 2, 1, 0, True, [b"a" * 32])],
        {},
    )
    worker.output_finished(inflight)
    with pytest.raises(RuntimeError, match="missing request"):
        worker._buffer.read("request", 0, 2)
    worker.close()


def test_worker_fails_if_request_finishes_before_block_gets_kv_hash(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    worker._pending_blocks["request"] = [
        (0, np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE))
    ]

    with pytest.raises(RuntimeError, match="uncommitted artifact blocks"):
        worker.begin_step(
            ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {"request": []})
        )
    worker._pending_blocks.clear()
    worker.close()


def test_worker_discards_uncommitted_blocks_for_aborted_request(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    worker._pending_blocks["request"] = [
        (0, np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE))
    ]
    assert worker._buffer is not None
    worker._buffer.capture("request", _BLOCK_SIZE, np.zeros((1, *_SHAPE)))

    worker.begin_step(ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {"request": None}))

    assert "request" not in worker._pending_blocks
    with pytest.raises(RuntimeError, match="missing request"):
        worker._buffer.read("request", _BLOCK_SIZE, _BLOCK_SIZE + 1)
    worker.close()


def test_worker_commits_pending_block_with_finished_request_hash(tmp_path):
    worker = _make_worker(tmp_path, 1)
    worker._generation = 0
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    worker._pending_requests["request"] = 1

    block_hash = b"a" * 32
    worker.begin_step(
        ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {"request": [block_hash]})
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

    assert list(metadata.finished_requests[request.request_id]) == block_hashes


def test_scheduler_connector_hash_snapshots_are_self_contained(tmp_path):
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
    assert list(second.requests[0].block_hashes) == [b"a" * 32, b"b" * 32]


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

    assert scheduler.reset_prefix_cache() is reset_successful
    if reset_successful:
        scheduler.artifact_connector.reset.assert_called_once_with()
    else:
        scheduler.artifact_connector.reset.assert_not_called()
