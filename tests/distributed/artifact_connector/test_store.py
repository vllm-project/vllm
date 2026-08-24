# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata as _ArtifactConnectorMetadata,
)
from vllm.distributed.artifact_connector.connector import (
    ArtifactRequestOutput,
    ArtifactSchedulerConnector,
    PackedBlockHashes,
)
from vllm.distributed.artifact_connector.routed_experts import (
    RoutedExpertsArtifactBuffer,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactObject,
    ArtifactStoreError,
    BackgroundArtifactStore,
    InProcessArtifactStore,
)
from vllm.distributed.artifact_connector.worker import (
    ArtifactWorkerConnector,
    _WorkerRequestState,
)
from vllm.v1.core.sched.interface import PauseState
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

    def get_concatenated(self, keys: list[str]) -> bytes:
        return b"".join(self.objects[key] for key in keys)

    def close(self) -> None:
        self.closed = True


def test_background_store_put_is_async_and_ordered():
    underlying = _BlockingArtifactStore()
    store = BackgroundArtifactStore(underlying, max_pending_batches=2)

    store.put([ArtifactObject("key", b"value")])
    assert underlying.started.wait(timeout=1)
    assert "key" not in underlying.objects

    underlying.release.set()
    assert store.get_concatenated(["key"]) == b"value"
    store.close()
    assert underlying.closed


def test_background_store_surfaces_publication_failure():
    underlying = Mock()
    underlying.put.side_effect = ArtifactStoreError("full")
    store = BackgroundArtifactStore(underlying, max_pending_batches=2)

    store.put([ArtifactObject("key", b"value")])
    with pytest.raises(ArtifactStoreError, match="publication failed"):
        store.get_concatenated(["key"])
    with pytest.raises(ArtifactStoreError, match="publication failed"):
        store.close()


def test_background_store_preserves_capacity_independent_batches():
    started = threading.Event()
    release = threading.Event()
    batches: list[list[str]] = []
    underlying = Mock()

    def put(objects):
        if not batches:
            started.set()
            release.wait()
        if len(objects) > 1:
            raise ArtifactStoreError("batch exceeds capacity")
        batches.append([obj.key for obj in objects])

    underlying.put.side_effect = put
    store = BackgroundArtifactStore(underlying, max_pending_batches=3)
    store.put([ArtifactObject("first", b"1")])
    assert started.wait(timeout=1)
    store.put([ArtifactObject("second", b"2")])
    store.put([ArtifactObject("third", b"3")])
    release.set()

    store.close()

    assert batches == [["first"], ["second"], ["third"]]


def _make_vllm_config(
    *,
    max_bytes: int | None = 1 << 20,
    num_experts: int = 256,
):
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=3),
        get_num_experts_per_tok=lambda: 2,
        get_num_experts=lambda: num_experts,
        get_total_num_hidden_layers=lambda: 3,
        max_model_len=4096,
    )
    return SimpleNamespace(
        artifact_config=SimpleNamespace(
            enabled=True,
            enable_return_routed_experts=True,
            max_bytes=max_bytes,
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0, rank=0),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        model_config=model_config,
        instance_id="instance",
    )


def _make_connector():
    return ArtifactSchedulerConnector()


@dataclass
class _Execution:
    request_id: str
    token_start: int
    num_tokens: int
    emit_start: int
    block_hashes: list[bytes] | PackedBlockHashes


@dataclass
class _TestConnectorMetadata(_ArtifactConnectorMetadata):
    _test_requests: list[_Execution]


def _request_metadata(
    request_id: str,
    token_start: int,
    num_tokens: int,
    emit_start: int,
    block_hashes,
) -> _Execution:
    return _Execution(request_id, token_start, num_tokens, emit_start, block_hashes)


def _metadata(
    generation: int,
    block_size: int,
    requests: list[_Execution],
    finished_requests: dict[str, list[bytes] | PackedBlockHashes],
    hash_updates: dict[str, list[bytes] | PackedBlockHashes] | None = None,
) -> _TestConnectorMetadata:
    del block_size
    block_hashes = {
        request.request_id: request.block_hashes
        for request in requests
        if request.block_hashes
    }
    if hash_updates:
        block_hashes.update(
            (request_id, hashes)
            for request_id, hashes in hash_updates.items()
            if hashes
        )
    block_hashes.update(
        (request_id, hashes)
        for request_id, hashes in finished_requests.items()
        if hashes
    )
    return _TestConnectorMetadata(
        generation,
        {request.request_id: request.emit_start for request in requests},
        block_hashes,
        set(finished_requests),
        requests,
    )


def _execution_ranges(metadata, request_ids):
    by_request = {request.request_id: request for request in metadata._test_requests}
    token_starts = tuple(
        by_request[request_id].token_start for request_id in request_ids
    )
    num_tokens = tuple(by_request[request_id].num_tokens for request_id in request_ids)
    return token_starts, num_tokens


def _make_worker(
    max_num_seqs: int,
    max_num_batched_tokens: int | None = None,
    max_concurrent_batches: int = 2,
) -> ArtifactWorkerConnector:
    store = _make_store(object_nbytes=_BLOCK_SIZE * int(np.prod(_SHAPE)))
    worker = object.__new__(ArtifactWorkerConnector)
    worker._store = store
    worker._buffer = RoutedExpertsArtifactBuffer(
        _DTYPE,
        _SHAPE,
        _BLOCK_SIZE,
        max_num_seqs,
        max_num_batched_tokens or max_num_seqs * _BLOCK_SIZE,
        max_concurrent_batches,
    )
    worker._requests = {}
    worker._generation = 0
    return worker


def test_worker_rejects_metadata_generation_rollback():
    worker = _make_worker(1)
    worker.begin_step(_metadata(1, _BLOCK_SIZE, [], {}))

    with pytest.raises(RuntimeError, match="generation moved backwards"):
        worker.begin_step(_metadata(0, _BLOCK_SIZE, [], {}))

    worker.close()


def test_worker_rejects_run_and_finish_in_one_step():
    worker = _make_worker(1)
    request = _request_metadata("request", 0, 1, 0, [])
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [request],
        {"request": []},
    )

    with pytest.raises(RuntimeError, match="cannot run and finish"):
        worker.begin_step(metadata)

    worker.close()


def test_non_output_rank_skips_capture_snapshot():
    worker = object.__new__(ArtifactWorkerConnector)
    worker._store = None
    worker._buffer = None
    worker._capturer = Mock()
    worker._step_metadata = Mock()
    assert worker.prepare_output([], np.array([]), np.array([]), Mock(), Mock()) is None
    worker._capturer.snapshot_routing_data.assert_not_called()


def test_worker_skips_artifacts_for_internal_warmup_step():
    worker = _make_worker(1)
    worker._capturer = Mock()

    worker.begin_step(None)

    assert worker.prepare_output([], np.array([]), np.array([]), Mock(), Mock()) is None
    worker._capturer.snapshot_routing_data.assert_not_called()
    worker.close()


def test_worker_encapsulates_step_output():
    worker = _make_worker(1)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = torch.empty(
        (0, *_SHAPE), dtype=torch.uint8
    )
    metadata = _metadata(0, _BLOCK_SIZE, [], {})

    worker.begin_step(metadata)
    empty = torch.empty(0, dtype=torch.int32)
    output = worker.prepare_output([], np.array([]), np.array([0]), empty, empty)

    assert output == {}
    worker._capturer.snapshot_routing_data.assert_called_once_with(0)
    worker.close()


def test_worker_rejects_invalid_rejected_token_count():
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )
    with pytest.raises(RuntimeError, match="rejected-token count is invalid"):
        _process_output(
            worker,
            metadata,
            np.zeros((1, *_SHAPE), dtype=_DTYPE),
            ["request"],
            np.array([-1]),
            token_starts=(0,),
            num_tokens=(1,),
        )

    worker.close()


def _process_output(
    worker,
    metadata,
    rows,
    request_ids,
    num_rejected,
    num_sampled=None,
    *,
    token_starts=None,
    num_tokens=None,
    query_start_loc=None,
):
    if num_sampled is None:
        num_sampled = np.ones(len(request_ids), dtype=np.int32)
    if token_starts is None or num_tokens is None:
        token_starts, num_tokens = _execution_ranges(metadata, request_ids)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
    worker.begin_step(metadata)
    if query_start_loc is None:
        query_start_loc = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(num_tokens, dtype=np.int32))
        )
    return worker.prepare_output(
        request_ids,
        np.asarray(token_starts),
        query_start_loc,
        torch.from_numpy(num_sampled),
        torch.from_numpy(num_rejected),
    )


def test_worker_ignores_cudagraph_query_padding():
    worker = _make_worker(2)
    logical = np.arange(2 * 3 * 2, dtype=np.uint8).reshape(2, 3, 2)
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [
            _request_metadata("first", 0, 1, 0, []),
            _request_metadata("second", 0, 1, 0, []),
        ],
        {},
    )

    output = _process_output(
        worker,
        metadata,
        logical,
        ["first", "second"],
        np.zeros(2, dtype=np.int32),
        query_start_loc=np.array([0, 1, 2, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(output["first"].rows, logical[:1])
    np.testing.assert_array_equal(output["second"].rows, logical[1:])
    worker.close()


def test_worker_rejects_scheduler_emit_cursor_ahead():
    worker = _make_worker(1)
    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [_request_metadata("request", 0, 1, 0, [])],
            {},
        )
    )

    with pytest.raises(RuntimeError, match="Scheduler emit cursor moved ahead"):
        worker.begin_step(
            _metadata(
                0,
                _BLOCK_SIZE,
                [_request_metadata("request", 1, 1, 1, [])],
                {},
            )
        )

    worker.close()


def test_worker_rejects_terminal_without_request_state():
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [],
        {"missing": []},
    )

    with pytest.raises(KeyError, match="missing"):
        worker.begin_step(metadata)

    worker.close()


def test_worker_rejects_hash_update_without_request_state():
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [],
        {},
        {"missing": PackedBlockHashes(b"a" * 32, 32)},
    )

    with pytest.raises(KeyError, match="missing"):
        worker.begin_step(metadata)

    worker.close()


@pytest.mark.parametrize(
    "routing",
    [
        np.zeros((1, _SHAPE[0], _SHAPE[1] + 1), dtype=_DTYPE),
        np.zeros((1, *_SHAPE), dtype=np.int32),
    ],
    ids=["shape", "dtype"],
)
def test_worker_rejects_mismatched_capture_profile(routing):
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )

    with pytest.raises(RuntimeError, match="capture profile changed"):
        _process_output(
            worker,
            metadata,
            routing,
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_store_rejects_invalid_payload_size():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = array.tobytes()

    store = _make_store(object_nbytes=len(payload))
    with pytest.raises(ArtifactStoreError, match="size"):
        store.put([ArtifactObject("key", payload[:-1])])
    store.close()


def test_logical_buffer_rejects_overlap():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,), 4, 2, 8, 2)
    assert (
        buffer.capture("request", 4, np.arange(4, 7, dtype=np.uint8).reshape(-1, 1))
        == []
    )
    with pytest.raises(RuntimeError, match="not contiguous"):
        buffer.capture("request", 6, np.array([[60], [70]], dtype=np.uint8))


def test_logical_buffer_rejects_gap():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,), 4, 1, 4, 1)
    assert buffer.capture("request", 0, np.array([[0]], dtype=np.uint8)) == []

    with pytest.raises(RuntimeError, match="not contiguous"):
        buffer.capture("request", 2, np.array([[2]], dtype=np.uint8))


def test_logical_buffer_rejects_unbacked_partial_block():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,), 4, 1, 4, 1)

    with pytest.raises(RuntimeError, match="not contiguous"):
        buffer.capture("request", 3, np.array([[3]], dtype=np.uint8))


def test_logical_buffer_captures_one_row_per_decode_step():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, 3, 2)

    completed = []
    for step in range(_BLOCK_SIZE):
        completed += buffer.capture("request", step, logical[step : step + 1])

    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0][1], logical)


def test_logical_buffer_retains_completed_tail_without_copy():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 4, 2)
    logical = np.arange(9 * 3 * 2, dtype=_DTYPE).reshape(9, *_SHAPE)

    assert not buffer.capture("request", 0, logical[:2])
    first = buffer.capture("request", 2, logical[2:6])
    first_retained = buffer.retain_block(first[0][1])
    assert first_retained is first[0][1]
    second = buffer.capture("request", 6, logical[6:])
    second_retained = buffer.retain_block(second[0][1])
    assert second_retained is second[0][1]

    np.testing.assert_array_equal(first_retained, logical[:4])
    np.testing.assert_array_equal(second_retained, logical[4:8])
    buffer.release_block(first_retained)
    buffer.release_block(second_retained)


def test_logical_buffer_copies_borrowed_block_when_retained():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 4, 2)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, *_SHAPE)

    expected = logical.copy()
    completed = buffer.capture("request", 0, logical)
    retained = buffer.retain_block(completed[0][1])
    logical[:] = 0

    assert not np.shares_memory(retained, logical)
    np.testing.assert_array_equal(retained, expected)
    buffer.release_block(retained)


def test_logical_buffer_retains_two_full_steps():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 16, 2)
    logical = np.arange(32 * 3 * 2, dtype=_DTYPE).reshape(32, *_SHAPE)

    retained = [
        buffer.retain_block(rows)
        for _, rows in buffer.capture("first", 0, logical[:16])
    ]
    retained += [
        buffer.retain_block(rows)
        for _, rows in buffer.capture("second", 0, logical[16:])
    ]

    assert len(retained) == 8
    for rows in retained:
        buffer.release_block(rows)


def _make_store(
    *,
    max_bytes: int = 1 << 20,
    object_nbytes: int = 4,
):
    return InProcessArtifactStore(
        max_bytes=max_bytes,
        object_nbytes=object_nbytes,
    )


def test_publish_routed_experts_publishes_full_blocks():
    store = _make_store(object_nbytes=_BLOCK_SIZE * int(np.prod(_SHAPE)))
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    hashes = [b"a" * 32, b"b" * 32]
    keys = routed_experts_keys(hashes, "0")
    blocks = buffer.capture("request", 0, logical)
    publish_routed_experts(
        store,
        batches=[(keys, blocks)],
        block_size=_BLOCK_SIZE,
    )
    expected = logical.copy()
    logical[:] = 0

    np.testing.assert_array_equal(
        materialize_routed_experts(
            store,
            keys,
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        expected,
    )
    store.close()


def test_worker_data_plane_publishes_blocks_and_reuses_prefix():
    worker = _make_worker(2)

    hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    logical = np.arange(10 * 3 * 2, dtype=np.uint8).reshape(10, 3, 2)
    first = _metadata(
        generation=0,
        block_size=_BLOCK_SIZE,
        requests=[_request_metadata("first", 0, 8, 0, hashes)],
        finished_requests={},
    )
    output = _process_output(worker, first, logical[:8], ["first"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["first"].rows, logical[:8])

    second = _metadata(
        generation=0,
        block_size=_BLOCK_SIZE,
        requests=[_request_metadata("second", 8, 2, 0, hashes)],
        finished_requests={"first": []},
    )
    output = _process_output(worker, second, logical[8:], ["second"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["second"].rows, logical)
    worker.close()


def test_worker_rejects_unaligned_initial_capture():
    worker = _make_worker(2)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("first", 0, 8, 0, hashes)],
        {},
    )
    _process_output(worker, first, logical, ["first"], np.array([0]))

    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("second", 7, 1, 0, hashes)],
        {"first": []},
    )
    with pytest.raises(RuntimeError, match="capture is not contiguous"):
        _process_output(worker, second, logical[7:], ["second"], np.array([0]))
    worker.close()


def test_worker_rejects_unbacked_capture_gap():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:1],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 2, 1, 0, [])],
        {},
    )

    with pytest.raises(RuntimeError, match="unbacked token gap"):
        _process_output(
            worker,
            second,
            logical[2:3],
            ["request"],
            np.array([0]),
            np.zeros(1, dtype=np.int32),
        )
    worker.close()


def test_worker_replays_rejected_speculative_gap():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 5, 0, [])],
        {},
    )
    first_output = _process_output(
        worker, first, logical[:5], ["request"], np.array([4])
    )

    queued = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 5, 4, 0, [])],
        {},
    )
    queued_output = _process_output(
        worker,
        queued,
        logical[1:],
        ["request"],
        np.array([3]),
    )

    rolled_back = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 6, 3, 0, [b"a" * 32])],
        {},
    )

    rolled_back_output = _process_output(
        worker,
        rolled_back,
        logical[2:],
        ["request"],
        np.array([0]),
    )

    np.testing.assert_array_equal(first_output["request"].rows, logical[:1])
    assert queued_output["request"].token_start == 1
    np.testing.assert_array_equal(
        queued_output["request"].rows,
        logical[1:2],
    )
    np.testing.assert_array_equal(
        rolled_back_output["request"].rows,
        logical[2:],
    )
    assert worker._requests["request"].capture_cursor == len(logical)
    worker.close()


def test_worker_rejects_overlapping_capture_rows():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 3, 0, [])],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:3],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    stale_overlap = logical[2:].copy()
    stale_overlap[0] += 100
    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 2, 2, 0, [b"a" * 32])],
        {},
    )
    with pytest.raises(RuntimeError, match="capture moved backwards"):
        _process_output(
            worker,
            second,
            stale_overlap,
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_worker_publishes_entire_batch_before_materializing_prefix():
    worker = _make_worker(2)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    metadata = _metadata(
        generation=0,
        block_size=_BLOCK_SIZE,
        requests=[
            _request_metadata("consumer", 4, 4, 0, hashes),
            _request_metadata("producer", 0, 4, 0, hashes),
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
    np.testing.assert_array_equal(output["consumer"].rows, logical)
    np.testing.assert_array_equal(output["producer"].rows, logical[:4])
    worker.close()


def test_worker_defers_full_block_until_kv_hash_arrives():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 4, 0, [])],
        {},
    )

    output = _process_output(
        worker,
        first,
        logical[:4],
        ["request"],
        np.array([0]),
    )

    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical[:4])
    assert worker._requests["request"].pending_blocks[0][0] == 0

    block_hash = b"a" * 32
    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 4, 1, 4, [block_hash])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    assert not worker._requests["request"].pending_blocks
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:4],
    )
    worker.close()


def test_worker_releases_newly_keyed_blocks_before_capture():
    worker = _make_worker(1, max_num_batched_tokens=16)
    connector = _make_connector()
    block_hashes = [b"a" * 32]
    request = _scheduler_request("request", block_hashes, num_tokens=100)
    logical = np.arange(52 * 3 * 2, dtype=np.uint8).reshape(52, 3, 2)

    for token_start, num_tokens in [(0, 4), (4, 16), (20, 16)]:
        request.num_computed_tokens = token_start
        metadata = connector.build_connector_meta(
            _step_output([request.request_id], [token_start], [num_tokens]),
            {request.request_id: request},
        )
        _process_output(
            worker,
            metadata,
            logical[token_start : token_start + num_tokens],
            [request.request_id],
            np.array([0]),
            token_starts=(token_start,),
            num_tokens=(num_tokens,),
        )

    block_hashes.extend(bytes([value]) * 32 for value in range(1, 5))
    request.num_computed_tokens = 36
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [36], [16]),
        {request.request_id: request},
    )
    _process_output(
        worker,
        metadata,
        logical[36:52],
        [request.request_id],
        np.array([0]),
        token_starts=(36,),
        num_tokens=(16,),
    )

    state = worker._requests[request.request_id]
    assert [start for start, _ in state.pending_blocks] == [
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
    ]
    worker.close()


def test_worker_publishes_pending_block_without_forward():
    worker = _make_worker(1)
    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [_request_metadata("request", 0, 1, 0, [])],
            {},
        )
    )
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    assert worker._buffer is not None
    worker._requests["request"].pending_blocks = [
        (0, worker._buffer.retain_block(logical))
    ]
    block_hash = b"a" * 32

    worker.begin_step(_metadata(0, _BLOCK_SIZE, [], {}, {"request": [block_hash]}))

    assert not worker._requests["request"].pending_blocks
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical,
    )
    worker.close()


def test_worker_releases_unscheduled_request_blocks_before_capture():
    worker = _make_worker(2, 20, max_concurrent_batches=1)
    logical = np.arange(48 * 3 * 2, dtype=np.uint8).reshape(48, 3, 2)
    prompt_hashes = [b"a" * 32, b"b" * 32]
    prompt = _metadata(
        0,
        _BLOCK_SIZE,
        [
            _request_metadata("first", 0, 4, 0, prompt_hashes[:1]),
            _request_metadata("second", 0, 4, 0, prompt_hashes[1:]),
        ],
        {},
    )
    _process_output(
        worker,
        prompt,
        logical[:8],
        ["first", "second"],
        np.zeros(2),
        np.zeros(2, dtype=np.int32),
    )

    first_decode = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("first", 4, 20, 0, [])],
        {},
    )
    _process_output(
        worker,
        first_decode,
        logical[8:28],
        ["first"],
        np.zeros(1),
        np.zeros(1, dtype=np.int32),
    )
    assert len(worker._requests["first"].pending_blocks) == 5

    first_hashes = [bytes([value]) * 32 for value in range(2, 7)]
    second_decode = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("second", 4, 20, 0, [])],
        {},
        {"first": first_hashes},
    )
    _process_output(
        worker,
        second_decode,
        logical[28:48],
        ["second"],
        np.zeros(1),
        np.zeros(1, dtype=np.int32),
    )

    assert not worker._requests["first"].pending_blocks
    assert len(worker._requests["second"].pending_blocks) == 5
    worker.close()


def test_worker_rejects_terminal_and_restart_in_same_step():
    worker = _make_worker(1)
    block_hash = b"a" * 32
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 4, 0, [])],
        {},
    )
    output = _process_output(
        worker,
        first,
        logical[:4],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    assert output is not None and not output

    restarted = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 4, 1, 0, [block_hash])],
        {"request": PackedBlockHashes(block_hash, len(block_hash))},
    )
    with pytest.raises(RuntimeError, match="cannot run and finish"):
        worker.begin_step(restarted)
    worker.close()


def test_worker_does_not_rematerialize_emitted_rows():
    worker = _make_worker(1)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 4, 0, hashes)],
        {},
    )
    output = _process_output(worker, first, logical[:4], ["request"], np.array([0]))
    assert output is not None

    assert worker._store is not None
    worker._store.get_concatenated = Mock(wraps=worker._store.get_concatenated)
    second = _metadata(
        0,
        _BLOCK_SIZE,
        # Async scheduling may build this before the scheduler consumes first.
        [_request_metadata("request", 4, 4, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    worker._store.get_concatenated.assert_not_called()
    worker.close()


def test_worker_resumes_from_scheduler_emit_boundary():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    resumed = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 4, 1, 4, [b"a" * 32])],
        {},
    )

    output = _process_output(worker, resumed, logical[4:], ["request"], np.array([0]))

    assert output["request"].token_start == 4
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    assert worker._requests["request"].emit_cursor == 5
    worker.close()


def test_worker_keeps_only_chunked_prefill_tail():
    worker = _make_worker(1)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 6, 0, hashes)],
        {},
    )
    output = _process_output(
        worker,
        first,
        logical[:6],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    assert output is not None and not output
    assert worker._buffer._rows.shape[1] == _BLOCK_SIZE

    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 6, 2, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[6:], ["request"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical)
    worker.close()


def test_worker_emits_mid_block_chunked_prefill():
    worker = _make_worker(1)
    logical = np.arange(3 * 3 * 2, dtype=np.uint8).reshape(3, 3, 2)

    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 2, 0, [])],
        {},
    )
    output = _process_output(
        worker,
        first,
        logical[:2],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    assert output is not None and not output

    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 2, 1, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output["request"].rows, logical)
    worker.close()


def test_worker_emits_published_block_and_mid_block_tail():
    worker = _make_worker(1)
    logical = np.arange(7 * 3 * 2, dtype=np.uint8).reshape(7, 3, 2)

    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 6, 0, [b"a" * 32])],
        {},
    )
    output = _process_output(
        worker,
        first,
        logical[:6],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    assert output is not None and not output

    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 6, 1, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[6:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output["request"].rows, logical)
    worker.close()


def test_worker_output_does_not_alias_released_tail_buffer():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 2, 0, [b"a" * 32])],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:2],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    second = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 2, 2, 0, [])],
        {},
    )

    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))
    emitted = output["request"].rows
    expected = logical.copy()

    replacement = np.full_like(logical, 99)
    reuse = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("reuse", 0, 4, 0, [b"b" * 32])],
        {},
    )
    _process_output(
        worker,
        reuse,
        replacement,
        ["reuse"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )

    np.testing.assert_array_equal(emitted, expected)
    worker.close()


def test_worker_excludes_rejected_speculative_rows():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, 5, 0, [b"a" * 32])],
        {},
    )

    output = _process_output(worker, metadata, logical, ["request"], np.array([1]))

    np.testing.assert_array_equal(output["request"].rows, logical[:4])
    worker.close()


def test_worker_discards_tail_on_finish():
    worker = _make_worker(1)
    worker._generation = 0
    assert worker._buffer is not None
    rows = np.zeros((2, *_SHAPE), dtype=_DTYPE)
    worker._buffer.capture("request", 0, rows)
    worker._requests["request"] = _WorkerRequestState()

    cleanup = _metadata(
        0,
        _BLOCK_SIZE,
        [],
        {"request": []},
    )
    worker.begin_step(cleanup)
    assert "request" not in worker._buffer._requests
    worker.close()


def test_worker_merges_block_hash_deltas():
    worker = _make_worker(1)
    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [_request_metadata("request", 0, 1, 0, [b"a" * 32, b"b" * 32])],
            {},
        )
    )
    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [
                _request_metadata(
                    "request",
                    1,
                    1,
                    0,
                    [b"c" * 32],
                )
            ],
            {},
        )
    )

    assert worker._requests["request"].artifact_keys == [
        "vllm-artifact/0/" + (b"a" * 32).hex(),
        "vllm-artifact/0/" + (b"b" * 32).hex(),
        "vllm-artifact/0/" + (b"c" * 32).hex(),
    ]
    worker.close()


def test_worker_discards_finished_block_without_kv_hash():
    worker = _make_worker(1)
    worker._generation = 0
    worker._requests["request"] = _WorkerRequestState(
        pending_blocks=[(0, np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE))]
    )

    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [],
            {"request": []},
        )
    )
    assert "request" not in worker._requests
    worker.close()


def test_worker_publishes_hashed_block_and_discards_tail_on_request_finish():
    worker = _make_worker(1)
    worker._generation = 0
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    assert worker._buffer is not None
    worker._requests["request"] = _WorkerRequestState(
        pending_blocks=[(0, worker._buffer.retain_block(logical))]
    )
    worker._buffer.capture("request", _BLOCK_SIZE, np.zeros((1, *_SHAPE), dtype=_DTYPE))
    block_hash = b"a" * 32

    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [],
            {"request": [block_hash]},
        )
    )

    assert "request" not in worker._requests
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical,
    )
    assert "request" not in worker._buffer._requests
    worker.close()


def test_worker_commits_pending_block_with_finished_request_hash():
    worker = _make_worker(1)
    worker._generation = 0
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    inflight = _metadata(
        0,
        _BLOCK_SIZE,
        [_request_metadata("request", 0, _BLOCK_SIZE, 0, [])],
        {},
    )
    _process_output(
        worker,
        inflight,
        logical,
        ["request"],
        np.array([0]),
        token_starts=(0,),
        num_tokens=(_BLOCK_SIZE,),
    )

    block_hash = b"a" * 32
    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [],
            {"request": [block_hash]},
        )
    )

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical,
    )
    worker.close()


def test_worker_fails_when_cached_artifact_is_missing():
    worker = _make_worker(1)
    worker._generation = 0
    metadata = _metadata(
        0,
        _BLOCK_SIZE,
        [
            _request_metadata(
                "request",
                8,
                1,
                0,
                [b"a" * 32, b"b" * 32],
            )
        ],
        {},
    )

    with pytest.raises(ArtifactStoreError, match="does not exist"):
        _process_output(
            worker,
            metadata,
            np.zeros((1, *_SHAPE), dtype=_DTYPE),
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_store_rejects_oversized_batch_without_partial_write():
    store = _make_store(max_bytes=6, object_nbytes=3)
    store.put([ArtifactObject("retained", b"rrr")])

    with pytest.raises(ArtifactStoreError, match="cannot retain"):
        store.put(
            [
                ArtifactObject("first", b"111"),
                ArtifactObject("second", b"222"),
                ArtifactObject("third", b"333"),
            ]
        )

    assert store.get_concatenated(["retained"]) == b"rrr"
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        store.get_concatenated(["first"])
    store.close()


def test_store_lru_and_immutable_put():
    store = _make_store(max_bytes=8)
    store.put([ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")])
    assert store.get_concatenated(["first"]) == b"1111"

    store.put([ArtifactObject("first", b"1111"), ArtifactObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(ArtifactStoreError, match="Increase artifact_config"):
        store.get_concatenated(["second"])
    store.close()


def test_store_rejects_access_after_close():
    store = _make_store(max_bytes=8)
    store.close()

    with pytest.raises(RuntimeError, match="closed"):
        store.put([ArtifactObject("first", b"1111")])
    with pytest.raises(RuntimeError, match="closed"):
        store.get_concatenated(["first"])


def test_store_reuses_evicted_slot_without_moving_live_objects():
    store = _make_store(max_bytes=12)
    store.put(
        [
            ArtifactObject("first", b"1111"),
            ArtifactObject("second", b"2222"),
            ArtifactObject("third", b"3333"),
        ]
    )
    assert store.get_concatenated(["second"]) == b"2222"
    store.put([ArtifactObject("fourth", b"4444")])

    assert store.get_concatenated(["second", "third", "fourth"]) == b"222233334444"
    store.close()


def _scheduler_request(
    request_id: str,
    block_hashes: list[bytes],
    *,
    num_tokens: int = 10,
    num_output_tokens: int = 1,
    prompt_start: int = 0,
):
    request = Mock()
    request.request_id = request_id
    request.block_hashes = block_hashes
    request.num_tokens = num_tokens
    request.num_output_tokens = num_output_tokens
    request.num_computed_tokens = num_tokens
    request.num_in_flight_tokens = 0
    request.num_prompt_tokens = num_tokens
    request.sampling_params = SimpleNamespace(routed_experts_prompt_start=prompt_start)
    request.is_finished.return_value = False
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


def test_scheduler_connector_builds_worker_metadata_and_forwards_output():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    request.num_computed_tokens = 0
    scheduler_output = _step_output([request.request_id], [0], [4])

    metadata = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert metadata.generation == 0
    assert metadata.requests == {request.request_id: 4}
    assert list(metadata.block_hashes[request.request_id]) == [b"a" * 32]

    routing = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    output = {"request": ArtifactRequestOutput(0, routing)}
    request.num_computed_tokens = 4
    np.testing.assert_array_equal(connector.take_output(request, output), routing)


def test_scheduler_connector_returns_uncropped_output():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=5, prompt_start=2)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    routing = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    output = {"request": ArtifactRequestOutput(0, routing)}

    np.testing.assert_array_equal(connector.take_output(request, output), routing)


def test_scheduler_connector_sends_finish_after_block_hashes():
    connector = _make_connector()
    block_hashes = [b"a" * 32]
    request = _scheduler_request("request", block_hashes)
    request.num_computed_tokens = 0
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [0]),
        {request.request_id: request},
    )
    request.num_computed_tokens = request.num_tokens
    connector.request_finished(request)
    scheduler_output = _step_output([], [], [])
    scheduler_output.finished_req_ids = {request.request_id}

    metadata = connector.build_connector_meta(scheduler_output, {})

    assert request.request_id in metadata.finished_requests
    assert request.request_id not in metadata.block_hashes


def test_scheduler_connector_sends_all_prompt_hashes_before_terminal():
    connector = _make_connector()
    block_hashes = [bytes([i]) * 32 for i in range(4)]
    request = _scheduler_request("request", block_hashes, num_tokens=16)
    request.num_computed_tokens = 0

    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [_BLOCK_SIZE]),
        {request.request_id: request},
    )
    request.num_computed_tokens = _BLOCK_SIZE
    connector.request_finished(request)
    cleanup = connector.build_connector_meta(_step_output([], [], []), {})

    assert list(metadata.block_hashes[request.request_id]) == block_hashes
    assert request.request_id in cleanup.finished_requests
    assert request.request_id not in cleanup.block_hashes


def test_scheduler_connector_terminal_keeps_computed_prefill_block():
    connector = _make_connector()
    block_hash = b"a" * 32
    request = _scheduler_request(
        "request",
        [block_hash],
        num_tokens=_BLOCK_SIZE,
        num_output_tokens=0,
    )

    connector.build_connector_meta(
        _step_output([request.request_id], [0], [_BLOCK_SIZE]),
        {request.request_id: request},
    )
    connector.request_finished(request)
    cleanup = connector.build_connector_meta(_step_output([], [], []), {})

    assert request.request_id in cleanup.finished_requests
    assert request.request_id not in cleanup.block_hashes


def test_worker_preemption_terminal_keeps_inflight_hashes_for_publish():
    worker = _make_worker(1)
    block_hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(2 * _BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(
        2 * _BLOCK_SIZE, 3, 2
    )
    first = _metadata(
        0,
        _BLOCK_SIZE,
        [
            _request_metadata(
                "request",
                0,
                _BLOCK_SIZE,
                0,
                block_hashes[:1],
            )
        ],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:_BLOCK_SIZE],
        ["request"],
        np.array([0]),
    )
    inflight = _metadata(
        0,
        _BLOCK_SIZE,
        [
            _request_metadata(
                "request",
                _BLOCK_SIZE,
                _BLOCK_SIZE,
                0,
                block_hashes[1:],
            )
        ],
        {},
    )
    _process_output(
        worker,
        inflight,
        logical[_BLOCK_SIZE:],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
        token_starts=(_BLOCK_SIZE,),
        num_tokens=(_BLOCK_SIZE,),
    )
    worker.begin_step(
        _metadata(
            0,
            _BLOCK_SIZE,
            [],
            {"request": []},
        )
    )

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys(block_hashes[:1], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:_BLOCK_SIZE],
    )
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys(block_hashes[1:], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[_BLOCK_SIZE:],
    )
    worker.close()


def test_terminal_request_publishes_hash_discovered_after_last_schedule():
    connector = _make_connector()
    worker = _make_worker(1)
    block_hashes: list[bytes] = []
    request = _scheduler_request("request", block_hashes, num_tokens=5)
    request.num_computed_tokens = 0
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)

    _process_output(
        worker,
        metadata,
        logical,
        [request.request_id],
        np.array([0]),
        token_starts=(0,),
        num_tokens=(_BLOCK_SIZE,),
    )
    assert worker._requests[request.request_id].pending_blocks

    block_hash = b"a" * 32
    block_hashes.append(block_hash)
    request.num_computed_tokens = _BLOCK_SIZE
    connector.request_finished(request)
    cleanup = connector.build_connector_meta(_step_output([], [], []), {})
    worker.begin_step(cleanup)

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical,
    )
    worker.close()


def test_scheduler_connector_recreates_preempted_request_state():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    connector.request_finished(request)
    cleanup = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )
    assert request.request_id in cleanup.finished_requests
    assert request.request_id not in cleanup.block_hashes

    resumed = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    assert list(resumed.block_hashes[request.request_id]) == [b"a" * 32]


def test_scheduler_connector_preemption_keeps_inflight_hashes():
    connector = _make_connector()
    block_hashes = [b"a" * 32, b"b" * 32]
    request = _scheduler_request("request", block_hashes, num_tokens=8)
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [8]),
        {request.request_id: request},
    )
    request.num_in_flight_tokens = _BLOCK_SIZE

    connector.request_finished(request)
    cleanup = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert list(metadata.block_hashes[request.request_id]) == block_hashes
    assert request.request_id in cleanup.finished_requests
    assert request.request_id not in cleanup.block_hashes


def test_scheduler_connector_abort_keeps_inflight_hashes():
    connector = _make_connector()
    block_hashes = [b"a" * 32, b"b" * 32]
    request = _scheduler_request("request", block_hashes, num_tokens=8)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [8]),
        {request.request_id: request},
    )
    request.num_in_flight_tokens = _BLOCK_SIZE

    connector.request_finished(request)
    cleanup = connector.build_connector_meta(_step_output([], [], []), {})

    assert request.request_id in cleanup.finished_requests
    assert request.request_id not in cleanup.block_hashes


def test_scheduler_consumes_ordered_stale_artifact_outputs():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=6)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    first_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    second_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2) + 40
    first = {"request": ArtifactRequestOutput(0, first_rows)}
    second = {"request": ArtifactRequestOutput(4, second_rows)}
    connector.request_finished(request)

    request.num_tokens = 5
    np.testing.assert_array_equal(connector.take_output(request, first), first_rows)
    request.num_tokens = 6
    np.testing.assert_array_equal(
        connector.take_output(request, second), second_rows[:1]
    )


def test_scheduler_rejects_stale_output_without_artifacts():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )

    with pytest.raises(RuntimeError, match="artifact worker output is missing"):
        connector.take_output(
            request,
            {},
        )


def test_scheduler_uses_worker_artifact_output_start():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=3)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [2]),
        {request.request_id: request},
    )
    output = {"request": ArtifactRequestOutput(1, np.zeros((1, 3, 2), dtype=np.uint8))}

    np.testing.assert_array_equal(
        connector.take_output(request, output), output["request"].rows
    )


def test_scheduler_rejects_missing_accepted_artifact_rows():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=2)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )
    output = {
        "request": ArtifactRequestOutput(
            0,
            np.empty((0, *_SHAPE), dtype=_DTYPE),
        )
    }

    with pytest.raises(RuntimeError, match="invalid token range"):
        connector.take_output(request, output)


def test_scheduler_rejects_empty_artifact_output_when_request_is_finished():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=2)
    request.is_finished.return_value = True
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )
    output = {
        "request": ArtifactRequestOutput(
            0,
            np.empty((0, *_SHAPE), dtype=_DTYPE),
        )
    }

    with pytest.raises(RuntimeError, match="invalid token range"):
        connector.take_output(request, output)


def test_scheduler_connector_sends_only_new_block_hashes():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=8)
    scheduler_output = _step_output([request.request_id], [0], [4])

    first = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )
    request.block_hashes.append(b"b" * 32)
    second = connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )

    assert list(first.block_hashes[request.request_id]) == [b"a" * 32]
    assert list(second.block_hashes[request.request_id]) == [b"b" * 32]


def test_scheduler_connector_sends_unscheduled_hash_update():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=4)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    request.block_hashes.append(b"b" * 32)
    request.num_tokens = 9
    request.num_computed_tokens = 8
    metadata = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert not metadata.requests
    assert list(metadata.block_hashes[request.request_id]) == [b"b" * 32]


def test_scheduler_connector_ignores_optimistic_unscheduled_hash_frontier():
    connector = _make_connector()
    block_hashes = [b"a" * 32, b"b" * 32]
    request = _scheduler_request("request", block_hashes, num_tokens=8)
    request.num_computed_tokens = 0
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [8]),
        {request.request_id: request},
    )
    request.num_in_flight_tokens = 8

    metadata = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert not metadata.block_hashes


def test_scheduler_connector_does_not_send_uncomputed_prompt_hashes():
    connector = _make_connector()
    block_hashes = [bytes([i]) * 32 for i in range(4)]
    request = _scheduler_request("request", block_hashes, num_tokens=16)
    request.num_computed_tokens = 0
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    request.num_computed_tokens = 4
    metadata = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert not metadata.block_hashes
    connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )


def test_scheduler_connector_does_not_resend_hashes_while_in_flight():
    connector = _make_connector()
    block_hashes = [bytes([i]) * 32 for i in range(3)]
    request = _scheduler_request("request", block_hashes, num_tokens=12)
    request.num_computed_tokens = 0
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    request.num_computed_tokens = 12
    request.num_in_flight_tokens = 4
    metadata = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert request.request_id not in metadata.block_hashes


def test_scheduler_connector_sends_prompt_hashes_once():
    connector = _make_connector()
    block_hashes = [bytes([i]) * 32 for i in range(4)]
    request = _scheduler_request("request", block_hashes, num_tokens=16)
    request.num_computed_tokens = 0

    first = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    request.num_computed_tokens = 4
    second = connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )

    assert list(first.block_hashes[request.request_id]) == block_hashes
    assert request.request_id not in second.block_hashes


def test_scheduler_connector_reset_derives_emit_start_from_request():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    request.num_prompt_tokens = 4
    request.num_computed_tokens = 0
    before_reset = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    assert list(before_reset.block_hashes[request.request_id]) == [b"a" * 32]

    connector.reset()
    empty = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )
    assert not empty.block_hashes
    request.num_computed_tokens = 4
    scheduler_output = _step_output([request.request_id], [4], [1])
    metadata = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert metadata.generation == 1
    assert metadata.requests[request.request_id] == 4
    assert list(metadata.block_hashes[request.request_id]) == [b"a" * 32]


def test_scheduler_only_resets_running_artifact_requests_after_pause_keep():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = Mock()
    request = SimpleNamespace(num_in_flight_tokens=0)
    scheduler.running = [request]
    scheduler.requests = {"request": request}
    scheduler._pause_state = PauseState.UNPAUSED

    with pytest.raises(RuntimeError, match=r"pause\(mode='keep'\)"):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    scheduler._pause_state = PauseState.PAUSED_ALL
    scheduler.running = []
    request.num_in_flight_tokens = 1
    with pytest.raises(RuntimeError, match="model output is in flight"):
        scheduler.reset_prefix_cache(reset_running_requests=True)


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
