# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
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
from vllm.distributed.artifact_connector.worker import ArtifactWorkerConnector
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

pytestmark = pytest.mark.cpu_test

_SHAPE = (3, 2)
_DTYPE = np.dtype("uint8")
_BLOCK_SIZE = 4


def test_background_store_publishes_without_blocking_caller():
    started = threading.Event()
    release = threading.Event()
    underlying = Mock()

    def put(*_args, **_kwargs):
        started.set()
        release.wait()

    underlying.put.side_effect = put
    store = BackgroundArtifactStore(underlying, max_pending_batches=1)

    store.put([ArtifactObject("key", b"value")])
    assert started.wait(timeout=1)
    release.set()
    store.close()

    underlying.close.assert_called_once_with()


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
class _WorkerStep:
    metadata: ArtifactConnectorMetadata
    executions: list[_Execution]


@dataclass
class _SchedulerRequest:
    request_id: str
    block_hashes: list[bytes]
    num_tokens: int
    num_output_tokens: int = 1
    num_computed_tokens: int = 0
    num_in_flight_tokens: int = 0
    finished: bool = False
    sampling_params: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(routed_experts_prompt_start=0)
    )

    def is_finished(self) -> bool:
        return self.finished


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
    requests: list[_Execution],
    finished_requests: dict[str, list[bytes] | PackedBlockHashes],
    hash_updates: dict[str, list[bytes] | PackedBlockHashes] | None = None,
) -> _WorkerStep:
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
    return _WorkerStep(
        ArtifactConnectorMetadata(
            generation,
            {request.request_id: request.emit_start for request in requests},
            block_hashes,
            tuple(finished_requests),
        ),
        requests,
    )


def _execution_ranges(metadata, request_ids):
    by_request = {request.request_id: request for request in metadata.executions}
    token_starts = tuple(
        by_request[request_id].token_start for request_id in request_ids
    )
    num_tokens = tuple(by_request[request_id].num_tokens for request_id in request_ids)
    return token_starts, num_tokens


def _begin_step(worker: ArtifactWorkerConnector, step: _WorkerStep) -> None:
    worker.begin_step(step.metadata)


def _make_worker(
    max_num_seqs: int,
    max_num_batched_tokens: int | None = None,
    max_concurrent_batches: int = 2,
    max_store_blocks: int | None = None,
) -> ArtifactWorkerConnector:
    object_nbytes = _BLOCK_SIZE * int(np.prod(_SHAPE))
    store = _make_store(
        max_bytes=(
            1 << 20 if max_store_blocks is None else max_store_blocks * object_nbytes
        ),
        object_nbytes=object_nbytes,
    )
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
    worker._pending_output = None
    return worker


def test_worker_rejects_metadata_generation_rollback():
    worker = _make_worker(1)
    _begin_step(worker, _metadata(1, [], {}))

    with pytest.raises(AssertionError, match="generation moved backwards"):
        _begin_step(worker, _metadata(0, [], {}))

    worker.close()


def test_worker_rejects_run_and_finish_in_one_step():
    worker = _make_worker(1)
    request = _request_metadata("request", 0, 1, 0, [])
    metadata = _metadata(
        0,
        [request],
        {"request": []},
    )

    with pytest.raises(AssertionError, match="cannot run and finish"):
        _begin_step(worker, metadata)

    worker.close()


def test_non_output_rank_skips_capture_snapshot():
    worker = object.__new__(ArtifactWorkerConnector)
    worker._store = None
    worker._buffer = None
    worker._capturer = Mock()
    worker._step_metadata = Mock()
    assert worker.prepare_output([], np.array([]), np.array([])) is None
    worker._capturer.snapshot_routing_data.assert_not_called()


def test_worker_skips_artifacts_for_internal_warmup_step():
    worker = _make_worker(1)
    worker._capturer = Mock()

    worker.begin_step(None)

    assert worker.prepare_output([], np.array([]), np.array([])) is None
    worker._capturer.snapshot_routing_data.assert_not_called()
    worker.close()


def test_worker_waits_for_previous_artifact_output_before_next_step():
    worker = _make_worker(1)
    finished = threading.Event()
    worker._pending_output = SimpleNamespace(finished=finished)
    entered = threading.Event()
    returned = threading.Event()

    def begin_step():
        entered.set()
        worker.begin_step(_metadata(0, [], {}).metadata)
        returned.set()

    thread = threading.Thread(target=begin_step)
    thread.start()
    assert entered.wait(1)
    assert not returned.wait(0.05)
    worker._pending_output = None
    finished.set()
    assert returned.wait(1)
    thread.join()
    assert worker._pending_output is None
    worker.close()


def test_worker_rejects_invalid_rejected_token_count():
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )
    with pytest.raises(AssertionError, match="rejected-token count is invalid"):
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
    step,
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
        assert isinstance(step, _WorkerStep)
        token_starts, num_tokens = _execution_ranges(step, request_ids)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
    metadata = step.metadata if isinstance(step, _WorkerStep) else step
    worker.begin_step(metadata)
    if query_start_loc is None:
        query_start_loc = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(num_tokens, dtype=np.int32))
        )
    pending = worker.prepare_output(
        request_ids,
        np.asarray(token_starts),
        query_start_loc,
    )
    assert pending is not None
    try:
        return worker.process_output(
            request_ids,
            pending.token_starts,
            pending.query_start_loc,
            pending.routed_experts.cpu().numpy(),
            num_sampled,
            num_rejected,
        )
    finally:
        pending.complete()


def test_worker_ignores_cudagraph_query_padding():
    worker = _make_worker(2)
    logical = np.arange(2 * 3 * 2, dtype=np.uint8).reshape(2, 3, 2)
    metadata = _metadata(
        0,
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
    _begin_step(
        worker,
        _metadata(
            0,
            [_request_metadata("request", 0, 1, 0, [])],
            {},
        ),
    )

    with pytest.raises(AssertionError, match="Scheduler emit cursor moved ahead"):
        _begin_step(
            worker,
            _metadata(
                0,
                [_request_metadata("request", 1, 1, 1, [])],
                {},
            ),
        )

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
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )

    with pytest.raises(AssertionError, match="capture profile changed"):
        _process_output(
            worker,
            metadata,
            routing,
            ["request"],
            np.array([0]),
        )
    worker.close()


@pytest.mark.parametrize(
    ("initial_start", "initial_rows", "invalid_start"),
    [
        (4, [4, 5, 6], 6),
        (0, [0], 2),
        (None, [], 3),
    ],
    ids=["overlap", "gap", "unaligned-initial-capture"],
)
def test_logical_buffer_rejects_noncontiguous_capture(
    initial_start, initial_rows, invalid_start
):
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,), 4, 1, 4, 1)
    if initial_start is not None:
        rows = np.asarray(initial_rows, dtype=np.uint8).reshape(-1, 1)
        assert buffer.capture("request", initial_start, rows) == []

    with pytest.raises(AssertionError, match="not contiguous"):
        buffer.capture(
            "request", invalid_start, np.array([[invalid_start]], dtype=np.uint8)
        )


def test_logical_buffer_captures_one_row_per_decode_step():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, 3, 2)

    completed = []
    for step in range(_BLOCK_SIZE):
        completed += buffer.capture("request", step, logical[step : step + 1])

    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0][1], logical)


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
        requests=[_request_metadata("first", 0, 8, 0, hashes)],
        finished_requests={},
    )
    output = _process_output(worker, first, logical[:8], ["first"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["first"].rows, logical[:8])

    second = _metadata(
        generation=0,
        requests=[_request_metadata("second", 8, 2, 0, hashes)],
        finished_requests={"first": []},
    )
    output = _process_output(worker, second, logical[8:], ["second"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["second"].rows, logical)
    worker.close()


def test_worker_matches_kv_eviction_order_after_requests_finish():
    worker = _make_worker(2, max_store_blocks=3)
    hashes = [bytes([value]) * 32 for value in range(5)]
    blocks = [
        np.full((_BLOCK_SIZE, *_SHAPE), value, dtype=_DTYPE) for value in range(5)
    ]

    def publish_and_finish(request_id, block_indexes):
        rows = np.concatenate([blocks[index] for index in block_indexes])
        step = _metadata(
            0,
            [
                _request_metadata(
                    request_id,
                    0,
                    len(rows),
                    0,
                    [hashes[index] for index in block_indexes],
                )
            ],
            {},
        )
        _process_output(worker, step, rows, [request_id], np.array([0]))
        _begin_step(worker, _metadata(0, [], {request_id: []}))

    publish_and_finish("s", [0])
    publish_and_finish("ab", [1, 2])
    publish_and_finish("c", [3])
    publish_and_finish("d", [4])

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([hashes[1], hashes[3], hashes[4]], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        np.concatenate([blocks[1], blocks[3], blocks[4]]),
    )
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([hashes[2]], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        )
    worker.close()


def test_worker_rejects_unbacked_capture_gap():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
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
        [_request_metadata("request", 2, 1, 0, [])],
        {},
    )

    with pytest.raises(AssertionError, match="unbacked token gap"):
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
        [_request_metadata("request", 0, 5, 0, [])],
        {},
    )
    first_output = _process_output(
        worker, first, logical[:5], ["request"], np.array([4])
    )

    queued = _metadata(
        0,
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
        [_request_metadata("request", 2, 2, 0, [b"a" * 32])],
        {},
    )
    with pytest.raises(AssertionError, match="capture moved backwards"):
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
    _begin_step(
        worker,
        _metadata(
            0,
            [_request_metadata("request", 0, 1, 0, [])],
            {},
        ),
    )
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    assert worker._buffer is not None
    worker._requests["request"].pending_blocks = [
        (0, worker._buffer.retain_block(logical))
    ]
    block_hash = b"a" * 32

    _begin_step(worker, _metadata(0, [], {}, {"request": [block_hash]}))

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


def test_worker_does_not_rematerialize_emitted_rows():
    worker = _make_worker(1)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    first = _metadata(
        0,
        [_request_metadata("request", 0, 4, 0, hashes)],
        {},
    )
    output = _process_output(worker, first, logical[:4], ["request"], np.array([0]))
    assert output is not None

    assert worker._store is not None
    worker._store.get_concatenated = Mock(wraps=worker._store.get_concatenated)
    second = _metadata(
        0,
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
        [_request_metadata("request", 4, 1, 4, [b"a" * 32])],
        {},
    )

    output = _process_output(worker, resumed, logical[4:], ["request"], np.array([0]))

    assert output["request"].token_start == 4
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    assert worker._requests["request"].emit_cursor == 5
    worker.close()


def test_worker_emits_mid_block_chunked_prefill():
    worker = _make_worker(1)
    logical = np.arange(3 * 3 * 2, dtype=np.uint8).reshape(3, 3, 2)

    first = _metadata(
        0,
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
        [_request_metadata("request", 2, 2, 0, [])],
        {},
    )

    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))
    emitted = output["request"].rows
    expected = logical.copy()

    replacement = np.full_like(logical, 99)
    reuse = _metadata(
        0,
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


def test_worker_merges_block_hash_deltas():
    worker = _make_worker(1)
    _begin_step(
        worker,
        _metadata(
            0,
            [_request_metadata("request", 0, 1, 0, [b"a" * 32, b"b" * 32])],
            {},
        ),
    )
    _begin_step(
        worker,
        _metadata(
            0,
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
        ),
    )

    assert worker._requests["request"].artifact_keys == [
        "vllm-artifact/0/" + (b"a" * 32).hex(),
        "vllm-artifact/0/" + (b"b" * 32).hex(),
        "vllm-artifact/0/" + (b"c" * 32).hex(),
    ]
    worker.close()


def test_worker_finish_publishes_keyed_block_and_discards_request_state():
    worker = _make_worker(1)
    logical = np.arange(2 * _BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(
        2 * _BLOCK_SIZE, 3, 2
    )
    running = _metadata(
        0,
        [_request_metadata("request", 0, len(logical), 0, [])],
        {},
    )
    _process_output(
        worker,
        running,
        logical,
        ["request"],
        np.array([0]),
        np.array([0]),
    )
    block_hash = b"a" * 32

    _begin_step(
        worker,
        _metadata(
            0,
            [],
            {"request": [block_hash]},
        ),
    )

    assert "request" not in worker._requests
    assert worker._buffer is not None
    assert "request" not in worker._buffer._requests
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:_BLOCK_SIZE],
    )
    worker.close()


def test_worker_fails_when_cached_artifact_is_missing():
    worker = _make_worker(1)
    worker._generation = 0
    metadata = _metadata(
        0,
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


def test_store_does_not_evict_referenced_artifacts():
    store = _make_store(max_bytes=8)
    store.put([ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")])
    store.put([], retain_keys=["first"])
    store.put([], retain_keys=["first"], release_keys=["first"])

    store.put([ArtifactObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.put([], release_keys=["first"])
    store.close()


def test_store_terminal_put_preserves_tail_first_eviction_order():
    store = _make_store(max_bytes=8)
    keys = ["first", "second", "third"]
    store.put(
        [ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")],
        retain_keys=keys,
    )

    store.put(
        [ArtifactObject("third", b"3333")],
        release_keys=reversed(keys),
    )

    assert store.get_concatenated(keys[:2]) == b"11112222"
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        store.get_concatenated(["third"])
    store.close()


def test_store_terminal_put_orders_survivors_for_later_eviction():
    store = _make_store(max_bytes=12)
    keys = ["first", "second", "third"]
    store.put(
        [ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")],
        retain_keys=keys,
    )

    store.put(
        [ArtifactObject("third", b"3333")],
        release_keys=reversed(keys),
    )
    store.put([ArtifactObject("fourth", b"4444")])

    assert store.get_concatenated(["first", "second", "fourth"]) == b"111122224444"
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        store.get_concatenated(["third"])
    store.close()


def test_store_terminal_put_keeps_a_shared_incoming_key():
    store = _make_store(max_bytes=8)
    keys = ["first", "second", "third"]
    store.put(
        [ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")],
        retain_keys=(*keys, "third"),
    )

    store.put(
        [ArtifactObject("third", b"3333")],
        release_keys=reversed(keys),
    )

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.put([], release_keys=["third"])
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
    return _SchedulerRequest(
        request_id=request_id,
        block_hashes=block_hashes,
        num_tokens=num_tokens,
        num_output_tokens=num_output_tokens,
        num_computed_tokens=num_tokens,
        sampling_params=SimpleNamespace(routed_experts_prompt_start=prompt_start),
    )


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


def test_scheduler_starts_worker_output_at_requested_prompt_token():
    connector = _make_connector()
    request = _scheduler_request(
        "request",
        [b"a" * 32],
        num_tokens=4,
        num_output_tokens=0,
        prompt_start=2,
    )

    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    assert metadata.requests == {request.request_id: 2}


def test_scheduler_connector_preserves_request_finish_order():
    connector = _make_connector()
    first = _scheduler_request("first", [b"a" * 32])
    second = _scheduler_request("second", [b"b" * 32])
    connector.build_connector_meta(
        _step_output([first.request_id, second.request_id], [0, 0], [0, 0]),
        {first.request_id: first, second.request_id: second},
    )
    connector.request_finished(second)
    connector.request_finished(first)

    metadata = connector.build_connector_meta(_step_output([], [], []), {})

    assert metadata.finished_requests == (second.request_id, first.request_id)
    assert not metadata.block_hashes


def test_worker_preemption_terminal_keeps_inflight_hashes_for_publish():
    worker = _make_worker(1)
    block_hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(2 * _BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(
        2 * _BLOCK_SIZE, 3, 2
    )
    first = _metadata(
        0,
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
    _begin_step(
        worker,
        _metadata(
            0,
            [],
            {"request": []},
        ),
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


def test_worker_releases_terminal_pins_before_same_step_publish():
    worker = _make_worker(2, max_store_blocks=1)
    victim_hash = b"a" * 32
    producer_hash = b"b" * 32
    victim_rows = np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE)
    producer_rows = np.ones((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE)
    running = _metadata(
        0,
        [
            _request_metadata("victim", 0, _BLOCK_SIZE, 0, [victim_hash]),
            _request_metadata("producer", 0, _BLOCK_SIZE, 0, []),
        ],
        {},
    )
    _process_output(
        worker,
        running,
        np.concatenate((victim_rows, producer_rows)),
        ["victim", "producer"],
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    )

    _begin_step(
        worker,
        _metadata(
            0,
            [],
            {"victim": []},
            {"producer": [producer_hash]},
        ),
    )

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([producer_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        producer_rows,
    )
    with pytest.raises(ArtifactStoreError, match="does not exist"):
        worker._store.get_concatenated(routed_experts_keys([victim_hash], "0"))
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

    with pytest.raises(AssertionError, match="artifact worker output is missing"):
        connector.take_output(
            request,
            {},
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

    with pytest.raises(AssertionError, match="invalid token range"):
        connector.take_output(request, output)


def test_scheduler_rejects_empty_artifact_output_when_request_is_finished():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    request.finished = True
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

    with pytest.raises(
        AssertionError, match="finished artifact output has no accepted"
    ):
        connector.take_output(request, output)


def test_scheduler_connector_sends_each_block_hash_once():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32, b"b" * 32], num_tokens=12)
    scheduler_output = _step_output([request.request_id], [0], [4])

    first = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )
    request.block_hashes.append(b"c" * 32)
    second = connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )
    third = connector.build_connector_meta(
        _step_output([request.request_id], [8], [4]),
        {request.request_id: request},
    )

    assert list(first.block_hashes[request.request_id]) == [b"a" * 32, b"b" * 32]
    assert list(second.block_hashes[request.request_id]) == [b"c" * 32]
    assert request.request_id not in third.block_hashes


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


def test_scheduler_connector_reset_derives_emit_start_from_request():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
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
