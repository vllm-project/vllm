# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.artifacts import (
    AuxOutputBuffer,
    aux_output_keys,
    materialize_aux_output,
    publish_aux_output,
)
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxOutputSchedulerConnector,
    AuxRequestOutput,
    PackedBlockHashes,
)
from vllm.distributed.aux_output_connector.memory import validate_shm_budget
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObject,
    BlockObjectStore,
    BlockObjectStoreError,
)
from vllm.distributed.aux_output_connector.worker import (
    AuxOutputWorkerConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.sample.output import SamplerOutput

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
    store = BackgroundBlockObjectStore(underlying, max_pending_batches=1)

    store.put([BlockObject("key", b"value")])
    assert started.wait(timeout=1)
    release.set()
    store.close()

    underlying.close.assert_called_once_with()


def _make_connector():
    return AuxOutputSchedulerConnector(_BLOCK_SIZE)


@dataclass
class _Execution:
    request_id: str
    token_start: int
    num_tokens: int
    emit_start: int
    block_hashes: list[bytes] | PackedBlockHashes


@dataclass
class _WorkerStep:
    metadata: AuxOutputConnectorMetadata
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

    @property
    def num_prompt_tokens(self) -> int:
        return self.num_tokens - self.num_output_tokens

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
        AuxOutputConnectorMetadata(
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


def _begin_step(worker: AuxOutputWorkerConnector, step: _WorkerStep) -> None:
    worker.begin_step(step.metadata)


def _make_worker(
    max_num_seqs: int,
    max_num_batched_tokens: int | None = None,
    max_concurrent_batches: int = 2,
    max_store_blocks: int | None = None,
    indexer_shape: tuple[int, int] | None = None,
    enable_r3: bool = True,
) -> AuxOutputWorkerConnector:
    r3_nbytes = int(np.prod(_SHAPE)) if enable_r3 else 0
    shape = (r3_nbytes + int(np.prod(indexer_shape)) * 4,) if indexer_shape else _SHAPE
    object_nbytes = _BLOCK_SIZE * int(np.prod(shape))
    store = _make_store(
        max_bytes=(
            1 << 20 if max_store_blocks is None else max_store_blocks * object_nbytes
        ),
        object_nbytes=object_nbytes,
    )
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._indexer = (
        SimpleNamespace(
            shape_per_token=indexer_shape,
            fixed_bytes_per_token=int(np.prod(indexer_shape)) * 4,
            namespace="synthetic-indexer",
            begin_step=lambda: None,
            release_snapshot=lambda slot: None,
            close=lambda: None,
        )
        if indexer_shape
        else None
    )
    worker._r3_nbytes = r3_nbytes
    worker._r3_dtype = _DTYPE
    worker._r3_shape = _SHAPE
    worker._store = store
    worker._buffer = AuxOutputBuffer(
        _DTYPE,
        shape,
        _BLOCK_SIZE,
        max_num_seqs,
        max_num_batched_tokens or max_num_seqs * _BLOCK_SIZE,
        max_concurrent_batches,
    )
    worker._requests = {}
    worker._generation = 0
    worker._step_metadata = None
    worker._pending_outputs = []
    worker._lock = threading.Lock()
    worker._max_concurrent_batches = max_concurrent_batches
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


def _input_batch(request_ids, token_starts, query_start_loc):
    return SimpleNamespace(
        req_ids=request_ids,
        num_computed_tokens_np=token_starts,
        query_start_loc_np=query_start_loc,
    )


def test_non_output_rank_skips_capture_snapshot():
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._indexer = None
    worker._store = None
    worker._buffer = None
    worker._capturer = Mock()
    worker._step_metadata = Mock()
    assert worker.prepare_output(_input_batch([], np.array([]), np.array([]))) is None
    worker._capturer.snapshot_routing_data.assert_not_called()


def test_worker_skips_aux_outputs_for_internal_warmup_step():
    worker = _make_worker(1)
    worker._capturer = Mock()

    worker.begin_step(None)

    assert worker.prepare_output(_input_batch([], np.array([]), np.array([]))) is None
    worker._capturer.snapshot_routing_data.assert_not_called()
    worker.close()


def test_next_step_does_not_consume_pending_output(monkeypatch):
    """begin_step must not wait for or consume an unconsumed step output."""
    event = Mock()
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(async_utils, "stream", lambda *args: nullcontext())
    worker = _make_worker(1)
    worker.begin_step(
        _metadata(0, [_request_metadata("request", 0, 1, 0, [])], {}).metadata
    )
    rows = torch.ones((1, *_SHAPE), dtype=torch.uint8)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = rows
    process_output = Mock(wraps=worker.process_output)
    monkeypatch.setattr(worker, "process_output", process_output)
    pending = worker.prepare_output(
        _input_batch(["request"], np.array([0]), np.array([0, 1]))
    )
    assert pending is not None
    output = async_utils.AsyncOutput(
        ModelRunnerOutput(["request"], {"request": 0}),
        SamplerOutput(
            torch.tensor([[7]]), None, None, None, num_rejected=torch.tensor([0])
        ),
        torch.tensor([1]),
        Mock(),
        Mock(),
        check_ep_fault=False,
        pending_aux_output=pending,
    )
    event.record.assert_called_once()
    assert worker._pending_outputs == [pending]

    worker.begin_step(_metadata(0, [], {}).metadata)
    process_output.assert_not_called()
    assert worker._pending_outputs == [pending]

    result = output.get_output()

    assert result.sampled_token_ids == [[7]]
    np.testing.assert_array_equal(
        result.aux_output_connector_output["request"].routed_rows, rows.numpy()
    )
    process_output.assert_called_once()
    assert worker._pending_outputs == []
    worker.close()


@pytest.mark.parametrize("num_pending", [1, 2])
def test_finished_request_teardown_waits_for_pending_output(num_pending):
    """Terminal cleanup waits for every outstanding output, not just the first."""
    worker = _make_worker(1)
    rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, *_SHAPE)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
    pending_outputs = []
    for i in range(num_pending):
        request = _request_metadata("request", 4 * i, 4, 4 * i, [bytes([i]) * 4])
        worker.begin_step(_metadata(0, [request], {}).metadata)
        pending = worker.prepare_output(
            _input_batch(["request"], np.array([4 * i]), np.array([0, 4]))
        )
        assert pending is not None
        pending.enqueue_cpu_copy(np.array([1]), np.array([0]))
        pending_outputs.append(pending)

    # The next step finishes the request while its output is unconsumed.
    worker.begin_step(_metadata(0, [], {"request": []}).metadata)
    assert "request" in worker._requests

    for i, pending in enumerate(pending_outputs):
        output = worker.process_output(pending)
        np.testing.assert_array_equal(output["request"].routed_rows, rows)
        if i + 1 < num_pending:
            assert "request" in worker._requests
            assert worker._store._references
    assert "request" not in worker._requests
    # The finished request's store keys must be released after commit.
    assert not worker._store._references
    worker.close()


def test_pending_outputs_bounded_by_max_concurrent_batches():
    worker = _make_worker(1, max_concurrent_batches=2)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.side_effect = lambda num_rows: torch.zeros(
        (num_rows, *_SHAPE), dtype=torch.uint8
    )
    metadata = _metadata(0, [_request_metadata("request", 0, 1, 0, [])], {}).metadata
    batch = _input_batch(["request"], np.array([0]), np.array([0, 1]))
    for _ in range(2):
        worker.begin_step(metadata)
        assert worker.prepare_output(batch) is not None

    worker.begin_step(metadata)
    with pytest.raises(AssertionError, match="not consumed in order"):
        worker.prepare_output(batch)

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
    assert worker._pending_outputs == []

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
    indexer_rows=None,
):
    if num_sampled is None:
        num_sampled = np.ones(len(request_ids), dtype=np.int32)
    if token_starts is None or num_tokens is None:
        assert isinstance(step, _WorkerStep)
        token_starts, num_tokens = _execution_ranges(step, request_ids)
    worker._capturer = None
    if worker._r3_nbytes:
        worker._capturer = Mock()
        worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
    if worker._indexer is not None:
        worker._indexer.snapshot = lambda n: SimpleNamespace(
            rows=torch.from_numpy(indexer_rows[:n].copy()),
            valid=torch.ones((n, 1), dtype=torch.bool),
            slot=0,
        )
    metadata = step.metadata if isinstance(step, _WorkerStep) else step
    worker.begin_step(metadata)
    if query_start_loc is None:
        query_start_loc = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(num_tokens, dtype=np.int32))
        )
    pending = worker.prepare_output(
        _input_batch(request_ids, np.asarray(token_starts), query_start_loc)
    )
    assert pending is not None
    pending.enqueue_cpu_copy(num_sampled, num_rejected)
    return worker.process_output(pending)


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

    np.testing.assert_array_equal(output["first"].routed_rows, logical[:1])
    np.testing.assert_array_equal(output["second"].routed_rows, logical[1:])
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
    buffer = AuxOutputBuffer(np.dtype("uint8"), (1,), 4, 1, 4, 1)
    if initial_start is not None:
        rows = np.asarray(initial_rows, dtype=np.uint8).reshape(-1, 1)
        assert buffer.capture("request", initial_start, rows) == []

    with pytest.raises(AssertionError, match="not contiguous"):
        buffer.capture(
            "request", invalid_start, np.array([[invalid_start]], dtype=np.uint8)
        )


def test_logical_buffer_captures_one_row_per_decode_step():
    buffer = AuxOutputBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, 3, 2)

    completed = []
    for step in range(_BLOCK_SIZE):
        completed += buffer.capture("request", step, logical[step : step + 1])

    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0][1], logical)


def test_logical_buffer_copies_borrowed_block_when_retained():
    buffer = AuxOutputBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 4, 2)
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
    return BlockObjectStore(
        max_bytes=max_bytes,
        object_nbytes=object_nbytes,
    )


def test_publish_aux_output_publishes_full_blocks():
    store = _make_store(object_nbytes=_BLOCK_SIZE * int(np.prod(_SHAPE)))
    buffer = AuxOutputBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    hashes = [b"a" * 32, b"b" * 32]
    keys = aux_output_keys(hashes, "0")
    blocks = buffer.capture("request", 0, logical)
    publish_aux_output(
        store,
        batches=[(keys, blocks)],
        block_size=_BLOCK_SIZE,
    )
    expected = logical.copy()
    logical[:] = 0

    np.testing.assert_array_equal(
        materialize_aux_output(
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
    np.testing.assert_array_equal(output["first"].routed_rows, logical[:8])

    second = _metadata(
        generation=0,
        requests=[_request_metadata("second", 8, 2, 0, hashes)],
        finished_requests={"first": []},
    )
    output = _process_output(worker, second, logical[8:], ["second"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["second"].routed_rows, logical)
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
        materialize_aux_output(
            worker._store,
            aux_output_keys([hashes[1], hashes[3], hashes[4]], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        np.concatenate([blocks[1], blocks[3], blocks[4]]),
    )
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        materialize_aux_output(
            worker._store,
            aux_output_keys([hashes[2]], "0"),
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

    np.testing.assert_array_equal(first_output["request"].routed_rows, logical[:1])
    assert queued_output["request"].token_start == 1
    np.testing.assert_array_equal(
        queued_output["request"].routed_rows,
        logical[1:2],
    )
    np.testing.assert_array_equal(
        rolled_back_output["request"].routed_rows,
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
    np.testing.assert_array_equal(output["consumer"].routed_rows, logical)
    np.testing.assert_array_equal(output["producer"].routed_rows, logical[:4])
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
    np.testing.assert_array_equal(output["request"].routed_rows, logical[:4])
    assert worker._requests["request"].pending_blocks[0][0] == 0

    block_hash = b"a" * 32
    second = _metadata(
        0,
        [_request_metadata("request", 4, 1, 4, [block_hash])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output["request"].routed_rows, logical[4:])
    assert not worker._requests["request"].pending_blocks
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_aux_output(
            worker._store,
            aux_output_keys([block_hash], "0"),
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


def test_worker_does_not_pin_unscheduled_prompt_blocks():
    worker = _make_worker(2, max_num_batched_tokens=16, max_store_blocks=4)
    connector = _make_connector()
    prompt_hashes = [bytes([value]) * 32 for value in range(4)]
    logical = np.arange(16 * 3 * 2, dtype=np.uint8).reshape(16, 3, 2)

    def run(requests, token_counts, rows):
        token_starts = [request.num_computed_tokens for request in requests]
        metadata = connector.build_connector_meta(
            _step_output(
                [request.request_id for request in requests],
                token_starts,
                token_counts,
            ),
            {request.request_id: request for request in requests},
        )
        _process_output(
            worker,
            metadata,
            rows,
            [request.request_id for request in requests],
            np.zeros(len(requests), dtype=np.int32),
            token_starts=tuple(token_starts),
            num_tokens=tuple(token_counts),
        )
        for request, num_tokens in zip(requests, token_counts):
            request.num_computed_tokens += num_tokens

    first = _scheduler_request("first", prompt_hashes, num_tokens=17)
    first.num_computed_tokens = 0
    run([first], [16], logical)
    connector.request_finished(first)
    worker.begin_step(connector.build_connector_meta(_step_output([], [], []), {}))

    second = _scheduler_request("second", prompt_hashes, num_tokens=17)
    second.num_computed_tokens = 0
    other = _scheduler_request("other", [b"x" * 32], num_tokens=5)
    other.num_computed_tokens = 0
    run([second, other], [4, 4], np.concatenate((logical[:4], logical[4:8])))

    np.testing.assert_array_equal(
        materialize_aux_output(
            worker._store,
            aux_output_keys([b"x" * 32], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[4:8],
    )
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
        materialize_aux_output(
            worker._store,
            aux_output_keys([block_hash], "0"),
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
    np.testing.assert_array_equal(output["request"].routed_rows, logical[4:])
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
    np.testing.assert_array_equal(output["request"].routed_rows, logical[4:])
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

    np.testing.assert_array_equal(output["request"].routed_rows, logical)
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

    np.testing.assert_array_equal(output["request"].routed_rows, logical)
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
    emitted = output["request"].routed_rows
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

    assert worker._requests["request"].aux_output_keys == [
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
        materialize_aux_output(
            worker._store,
            aux_output_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:_BLOCK_SIZE],
    )
    worker.close()


def test_worker_fails_when_cached_aux_output_is_missing():
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

    with pytest.raises(BlockObjectStoreError, match="does not exist"):
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
    store.put([BlockObject("retained", b"rrr")])

    with pytest.raises(BlockObjectStoreError, match="cannot retain"):
        store.put(
            [
                BlockObject("first", b"111"),
                BlockObject("second", b"222"),
                BlockObject("third", b"333"),
            ]
        )

    assert store.get_concatenated(["retained"]) == b"rrr"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["first"])
    store.close()


def test_store_lru_and_immutable_put():
    store = _make_store(max_bytes=8)
    store.put([BlockObject("first", b"1111"), BlockObject("second", b"2222")])
    assert store.get_concatenated(["first"]) == b"1111"

    store.put([BlockObject("first", b"1111"), BlockObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="Increase aux_output_config"):
        store.get_concatenated(["second"])
    store.close()


def test_store_does_not_evict_referenced_aux_outputs():
    store = _make_store(max_bytes=8)
    store.put([BlockObject("first", b"1111"), BlockObject("second", b"2222")])
    store.put([], retain_keys=["first"])
    store.put([], retain_keys=["first"], release_keys=["first"])

    store.put([BlockObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.put([], release_keys=["first"])
    store.close()


def test_store_eviction_does_not_rescan_referenced_keys():
    hashes = 0

    class CountingKey(str):
        def __hash__(self):
            nonlocal hashes
            hashes += 1
            return super().__hash__()

    store = _make_store(max_bytes=4 * 1000)
    referenced = [CountingKey(f"referenced-{i}") for i in range(900)]
    store.put([BlockObject(key, b"rrrr") for key in referenced], retain_keys=referenced)
    old = [f"old-{i}" for i in range(100)]
    store.put([BlockObject(key, b"oooo") for key in old])
    hashes = 0

    for i in range(100):
        store.put([BlockObject(f"new-{i}", b"nnnn")])

    assert hashes < len(referenced)
    assert store.get_concatenated(referenced) == b"rrrr" * 900
    for key in old:
        with pytest.raises(BlockObjectStoreError):
            store.get_concatenated([key])
    store.close()


def test_store_terminal_put_preserves_tail_first_eviction_order():
    store = _make_store(max_bytes=8)
    keys = ["first", "second", "third"]
    store.put(
        [BlockObject("first", b"1111"), BlockObject("second", b"2222")],
        retain_keys=keys,
    )

    store.put(
        [BlockObject("third", b"3333")],
        release_keys=reversed(keys),
    )

    assert store.get_concatenated(keys[:2]) == b"11112222"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["third"])
    store.close()


def test_store_terminal_put_orders_survivors_for_later_eviction():
    store = _make_store(max_bytes=12)
    keys = ["first", "second", "third"]
    store.put(
        [BlockObject("first", b"1111"), BlockObject("second", b"2222")],
        retain_keys=keys,
    )

    store.put(
        [BlockObject("third", b"3333")],
        release_keys=reversed(keys),
    )
    store.put([BlockObject("fourth", b"4444")])

    assert store.get_concatenated(["first", "second", "fourth"]) == b"111122224444"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["third"])
    store.close()


def test_store_terminal_put_keeps_a_shared_incoming_key():
    store = _make_store(max_bytes=8)
    keys = ["first", "second", "third"]
    store.put(
        [BlockObject("first", b"1111"), BlockObject("second", b"2222")],
        retain_keys=(*keys, "third"),
    )

    store.put(
        [BlockObject("third", b"3333")],
        release_keys=reversed(keys),
    )

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.put([], release_keys=["third"])
    store.close()


def test_store_rejects_access_after_close():
    store = _make_store(max_bytes=8)
    store.close()

    with pytest.raises(RuntimeError, match="closed"):
        store.put([BlockObject("first", b"1111")])
    with pytest.raises(RuntimeError, match="closed"):
        store.get_concatenated(["first"])


def test_store_reuses_evicted_slot_without_moving_live_objects():
    store = _make_store(max_bytes=12)
    store.put(
        [
            BlockObject("first", b"1111"),
            BlockObject("second", b"2222"),
            BlockObject("third", b"3333"),
        ]
    )
    assert store.get_concatenated(["second"]) == b"2222"
    store.put([BlockObject("fourth", b"4444")])

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
    output = {"request": AuxRequestOutput(0, routing)}
    request.num_computed_tokens = 4
    np.testing.assert_array_equal(connector.take_output(request, output)[0], routing)


@pytest.mark.parametrize("finish", [False, True], ids=["hash-only-update", "finish"])
def test_scheduler_connector_sends_only_computed_hashes_off_schedule(finish):
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=9)
    request.num_computed_tokens = 0
    requests = {request.request_id: request}
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]), requests
    )
    assert list(metadata.block_hashes[request.request_id]) == [b"a" * 32]

    request.block_hashes.append(b"b" * 32)
    request.num_computed_tokens = 7
    if finish:
        connector.request_finished(request)
    metadata = connector.build_connector_meta(_step_output([], [], []), requests)
    assert request.request_id not in metadata.block_hashes
    assert metadata.finished_requests == ((request.request_id,) if finish else ())

    if not finish:
        request.num_computed_tokens = 8
        metadata = connector.build_connector_meta(_step_output([], [], []), requests)
        assert list(metadata.block_hashes[request.request_id]) == [b"b" * 32]


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

    # A P/D prefiller cuts its prompt short of the last token, which can pass
    # the requested start; the rows then start at the end of the cut prompt.
    connector = _make_connector()
    request = _scheduler_request(
        "cut", [b"a" * 32], num_tokens=3, num_output_tokens=0, prompt_start=4
    )
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [3]),
        {request.request_id: request},
    )
    assert metadata.requests == {request.request_id: 3}


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
        materialize_aux_output(
            worker._store,
            aux_output_keys(block_hashes[:1], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:_BLOCK_SIZE],
    )
    np.testing.assert_array_equal(
        materialize_aux_output(
            worker._store,
            aux_output_keys(block_hashes[1:], "0"),
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
        materialize_aux_output(
            worker._store,
            aux_output_keys([producer_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        producer_rows,
    )
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        worker._store.get_concatenated(aux_output_keys([victim_hash], "0"))
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
        materialize_aux_output(
            worker._store,
            aux_output_keys([block_hash], "0"),
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


def test_scheduler_consumes_ordered_stale_aux_outputs():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=6, num_output_tokens=0)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    first_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    second_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2) + 40
    first = {"request": AuxRequestOutput(0, first_rows)}
    second = {"request": AuxRequestOutput(4, second_rows)}
    connector.request_finished(request)

    request.num_tokens = 5
    np.testing.assert_array_equal(connector.take_output(request, first)[0], first_rows)
    request.num_tokens = 6
    np.testing.assert_array_equal(
        connector.take_output(request, second)[0], second_rows[:1]
    )


def test_scheduler_rejects_stale_output_without_aux_outputs():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )

    with pytest.raises(
        AssertionError, match="auxiliary output worker output is missing"
    ):
        connector.take_output(
            request,
            {},
        )


def test_scheduler_rejects_missing_accepted_aux_output_rows():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=2)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )
    output = {
        "request": AuxRequestOutput(
            0,
            np.empty((0, *_SHAPE), dtype=_DTYPE),
        )
    }

    with pytest.raises(AssertionError, match="invalid token range"):
        connector.take_output(request, output)[0]


def test_scheduler_rejects_aux_output_past_finished_request():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    request.finished = True
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )
    output = {
        "request": AuxRequestOutput(
            1,
            np.empty((0, *_SHAPE), dtype=_DTYPE),
        )
    }

    with pytest.raises(
        AssertionError, match="finished auxiliary output output has no accepted"
    ):
        connector.take_output(request, output)[0]


@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("max_tokens", [1, 2])
def test_prompt_end_offset_through_worker_and_scheduler(chunk_size, max_tokens):
    """Omit prompt R3 without losing the empty first output or later decode rows."""
    connector = _make_connector()
    worker = _make_worker(1)
    request = _scheduler_request(
        "request", [b"a" * 32], num_tokens=4, num_output_tokens=0, prompt_start=4
    )
    rows = np.arange(5 * 3 * 2, dtype=_DTYPE).reshape(5, *_SHAPE)
    try:
        for start in range(0, 4, chunk_size):
            metadata = connector.build_connector_meta(
                _step_output([request.request_id], [start], [chunk_size]),
                {request.request_id: request},
            )
            output = _process_output(
                worker,
                metadata,
                rows[start : start + chunk_size],
                [request.request_id],
                np.array([0]),
                num_sampled=np.array([int(start + chunk_size == 4)]),
                token_starts=(start,),
                num_tokens=(chunk_size,),
            )
            if start + chunk_size < 4:
                assert output == {}
        request.num_tokens = 5
        request.num_output_tokens = 1
        request.finished = max_tokens == 1
        result = connector.take_output(request, output)[0]
        np.testing.assert_array_equal(result, rows[:0])

        if max_tokens == 2:
            metadata = connector.build_connector_meta(
                _step_output([request.request_id], [4], [1]),
                {request.request_id: request},
            )
            output = _process_output(
                worker,
                metadata,
                rows[4:],
                [request.request_id],
                np.array([0]),
                token_starts=(4,),
                num_tokens=(1,),
            )
            request.num_tokens = 6
            request.finished = True
            np.testing.assert_array_equal(
                connector.take_output(request, output)[0], rows[4:]
            )
    finally:
        worker.close()


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


@pytest.mark.parametrize("enable_r3", [False, True])
def test_indexer_shares_spec_acceptance_and_shm_prefix_lifecycle(enable_r3):
    worker = _make_worker(2, indexer_shape=(1, 2), enable_r3=enable_r3)
    indexer = np.arange(12, dtype=np.int32).reshape(6, 1, 2)
    routing = np.arange(36, dtype=np.uint8).reshape(6, *_SHAPE)
    first = _metadata(0, [_request_metadata("producer", 0, 6, 0, [b"a" * 32])], {})
    output = _process_output(
        worker,
        first,
        routing,
        ["producer"],
        np.array([2]),
        indexer_rows=indexer,
    )["producer"]
    np.testing.assert_array_equal(output.indexer_rows, indexer[:4])
    assert output.get_num_rows() == 4
    if enable_r3:
        np.testing.assert_array_equal(output.routed_rows, routing[:4])
    else:
        assert output.routed_rows is None
    # A different request skips the cached prefix's forward entirely.
    cached = _metadata(0, [_request_metadata("cached", 4, 1, 0, [b"a" * 32])], {})
    output = _process_output(
        worker,
        cached,
        routing[4:5],
        ["cached"],
        np.array([0]),
        indexer_rows=indexer[4:5],
    )["cached"]
    np.testing.assert_array_equal(output.indexer_rows, indexer[:5])
    if enable_r3:
        np.testing.assert_array_equal(output.routed_rows, routing[:5])
    worker.close()


def test_preemption_discards_partial_block_before_request_resumes():
    worker = _make_worker(1)
    rows = np.arange(30, dtype=np.uint8).reshape(5, *_SHAPE)
    first = _metadata(0, [_request_metadata("request", 0, 3, 0, [])], {})
    _process_output(worker, first, rows[:3] + 100, ["request"], np.array([0]))
    _begin_step(worker, _metadata(0, [], {"request": []}))
    resumed = _metadata(0, [_request_metadata("request", 0, 5, 3, [b"a" * 32])], {})
    output = _process_output(worker, resumed, rows, ["request"], np.array([0]))[
        "request"
    ]
    assert output.token_start == 3
    np.testing.assert_array_equal(output.routed_rows, rows[3:])
    worker.close()


@pytest.mark.parametrize(
    ("routed", "indexer_enabled"), [(True, False), (False, True), (True, True)]
)
def test_indexer_and_routing_prompt_offsets_are_independent(routed, indexer_enabled):
    connector = AuxOutputSchedulerConnector(
        _BLOCK_SIZE, enable_routed_experts=routed, enable_indexer=indexer_enabled
    )
    request = Request(
        "request",
        [0, 1, 2, 3],
        SamplingParams(
            max_tokens=2, routed_experts_prompt_start=3, indexer_topk_prompt_start=1
        ),
        None,
    )
    metadata = connector.build_connector_meta(
        _step_output(["request"], [0], [4]), {"request": request}
    )
    assert metadata.requests["request"] == (1 if indexer_enabled else 3)
    request.append_output_token_ids([4, 5])
    routing = np.arange(24, dtype=np.uint8).reshape(4, *_SHAPE)
    indexer = np.arange(8, dtype=np.int32).reshape(4, 1, 2)
    outputs = {
        "request": AuxRequestOutput(
            1, routing if routed else None, indexer if indexer_enabled else None
        )
    }
    assert outputs["request"].get_num_rows() == 4
    routed_rows, selections = connector.take_output(request, outputs)
    if routed:
        np.testing.assert_array_equal(routed_rows, routing[2:])
    else:
        assert routed_rows is None
    if indexer_enabled:
        np.testing.assert_array_equal(selections, indexer)
    else:
        assert selections is None


def test_shared_output_validation_rejects_mismatched_artifact_rows():
    connector = AuxOutputSchedulerConnector(_BLOCK_SIZE, enable_indexer=True)
    request = Request("request", [0, 1], SamplingParams(max_tokens=1), None)
    output = AuxRequestOutput(
        0, np.empty((1, 1, 2), dtype=np.uint8), np.empty((2, 1, 2), dtype=np.int32)
    )
    with pytest.raises(AssertionError, match="artifact row counts differ"):
        connector.take_output(request, {"request": output})


def test_indexer_only_checks_accepted_range_without_routing_placeholder():
    connector = AuxOutputSchedulerConnector(
        _BLOCK_SIZE, enable_routed_experts=False, enable_indexer=True
    )
    request = Request("request", [0, 1, 2, 3], SamplingParams(max_tokens=1), None)
    request.append_output_token_ids(4)
    output = AuxRequestOutput(0, indexer_rows=np.empty((3, 1, 2), dtype=np.int32))
    with pytest.raises(AssertionError, match="invalid token range"):
        connector.take_output(request, {"request": output})


def test_fixed_store_rejects_payload_size_before_retaining_keys():
    store = BlockObjectStore(max_bytes=16, object_nbytes=8)
    with pytest.raises(ValueError, match="fixed slot size"):
        store.put([BlockObject("bad", b"short")], retain_keys=["bad"])
    assert not store._references
    store.close()


def test_budget_covers_every_kv_block_without_silent_truncation():
    assert (
        validate_shm_budget(
            num_blocks=10,
            scheduler_block_size=16,
            hash_block_size=16,
            bytes_per_token=20,
            max_bytes=None,
            available_host_bytes=2 << 30,
            staging_bytes=0,
        )
        == 3200
    )
    with pytest.raises(ValueError, match="Maximum Supported Blocks: 9"):
        validate_shm_budget(
            num_blocks=10,
            scheduler_block_size=16,
            bytes_per_token=20,
            max_bytes=3199,
            hash_block_size=16,
            available_host_bytes=2 << 30,
            staging_bytes=0,
        )


def test_host_budget_counts_local_workers_and_hash_object_metadata():
    """The inferred KV limit includes every local arena and all hash slots."""
    validate_budget = partial(
        validate_shm_budget,
        scheduler_block_size=16,
        hash_block_size=4,
        bytes_per_token=20,
        max_bytes=None,
        available_host_bytes=(512 << 20) + 2 * (4096 + 8 * (80 + 512)),
        staging_bytes=4096,
        local_output_workers=2,
    )
    assert validate_budget(num_blocks=2) == 640
    with pytest.raises(ValueError, match="Maximum Supported Blocks: 2") as error:
        validate_budget(num_blocks=3)
    assert "--num-gpu-blocks-override 2" in str(error.value)


def test_explicit_arena_larger_than_ram_requires_resizing_the_arena():
    """Fewer KV blocks alone do not shrink an explicit arena allocation."""
    with pytest.raises(ValueError, match="Maximum Supported Blocks: 3") as error:
        validate_shm_budget(
            num_blocks=2,
            scheduler_block_size=16,
            hash_block_size=16,
            bytes_per_token=20,
            max_bytes=4 * 320,
            available_host_bytes=(512 << 20) + 4096 + 3 * (320 + 512),
            staging_bytes=4096,
        )
    assert "--num-gpu-blocks-override 2" in str(error.value)
    assert "Unset aux_output_config.max_bytes" in str(error.value)


def test_arena_budget_matches_complete_store_slots():
    assert (
        validate_shm_budget(
            num_blocks=2,
            scheduler_block_size=16,
            hash_block_size=16,
            bytes_per_token=20,
            max_bytes=3 * 320 - 1,
            available_host_bytes=(512 << 20) + 2 * (320 + 512),
            staging_bytes=0,
        )
        == 640
    )


def test_staging_exhaustion_does_not_suggest_zero_gpu_blocks():
    with pytest.raises(ValueError, match="Maximum Supported Blocks: 0") as error:
        validate_shm_budget(
            num_blocks=1,
            scheduler_block_size=16,
            hash_block_size=16,
            bytes_per_token=20,
            max_bytes=None,
            available_host_bytes=(512 << 20) + 4096,
            staging_bytes=4096,
        )
    assert "No GPU KV block fits" in str(error.value)
    assert "--num-gpu-blocks-override 0" not in str(error.value)


def test_sub_block_arena_limit_is_distinguished_from_ram_exhaustion():
    with pytest.raises(ValueError, match="cannot hold one KV block") as error:
        validate_shm_budget(
            num_blocks=1,
            scheduler_block_size=16,
            hash_block_size=16,
            bytes_per_token=20,
            max_bytes=319,
            available_host_bytes=2 << 30,
            staging_bytes=0,
        )
    assert "No GPU KV block fits the current host budget" not in str(error.value)
