# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from safetensors.numpy import load

from tests.distributed.aux_output_connector.utils import fake_store
from tests.v1.core.test_prefix_caching import _make_hybrid_kv_cache_config
from tests.v1.core.utils import new_kv_cache_group
from vllm.distributed.aux_output_connector import connector as connector_module
from vllm.distributed.aux_output_connector import (
    session_objects as session_objects_module,
)
from vllm.distributed.aux_output_connector import store as store_module
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
    retained_objects,
)
from vllm.distributed.aux_output_connector.object_store import read_aux_rows
from vllm.distributed.aux_output_connector.session_objects import (
    SessionObjectWriter,
    aux_owned_keys,
)
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObject,
    BlockObjectStore,
    BlockObjectStoreError,
)
from vllm.distributed.aux_output_connector.worker import (
    AuxOutputWorkerConnector,
    _WorkerRequestState,
    compute_topk_logprobs,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec
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


def _kv_cache_manager(
    cached: dict, hashes_per_block: int = 1, num_blocks: int = 0
) -> SimpleNamespace:
    block_pool = SimpleNamespace(
        get_cached_block=lambda block_hash, _: cached.get(block_hash),
        hash_block_size=_BLOCK_SIZE,
    )
    manager = SimpleNamespace(
        block_size=hashes_per_block * _BLOCK_SIZE, block_pool=block_pool
    )
    return SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[manager], scheduler_block_size=manager.block_size
        ),
        block_pool=block_pool,
        kv_cache_config=SimpleNamespace(num_blocks=num_blocks),
    )


def _make_connector():
    return AuxOutputSchedulerConnector(_kv_cache_manager({}), 0)


@dataclass(slots=True)
class _Execution:
    request_id: str
    token_start: int
    num_tokens: int
    emit_start: int
    block_hashes: list[bytes] | PackedBlockHashes


@dataclass(slots=True)
class _WorkerStep:
    metadata: AuxOutputConnectorMetadata
    executions: list[_Execution]


@dataclass(slots=True)
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
    *,
    shape_per_token: tuple[int, int] = _SHAPE,
    dtype: np.dtype = _DTYPE,
    block_size: int = _BLOCK_SIZE,
) -> AuxOutputWorkerConnector:
    r3_nbytes = int(np.prod(shape_per_token)) * dtype.itemsize if enable_r3 else 0
    shape = (
        (r3_nbytes + int(np.prod(indexer_shape)) * 4,)
        if indexer_shape
        else shape_per_token
    )
    storage_dtype = np.dtype("uint8") if indexer_shape else dtype
    object_nbytes = block_size * int(np.prod(shape)) * storage_dtype.itemsize
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
            close=lambda: None,
        )
        if indexer_shape
        else None
    )
    worker._routed_experts_capturer = Mock() if enable_r3 else None
    worker._r3_nbytes = r3_nbytes
    worker._r3_dtype = dtype
    worker._r3_shape = shape_per_token
    worker._store = store
    worker._buffer = AuxOutputBuffer(
        storage_dtype,
        shape,
        block_size,
        max_num_seqs,
        max_num_batched_tokens or max_num_seqs * block_size,
        max_concurrent_batches,
    )
    worker._requests = {}
    worker._generation = 0
    worker._step_metadata = None
    worker._pending_outputs = []
    worker._lock = threading.Lock()
    worker._max_concurrent_batches = max_concurrent_batches
    worker._object_writer = None
    worker._object_size = 0
    worker._session_request_ids = set()
    worker._session_generation = 0
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
    worker._routed_experts_capturer = Mock()
    worker._step_metadata = Mock()
    assert worker.prepare_output(_input_batch([], np.array([]), np.array([]))) is None
    worker._routed_experts_capturer.snapshot_routing_data.assert_not_called()


def test_worker_skips_aux_outputs_for_internal_warmup_step():
    worker = _make_worker(1)
    worker._routed_experts_capturer = Mock()

    worker.begin_step(None)

    assert worker.prepare_output(_input_batch([], np.array([]), np.array([]))) is None
    worker._routed_experts_capturer.snapshot_routing_data.assert_not_called()
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
    worker._routed_experts_capturer = Mock()
    worker._routed_experts_capturer.snapshot_routing_data.return_value = rows
    process_output = Mock(wraps=worker.process_output)
    monkeypatch.setattr(worker, "process_output", process_output)
    pending = worker.prepare_output(
        _input_batch(["request"], np.array([0]), np.array([0, 1]))
    )
    assert pending is not None
    output = async_utils.AsyncOutput(
        ModelRunnerOutput(["request"]),
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
    assert result.aux_output_connector_output is not None
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
    worker._routed_experts_capturer = Mock()
    worker._routed_experts_capturer.snapshot_routing_data.return_value = (
        torch.from_numpy(rows)
    )
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

    store = worker._store
    assert store is not None
    references = cast(Any, store)._references
    for i, pending in enumerate(pending_outputs):
        output = worker.process_output(pending)
        np.testing.assert_array_equal(output["request"].routed_rows, rows)
        if i + 1 < num_pending:
            assert "request" in worker._requests
            assert references
    assert "request" not in worker._requests
    # The finished request's store keys must be released after commit.
    assert not references
    worker.close()


def test_pending_outputs_bounded_by_max_concurrent_batches():
    worker = _make_worker(1, max_concurrent_batches=2)
    worker._routed_experts_capturer = Mock()
    worker._routed_experts_capturer.snapshot_routing_data.side_effect = (
        lambda num_rows: torch.zeros((num_rows, *_SHAPE), dtype=torch.uint8)
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


def _prepare_output(
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
    hidden_states=None,
    logits_fn=None,
):
    if num_sampled is None:
        num_sampled = np.ones(len(request_ids), dtype=np.int32)
    if token_starts is None or num_tokens is None:
        assert isinstance(step, _WorkerStep)
        token_starts, num_tokens = _execution_ranges(step, request_ids)
    worker._routed_experts_capturer = None
    if worker._r3_nbytes:
        worker._routed_experts_capturer = Mock()
        worker._routed_experts_capturer.snapshot_routing_data.return_value = (
            torch.from_numpy(rows)
        )
    if worker._indexer is not None:
        worker._indexer.snapshot = lambda n: torch.from_numpy(indexer_rows[:n].copy())
    metadata = step.metadata if isinstance(step, _WorkerStep) else step
    worker.begin_step(metadata)
    if query_start_loc is None:
        query_start_loc = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(num_tokens, dtype=np.int32))
        )
    pending = worker.prepare_output(
        _input_batch(request_ids, np.asarray(token_starts), query_start_loc),
        hidden_states,
        logits_fn,
    )
    assert pending is not None
    pending.enqueue_cpu_copy(num_sampled, num_rejected)
    return pending


def _process_output(worker, *args, **kwargs):
    return _prepare_output(worker, *args, **kwargs).process_output()


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
    store = worker._store
    assert store is not None
    cast(Any, store).get_concatenated = Mock(wraps=store.get_concatenated)
    second = _metadata(
        0,
        # Async scheduling may build this before the scheduler consumes first.
        [_request_metadata("request", 4, 4, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output["request"].routed_rows, logical[4:])
    cast(Any, worker._store).get_concatenated.assert_not_called()
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


@pytest.mark.parametrize("emit_start", [0, 5])
@pytest.mark.parametrize("token_end", [7, 8, 10])
def test_worker_emits_published_blocks_with_unkeyed_rows(token_end, emit_start):
    worker = _make_worker(1)
    logical = np.arange(token_end * 6, dtype=np.uint8).reshape(token_end, *_SHAPE)

    first = _metadata(
        0,
        [_request_metadata("request", 0, 6, emit_start, [b"a" * 32])],
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
        [_request_metadata("request", 6, token_end - 6, emit_start, [])],
        {},
    )
    output = _process_output(worker, second, logical[6:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output["request"].routed_rows, logical[emit_start:])
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


def test_store_grows_with_a_warning_when_every_object_is_referenced():
    store = _make_store(max_bytes=6, object_nbytes=3)
    store.put([BlockObject("unused", b"uuu"), BlockObject("kept", b"kkk")])
    store.put([], retain_keys=["kept"])

    with patch.object(store_module.logger, "warning") as warning:
        store.put([BlockObject("first", b"111"), BlockObject("second", b"222")])

    warning.assert_called_once()
    assert store.get_concatenated(["kept", "first", "second"]) == b"kkk111222"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["unused"])
    store.close()


def test_store_lru_and_immutable_put():
    store = _make_store(max_bytes=8)
    store.put([BlockObject("first", b"1111"), BlockObject("second", b"2222")])
    assert store.get_concatenated(["first"]) == b"1111"

    store.put([BlockObject("first", b"1111"), BlockObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.close()


def test_store_replaces_only_unreferenced_aux_outputs():
    store = _make_store(max_bytes=8)
    store.put([BlockObject("block", b"1111")], retain_keys=["block"])
    store.put([], release_keys=["block"])
    store.put([BlockObject("block", b"2222")], retain_keys=["block"])
    assert store.get_concatenated(["block"]) == b"2222"

    store.put([BlockObject("block", b"3333")])
    assert store.get_concatenated(["block"]) == b"2222"
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
        new_token_ids=[[] for _ in request_ids],
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


def test_worker_skips_initial_partial_block():
    worker = _make_worker(1)
    rows = np.arange(9 * 6, dtype=_DTYPE).reshape(9, *_SHAPE)
    try:
        for start in range(5, 9):
            emit = max(6, start)
            step = _metadata(0, [_request_metadata("decode", start, 1, emit, [])], {})
            output = _process_output(
                worker, step, rows[start : start + 1], ["decode"], np.array([0])
            )["decode"]
            assert output.token_start == emit
            np.testing.assert_array_equal(output.routed_rows, rows[emit : start + 1])
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
        _kv_cache_manager({}),
        0,
        enable_routed_experts=routed,
        enable_indexer=indexer_enabled,
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
    connector = AuxOutputSchedulerConnector(
        _kv_cache_manager({}), 0, enable_indexer=True
    )
    request = Request("request", [0, 1], SamplingParams(max_tokens=1), None)
    output = AuxRequestOutput(
        0, np.empty((1, 1, 2), dtype=np.uint8), np.empty((2, 1, 2), dtype=np.int32)
    )
    with pytest.raises(AssertionError, match="artifact row counts differ"):
        connector.take_output(request, {"request": output})


def test_indexer_only_checks_accepted_range_without_routing_placeholder():
    connector = AuxOutputSchedulerConnector(
        _kv_cache_manager({}), 0, enable_routed_experts=False, enable_indexer=True
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
    assert not cast(Any, store)._references
    store.close()


_HASHES = [bytes([index]) * 32 for index in range(4)]


def _kv_cache_metadata(connector: AuxOutputSchedulerConnector):
    metadata = connector.build_connector_meta(_step_output([], [], []), {})
    retained, released = metadata.retained_block_hashes, metadata.released_block_hashes
    return list(retained or ()), list(released or ())


def _finished_request_with_cached_blocks(num_blocks: int = 1):
    cached = {_HASHES[1]: [Mock()], _HASHES[3]: [Mock()]}
    connector = AuxOutputSchedulerConnector(
        _kv_cache_manager(cached, 2, num_blocks), 2 * num_blocks
    )
    request = _SchedulerRequest(
        "request", list(_HASHES), 17, num_output_tokens=0, num_computed_tokens=16
    )
    connector.request_finished(request)
    assert _kv_cache_metadata(connector) == (_HASHES, [])
    return cached, connector, request


def _finish_empty_request(connector: AuxOutputSchedulerConnector):
    connector.request_finished(_SchedulerRequest("empty", [], 1))
    return _kv_cache_metadata(connector)


@pytest.mark.parametrize(
    "max_bytes,kv_transfer_config,expected",
    [(None, None, 6), (None, object(), 12), (400, object(), 5), (1, None, 1)],
    ids=["kv-cache", "kv-connector", "max-bytes", "max-bytes-floor"],
)
def test_retained_objects_follow_max_bytes(
    monkeypatch, max_bytes, kv_transfer_config, expected
):
    monkeypatch.setattr(
        connector_module, "resolve_kv_cache_block_sizes", lambda *_: (16, 8)
    )
    monkeypatch.setattr(
        connector_module, "get_routed_experts_bytes_per_token", lambda _: 10
    )
    vllm_config = SimpleNamespace(
        aux_output_config=SimpleNamespace(max_bytes=max_bytes),
        kv_transfer_config=kv_transfer_config,
    )
    kv_cache_config = SimpleNamespace(num_blocks=3)
    assert retained_objects(vllm_config, kv_cache_config) == (expected, 8)


@pytest.mark.parametrize(
    "num_output_tokens,prompt_start",
    [(0, 17), (1, 0)],
    ids=["prompt-start", "resumed"],
)
def test_connector_hit_needs_rows_from_emit_start(num_output_tokens, prompt_start):
    connector = _make_connector()
    request = _SchedulerRequest("request", [], 17, num_output_tokens=num_output_tokens)
    request.sampling_params.routed_experts_prompt_start = prompt_start

    assert connector.max_cache_hit(request) == 16


def test_connector_forgets_loaded_blocks_once_evicted():
    cached: dict = {}
    connector = AuxOutputSchedulerConnector(_kv_cache_manager(cached, 2, 1), 2)
    decode = _SchedulerRequest("decode", list(_HASHES), 17, num_output_tokens=0)
    decode.sampling_params.routed_experts_prompt_start = 16
    request = _SchedulerRequest("request", list(_HASHES), 17, num_output_tokens=0)
    blocks = {_HASHES[1]: [Mock()], _HASHES[3]: [Mock()]}

    assert connector.cap_external_hit(decode, 0, 16) == 16
    connector.request_loading(decode, 16)
    _finish_empty_request(connector)
    cached.update(blocks)
    assert connector.max_cache_hit(request) == 0

    cached.clear()
    connector.request_finished(decode)
    cached.update(blocks)
    assert connector.max_cache_hit(request) == 16


def test_connector_keeps_mark_of_partly_loaded_block():
    cached = {_HASHES[0]: [Mock()], _HASHES[1]: [Mock()]}
    connector = AuxOutputSchedulerConnector(_kv_cache_manager(cached), 0)
    decode = _SchedulerRequest("decode", list(_HASHES[:3]), 13, num_output_tokens=0)
    decode.sampling_params.routed_experts_prompt_start = 9
    request = _SchedulerRequest("request", list(_HASHES[:3]), 13, num_output_tokens=0)

    assert connector.cap_external_hit(decode, 8, 1) == 1
    connector.request_loading(decode, 9)
    _finish_empty_request(connector)
    cached[_HASHES[2]] = [Mock()]
    assert connector.max_cache_hit(request) == 8


def test_connector_marks_partly_computed_block_once_its_hash_exists():
    cached = {_HASHES[0]: [Mock()], _HASHES[1]: [Mock()]}
    connector = AuxOutputSchedulerConnector(_kv_cache_manager(cached), 0)
    decode = _SchedulerRequest("decode", [_HASHES[0]], 7, num_computed_tokens=5)
    request = _SchedulerRequest("request", list(_HASHES[:2]), 9, num_output_tokens=0)
    connector.build_connector_meta(
        _step_output(["decode"], [5], [1]), {"decode": decode}
    )
    decode.block_hashes.append(_HASHES[1])
    decode.num_tokens = 8
    rows = np.zeros((2, *_SHAPE), dtype=_DTYPE)
    connector.take_output(decode, {"decode": AuxRequestOutput(5, rows)})

    assert connector.max_cache_hit(request) == 4


def test_connector_releases_oldest_uncached_blocks_past_capacity():
    cached, connector, request = _finished_request_with_cached_blocks()
    assert _finish_empty_request(connector) == ([], [])
    assert connector.max_cache_hit(request) == 16

    del cached[_HASHES[3]]
    assert _finish_empty_request(connector) == ([], _HASHES[2:])
    assert connector.max_cache_hit(request) == 8

    del cached[_HASHES[1]]
    assert _finish_empty_request(connector) == ([], [])
    assert connector.max_cache_hit(request) == 8


def test_connector_releases_retained_blocks_on_reset():
    cached, connector, _ = _finished_request_with_cached_blocks()
    connector.reset()
    metadata = connector.build_connector_meta(_step_output([], [], []), {})
    assert metadata.generation == 1
    assert list(metadata.released_block_hashes) == _HASHES


def test_connector_keeps_pending_load_blocks_retained():
    cached, connector, request = _finished_request_with_cached_blocks()
    connector.request_loading(request, 16)
    cached.clear()
    assert _finish_empty_request(connector) == ([], [])

    connector.request_finished(request)
    assert _kv_cache_metadata(connector) == ([], _HASHES[:2])


def test_worker_retains_kv_cache_blocks_in_the_sending_generation():
    worker = _make_worker(1)
    store = worker._store
    assert store is not None
    references = cast(Any, store)._references
    keys = aux_output_keys(_HASHES[:2], "0")
    step = _metadata(0, [], {})
    step.metadata.retained_block_hashes = PackedBlockHashes(b"".join(_HASHES[:2]), 32)
    _begin_step(worker, step)
    assert all(references[key] == 1 for key in keys)

    step = _metadata(0, [], {})
    step.metadata.released_block_hashes = PackedBlockHashes(_HASHES[0], 32)
    _begin_step(worker, step)
    assert keys[0] not in references

    step = _metadata(1, [], {})
    step.metadata.released_block_hashes = PackedBlockHashes(_HASHES[1], 32)
    _begin_step(worker, step)
    assert not references


class _SnapshotIndexer:
    shape_per_token = (2, 2)
    fixed_bytes_per_token = 16
    namespace = "fake-indexer"

    def snapshot(self, num_rows):
        return torch.full((num_rows, 2, 2), num_rows, dtype=torch.int32)

    def close(self):
        pass


def test_two_pending_steps_keep_independent_indexer_snapshots():
    worker = _make_worker(2, indexer_shape=(2, 2), enable_r3=False)
    worker._indexer = _SnapshotIndexer()
    worker.begin_step(
        _metadata(0, [_request_metadata("request", 0, 3, 0, [])], {}).metadata
    )

    first = worker.prepare_output(
        _input_batch(["request"], np.array([0]), np.array([0, 2]))
    )
    second = worker.prepare_output(
        _input_batch(["request"], np.array([2]), np.array([0, 3]))
    )

    assert first is not None and second is not None
    first_snapshot = first.indexer_snapshot
    second_snapshot = second.indexer_snapshot
    assert first_snapshot is not None and second_snapshot is not None
    assert first_snapshot.data_ptr() != second_snapshot.data_ptr()
    assert first_snapshot[0, 0, 0].item() == 2
    assert second_snapshot[0, 0, 0].item() == 3
    worker.close()


@pytest.mark.parametrize(
    ("enable_r3", "enable_indexer"), [(True, False), (False, True), (True, True)]
)
def test_current_step_outputs_reuse_only_accepted_host_rows(enable_r3, enable_indexer):
    worker = _make_worker(
        2,
        indexer_shape=(3, 2) if enable_indexer else None,
        enable_r3=enable_r3,
    )
    routed = np.arange(42, dtype=_DTYPE).reshape(7, *_SHAPE)
    indexer = np.arange(42, dtype=np.int32).reshape(7, 3, 2) - 1
    step = _metadata(
        0,
        [
            _request_metadata("first", 0, 3, 1, []),
            _request_metadata("second", 0, 4, 1, []),
        ],
        {},
    )
    pending = _prepare_output(
        worker,
        step,
        routed,
        ["first", "second"],
        np.array([1, 2]),
        indexer_rows=indexer,
    )

    outputs = pending.process_output()

    for request_id, expected_slice in (("first", slice(1, 2)), ("second", slice(4, 5))):
        output = outputs[request_id]
        assert output.token_start == 1
        for actual, host, expected in (
            (output.routed_rows, pending.routed_experts, routed),
            (output.indexer_rows, pending.indexer, indexer),
        ):
            if host is None:
                assert actual is None
            else:
                assert actual is not None
                assert np.shares_memory(actual, host)
                assert actual.flags.c_contiguous and actual.flags.writeable
                np.testing.assert_array_equal(actual, expected[expected_slice])

    del pending
    worker.close()
    if enable_indexer:
        np.testing.assert_array_equal(outputs["second"].indexer_rows, indexer[4:5])
    if enable_r3:
        np.testing.assert_array_equal(outputs["second"].routed_rows, routed[4:5])


def test_indexer_only_output_is_a_typed_view_of_the_host_rows():
    indexer_shape = (3, 2)
    worker = _make_worker(1, indexer_shape=indexer_shape, enable_r3=False)
    typed = np.arange(12, dtype=np.int32).reshape(2, *indexer_shape)

    output = worker._make_output(0, typed.view(np.uint8).reshape(2, -1))

    assert output.routed_rows is None
    indexer_rows = output.indexer_rows
    assert indexer_rows is not None
    assert indexer_rows.dtype == np.int32
    assert indexer_rows.shape == (2, *indexer_shape)
    assert not indexer_rows.flags.owndata
    np.testing.assert_array_equal(indexer_rows, typed)


def test_indexer_only_output_copies_read_only_store_rows():
    indexer_shape = (3, 2)
    worker = _make_worker(1, indexer_shape=indexer_shape, enable_r3=False)
    typed = np.arange(12, dtype=np.int32).reshape(2, *indexer_shape)
    packed = typed.view(np.uint8).reshape(2, -1)
    packed.setflags(write=False)

    output = worker._make_output(0, packed)

    indexer_rows = output.indexer_rows
    assert indexer_rows is not None and indexer_rows.flags.writeable
    np.testing.assert_array_equal(indexer_rows, typed)


def test_combined_output_artifacts_are_independent_copies():
    indexer_shape = (3, 2)
    worker = _make_worker(1, indexer_shape=indexer_shape, enable_r3=True)
    routed = np.arange(12, dtype=_DTYPE).reshape(2, *_SHAPE)
    indexer = np.arange(12, dtype=np.int32).reshape(2, *indexer_shape)
    packed = np.concatenate(
        (routed.reshape(2, -1), indexer.view(np.uint8).reshape(2, -1)), axis=1
    )

    output = worker._make_output(0, packed)

    routed_rows = output.routed_rows
    indexer_rows = output.indexer_rows
    assert routed_rows is not None and indexer_rows is not None
    np.testing.assert_array_equal(routed_rows, routed)
    np.testing.assert_array_equal(indexer_rows, indexer)
    assert not np.shares_memory(routed_rows, indexer_rows)
    assert not np.shares_memory(routed_rows, packed)
    assert not np.shares_memory(indexer_rows, packed)
    indexer_rows[:] = -1
    np.testing.assert_array_equal(routed_rows, routed)


def test_connector_close_is_idempotent():
    worker = _make_worker(1)
    worker.close()
    worker.close()
    assert worker._store is None
    assert worker._buffer is None
    assert worker._routed_experts_capturer is None
    assert worker._indexer is None


def test_connector_rolls_back_routed_experts_when_indexer_capture_fails(monkeypatch):
    import vllm.distributed.aux_output_connector.worker as aux_worker

    routed = Mock()
    monkeypatch.setattr(aux_worker, "RoutedExpertsCapturer", Mock(return_value=routed))
    monkeypatch.setattr(aux_worker, "bind_routed_experts_capturer", Mock())
    monkeypatch.setattr(
        aux_worker,
        "IndexerCapturer",
        Mock(side_effect=RuntimeError("indexer init failed")),
    )
    monkeypatch.setattr(
        aux_worker,
        "get_tp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(
            enable_return_routed_experts=True, enable_return_indexer_topk=True
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        max_concurrent_batches=2,
    )

    with pytest.raises(RuntimeError, match="indexer init failed"):
        AuxOutputWorkerConnector(vllm_config=config, model=Mock())

    routed.close.assert_called_once()


def test_connector_rolls_back_capture_when_routed_experts_binding_fails(monkeypatch):
    import vllm.distributed.aux_output_connector.worker as aux_worker

    routed = Mock()
    monkeypatch.setattr(aux_worker, "RoutedExpertsCapturer", Mock(return_value=routed))
    monkeypatch.setattr(
        aux_worker,
        "bind_routed_experts_capturer",
        Mock(side_effect=ValueError("no supported MoE router")),
    )
    monkeypatch.setattr(
        aux_worker,
        "get_tp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(
            enable_return_routed_experts=True, enable_return_indexer_topk=False
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        max_concurrent_batches=2,
    )

    with pytest.raises(ValueError, match="no supported MoE router"):
        AuxOutputWorkerConnector(vllm_config=config, model=Mock())

    routed.close.assert_called_once()


@pytest.mark.parametrize(
    "groups,kwargs,error",
    [
        (["full"], {}, False),
        (["full", "sliding_window"], {}, False),
        (["full", "mamba"], {"hash_block_size": 8}, True),
        (["full"], {"num_hash_lookahead_tokens": 2}, True),
        (["sliding_window"], {}, True),
        (["full", "replay"], {}, True),
    ],
    ids=[
        "full-attention",
        "sliding-window",
        "partial-hash-hits",
        "hash-lookahead",
        "no-full-attention",
        "prefix-replay",
    ],
)
def test_sessions_reject_kv_layouts_that_break_objects(groups, kwargs, error):
    config = _make_hybrid_kv_cache_config(
        16, 10, [group for group in groups if group != "replay"]
    )
    if "replay" in groups:
        spec = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
        )
        layer = SlidingWindowManager.layer_config(
            spec, sliding_window=32, bounded_replay=True
        )
        config.kv_cache_groups.append(new_kv_cache_group(["replay"], layer))
    kwargs = {"hash_block_size": 16, "scheduler_block_size": 16, **kwargs}
    manager = KVCacheManager(config, max_model_len=8192, **kwargs)
    AuxOutputSchedulerConnector(manager, 0)
    if not error:
        AuxOutputSchedulerConnector(manager, 0, key_prefix="job")
        return
    with pytest.raises(ValueError, match="object_key_prefix needs"):
        AuxOutputSchedulerConnector(manager, 0, key_prefix="job")


def _position_rows(start, end):
    return np.arange(start, end, dtype=np.uint8).repeat(6).reshape(-1, *_SHAPE)


def _session_worker(store):
    worker = _make_worker(2)
    worker._object_writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=4,
        fields=worker._object_fields,
        put_stall_timeout_s=0.5,
    )
    return worker


def _session_step(
    worker, rows, start, end, hashes, *, object_blocks=1, hidden_states=None
):
    step = _metadata(
        0, [_request_metadata("session", start, end - start, 0, hashes)], {}
    )
    step.metadata.session_object_size = object_blocks * _BLOCK_SIZE
    if start == 0:
        step.metadata.sessions = {"session": "s"}
    _begin_step(worker, step)
    worker._routed_experts_capturer = Mock()
    worker._routed_experts_capturer.snapshot_routing_data.return_value = (
        torch.from_numpy(rows[start:end])
    )
    pending = worker.prepare_output(
        _input_batch(["session"], np.array([start]), np.array([0, end - start])),
        hidden_states,
    )
    pending.enqueue_cpu_copy(np.array([0]), np.array([0]))
    return pending


def test_worker_names_session_objects_by_their_last_hash_block():
    client, store = fake_store()
    worker = _session_worker(store)
    rows = _position_rows(0, 2 * _BLOCK_SIZE)
    for block in range(2):
        start = block * _BLOCK_SIZE
        step = _metadata(
            0,
            [
                _request_metadata(
                    "session", start, _BLOCK_SIZE, 0, _HASHES[block : block + 1]
                ),
                _request_metadata(
                    "plain", start, _BLOCK_SIZE, 0, _HASHES[block + 2 : block + 3]
                ),
            ],
            {},
        )
        step.metadata.session_object_size = _BLOCK_SIZE
        if block == 0:
            step.metadata.sessions = {"session": "s"}
        _begin_step(worker, step)
        worker._routed_experts_capturer = Mock()
        worker._routed_experts_capturer.snapshot_routing_data.return_value = (
            torch.from_numpy(np.concatenate([rows[start : start + _BLOCK_SIZE]] * 2))
        )
        pending = worker.prepare_output(
            _input_batch(
                ["session", "plain"],
                np.array([start, start]),
                np.array([0, _BLOCK_SIZE, 2 * _BLOCK_SIZE]),
            )
        )
        pending.enqueue_cpu_copy(np.array([0, 0]), np.array([0, 0]))
        worker.process_output(pending)
    local_store = worker._store
    assert len(local_store.get_concatenated(aux_output_keys(_HASHES[2:], "0")))
    with pytest.raises(BlockObjectStoreError):
        local_store.get_concatenated(aux_output_keys(_HASHES[:1], "0"))
    worker.close()

    keys = [f"job/s/{_HASHES[0].hex()}", f"job/s/{_HASHES[1].hex()}"]
    assert client.writes == [keys[:1], keys[1:]]
    np.testing.assert_array_equal(read_aux_rows(store, keys, "routed_experts"), rows)


@pytest.mark.parametrize(
    "terminal_first", [True, False], ids=["terminal-first", "output-first"]
)
def test_worker_writes_a_session_object_of_two_hash_blocks_once(terminal_first):
    client, store = fake_store()
    worker = _session_worker(store)
    rows = _position_rows(0, 2 * _BLOCK_SIZE)
    pending = _session_step(worker, rows, 0, 2 * _BLOCK_SIZE, [], object_blocks=2)
    terminal = _metadata(0, [], {"session": _HASHES[:2]})
    terminal.metadata.session_object_size = 2 * _BLOCK_SIZE
    if terminal_first:
        _begin_step(worker, terminal)
        worker.process_output(pending)
    else:
        worker.process_output(pending)
        _begin_step(worker, terminal)
    assert worker._requests == {}
    worker.close()

    key = f"job/s/{_HASHES[1].hex()}"
    assert client.writes == [[key]]
    np.testing.assert_array_equal(read_aux_rows(store, [key], "routed_experts"), rows)


@pytest.mark.parametrize(
    "hashes,object_blocks,tail_start",
    [(_HASHES[:1], 1, _BLOCK_SIZE), ([], 1, 0), (_HASHES[:1], 2, 0)],
    ids=["after-an-object", "unhashed-block", "staged-block"],
)
def test_worker_writes_the_session_tail_object_when_the_request_finishes(
    hashes, object_blocks, tail_start
):
    client, store = fake_store()
    worker = _session_worker(store)
    end = _BLOCK_SIZE + 3
    rows = _position_rows(0, end)
    pending = _session_step(worker, rows, 0, end, hashes, object_blocks=object_blocks)
    worker.process_output(pending)
    terminal = _metadata(0, [], {"session": []})
    terminal.metadata.session_object_size = object_blocks * _BLOCK_SIZE
    terminal.metadata.session_tails = {"session": ("job/s/tail/t", tail_start, end)}
    _begin_step(worker, terminal)
    worker.close()

    keys = [f"job/s/{_HASHES[0].hex()}"] if tail_start else []
    assert client.writes == [[key] for key in [*keys, "job/s/tail/t"]]
    np.testing.assert_array_equal(
        read_aux_rows(store, [*keys, "job/s/tail/t"], "routed_experts"), rows
    )


def test_worker_keeps_stepping_after_a_session_object_write_failed():
    client, store = fake_store(-800)
    worker = _session_worker(store)
    pending = _session_step(
        worker, _position_rows(0, _BLOCK_SIZE), 0, _BLOCK_SIZE, _HASHES[:1]
    )
    worker.process_output(pending)
    deadline = time.monotonic() + 0.8
    while time.monotonic() < deadline:
        _begin_step(worker, _metadata(0, [], {}))
        time.sleep(0.01)
    worker._object_writer.drain(timeout_s=10)
    assert len(client.writes) > 1 and not client.objects
    client.code = 0
    rows = _position_rows(0, 2 * _BLOCK_SIZE)
    pending = _session_step(worker, rows, _BLOCK_SIZE, 2 * _BLOCK_SIZE, _HASHES[1:2])
    worker.process_output(pending)
    worker._object_writer.drain(timeout_s=10)
    worker.close()
    assert list(client.objects) == [f"job/s/{_HASHES[1].hex()}"]
    assert client.closed and worker._store is None


def _writer_with_a_queued_write(code, release):
    client, store = fake_store(code)
    put_batch = client.put_batch
    client.put_batch = lambda *args: release.wait() and put_batch(*args)
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=4,
        fields=lambda rows: {"routed_experts": rows},
        put_stall_timeout_s=0.5,
    )
    writer.add_tail("job/s/tail/0", np.zeros((1, 1), dtype=np.uint8), {})
    return client, writer


def test_session_object_writer_drain_raises_past_its_timeout():
    release = threading.Event()
    _, writer = _writer_with_a_queued_write(0, release)
    try:
        with pytest.raises(BlockObjectStoreError, match="not written in time"):
            writer.drain(timeout_s=0.05)
    finally:
        release.set()
        assert _run_within(writer.close) is None


def test_session_object_writer_drops_writes_its_queue_cannot_take_behind_a_hang():
    release = threading.Event()
    client, store = fake_store()
    put_batch = client.put_batch
    client.put_batch = lambda *args: release.wait() and put_batch(*args)
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=1,
        fields=lambda rows: {"routed_experts": rows},
        put_stall_timeout_s=0.5,
    )
    rows = np.zeros((1, 1), dtype=np.uint8)

    def produce():
        for index in range(3):
            writer.add_tail(f"job/s/tail/{index}", rows, {})

    try:
        assert _run_within(produce) is None
        assert _run_within(writer.drain) is None
        writer.check()
        release.set()
        writer.drain(timeout_s=10)
    finally:
        release.set()
        assert _run_within(writer.close) is None
    assert list(client.objects) == ["job/s/tail/0", "job/s/tail/1"]
    assert writer._dropped == 0


@pytest.mark.parametrize("failures", [3, None], ids=["store-recovers", "store-stalls"])
def test_session_object_writer_retries_refused_puts_until_the_stall_deadline(
    failures,
):
    client, store = fake_store()
    put_batch = client.put_batch
    attempts = []

    def refusing_put(keys, values, config):
        attempts.append(keys)
        if failures is None or len(attempts) <= failures:
            return -200
        return put_batch(keys, values, config)

    client.put_batch = refusing_put
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=1,
        fields=lambda rows: {"routed_experts": rows},
        put_stall_timeout_s=1.0,
    )
    writer.add_tail("job/s/tail/0", np.zeros((1, 1), dtype=np.uint8), {})

    try:
        writer.drain(timeout_s=10)
        writer.check()
    finally:
        assert _run_within(writer.close) is None
    if failures is None:
        assert len(attempts) > 1 and not client.objects
    else:
        assert len(attempts) == failures + 1
        assert list(client.objects) == ["job/s/tail/0"]


def test_session_object_writer_never_writes_an_object_again():
    client, store = fake_store()
    put_batch = client.put_batch
    state = {"space": 1, "refused": False}

    def partial_put(keys, values, config):
        if state["refused"]:
            assert client.batch_remove(["job/s/a"], True) == [0]
            state.update(space=len(keys), refused=False)
        written = min(state["space"], len(keys))
        state["space"] -= written
        put_batch(keys[:written], values[:written], config)
        if written < len(keys):
            state["refused"] = True
            return -200
        return 0

    client.put_batch = partial_put
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=1,
        fields=lambda rows: {"routed_experts": rows},
    )
    writer._put((["job/s/a", "job/s/b"], [b"a", b"b"]))
    writer.drain(timeout_s=10)
    writer.close()
    assert list(client.objects) == ["job/s/b"]


def test_session_object_writer_counts_each_written_object_as_progress():
    client, writer = _timed_writer(lambda keys: time.sleep(0.05), 0.3, 1)
    keys = [f"job/s/{index}" for index in range(20)]
    writer._put((keys, [b"x"] * len(keys)))
    try:
        writer.drain(timeout_s=30)
    finally:
        assert _run_within(writer.close) is None
    assert list(client.objects) == keys


@pytest.mark.parametrize("code", [-703, -706], ids=["not-ready", "has-lease"])
def test_session_object_writer_removes_an_object_the_store_refused_once(code):
    client, store = fake_store()
    batch_remove = client.batch_remove
    refused = []

    def refuse_once(keys, force):
        if not refused:
            refused.append(keys)
            return [code] * len(keys)
        return batch_remove(keys, force)

    client.batch_remove = refuse_once
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=2,
        fields=lambda rows: {"routed_experts": rows},
    )
    writer._put((["job/s/a"], [b"a"]))
    writer.remove(["job/s/a"])
    writer.drain(timeout_s=10)
    writer.drain(timeout_s=10)
    writer.close()
    assert refused and not client.objects and not writer._unremoved


def test_session_object_writer_frees_space_by_retrying_refused_removals():
    client, store = fake_store()
    put_batch = client.put_batch
    batch_remove = client.batch_remove
    removals = []

    def put_one(keys, values, config):
        return -200 if client.objects else put_batch(keys, values, config)

    def refuse_twice(keys, force):
        removals.append(list(keys))
        return [-703] * len(keys) if len(removals) <= 2 else batch_remove(keys, force)

    client.put_batch = put_one
    client.batch_remove = refuse_twice
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=3,
        fields=lambda rows: {"routed_experts": rows},
        put_stall_timeout_s=2.0,
    )
    writer._put((["job/s/a"], [b"a"]))
    writer.remove(["job/s/a"])
    writer._put((["job/s/b"], [b"b"]))
    try:
        writer.drain(timeout_s=30)
    finally:
        assert _run_within(writer.close) is None
    assert list(client.objects) == ["job/s/b"] and not writer._unremoved


def test_session_object_writer_backs_off_retrying_refused_removals():
    client, store = fake_store()
    put_batch = client.put_batch
    removals = []

    def put_one(keys, values, config):
        return -200 if client.objects else put_batch(keys, values, config)

    def refuse(keys, force):
        removals.append(list(keys))
        return [-706] * len(keys)

    client.put_batch = put_one
    client.batch_remove = refuse
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=3,
        fields=lambda rows: {"routed_experts": rows},
        put_stall_timeout_s=2.0,
    )
    writer._put((["job/s/a"], [b"a"]))
    writer.remove(["job/s/a"])
    writer._put((["job/s/b"], [b"b"]))
    try:
        writer.drain(timeout_s=30)
    finally:
        assert _run_within(writer.close) is None
    assert 1 < len(removals) <= 6


def test_session_object_writer_retries_refused_removals_every_32_attempts(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm.distributed.aux_output_connector.session_objects._PUT_RETRY_INTERVAL_S",
        0,
    )
    client, store = fake_store()
    put_batch = client.put_batch
    attempts = []
    removals = []

    def put_late(keys, values, config):
        attempts.append(keys)
        return -200 if len(attempts) <= 96 else put_batch(keys, values, config)

    def refuse(keys, force):
        removals.append(len(attempts))
        return [-706] * len(keys)

    client.put_batch = put_late
    client.batch_remove = refuse
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=3,
        fields=lambda rows: {"routed_experts": rows},
    )
    writer.remove(["job/s/a"])
    writer._put((["job/s/b"], [b"b"]))
    try:
        writer.drain(timeout_s=30)
    finally:
        assert _run_within(writer.close) is None
    assert removals[:9] == [0, 1, 2, 4, 8, 16, 32, 64, 96]
    assert list(client.objects) == ["job/s/b"]


def test_session_object_writer_does_not_wait_for_an_object_that_is_not_there():
    client, store = fake_store()
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=2,
        fields=lambda rows: {"routed_experts": rows},
    )
    writer.remove(["job/s/absent"])
    writer._put((["job/s/a"], [b"a"]))
    writer.drain(timeout_s=2)
    writer.close()
    assert list(client.objects) == ["job/s/a"] and not writer._unremoved


def test_session_object_writer_counts_objects_it_could_not_remove():
    client, store = fake_store()
    client.batch_remove = lambda keys, force: [-706] * len(keys)
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=2,
        fields=lambda rows: {"routed_experts": rows},
    )
    writer._put((["job/s/a", "job/s/b"], [b"a", b"b"]))
    writer.remove(["job/s/a", "job/s/b"])
    writer.drain(timeout_s=10)
    writer.close()
    assert len(writer._unremoved) == 2


def _run_within(call, timeout_s=30.0):
    errors = []

    def run():
        try:
            call()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_s)
    assert not thread.is_alive()
    return errors[0] if errors else None


def _timed_writer(put, put_stall_timeout_s, max_pending_batches):
    client, store = fake_store()
    put_batch = client.put_batch

    def timed_put_batch(keys, values, config):
        return put(keys) or put_batch(keys, values, config)

    client.put_batch = timed_put_batch
    writer = SessionObjectWriter(
        store,
        key_prefix="job",
        max_pending_batches=max_pending_batches,
        fields=lambda rows: {"routed_experts": rows},
        put_stall_timeout_s=put_stall_timeout_s,
    )
    return client, writer


@pytest.mark.parametrize(
    "refuse_s,hang,stall_s,written",
    [(1.0, False, 5.0, 4), (None, False, 1.0, 0), (None, True, 1.0, 0)],
    ids=["full-within-deadline", "full-past-deadline", "put-hangs"],
)
def test_session_object_writer_waits_for_a_full_store_until_the_stall_deadline(
    refuse_s, hang, stall_s, written
):
    start = time.monotonic()
    release = threading.Event()

    def put(keys):
        if hang:
            release.wait(30)
            return -900
        if refuse_s is None or time.monotonic() - start < refuse_s:
            return -200
        return 0

    client, writer = _timed_writer(put, stall_s, 1)
    keys = [f"job/s/tail/{index}" for index in range(4)]

    def produce():
        for key in keys:
            writer.add_tail(key, np.zeros((1, 1), dtype=np.uint8), {})
        writer.drain(timeout_s=30)

    try:
        assert _run_within(produce) is None
        assert time.monotonic() - start < stall_s + 5
        writer.check()
    finally:
        release.set()
        assert _run_within(writer.close) is None
    assert list(client.objects) == keys[:written]


def test_session_object_writer_drain_waits_while_writes_progress(monkeypatch):
    now = [0.0]
    attempts: dict[str, int] = {}
    monkeypatch.setattr(
        session_objects_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )

    def slow_put(keys):
        now[0] += 10.0
        attempts[keys[0]] = attempts.get(keys[0], 0) + 1
        return -200 if attempts[keys[0]] < 7 else 0

    client, writer = _timed_writer(slow_put, 100.0, 3)
    keys = [f"job/s/tail/{index}" for index in range(8)]

    def produce():
        for key in keys:
            writer.add_tail(key, np.zeros((1, 1), dtype=np.uint8), {})
        writer.drain()

    try:
        assert _run_within(produce) is None
    finally:
        assert _run_within(writer.close) is None
    assert list(client.objects) == keys
    assert now[0] == 70.0 * len(keys)


@pytest.mark.parametrize(
    "code", [-200, -800, -900], ids=["no-memory", "holder-gone", "master-gone"]
)
def test_session_object_writer_drops_what_the_store_refuses_and_recovers(code):
    attempts: list[str] = []
    client, writer = _timed_writer(lambda keys: attempts.append(keys[0]), 1.0, 1)
    rows = np.zeros((1, 1), dtype=np.uint8)
    client.code = code
    try:
        writer.add_tail("job/s/tail/0", rows, {})
        writer.drain(timeout_s=10)
        writer.add_tail("job/s/tail/1", rows, {})
        writer.drain(timeout_s=0.5)
        client.code = 0
        writer.add_tail("job/s/tail/2", rows, {})
        writer.drain(timeout_s=10)
        client.code = code
        writer.add_tail("job/s/tail/3", rows, {})
        while attempts.count("job/s/tail/3") < 2:
            time.sleep(0.01)
        client.code = 0
        writer.drain(timeout_s=10)
        writer.check()
    finally:
        assert _run_within(writer.close) is None
    assert attempts.count("job/s/tail/0") > 1 and attempts.count("job/s/tail/1") == 1
    assert list(client.objects) == ["job/s/tail/2", "job/s/tail/3"]
    assert writer._dropped == 0


def test_session_object_writer_drain_stops_waiting_once_a_put_hangs():
    entered = threading.Event()
    release = threading.Event()

    def put(keys):
        entered.set()
        release.wait(30)
        return 0

    client, writer = _timed_writer(put, 1.0, 4)
    try:
        writer.add_tail("job/s/tail/0", np.zeros((1, 1), dtype=np.uint8), {})
        assert entered.wait(10)
        with pytest.raises(BlockObjectStoreError, match="not written in time"):
            writer.drain(timeout_s=0.5)
        start = time.monotonic()
        writer.drain(timeout_s=10)
        assert time.monotonic() - start < 1.5
        writer.check()
    finally:
        release.set()
        assert _run_within(writer.close) is None
    assert list(client.objects) == ["job/s/tail/0"]


def test_session_object_writer_close_stops_retrying():
    attempts = []
    client, writer = _timed_writer(lambda keys: attempts.append(keys), 600.0, 1)
    client.code = -200
    writer.add_tail("job/s/tail/0", np.zeros((1, 1), dtype=np.uint8), {})
    try:
        deadline = time.monotonic() + 10
        while not attempts and time.monotonic() < deadline:
            time.sleep(0.01)
        assert attempts
    finally:
        assert _run_within(writer.close, 5) is None
    writer.check()
    assert client.closed and not client.objects and writer._dropped == 1


@pytest.mark.parametrize("idle_s", [601.0, 6000.0], ids=["past-deadline", "long-idle"])
def test_session_object_writer_does_not_count_idle_time_toward_the_stall_deadline(
    monkeypatch, idle_s
):
    now = [0.0]
    writer_waiting = threading.Event()
    checked = threading.Event()

    def monotonic():
        if threading.current_thread().name == "vllm-aux-object-writer":
            writer_waiting.set()
            checked.wait(10)
        elif writer_waiting.is_set():
            checked.set()
        return now[0]

    monkeypatch.setattr(
        session_objects_module, "time", SimpleNamespace(monotonic=monotonic)
    )
    client, writer = _timed_writer(lambda keys: None, 600.0, 1)
    keys = [f"job/s/tail/{index}" for index in range(2)]
    try:
        for key in keys:
            now[0] += idle_s
            writer.check()
            writer.add_tail(key, np.zeros((1, 1), dtype=np.uint8), {})
            assert _run_within(writer.drain) is None
    finally:
        assert _run_within(writer.close) is None
    assert list(client.objects) == keys


def test_session_object_writer_drain_returns_after_close():
    release = threading.Event()
    release.set()
    client, writer = _writer_with_a_queued_write(0, release)
    assert _run_within(writer.close) is None
    writer.drain(timeout_s=0.05)
    assert list(client.objects) == ["job/s/tail/0"]


def test_session_object_holds_each_field():
    client, store = fake_store()
    routed = np.arange(_BLOCK_SIZE * 12, dtype=_DTYPE).reshape(_BLOCK_SIZE, 4, 3)
    indexer = 1000 - np.arange(_BLOCK_SIZE * 12, dtype=np.int32).reshape(
        _BLOCK_SIZE, 4, 3
    )
    worker = _make_worker(2, indexer_shape=(4, 3), shape_per_token=(4, 3))
    worker._object_writer = SessionObjectWriter(
        store, key_prefix="job", max_pending_batches=4, fields=worker._object_fields
    )
    step = _metadata(
        0, [_request_metadata("session", 0, _BLOCK_SIZE, 0, _HASHES[:1])], {}
    )
    step.metadata.session_object_size = _BLOCK_SIZE
    step.metadata.sessions = {"session": "s"}
    pending = _prepare_output(
        worker, step, routed, ["session"], np.array([0]), indexer_rows=indexer
    )
    worker.process_output(pending)
    worker.close()

    key = f"job/s/{_HASHES[0].hex()}"
    assert client.writes == [[key]]
    for name, rows in (("routed_experts", routed), ("indexer_topk", indexer)):
        stored = read_aux_rows(store, [key], name)
        assert stored.dtype == rows.dtype
        np.testing.assert_array_equal(stored, rows)


@pytest.mark.parametrize(
    "hashes,tail_start",
    [(_HASHES[:1], _BLOCK_SIZE), ([], 0)],
    ids=["after-an-object", "unhashed-block"],
)
def test_worker_writes_hidden_states_into_session_objects_and_the_tail(
    hashes, tail_start
):
    client, store = fake_store()
    worker = _session_worker(store)
    worker._return_hidden_states = True
    end = _BLOCK_SIZE + 3
    rows = _position_rows(0, end)
    hidden = torch.arange(2 * end, dtype=torch.float32).reshape(end, 2)
    for start, stop, step_hashes in [(0, 2, []), (2, end, hashes)]:
        pending = _session_step(
            worker, rows, start, stop, step_hashes, hidden_states=hidden[start:stop]
        )
        worker.process_output(pending)
    terminal = _metadata(0, [], {"session": []})
    terminal.metadata.session_object_size = _BLOCK_SIZE
    terminal.metadata.session_tails = {"session": ("job/s/tail/t", tail_start, end)}
    _begin_step(worker, terminal)
    worker.close()

    keys = [f"job/s/{_HASHES[0].hex()}"] if tail_start else []
    keys.append("job/s/tail/t")
    np.testing.assert_array_equal(
        read_aux_rows(store, keys, "hidden_states"), hidden.numpy()
    )
    np.testing.assert_array_equal(read_aux_rows(store, keys, "routed_experts"), rows)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32], ids=["bfloat16", "float32"]
)
def test_worker_snapshots_hidden_states_of_session_rows_only(dtype):
    worker = _make_worker(2)
    worker._return_hidden_states = True
    step = _metadata(
        0,
        [
            _request_metadata("plain", 0, 2, 0, []),
            _request_metadata("session", 0, 3, 0, []),
        ],
        {},
    )
    step.metadata.sessions = {"session": "s"}
    _begin_step(worker, step)
    worker._routed_experts_capturer.snapshot_routing_data.return_value = torch.zeros(
        (5, *_SHAPE), dtype=torch.uint8
    )
    hidden_states = torch.randn(6, 4).to(dtype)
    expected = hidden_states[2:5].clone()
    pending = worker.prepare_output(
        _input_batch(["plain", "session"], np.array([0, 0]), np.array([0, 2, 5])),
        hidden_states,
    )
    hidden_states.zero_()
    pending.enqueue_cpu_copy(np.array([0, 0]), np.array([0, 0]))
    rows = torch.from_numpy(pending.fields["hidden_states"]).view(dtype)
    assert torch.equal(rows, expected)
    worker.close()


def test_session_fields_follow_their_requests_in_a_batch():
    client, store = fake_store()
    worker = _make_worker(3, max_num_batched_tokens=16)
    worker._object_writer = SessionObjectWriter(
        store, key_prefix="job", max_pending_batches=4, fields=worker._object_fields
    )
    worker._return_hidden_states = True
    step = _metadata(
        0,
        [
            _request_metadata("a", 0, 5, 0, []),
            _request_metadata("plain", 0, 3, 0, []),
            _request_metadata("b", 0, 6, 0, []),
        ],
        {},
    )
    step.metadata.session_object_size = _BLOCK_SIZE
    step.metadata.sessions = {"a": "s", "b": "t"}
    lengths = {"a": 5, "plain": 3, "b": 6}
    tags = {"a": 100, "plain": 200, "b": 300}
    routed = np.concatenate([_position_rows(0, n) for n in lengths.values()])
    hidden = torch.cat(
        [torch.arange(n, dtype=torch.float32) + tags[r] for r, n in lengths.items()]
    )[:, None]
    pending = _prepare_output(
        worker,
        step,
        routed,
        list(lengths),
        np.array([1, 0, 2]),
        hidden_states=hidden,
    )
    worker.process_output(pending)
    for request in ("a", "b"):
        (staged,) = worker._requests[request].fields["hidden_states"][2]
        assert not np.shares_memory(staged, pending.fields["hidden_states"])
    pending.fields["hidden_states"][...] = -1
    terminal = _metadata(0, [], {"a": _HASHES[:1], "plain": [], "b": _HASHES[1:2]})
    terminal.metadata.session_object_size = _BLOCK_SIZE
    _begin_step(worker, terminal)
    worker.close()

    for session, block_hash, tag in [("s", _HASHES[0], 100), ("t", _HASHES[1], 300)]:
        key = f"job/{session}/{block_hash.hex()}"
        rows = read_aux_rows(store, [key], "hidden_states")
        assert rows.ravel().tolist() == [tag + p for p in range(_BLOCK_SIZE)]


def test_session_fields_follow_rejected_speculative_rows():
    client, store = fake_store()
    worker = _session_worker(store)
    worker._return_hidden_states = True
    worker._topk_logprobs = 1
    routed = _position_rows(0, 5)
    hidden = torch.arange(5, dtype=torch.float32)[:, None]

    def logits_fn(rows):
        return torch.nn.functional.one_hot(rows[:, 0].long(), 5).float()

    for token_start, first, num_rejected, hashes in [
        (0, 0, 4, []),
        (5, 1, 3, []),
        (6, 2, 0, _HASHES[:1]),
    ]:
        step = _metadata(
            0,
            [_request_metadata("request", token_start, 5 - first, 0, hashes)],
            {},
        )
        step.metadata.session_object_size = _BLOCK_SIZE
        if token_start == 0:
            step.metadata.sessions = {"request": "s"}
        pending = _prepare_output(
            worker,
            step,
            routed[first:],
            ["request"],
            np.array([num_rejected]),
            hidden_states=hidden[first:],
            logits_fn=logits_fn,
        )
        worker.process_output(pending)
    terminal = _metadata(0, [], {"request": []})
    terminal.metadata.session_object_size = _BLOCK_SIZE
    terminal.metadata.session_tails = {"request": ("job/s/tail/t", _BLOCK_SIZE, 5)}
    _begin_step(worker, terminal)
    worker.close()

    keys = [f"job/s/{_HASHES[0].hex()}", "job/s/tail/t"]
    np.testing.assert_array_equal(read_aux_rows(store, keys, "routed_experts"), routed)
    np.testing.assert_array_equal(
        read_aux_rows(store, keys, "hidden_states"), hidden.numpy()
    )
    np.testing.assert_array_equal(
        read_aux_rows(store, keys, "topk_token_ids"), hidden.int().numpy()
    )


@pytest.mark.parametrize(
    "start,end,shared",
    [(0, 4, True), (0, 8, False), (2, 8, False)],
    ids=["most-rows-left", "few-rows-left", "after-skipped-rows"],
)
def test_session_fields_copy_only_small_remainders(start, end, shared):
    chunks = [np.arange(10)[:, None], np.arange(10, 12)[:, None]]
    state = _WorkerRequestState(session="s")
    state.fields["hidden_states"] = (0, 12, list(chunks))
    rows = SessionObjectWriter._take_fields(state, start, end)["hidden_states"]
    assert rows.ravel().tolist() == list(range(start, end))
    first, stop, rest = state.fields["hidden_states"]
    assert (first, stop) == (end, 12)
    assert np.concatenate(rest).ravel().tolist() == list(range(end, 12))
    assert np.shares_memory(rest[0], chunks[0]) == shared
    assert rest[1] is chunks[1]


def test_session_fields_detach_the_rows_left_before_a_withheld_hash():
    chunk = np.arange(16)[:, None]
    state = _WorkerRequestState(session="s")
    state.fields["hidden_states"] = (0, 16, [chunk])
    for start in range(0, 14, 2):
        rows = SessionObjectWriter._take_fields(state, start, start + 2)
        assert rows["hidden_states"].ravel().tolist() == [start, start + 1]
    _, _, (rest,) = state.fields["hidden_states"]
    assert rest.ravel().tolist() == [14, 15]
    assert not np.shares_memory(rest, chunk)


def test_session_object_with_incomplete_field_rows_is_rejected():
    writer = SessionObjectWriter(
        fake_store()[1],
        key_prefix="job",
        max_pending_batches=4,
        fields=lambda rows: {"routed_experts": rows},
    )
    state = _WorkerRequestState(session="s", block_hashes=_HASHES[:1])
    state.fields["hidden_states"] = (1, 4, [np.zeros((3, 2), dtype=np.float32)])
    try:
        with pytest.raises(AssertionError, match="incomplete"):
            writer.add_blocks(state, [(0, _position_rows(0, _BLOCK_SIZE))], _BLOCK_SIZE)
    finally:
        assert _run_within(writer.close) is None


@pytest.mark.parametrize("k,calls", [(2, [2, 2, 1]), (5, [1] * 5)], ids=["k-2", "k-5"])
def test_topk_logprob_chunks_are_bounded_by_k(monkeypatch, k, calls):
    monkeypatch.setattr(
        "vllm.distributed.aux_output_connector.worker.TOPK_MAX_ELEMENTS",
        6,
        raising=False,
    )
    weight = torch.randn(7, 3)
    sizes = []

    def logits_fn(rows):
        sizes.append(len(rows))
        return rows @ weight.T

    compute_topk_logprobs(torch.randn(5, 3), logits_fn, k)
    assert sizes == calls


@pytest.mark.parametrize(
    "gathered", [True, False], ids=["logits-rank", "non-logits-rank"]
)
def test_topk_logprobs_are_computed_in_chunks(gathered):
    hidden_states = torch.randn(5, 3)
    weight = torch.randn(7, 3)
    calls = []

    def logits_fn(rows):
        calls.append(len(rows))
        return rows @ weight.T if gathered else None

    topk = compute_topk_logprobs(hidden_states, logits_fn, 2, chunk_size=2)
    assert calls == [2, 2, 1]
    if not gathered:
        assert topk is None
        return
    token_ids, logprobs = topk
    expected = torch.log_softmax(hidden_states @ weight.T, dim=-1).topk(2, dim=-1)
    assert token_ids.dtype == torch.int32
    assert token_ids.tolist() == expected.indices.tolist()
    torch.testing.assert_close(logprobs, expected.values)


def test_ranks_without_storage_compute_only_the_logits_collective():
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._buffer = None
    worker._store = None
    worker._object_writer = None
    worker._topk_logprobs = 2
    worker._session_request_ids = set()
    worker._session_generation = 0
    step = _metadata(0, [], {})
    step.metadata.sessions = {"a": "s"}
    worker.begin_step(step.metadata)
    calls = []

    def logits_fn(rows):
        calls.append(len(rows))
        return torch.zeros(len(rows), 1)

    batch = _input_batch(["a"], np.array([0]), np.array([0, 3]))
    assert worker.prepare_output(batch, torch.zeros(4, 2), logits_fn) is None
    assert calls == [3]


def test_worker_snapshots_topk_logprobs_of_session_rows_only():
    worker = _make_worker(2)
    worker._topk_logprobs = 2
    step = _metadata(
        0,
        [
            _request_metadata("plain", 0, 2, 0, []),
            _request_metadata("session", 0, 3, 0, []),
        ],
        {},
    )
    step.metadata.sessions = {"session": "s"}
    _begin_step(worker, step)
    worker._routed_experts_capturer.snapshot_routing_data.return_value = torch.zeros(
        (5, *_SHAPE), dtype=torch.uint8
    )
    hidden_states = torch.randn(6, 3)
    weight = torch.randn(7, 3)
    pending = worker.prepare_output(
        _input_batch(["plain", "session"], np.array([0, 0]), np.array([0, 2, 5])),
        hidden_states,
        lambda rows: rows @ weight.T,
    )
    pending.enqueue_cpu_copy(np.array([0, 0]), np.array([0, 0]))
    expected = torch.log_softmax(hidden_states[2:5] @ weight.T, dim=-1).topk(2)
    assert pending.fields["topk_token_ids"].tolist() == expected.indices.tolist()
    np.testing.assert_allclose(
        pending.fields["topk_logprobs"], expected.values.numpy(), rtol=1e-6
    )
    worker.close()


@pytest.mark.parametrize(
    "sessions,expected_rows",
    [(set(), []), ({"b"}, [2, 3, 4]), ({"a", "b"}, [0, 1, 2, 3, 4])],
    ids=["no-session", "one-session", "all-sessions"],
)
def test_topk_logprobs_cover_only_session_rows(sessions, expected_rows):
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._buffer = None
    worker._store = None
    worker._object_writer = None
    worker._topk_logprobs = 2
    worker._session_request_ids = set()
    worker._session_generation = 0
    step = _metadata(0, [], {})
    step.metadata.sessions = {request: "s" for request in sessions}
    worker.begin_step(step.metadata)
    hidden_states = torch.randn(6, 3)
    computed = []

    def logits_fn(rows):
        computed.append(rows)
        return torch.zeros(len(rows), 4)

    worker.prepare_output(
        _input_batch(["a", "b"], np.array([0, 0]), np.array([0, 2, 5])),
        hidden_states,
        logits_fn,
    )
    selected = torch.cat(computed) if computed else hidden_states[:0]
    assert torch.equal(selected, hidden_states[expected_rows])


def test_session_objects_without_capture_hold_only_the_other_fields():
    client, store = fake_store()
    worker = _make_worker(2, enable_r3=False, shape_per_token=(1,))
    worker._return_hidden_states = True
    worker._object_writer = SessionObjectWriter(
        store, key_prefix="job", max_pending_batches=4, fields=worker._object_fields
    )
    end = _BLOCK_SIZE + 2
    hidden = torch.arange(end, dtype=torch.float32)[:, None]
    for token_start, num_tokens, hashes in [
        (0, _BLOCK_SIZE, _HASHES[:1]),
        (_BLOCK_SIZE, 1, []),
        (_BLOCK_SIZE + 1, 1, []),
    ]:
        step = _metadata(
            0, [_request_metadata("session", token_start, num_tokens, 0, hashes)], {}
        )
        step.metadata.session_object_size = _BLOCK_SIZE
        if token_start == 0:
            step.metadata.sessions = {"session": "s"}
        _begin_step(worker, step)
        rows = hidden[token_start : token_start + num_tokens]
        pending = worker.prepare_output(
            _input_batch(
                ["plain", "session"],
                np.array([token_start, token_start]),
                np.array([0, num_tokens, 2 * num_tokens]),
            ),
            torch.cat([rows + 100, rows]),
        )
        pending.enqueue_cpu_copy(np.array([1, 1]), np.array([0, 0]))
        assert worker.process_output(pending) == {}
    assert "plain" not in worker._requests
    terminal = _metadata(0, [], {"session": []})
    terminal.metadata.session_object_size = _BLOCK_SIZE
    terminal.metadata.session_tails = {"session": ("job/s/tail/t", _BLOCK_SIZE, end)}
    _begin_step(worker, terminal)
    worker.close()

    keys = [f"job/s/{_HASHES[0].hex()}", "job/s/tail/t"]
    assert [list(load(client.objects[key])) for key in keys] == [["hidden_states"]] * 2
    np.testing.assert_array_equal(
        read_aux_rows(store, keys, "hidden_states"), hidden.numpy()
    )


class _RequestsTornDownWhileIterated(dict):
    def __init__(self, requests, teardown):
        super().__init__(requests)
        self._teardown = teardown

    def items(self):
        items = iter(super().items())
        yield next(items)
        self._teardown()
        yield from items


class _SessionsTornDownWhenRead(set):
    def __init__(self, teardown):
        super().__init__()
        self._teardown = teardown

    def __contains__(self, request_id):
        self._teardown()
        return super().__contains__(request_id)


def test_step_preparation_survives_a_concurrent_teardown():
    worker = _make_worker(2)
    first = _prepare_output(
        worker,
        _metadata(
            0,
            [
                _request_metadata("finished", 0, 2, 0, []),
                _request_metadata("running", 0, 2, 0, []),
            ],
            {},
        ),
        _position_rows(0, 4),
        ["finished", "running"],
        np.array([0, 0]),
    )
    torn_down = threading.Event()

    def teardown():
        if torn_down.is_set():
            return
        thread = threading.Thread(target=worker.process_output, args=(first,))
        thread.start()
        thread.join()
        torn_down.set()

    worker._requests = _RequestsTornDownWhileIterated(worker._requests, teardown)
    worker._session_request_ids = _SessionsTornDownWhenRead(teardown)
    second = _prepare_output(
        worker,
        _metadata(0, [_request_metadata("running", 2, 1, 0, [])], {"finished": []}),
        _position_rows(2, 3),
        ["running"],
        np.array([0]),
    )
    assert torn_down.is_set()
    worker.process_output(second)
    assert list(worker._requests) == ["running"]
    worker.close()


def test_worker_without_capture_skips_steps_without_session_rows():
    worker = _make_worker(2, enable_r3=False, shape_per_token=(1,))
    worker._return_hidden_states = True
    _begin_step(worker, _metadata(0, [], {}))
    batch = _input_batch(["plain"], np.array([0]), np.array([0, 3]))
    assert worker.prepare_output(batch, torch.zeros(3, 2)) is None
    assert worker._pending_outputs == []
    worker.close()


@pytest.mark.parametrize(
    "segments,owned",
    [
        ([(0, "r@v1"), (64, None)], "abcdt"),
        ([(0, "p@v1"), (32, "r@v2"), (48, "r@v3"), (64, None)], "cdt"),
        ([(0, "p@v1"), (40, "r@v2"), (64, None)], "dt"),
        ([(0, "p@v1"), (64, None)], "t"),
        ([(0, "r@x@v1"), (32, "r@v2"), (64, None)], "cdt"),
    ],
    ids=[
        "all-own",
        "own-from-an-object",
        "own-inside-an-object",
        "all-hit",
        "hit-of-a-request-whose-id-starts-with-it",
    ],
)
@pytest.mark.parametrize("prefix", ["j", "tail", "j/tail"])
def test_aux_owned_keys(segments, owned, prefix):
    keys = [f"{prefix}/s/{name}" for name in ("a", "b", "c", "d", "tail/t")]
    assert aux_owned_keys("r", keys, segments) == [
        key for key in keys if key[-1] in owned
    ]
