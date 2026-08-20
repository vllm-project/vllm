# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    routed_experts_keys,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
    ArtifactStoreError,
    BackgroundArtifactStore,
    InProcessArtifactStore,
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

    def get_concatenated(self, keys: list[str], *, object_size: int) -> bytes:
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
    assert store.get_concatenated(["key"], object_size=5) == b"value"
    store.close()
    assert underlying.closed


def test_background_store_surfaces_publication_failure():
    underlying = Mock()
    underlying.put.side_effect = ArtifactCapacityError("full")
    store = BackgroundArtifactStore(underlying, max_pending_batches=2)

    store.put([ArtifactObject("key", b"value")])
    with pytest.raises(ArtifactStoreError, match="publication failed"):
        store.get_concatenated(["key"], object_size=5)
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
            raise ArtifactCapacityError("batch exceeds capacity")
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
    enable_prefix_caching: bool = True,
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
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        model_config=model_config,
        instance_id="instance",
    )


@pytest.mark.parametrize(
    ("num_experts", "dtype"),
    [(256, "uint8"), (257, "uint16"), (65536, "uint16"), (65537, "int32")],
)
def test_routed_experts_shape_uses_model_arch_config(num_experts, dtype):
    config = _make_vllm_config(num_experts=num_experts)

    assert get_routing_shape_and_dtype(config) == ((3, 2), dtype)


def _make_connector(*, enable_prefix_caching: bool = True):
    return ArtifactSchedulerConnector(
        _make_vllm_config(
            enable_prefix_caching=enable_prefix_caching,
        ),
        block_size=_BLOCK_SIZE,
    )


def _make_worker(
    max_num_seqs: int,
    max_num_batched_tokens: int | None = None,
) -> ArtifactWorkerConnector:
    store = BackgroundArtifactStore(
        _make_store(object_nbytes=_BLOCK_SIZE * int(np.prod(_SHAPE))),
        max_pending_batches=2,
    )
    worker = object.__new__(ArtifactWorkerConnector)
    worker._store = store
    worker._buffer = RoutedExpertsArtifactBuffer(
        _DTYPE,
        _SHAPE,
        _BLOCK_SIZE,
        max_num_seqs,
        max_num_batched_tokens or max_num_seqs * _BLOCK_SIZE,
    )
    worker._requests = {}
    worker._generation = -1
    worker._shape_per_token = _SHAPE
    worker._dtype = _DTYPE
    worker._lock = threading.Lock()
    return worker


def test_non_output_rank_skips_capture_snapshot():
    worker = object.__new__(ArtifactWorkerConnector)
    worker._store = None
    worker._capturer = Mock()

    worker._step_metadata = Mock()
    worker._pending_capture = None

    assert worker.capture_step(1) is None
    worker._capturer.get_routing_data.assert_not_called()


def test_worker_encapsulates_step_output():
    worker = _make_worker(1)
    worker._capturer = Mock()
    worker._pending_capture = None
    metadata = ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {})

    worker.begin_step(metadata)
    worker.capture_step(1)
    output = worker.prepare_output([], Mock())

    assert output is not None
    worker._capturer.get_routing_data.assert_called_once_with(1)
    assert worker.prepare_output([], Mock()) is None
    worker.close()


def _process_output(worker, metadata, rows, request_ids, num_rejected):
    worker.begin_step(metadata)
    return worker.process_output(metadata, rows, request_ids, num_rejected)


def test_worker_rejects_mismatched_capture_shape():
    worker = _make_worker(1)
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


def test_materialize_rejects_invalid_object_size():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = array.tobytes()

    store = _make_store(object_nbytes=len(payload))
    with pytest.raises(ArtifactCorruptionError, match="size"):
        store.put([ArtifactObject("key", payload[:-1])])
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


def test_logical_buffer_moves_completed_tail_to_retained_pool():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 4)
    logical = np.arange(9 * 3 * 2, dtype=_DTYPE).reshape(9, *_SHAPE)

    assert not buffer.capture("request", 0, logical[:2])
    first = buffer.capture("request", 2, logical[2:6])
    first_retained = buffer.retain_block(first[0][1])
    second = buffer.capture("request", 6, logical[6:])
    second_retained = buffer.retain_block(second[0][1])

    np.testing.assert_array_equal(first_retained, logical[:4])
    np.testing.assert_array_equal(second_retained, logical[4:8])
    buffer.release_block(first_retained)
    buffer.release_block(second_retained)


def test_logical_buffer_retains_two_full_steps():
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 16)
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
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8)
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    hashes = [b"a" * 32, b"b" * 32]
    keys = routed_experts_keys(hashes, "0")
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


def test_worker_data_plane_publishes_blocks_and_reuses_prefix():
    worker = _make_worker(2)

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


def test_worker_reuses_prefix_when_execution_starts_inside_block():
    worker = _make_worker(2)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("first", 0, 8, 0, True, hashes)],
        {},
    )
    _process_output(worker, first, logical, ["first"], np.array([0]))

    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("second", 7, 1, 0, True, hashes)],
        {("first", 0): hashes},
    )
    output = _process_output(worker, second, logical[7:], ["second"], np.array([0]))

    np.testing.assert_array_equal(output.requests["second"].rows, logical)
    worker.close()


def test_worker_fills_capture_gap_from_published_artifact():
    worker = _make_worker(2)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    producer = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("producer", 0, 8, 0, True, hashes)],
        {},
    )
    _process_output(worker, producer, logical, ["producer"], np.array([0]))

    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("consumer", 0, 4, 0, False, hashes)],
        {("producer", 0): hashes},
    )
    _process_output(worker, first, logical[:4], ["consumer"], np.array([0]))
    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("consumer", 7, 1, 0, True, hashes)],
        {},
    )
    output = _process_output(worker, second, logical[7:], ["consumer"], np.array([0]))

    np.testing.assert_array_equal(output.requests["consumer"].rows, logical)
    worker.close()


def test_worker_rejects_unbacked_capture_gap():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 1, 0, False, [])],
        {},
    )
    _process_output(worker, first, logical[:1], ["request"], np.array([0]))
    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 2, 1, 0, False, [])],
        {},
    )

    with pytest.raises(RuntimeError, match="unbacked token gap"):
        _process_output(worker, second, logical[2:3], ["request"], np.array([0]))
    worker.close()


def test_worker_drops_overlapping_capture_rows():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 0, 3, 0, False, [])],
        {},
    )
    _process_output(worker, first, logical[:3], ["request"], np.array([0]))
    stale_overlap = logical[2:].copy()
    stale_overlap[0] += 100
    second = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("request", 2, 2, 0, True, [b"a" * 32])],
        {},
    )
    output = _process_output(
        worker,
        second,
        stale_overlap,
        ["request"],
        np.array([0]),
    )

    np.testing.assert_array_equal(output.requests["request"].rows, logical)
    worker.close()


def test_worker_publishes_entire_batch_before_materializing_prefix():
    worker = _make_worker(2)
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


def test_worker_defers_full_block_until_kv_hash_arrives():
    worker = _make_worker(1)
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
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        ),
        logical[:4],
    )
    worker.close()


def test_worker_releases_newly_keyed_blocks_before_capture():
    worker = _make_worker(1, max_num_batched_tokens=16)
    connector = _make_connector()
    block_hashes = [b"a" * 32]
    request = _scheduler_request("request", block_hashes, num_tokens=100)
    connector.request_started(request)
    logical = np.arange(52 * 3 * 2, dtype=np.uint8).reshape(52, 3, 2)

    for token_start, num_tokens in [(0, 4), (4, 16), (20, 16)]:
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
        )

    block_hashes.extend(bytes([value]) * 32 for value in range(1, 5))
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
    )

    state = worker._requests[(request.request_id, 0)]
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


def test_worker_publishes_full_block_before_restarted_epoch():
    worker = _make_worker(1)
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


def test_worker_does_not_rematerialize_emitted_rows():
    worker = _make_worker(1)
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


def test_worker_keeps_only_chunked_prefill_tail():
    worker = _make_worker(1)
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


def test_worker_emits_mid_block_chunked_prefill():
    worker = _make_worker(1)
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


def test_worker_emits_published_block_and_mid_block_tail():
    worker = _make_worker(1)
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


def test_worker_output_does_not_alias_released_tail_buffer():
    worker = _make_worker(1)
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


def test_worker_assembles_output_before_reusing_tail_buffer():
    worker = _make_worker(1)
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
    reuse = ArtifactConnectorMetadata(
        0,
        _BLOCK_SIZE,
        [ArtifactRequestMetadata("reuse", 0, 4, 0, False, [b"b" * 32])],
        {},
    )
    replacement = np.full_like(logical, 99)
    assembly_started = threading.Event()
    allow_assembly = threading.Event()
    reuse_finished = threading.Event()
    result = {}
    errors = []
    assemble = worker._assemble_segments

    def blocking_assemble(*args):
        assembly_started.set()
        assert allow_assembly.wait(timeout=2)
        return assemble(*args)

    def emit_output():
        try:
            result["output"] = _process_output(
                worker, second, logical[2:], ["request"], np.array([0])
            )
        except BaseException as error:
            errors.append(error)

    def reuse_buffer():
        try:
            _process_output(worker, reuse, replacement, ["reuse"], np.array([0]))
        except BaseException as error:
            errors.append(error)
        finally:
            reuse_finished.set()

    with patch.object(worker, "_assemble_segments", side_effect=blocking_assemble):
        output_thread = threading.Thread(target=emit_output)
        output_thread.start()
        assert assembly_started.wait(timeout=2)
        reuse_thread = threading.Thread(target=reuse_buffer)
        reuse_thread.start()
        assert not reuse_finished.wait(timeout=0.1)
        allow_assembly.set()
        output_thread.join(timeout=2)
        reuse_thread.join(timeout=2)

    assert not errors
    np.testing.assert_array_equal(result["output"].requests["request"].rows, logical)
    worker.close()


def test_worker_excludes_rejected_speculative_rows():
    worker = _make_worker(1)
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


def test_worker_retains_tail_until_inflight_output_finishes():
    worker = _make_worker(1)
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


def test_worker_discards_inflight_output_from_old_generation():
    worker = _make_worker(1)
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


def test_worker_merges_block_hash_deltas():
    worker = _make_worker(1)
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


def test_worker_discards_finished_block_without_kv_hash():
    worker = _make_worker(1)
    worker._generation = 0
    worker._requests[("request", 0)] = _WorkerRequestState(
        pending_blocks=[(0, np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE))]
    )

    worker.begin_step(
        ArtifactConnectorMetadata(0, _BLOCK_SIZE, [], {("request", 0): []})
    )
    assert ("request", 0) not in worker._requests
    worker.close()


def test_worker_discards_uncommitted_blocks_for_aborted_request():
    worker = _make_worker(1)
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


def test_worker_commits_pending_block_with_finished_request_hash():
    worker = _make_worker(1)
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
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
            rows_per_object=_BLOCK_SIZE,
        ),
        logical,
    )
    worker.close()


def test_worker_fails_when_cached_artifact_is_missing():
    worker = _make_worker(1)
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


def test_store_rejects_oversized_batch_without_partial_write():
    store = _make_store(max_bytes=6, object_nbytes=3)
    store.put([ArtifactObject("retained", b"rrr")])

    with pytest.raises(ArtifactCapacityError):
        store.put(
            [
                ArtifactObject("first", b"111"),
                ArtifactObject("second", b"222"),
                ArtifactObject("third", b"333"),
            ]
        )

    assert store.get_concatenated(["retained"], object_size=3) == b"rrr"
    with pytest.raises(ArtifactNotFoundError):
        store.get_concatenated(["first"], object_size=3)
    store.close()


def test_store_lru_and_immutable_put():
    store = _make_store(max_bytes=8)
    store.put([ArtifactObject("first", b"1111"), ArtifactObject("second", b"2222")])
    assert store.get_concatenated(["first"], object_size=4) == b"1111"

    store.put([ArtifactObject("first", b"xxxx"), ArtifactObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"], object_size=4) == b"11113333"
    with pytest.raises(ArtifactNotFoundError, match="Increase artifact_config"):
        store.get_concatenated(["second"], object_size=4)
    store.close()


def test_store_reuses_evicted_slot_without_moving_live_objects():
    store = _make_store(max_bytes=12)
    store.put(
        [
            ArtifactObject("first", b"1111"),
            ArtifactObject("second", b"2222"),
            ArtifactObject("third", b"3333"),
        ]
    )
    assert store.get_concatenated(["second"], object_size=4) == b"2222"
    store.put([ArtifactObject("fourth", b"4444")])

    assert (
        store.get_concatenated(["second", "third", "fourth"], object_size=4)
        == b"222233334444"
    )
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


def test_scheduler_connector_builds_worker_metadata_and_forwards_output():
    connector = _make_connector()
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


def test_scheduler_connector_sends_final_block_hashes():
    connector = _make_connector()
    block_hashes = [b"a" * 32]
    request = _scheduler_request("request", block_hashes)
    connector.request_started(request)
    connector.request_finished(request)
    scheduler_output = _step_output([], [], [])
    scheduler_output.finished_req_ids = {request.request_id}

    metadata = connector.build_connector_meta(scheduler_output, {})

    assert list(metadata.finished_requests[(request.request_id, 0)]) == block_hashes


def test_scheduler_connector_restarts_preempted_request_epoch():
    connector = _make_connector()
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


def test_scheduler_connector_skips_preempted_request_that_was_freed():
    connector = _make_connector()
    preempted = _step_output([], [], [])
    preempted.preempted_req_ids = {"finished"}

    metadata = connector.build_connector_meta(preempted, {})

    assert not metadata.requests


def test_scheduler_consumes_ordered_stale_artifact_outputs():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=6)
    connector.request_started(request)
    first_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    second_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2) + 40
    first = ArtifactConnectorOutput({"request": ArtifactRequestOutput(0, first_rows)})
    second = ArtifactConnectorOutput({"request": ArtifactRequestOutput(4, second_rows)})

    np.testing.assert_array_equal(
        connector.take_output(request, True, first, is_stale=True), first_rows
    )
    np.testing.assert_array_equal(
        connector.take_output(request, True, second, is_stale=True), second_rows[:1]
    )
    later = ArtifactConnectorOutput({"request": ArtifactRequestOutput(6, second_rows)})
    assert connector.take_output(request, True, later, is_stale=True) is None


def test_scheduler_ignores_stale_output_without_new_artifacts():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    connector.request_started(request)

    assert (
        connector.take_output(
            request,
            True,
            ArtifactConnectorOutput({}),
            is_stale=True,
        )
        is None
    )


def test_scheduler_rejects_stale_artifact_token_gap():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=3)
    connector.request_started(request)
    output = ArtifactConnectorOutput(
        {"request": ArtifactRequestOutput(1, np.zeros((1, 3, 2), dtype=np.uint8))}
    )

    with pytest.raises(RuntimeError, match="token gap"):
        connector.take_output(request, True, output, is_stale=True)


def test_scheduler_connector_sends_only_new_block_hashes():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=8)
    connector.request_started(request)
    scheduler_output = _step_output([request.request_id], [0], [4])

    first = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )
    request.block_hashes.append(b"b" * 32)
    second = connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )

    assert list(first.requests[0].block_hashes) == [b"a" * 32]
    assert first.requests[0].block_hash_start == 0
    assert list(second.requests[0].block_hashes) == [b"b" * 32]
    assert second.requests[0].block_hash_start == 1


def test_scheduler_connector_defers_unscheduled_block_hashes():
    connector = _make_connector()
    block_hashes = [bytes([i]) * 32 for i in range(4)]
    request = _scheduler_request("request", block_hashes, num_tokens=16)
    connector.request_started(request)

    first = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    second = connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )

    assert list(first.requests[0].block_hashes) == block_hashes[:1]
    assert list(second.requests[0].block_hashes) == block_hashes[1:2]


def test_scheduler_connector_uses_fixed_size_synthetic_hashes():
    connector = _make_connector(enable_prefix_caching=False)
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


def test_scheduler_connector_does_not_reuse_synthetic_hashes():
    connector = _make_connector(enable_prefix_caching=False)
    first_request = _scheduler_request("request", [], num_tokens=4)
    connector.request_started(first_request)
    first = connector.build_connector_meta(
        _step_output([first_request.request_id], [0], [4]),
        {first_request.request_id: first_request},
    )
    connector.request_finished(first_request)

    second_request = _scheduler_request("request", [], num_tokens=4)
    connector.request_started(second_request)
    second = connector.build_connector_meta(
        _step_output([second_request.request_id], [0], [4]),
        {second_request.request_id: second_request},
    )

    assert list(first.requests[0].block_hashes) != list(second.requests[0].block_hashes)


def test_scheduler_connector_reset_preserves_emit_cursor():
    connector = _make_connector()
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
