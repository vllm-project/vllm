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
from vllm.distributed.artifact_connector.connector import ArtifactSchedulerConnector
from vllm.distributed.artifact_connector.request_core import (
    RoutedExpertsRequestCore,
    decode_routed_experts_array,
    encode_routed_experts_array,
    materialize_routed_experts,
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
        SimpleNamespace(num_blocks=8),
        kv_connector=None,
        block_size=_BLOCK_SIZE,
    )


@pytest.mark.parametrize(
    ("num_cpu_blocks", "expected_blocks"),
    [(None, 3), (2, 3), (10, 10)],
)
def test_connector_capacity_covers_largest_kv_tier(
    tmp_path,
    num_cpu_blocks,
    expected_blocks,
):
    config = _make_vllm_config(tmp_path, max_shm_bytes=None)
    kv_connector = None
    if num_cpu_blocks is not None:
        kv_connector = Mock()
        kv_connector.scheduler_manager = SimpleNamespace(num_cpu_blocks=num_cpu_blocks)

    connector = ArtifactSchedulerConnector(
        config,
        SimpleNamespace(num_blocks=3),
        kv_connector=kv_connector,
        block_size=8,
    )

    assert connector._store._store.max_bytes == expected_blocks * 8 * 3 * 2
    connector.shutdown()


def test_connector_rejects_capacity_below_kv_minimum(tmp_path):
    with pytest.raises(ValueError, match="gpu_blocks=3"):
        ArtifactSchedulerConnector(
            _make_vllm_config(tmp_path, max_shm_bytes=1),
            SimpleNamespace(num_blocks=3),
            kv_connector=None,
            block_size=8,
        )


def test_raw_object_round_trip_and_size_validation():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = encode_routed_experts_array(array)
    decoded = decode_routed_experts_array(
        payload,
        shape_per_token=_SHAPE,
        dtype=_DTYPE,
    )
    np.testing.assert_array_equal(decoded, array)

    with pytest.raises(ArtifactCorruptionError, match="size"):
        decode_routed_experts_array(
            payload[:-1],
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        )


def test_logical_buffer_handles_overlap_and_release():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,))
    buffer.capture("request", 4, np.arange(4, 8, dtype=np.int32).reshape(-1, 1))
    buffer.capture("request", 6, np.array([[60], [70], [80]], dtype=np.uint8))

    np.testing.assert_array_equal(
        buffer.read("request", 4, 9).ravel(),
        [4, 5, 60, 70, 80],
    )
    buffer.release_through("request", 8)
    np.testing.assert_array_equal(buffer.read("request", 8, 9).ravel(), [80])


def _make_store(tmp_path, *, max_bytes: int = 1 << 20, instance="instance"):
    return LocalSharedMemoryArtifactStore(
        str(tmp_path),
        instance,
        0,
        max_bytes=max_bytes,
        ttl_seconds=60,
    )


def test_request_core_publishes_full_blocks(tmp_path):
    store = _make_store(tmp_path)
    buffer = RoutedExpertsArtifactBuffer(_DTYPE, _SHAPE)
    core = RoutedExpertsRequestCore(store, buffer)
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    hashes = [b"a" * 32, b"b" * 32]
    buffer.capture("request", 0, logical)

    core.commit(
        request_id="request",
        artifact_namespace="0",
        block_hashes=hashes,
        block_start=0,
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
    with pytest.raises(RuntimeError, match="buffer is missing"):
        buffer.read("request", 0, 1)
    store.close()


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
        *stale.objects_dir.iterdir(),
        stale.objects_dir,
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


def _capture(
    connector: ArtifactSchedulerConnector,
    request,
    rows: np.ndarray,
    *,
    token_start: int = 0,
) -> None:
    connector.capture_step(
        _step_output([request.request_id], [token_start], [len(rows)]),
        rows,
        [request.request_id],
    )


def test_connector_splits_snapshot_by_request(tmp_path):
    connector = _make_connector(tmp_path)
    requests = [
        _scheduler_request("request-a", [b"a" * 32], num_tokens=5),
        _scheduler_request("request-b", [b"b" * 32], num_tokens=5),
    ]
    for request in requests:
        connector.request_started(
            request=request, cached_token_end=0, hash_block_size=_BLOCK_SIZE
        )
    routing = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    connector.capture_step(
        _step_output([request.request_id for request in requests], [0, 0], [4, 4]),
        routing,
        [request.request_id for request in requests],
    )

    for request in requests:
        connector.request_progress(
            request=request, accepted_token_end=4, hash_block_size=_BLOCK_SIZE
        )
    np.testing.assert_array_equal(
        connector.take_output(
            request=requests[0], token_end=4, hash_block_size=_BLOCK_SIZE
        ),
        routing[:4],
    )
    np.testing.assert_array_equal(
        connector.take_output(
            request=requests[1], token_end=4, hash_block_size=_BLOCK_SIZE
        ),
        routing[4:],
    )
    for request in requests:
        connector.request_finished(request.request_id)
    connector.shutdown()


def test_connector_reuses_cached_prefix_and_returns_suffix(tmp_path):
    connector = _make_connector(tmp_path)
    hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    logical = np.arange(10 * 3 * 2, dtype=np.uint8).reshape(10, 3, 2)
    first = _scheduler_request("first", hashes, num_tokens=9)
    connector.request_started(
        request=first, cached_token_end=0, hash_block_size=_BLOCK_SIZE
    )
    _capture(connector, first, logical[:8])
    connector.request_progress(
        request=first, accepted_token_end=8, hash_block_size=_BLOCK_SIZE
    )
    connector.take_output(request=first, token_end=8, hash_block_size=_BLOCK_SIZE)
    connector.request_finished(first.request_id)

    second = _scheduler_request("second", hashes, num_tokens=11)
    connector.request_started(
        request=second, cached_token_end=8, hash_block_size=_BLOCK_SIZE
    )
    _capture(connector, second, logical[8:10], token_start=8)
    np.testing.assert_array_equal(
        connector.take_output(
            request=second, token_end=10, hash_block_size=_BLOCK_SIZE
        ),
        logical,
    )
    connector.request_finished(second.request_id)
    connector.shutdown()


def test_connector_private_keys_without_prefix_cache(tmp_path):
    connector = _make_connector(tmp_path, enable_prefix_caching=False)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)

    for request_id in ("first", "second"):
        request = _scheduler_request(request_id, [], num_tokens=5)
        connector.request_started(
            request=request, cached_token_end=0, hash_block_size=_BLOCK_SIZE
        )
        _capture(connector, request, logical)
        connector.request_progress(
            request=request, accepted_token_end=4, hash_block_size=_BLOCK_SIZE
        )
        np.testing.assert_array_equal(
            connector.take_output(
                request=request, token_end=4, hash_block_size=_BLOCK_SIZE
            ),
            logical,
        )
        connector.request_finished(request_id)

    connector.shutdown()
    assert len(connector._store._store._lru) == 2


def test_cached_kv_with_missing_artifact_fails_at_get(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(
        request=request, cached_token_end=4, hash_block_size=_BLOCK_SIZE
    )

    with pytest.raises(ArtifactNotFoundError):
        connector.take_output(request=request, token_end=4, hash_block_size=_BLOCK_SIZE)
    connector.shutdown()


def test_reset_changes_namespace_and_preserves_emit_cursor(tmp_path):
    connector = _make_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(
        request=request, cached_token_end=0, hash_block_size=_BLOCK_SIZE
    )
    connector._state(request.request_id).emit_cursor = 3
    old_namespace = connector._artifact_namespace

    connector.reset()
    connector.request_started(
        request=request, cached_token_end=0, hash_block_size=_BLOCK_SIZE
    )

    assert connector._state(request.request_id).emit_cursor == 3
    assert connector._artifact_namespace != old_namespace
    assert routed_experts_key(b"a" * 32, connector._artifact_namespace) != (
        routed_experts_key(b"a" * 32, old_namespace)
    )
    connector.shutdown()


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
