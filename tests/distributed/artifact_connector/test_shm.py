# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.connector import ArtifactSchedulerConnector
from vllm.distributed.artifact_connector.request_core import (
    ArtifactCommit,
    ArtifactFinalize,
    RoutedExpertsRequestCore,
    decode_routed_experts_array,
    encode_routed_experts_array,
    materialize_routed_experts,
    routed_experts_key,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactReader,
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


def test_background_store_put_returns_before_publication():
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
        hf_text_config=SimpleNamespace(
            num_hidden_layers=3,
            num_experts_per_tok=2,
            num_experts=256,
        ),
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
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_hash_algo="sha256",
        ),
        model_config=model_config,
        instance_id="instance",
    )


def _make_scheduler_connector(tmp_path, *, enable_prefix_caching: bool = True):
    config = _make_vllm_config(
        tmp_path,
        enable_prefix_caching=enable_prefix_caching,
    )
    return ArtifactSchedulerConnector(
        config,
        SimpleNamespace(num_blocks=8),
        kv_connector=None,
        block_size=4,
    )


@pytest.mark.parametrize(
    ("num_cpu_blocks", "expected_blocks"),
    [(None, 3), (2, 3), (10, 10)],
)
def test_connector_derives_shm_capacity_from_largest_kv_tier(
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

    block_nbytes = 8 * 3 * 2 * np.dtype(np.uint8).itemsize
    assert connector._store._store.max_bytes == expected_blocks * block_nbytes
    connector.shutdown()


def test_connector_rejects_explicit_shm_capacity_below_kv_minimum(tmp_path):
    config = _make_vllm_config(tmp_path, max_shm_bytes=1)

    with pytest.raises(ValueError, match="gpu_blocks=3"):
        ArtifactSchedulerConnector(
            config,
            SimpleNamespace(num_blocks=3),
            kv_connector=None,
            block_size=8,
        )


@dataclass
class _CoreHarness:
    store: LocalSharedMemoryArtifactStore
    buffer: RoutedExpertsArtifactBuffer
    core: RoutedExpertsRequestCore
    logical: np.ndarray

    def close(self) -> None:
        self.store.close()


def _make_core_harness(tmp_path, *, max_bytes: int = 1 << 20) -> _CoreHarness:
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=max_bytes,
        ttl_seconds=60,
    )
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (3, 2))
    logical = np.arange(12 * 3 * 2, dtype=np.uint8).reshape(12, 3, 2)
    return _CoreHarness(
        store=store,
        buffer=buffer,
        core=RoutedExpertsRequestCore(store, buffer),
        logical=logical,
    )


def _block_key(
    block_hash: bytes,
    artifact_namespace: str = "default",
) -> str:
    return routed_experts_key(block_hash, artifact_namespace)


def _request(
    request_id: str,
    request_attempt_id: str,
    *,
    token_end: int = 10,
) -> ArtifactFinalize:
    return ArtifactFinalize(
        request_id=request_id,
        artifact_namespace="default",
        block_hashes=[bytes([index]) * 32 for index in range(3)],
        tail_block_hash=b"t" * 32 if token_end % 4 else None,
        token_end=token_end,
        hash_block_size=4,
    )


def _scheduler_request(
    request_id: str,
    block_hashes: list[bytes],
    *,
    num_tokens: int = 10,
    prompt_start: int | None = 0,
):
    request = Mock()
    request.request_id = request_id
    request.block_hashes = block_hashes
    request.num_tokens = num_tokens
    request.all_token_ids = list(range(num_tokens))
    request.mm_features = []
    request.lora_request = None
    request.cache_salt = None
    request.prompt_embeds = None
    request.num_prompt_tokens = num_tokens
    request.sampling_params = SimpleNamespace(routed_experts_prompt_start=prompt_start)
    return request


def _commit_request(request: ArtifactFinalize) -> ArtifactCommit | None:
    block_end = request.token_end // request.hash_block_size * request.hash_block_size
    if block_end <= 0:
        return None
    return ArtifactCommit(
        request_id=request.request_id,
        artifact_namespace=request.artifact_namespace,
        block_hashes=request.block_hashes[: block_end // request.hash_block_size],
        block_start=0,
        hash_block_size=request.hash_block_size,
    )


def _prepare_and_finalize(
    harness: _CoreHarness,
    request: ArtifactFinalize,
    logical: np.ndarray,
):
    harness.buffer.capture(request.request_id, 0, logical[: request.token_end])
    commit = _commit_request(request)
    if commit is not None:
        harness.core.commit([commit])
    return harness.core.finalize(request)


def _read_in_child(root: str, store_id: str, keys: list[str], result_queue) -> None:
    reader = LocalSharedMemoryArtifactReader(root, store_id)
    result_queue.put(materialize_routed_experts(reader, keys).tolist())


def test_object_envelope_round_trip():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = encode_routed_experts_array(
        key="key",
        kind="block",
        array=array,
        source_token_start=0,
    )

    decoded, header = decode_routed_experts_array(payload, expected_key="key")

    np.testing.assert_array_equal(decoded, array)
    assert header["shape"] == [4, 3, 2]


def test_object_envelope_rejects_corruption():
    payload = encode_routed_experts_array(
        key="key",
        kind="tail",
        array=np.zeros((2, 1), dtype=np.uint8),
        source_token_start=0,
    )
    corrupted = payload[:-1] + bytes([payload[-1] ^ 1])

    with pytest.raises(ArtifactCorruptionError, match="checksum"):
        decode_routed_experts_array(corrupted, expected_key="key")


@pytest.mark.parametrize(
    ("expected_shape", "expected_dtype", "error"),
    [
        ((3, 1), np.dtype("uint8"), "shape"),
        ((3, 2), np.dtype("uint16"), "dtype"),
    ],
)
def test_materialize_rejects_model_schema_mismatch(
    tmp_path, expected_shape, expected_dtype, error
):
    harness = _make_core_harness(tmp_path)
    request = _request("request-a", "a" * 32, token_end=4)
    keys = _prepare_and_finalize(harness, request, harness.logical)

    with pytest.raises(ArtifactCorruptionError, match=error):
        materialize_routed_experts(
            harness.store,
            keys,
            expected_shape_per_token=expected_shape,
            expected_dtype=expected_dtype,
        )
    harness.close()


@pytest.mark.parametrize("expected_token_end", [9, 12])
def test_materialize_rejects_terminal_range_mismatch(tmp_path, expected_token_end):
    harness = _make_core_harness(tmp_path)
    request = _request("request-a", "a" * 32, token_end=10)
    keys = _prepare_and_finalize(harness, request, harness.logical)

    with pytest.raises(ArtifactCorruptionError, match="terminal token range"):
        materialize_routed_experts(
            harness.store,
            keys,
            expected_token_end=expected_token_end,
            hash_block_size=4,
        )
    harness.close()


def test_logical_buffer_survives_recompute_and_release():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,))
    buffer.capture("request", 4, np.arange(4, 8, dtype=np.uint8).reshape(-1, 1))
    buffer.capture("request", 6, np.array([[60], [70], [80]], dtype=np.uint8))

    np.testing.assert_array_equal(
        buffer.read("request", 4, 9).ravel(),
        [4, 5, 60, 70, 80],
    )
    buffer.release_through("request", 8)
    np.testing.assert_array_equal(buffer.read("request", 8, 9).ravel(), [80])
    buffer.capture("request", 0, np.arange(9, dtype=np.uint8).reshape(-1, 1))
    np.testing.assert_array_equal(buffer.read("request", 8, 9).ravel(), [8])


def test_logical_buffer_encodes_router_ids_to_artifact_dtype():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,))

    buffer.capture("request", 0, np.array([[1], [2]], dtype=np.int32))

    encoded = buffer.read("request", 0, 2)
    assert encoded.dtype == np.uint8
    np.testing.assert_array_equal(encoded.ravel(), [1, 2])


def test_core_returns_ordered_keys_and_shm_value(tmp_path):
    harness = _make_core_harness(tmp_path)
    request = _request("request-a", "a" * 32)

    keys = _prepare_and_finalize(harness, request, harness.logical)

    assert len(keys) == 3
    assert keys[:2] == [
        _block_key(block_hash) for block_hash in request.block_hashes[:2]
    ]
    assert keys[2] == _block_key(request.tail_block_hash)  # type: ignore[arg-type]
    np.testing.assert_array_equal(
        materialize_routed_experts(harness.store, keys),
        harness.logical[:10],
    )
    harness.close()


def test_exact_block_request_has_no_tail(tmp_path):
    harness = _make_core_harness(tmp_path)
    request = _request("request-a", "a" * 32, token_end=8)

    keys = _prepare_and_finalize(harness, request, harness.logical)

    assert len(keys) == 2
    assert keys == [_block_key(block_hash) for block_hash in request.block_hashes[:2]]
    np.testing.assert_array_equal(
        materialize_routed_experts(harness.store, keys), harness.logical[:8]
    )
    harness.close()


def test_finalize_recovers_tail_from_released_full_block(tmp_path):
    harness = _make_core_harness(tmp_path)
    harness.buffer.capture("request-a", 0, harness.logical[:8])
    commit = ArtifactCommit(
        request_id="request-a",
        artifact_namespace="default",
        block_hashes=[bytes([index]) * 32 for index in range(2)],
        block_start=0,
        hash_block_size=4,
    )
    harness.core.commit([commit])

    request = _request("request-a", "attempt-a", token_end=6)
    keys = harness.core.finalize(request)

    assert len(keys) == 2
    np.testing.assert_array_equal(
        materialize_routed_experts(harness.store, keys), harness.logical[:6]
    )
    harness.close()


@pytest.mark.parametrize("token_end", [1, 4, 5, 8, 9, 12])
def test_key_count_is_ceiling_of_executed_tokens(tmp_path, token_end):
    harness = _make_core_harness(tmp_path)
    request = _request(
        f"request-{token_end}",
        f"{token_end:032x}",
        token_end=token_end,
    )

    keys = _prepare_and_finalize(harness, request, harness.logical)

    assert len(keys) == (token_end + 3) // 4
    harness.close()


def test_cached_blocks_are_reused_without_put(tmp_path):
    harness = _make_core_harness(tmp_path)
    first = _request("request-a", "a" * 32, token_end=8)
    first_keys = _prepare_and_finalize(harness, first, harness.logical)
    second = _request(
        "request-b",
        "b" * 32,
        token_end=10,
    )
    harness.buffer.capture(second.request_id, 8, harness.logical[8:10])
    harness.store.put = Mock(wraps=harness.store.put)

    keys = harness.core.finalize(second)

    assert harness.store.put.call_count == 1
    assert len(harness.store.put.call_args.args[0]) == 1
    assert keys[:2] == first_keys
    np.testing.assert_array_equal(
        materialize_routed_experts(harness.store, keys), harness.logical[:10]
    )
    harness.close()


def test_missing_cached_block_fails_closed(tmp_path):
    harness = _make_core_harness(tmp_path)
    request = _request(
        "request-a",
        "a" * 32,
        token_end=4,
    )

    keys = harness.core.finalize(request)
    with pytest.raises(RuntimeError, match="does not exist"):
        materialize_routed_experts(harness.store, keys)
    harness.close()


def test_reader_materializes_from_another_process(tmp_path):
    harness = _make_core_harness(tmp_path)
    request = _request("request-a", "a" * 32)
    keys = _prepare_and_finalize(harness, request, harness.logical)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_read_in_child,
        args=(
            str(tmp_path),
            harness.store.store_id,
            keys,
            result_queue,
        ),
    )

    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)

    assert process.exitcode == 0
    np.testing.assert_array_equal(np.asarray(result_queue.get()), harness.logical[:10])
    harness.close()


def test_store_rejects_object_larger_than_capacity_without_eviction(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=5,
        ttl_seconds=60,
    )
    store.put([ArtifactObject("small", b"1234")])

    with pytest.raises(ArtifactCapacityError):
        store.put([ArtifactObject("large", b"123456")])

    assert store.get(["small"]) == [b"1234"]
    with pytest.raises(ArtifactNotFoundError):
        store.get(["large"])
    store.close()


def test_store_rejects_batch_larger_than_capacity_without_partial_write(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=5,
        ttl_seconds=60,
    )
    store.put([ArtifactObject("retained", b"r")])

    with pytest.raises(ArtifactCapacityError):
        store.put(
            [
                ArtifactObject("first", b"111"),
                ArtifactObject("second", b"222"),
            ]
        )

    assert store.get(["retained"]) == [b"r"]
    with pytest.raises(ArtifactNotFoundError):
        store.get(["first"])
    with pytest.raises(ArtifactNotFoundError):
        store.get(["second"])
    store.close()


def test_store_lru_evicts_least_recently_read_object(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=8,
        ttl_seconds=60,
    )
    store.put(
        [
            ArtifactObject("first", b"1111"),
            ArtifactObject("second", b"2222"),
        ]
    )
    assert store.get(["first"]) == [b"1111"]

    store.put([ArtifactObject("third", b"3333")])

    assert store.get(["first", "third"]) == [b"1111", b"3333"]
    with pytest.raises(ArtifactNotFoundError, match="Increase artifact_config"):
        store.get(["second"])
    store.close()


def test_store_duplicate_put_keeps_first_value_and_refreshes_lru(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=2,
        ttl_seconds=60,
    )
    store.put([ArtifactObject("key", b"a")])
    store.put([ArtifactObject("other", b"o")])
    store.put(
        [
            ArtifactObject("key", b"b"),
            ArtifactObject("new", b"n"),
        ]
    )

    assert store.get(["key"]) == [b"a"]
    assert store.get(["new"]) == [b"n"]
    with pytest.raises(ArtifactNotFoundError):
        store.get(["other"])
    store.close()


def test_live_store_retains_objects_older_than_ttl(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    store.put([ArtifactObject("key", b"value")])
    object_path = store._path("key")
    partial = store.objects_dir / ".orphan.partial"
    partial.write_bytes(b"partial")
    old_time = time.time() - 5
    os.utime(object_path, (old_time, old_time))
    os.utime(partial, (old_time, old_time))

    store.put([ArtifactObject("second", b"value")])

    assert object_path.exists()
    store.close()

    reopened = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    assert reopened.get(["key"]) == [b"value"]
    assert not partial.exists()
    reopened.close()


def test_ttl_removes_inactive_engine_store(tmp_path):
    stale = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "stale-instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
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

    fresh = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "fresh-instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )

    assert not stale_root.exists()
    fresh.close()


def test_ttl_keeps_expired_store_with_live_writer(tmp_path):
    live = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "live-instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    live.put([ArtifactObject("key", b"value")])
    live_root = live.root

    old_time = time.time() - 5
    for path in [
        *live.objects_dir.iterdir(),
        live.objects_dir,
        live_root / ".writer.lock",
        live_root,
    ]:
        os.utime(path, (old_time, old_time))

    collector = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "collector-instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )

    assert live_root.exists()
    assert live.get(["key"]) == [b"value"]
    collector.close()
    live.close()


def test_writer_lock_retries_if_collector_unlinks_open_inode(tmp_path, monkeypatch):
    store = object.__new__(LocalSharedMemoryArtifactStore)
    store.root = tmp_path / "store"
    store.ttl_seconds = 60

    real_flock = fcntl.flock
    first_call = True

    def unlink_on_first_flock(fd, operation):
        nonlocal first_call
        real_flock(fd, operation)
        if first_call:
            first_call = False
            (store.root / ".writer.lock").unlink()
            (store.root / ".writer.lock").touch(mode=0o600)

    monkeypatch.setattr(fcntl, "flock", unlink_on_first_flock)
    fd = store._acquire_writer_lock()
    try:
        opened = os.fstat(fd)
        current = (store.root / ".writer.lock").stat()
        assert (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
    finally:
        os.close(fd)


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


def test_connector_splits_50721_snapshot_by_request(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    requests = [
        _scheduler_request("request-a", [b"a" * 32], num_tokens=5),
        _scheduler_request("request-b", [b"b" * 32], num_tokens=5),
    ]
    for request in requests:
        connector.request_started(
            request=request,
            cached_token_end=0,
            hash_block_size=4,
        )
    routing = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    connector.capture_step(
        _step_output([request.request_id for request in requests], [0, 0], [4, 4]),
        routing,
        [request.request_id for request in requests],
    )

    for request in requests:
        connector.request_progress(
            request=request,
            accepted_token_end=4,
            hash_block_size=4,
        )
    np.testing.assert_array_equal(
        connector.take_output(request=requests[0], token_end=4, hash_block_size=4),
        routing[:4],
    )
    np.testing.assert_array_equal(
        connector.take_output(request=requests[1], token_end=4, hash_block_size=4),
        routing[4:],
    )
    connector.request_finished(request=requests[0], token_end=4, hash_block_size=4)
    connector.request_finished(request=requests[1], token_end=4, hash_block_size=4)
    connector.shutdown()


@pytest.mark.parametrize("prompt_start", [2, None])
def test_connector_applies_prompt_start_to_inline_output(tmp_path, prompt_start):
    connector = _make_scheduler_connector(tmp_path)
    request = _scheduler_request(
        "request-a",
        [b"a" * 32],
        num_tokens=5,
        prompt_start=prompt_start,
    )
    connector.request_started(request=request, cached_token_end=0, hash_block_size=4)
    routing = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    connector.capture_step(
        _step_output([request.request_id], [0], [4]),
        routing,
        [request.request_id],
    )
    connector.request_progress(request=request, accepted_token_end=4, hash_block_size=4)

    expected_start = prompt_start or 0
    np.testing.assert_array_equal(
        connector.take_output(request=request, token_end=4, hash_block_size=4),
        routing[expected_start:],
    )
    connector.request_finished(request=request, token_end=4, hash_block_size=4)
    connector.shutdown()


@pytest.mark.parametrize("prompt_start", [-1, 5])
def test_connector_rejects_invalid_prompt_start(tmp_path, prompt_start):
    connector = _make_scheduler_connector(tmp_path)
    request = _scheduler_request(
        "request-a",
        [b"a" * 32],
        num_tokens=5,
        prompt_start=prompt_start,
    )

    with pytest.raises(ValueError, match="routed_experts_prompt_start"):
        connector.request_started(
            request=request,
            cached_token_end=0,
            hash_block_size=4,
        )
    connector.shutdown()


def test_connector_reuses_cached_blocks_and_captures_only_suffix(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    block_hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    first = _scheduler_request("request-a", block_hashes, num_tokens=9)
    logical = np.arange(10 * 3 * 2, dtype=np.uint8).reshape(10, 3, 2)
    connector.request_started(request=first, cached_token_end=0, hash_block_size=4)
    connector.capture_step(
        _step_output([first.request_id], [0], [8]),
        logical[:8],
        [first.request_id],
    )
    connector.request_progress(request=first, accepted_token_end=8, hash_block_size=4)
    np.testing.assert_array_equal(
        connector.take_output(request=first, token_end=8, hash_block_size=4),
        logical[:8],
    )
    connector.request_finished(request=first, token_end=8, hash_block_size=4)

    second = _scheduler_request("request-b", block_hashes, num_tokens=12)
    connector.request_started(request=second, cached_token_end=8, hash_block_size=4)
    connector.capture_step(
        _step_output([second.request_id], [8], [2]),
        logical[8:10],
        [second.request_id],
    )
    connector.request_progress(request=second, accepted_token_end=10, hash_block_size=4)

    np.testing.assert_array_equal(
        connector.take_output(request=second, token_end=10, hash_block_size=4),
        logical,
    )
    next_row = np.full((1, 3, 2), 255, dtype=np.uint8)
    connector.capture_step(
        _step_output([second.request_id], [10], [1]),
        next_row,
        [second.request_id],
    )
    connector.request_progress(request=second, accepted_token_end=11, hash_block_size=4)
    np.testing.assert_array_equal(
        connector.take_output(request=second, token_end=11, hash_block_size=4),
        next_row,
    )
    connector.request_finished(request=second, token_end=11, hash_block_size=4)
    connector.shutdown()


def test_connector_uses_request_private_keys_without_prefix_caching(tmp_path):
    connector = _make_scheduler_connector(tmp_path, enable_prefix_caching=False)
    logical = np.arange(6 * 3 * 2, dtype=np.uint8).reshape(6, 3, 2)
    all_keys = []

    for request_id in ("request-a", "request-b"):
        request = _scheduler_request(request_id, [], num_tokens=6)
        connector.request_started(
            request=request,
            cached_token_end=0,
            hash_block_size=4,
        )
        connector.capture_step(
            _step_output([request_id], [0], [6]),
            logical,
            [request_id],
        )
        connector.request_progress(
            request=request,
            accepted_token_end=6,
            hash_block_size=4,
        )
        keys = connector.request_finished(
            request=request,
            token_end=6,
            hash_block_size=4,
        )
        all_keys.append(keys)
        np.testing.assert_array_equal(
            materialize_routed_experts(connector._store, keys),
            logical,
        )

    assert all_keys[0] != all_keys[1]
    connector.shutdown()


def test_cached_kv_with_missing_artifact_fails_at_get(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    request = _scheduler_request("request-a", [b"a" * 32], num_tokens=5)
    connector.request_started(request=request, cached_token_end=4, hash_block_size=4)

    with pytest.raises(ArtifactNotFoundError):
        connector.take_output(request=request, token_end=4, hash_block_size=4)
    connector.shutdown()


def test_capture_requires_50721_output_for_nonempty_step(tmp_path):
    connector = _make_scheduler_connector(tmp_path)

    with pytest.raises(RuntimeError, match="capture output is missing"):
        connector.capture_step(_step_output(["request"], [0], [1]), None, ["request"])
    connector.shutdown()


def test_capture_drops_output_for_an_aborted_request(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    connector.request_aborted("request")
    connector.capture_step(
        _step_output(["request"], [0], [1]),
        np.zeros((1, 3, 2), dtype=np.uint8),
        ["request"],
    )

    with pytest.raises(RuntimeError, match="buffer is missing"):
        connector._buffer.read("request", 0, 1)
    connector.shutdown()


def test_capture_drops_stale_output_after_cache_reset(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(request=request, cached_token_end=0, hash_block_size=4)
    step_output = _step_output([request.request_id], [0], [1])
    rows = np.zeros((1, 3, 2), dtype=np.uint8)

    connector.capture_step(
        step_output,
        rows,
        [request.request_id],
        {request.request_id},
    )
    with pytest.raises(RuntimeError, match="buffer is missing"):
        connector._buffer.read(request.request_id, 0, 1)

    connector.capture_step(step_output, rows, [request.request_id])
    np.testing.assert_array_equal(
        connector._buffer.read(request.request_id, 0, 1),
        rows,
    )
    connector.shutdown()


def test_cache_generation_preserves_delivery_cursor_and_changes_namespace(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.request_started(request=request, cached_token_end=0, hash_block_size=4)
    connector._state(request.request_id).emit_cursor = 3
    connector._buffer.capture(
        request.request_id,
        0,
        np.zeros((3, 3, 2), dtype=np.uint8),
    )
    old_namespace = connector._state(request.request_id).artifact_namespace

    connector.reset()

    assert request.request_id not in connector._states
    assert connector._resume_emit_cursors[request.request_id] == 3
    with pytest.raises(RuntimeError, match="buffer is missing"):
        connector._buffer.read(request.request_id, 0, 1)

    connector.request_started(request=request, cached_token_end=0, hash_block_size=4)
    state = connector._state(request.request_id)
    new_namespace = state.artifact_namespace
    assert state.emit_cursor == 3
    assert request.request_id not in connector._resume_emit_cursors
    assert new_namespace != old_namespace
    assert routed_experts_key(b"a" * 32, new_namespace) != routed_experts_key(
        b"a" * 32,
        old_namespace,
    )

    connector.shutdown()


@pytest.mark.parametrize("reset_successful", [False, True])
def test_scheduler_resets_artifacts_only_after_successful_kv_reset(
    reset_successful,
):
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
