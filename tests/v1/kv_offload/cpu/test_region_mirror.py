# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the multi-node CPU offload region mirror."""

import threading
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

import vllm.v1.kv_offload.cpu.region_mirror as region_mirror
from vllm.v1.kv_offload.base import TransferResult
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.region_mirror import (
    GPUMirrorClient,
    MirrorStuckError,
    RegionMirrorClient,
    RegionMirrorHead,
    RegionMirrorWorker,
    SlotLayout,
    resolve_mirror_source,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

SLOT = 4096
NUM_CHUNKS = 8
HEAD_LAYOUT = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=2 * SLOT, size=SLOT)
LOCAL_LAYOUT = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=SLOT, size=SLOT)


def _rows(region: SharedOffloadRegion, layout: SlotLayout) -> np.ndarray:
    assert region.mmap_obj is not None
    flat = np.frombuffer(region.mmap_obj, dtype=np.uint8)
    return flat[: layout.num_chunks * layout.row_stride].reshape(
        layout.num_chunks, layout.row_stride
    )


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def nixl_mirror():
    """A head region and a remote worker's compact region joined by NIXL."""
    pytest.importorskip("nixl")
    tag = uuid.uuid4().hex[:8]
    head_region = SharedOffloadRegion(
        engine_id=f"mirror-head-{tag}",
        num_chunks=NUM_CHUNKS,
        rank=None,
        kv_bytes_per_chunk=HEAD_LAYOUT.row_stride,
        cpu_page_size=SLOT,
    )
    local_region = SharedOffloadRegion(
        engine_id=f"mirror-local-{tag}",
        num_chunks=NUM_CHUNKS,
        rank=0,
        kv_bytes_per_chunk=LOCAL_LAYOUT.row_stride,
        cpu_page_size=SLOT,
    )
    head = RegionMirrorHead(head_region, HEAD_LAYOUT, f"head-{tag}")
    client = RegionMirrorClient(
        local_region,
        LOCAL_LAYOUT,
        HEAD_LAYOUT,
        1,
        head.metadata,
        head.base,
        f"client-{tag}",
        timeout_s=10,
        max_transfer_bytes=2 * SLOT,
    )
    head.add_peers([client.metadata])
    head_rows = _rows(head_region, HEAD_LAYOUT)
    local_rows = _rows(local_region, LOCAL_LAYOUT)
    try:
        yield head_rows, local_rows, client
    finally:
        del head_rows, local_rows
        client.shutdown()
        head.shutdown()
        local_region.cleanup()
        head_region.cleanup()


def test_push_writes_only_the_remote_slot(nixl_mirror):
    head_rows, local_rows, client = nixl_mirror
    head_rows[:] = 7
    for chunk in range(NUM_CHUNKS):
        local_rows[chunk] = 100 + chunk

    client.push(np.array([2, 5]))

    for chunk in range(NUM_CHUNKS):
        # Rank 1's slot of a pushed chunk, nothing else.
        expected = 100 + chunk if chunk in (2, 5) else 7
        assert (head_rows[chunk, SLOT:] == expected).all()
        assert (head_rows[chunk, :SLOT] == 7).all()


def test_pull_reads_the_remote_slot(nixl_mirror):
    head_rows, local_rows, client = nixl_mirror
    local_rows[:] = 0
    for chunk in range(NUM_CHUNKS):
        head_rows[chunk, :SLOT] = 1
        head_rows[chunk, SLOT:] = 50 + chunk

    client.pull(np.array([1, 3, 6]))

    for chunk in range(NUM_CHUNKS):
        expected = 50 + chunk if chunk in (1, 3, 6) else 0
        assert (local_rows[chunk] == expected).all()


class _FakeClient:
    """Stands in for RegionMirrorClient: copies run when the test lets them."""

    def __init__(self, chunks_per_transfer: int = 64):
        self.chunks_per_transfer = chunks_per_transfer
        self.calls: list[tuple[str, list[int]]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail: str | None = None
        self.stuck: str | None = None
        self.stopped = False

    def _copy(self, op: str, chunk_ids: np.ndarray) -> None:
        self.gate.wait()
        self.calls.append((op, chunk_ids.tolist()))
        if self.stuck == op:
            raise MirrorStuckError(f"{op} stuck")
        if self.fail == op:
            raise RuntimeError(f"{op} failed")

    def push(self, chunk_ids: np.ndarray) -> None:
        self._copy("push", chunk_ids)

    def pull(self, chunk_ids: np.ndarray) -> None:
        self._copy("pull", chunk_ids)

    def stop(self, reason: str = "") -> None:
        self.stopped = True
        self.gate.set()

    def shutdown(self) -> None:
        pass


def _mirror_worker(client: _FakeClient) -> tuple[RegionMirrorWorker, MagicMock]:
    inner = MagicMock()
    inner.submit_store.return_value = True
    inner.submit_load.return_value = True
    inner.get_finished.return_value = []
    return RegionMirrorWorker(inner, client=client), inner  # type: ignore[arg-type]


def _store(worker: RegionMirrorWorker, inner: MagicMock, job_id: int, chunks):
    assert worker.submit_store(job_id, MagicMock(), CPULoadStoreSpec(chunks))
    inner.get_finished.return_value = [TransferResult(job_id=job_id, success=True)]
    assert worker.get_finished() == []
    inner.get_finished.return_value = []


def _finished(worker: RegionMirrorWorker) -> list[TransferResult]:
    results: list[TransferResult] = []

    def collected() -> bool:
        results.extend(worker.get_finished())
        return bool(results)

    _wait_for(collected)
    return results


def test_store_is_reported_only_after_its_push():
    client = _FakeClient()
    client.gate.clear()
    worker, inner = _mirror_worker(client)
    try:
        _store(worker, inner, 1, [3, 4])
        time.sleep(0.05)
        assert worker.get_finished() == []

        client.gate.set()
        assert [r.job_id for r in _finished(worker)] == [1]
        assert client.calls == [("push", [3, 4])]
    finally:
        worker.shutdown()


def test_load_dma_starts_only_after_its_pull():
    client = _FakeClient()
    client.gate.clear()
    worker, inner = _mirror_worker(client)
    try:
        assert worker.submit_load(2, CPULoadStoreSpec([5]), MagicMock())
        time.sleep(0.05)
        worker.get_finished()
        inner.submit_load.assert_not_called()

        client.gate.set()
        _wait_for(lambda: worker.get_finished() == [] and inner.submit_load.called)
        assert client.calls == [("pull", [5])]
    finally:
        worker.shutdown()


def test_wait_submits_a_pending_load_before_waiting():
    client = _FakeClient()
    worker, inner = _mirror_worker(client)
    try:
        assert worker.submit_load(3, CPULoadStoreSpec([1]), MagicMock())
        worker.wait({3})
        inner.submit_load.assert_called_once()
        inner.wait.assert_called_once_with({3})
    finally:
        worker.shutdown()


@pytest.mark.parametrize("op", ["push", "pull"])
def test_failed_copy_reports_a_failed_transfer(op):
    client = _FakeClient()
    client.fail = op
    worker, inner = _mirror_worker(client)
    try:
        if op == "push":
            _store(worker, inner, 4, [2])
        else:
            assert worker.submit_load(4, CPULoadStoreSpec([2]), MagicMock())
        assert _finished(worker) == [TransferResult(job_id=4, success=False)]
        inner.submit_load.assert_not_called()
    finally:
        worker.shutdown()


def test_a_stuck_transfer_is_never_reported_and_stops_the_engine():
    """A transfer that may still land must not let the chunk be reused."""
    client = _FakeClient()
    client.stuck = "push"
    worker, inner = _mirror_worker(client)
    try:
        _store(worker, inner, 5, [6])
        _wait_for(lambda: client.stopped)
        with pytest.raises(RuntimeError, match="may still land"):
            worker.get_finished()
    finally:
        worker.shutdown()
    # Its region may still be written, so it stays mapped.
    inner.shutdown.assert_not_called()


def test_loads_overtake_a_store_between_its_transfers():
    client = _FakeClient(chunks_per_transfer=1)
    client.gate.clear()
    worker, inner = _mirror_worker(client)
    try:
        _store(worker, inner, 6, [0, 1, 2])
        _wait_for(lambda: not worker._stores)
        assert worker.submit_load(7, CPULoadStoreSpec([5]), MagicMock())
        client.gate.set()
        _wait_for(lambda: len(client.calls) == 4)
        assert client.calls == [
            ("push", [0]),
            ("pull", [5]),
            ("push", [1]),
            ("push", [2]),
        ]
    finally:
        worker.shutdown()


def test_shutdown_is_prompt_with_copies_waiting():
    client = _FakeClient()
    client.gate.clear()
    worker, inner = _mirror_worker(client)
    _store(worker, inner, 8, [1])
    started = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - started < 5
    inner.shutdown.assert_called_once()


def test_gpu_client_maps_compact_addresses_to_its_head_slot():
    """A byte of the compact region maps to the same byte of the rank's slot."""
    client = object.__new__(GPUMirrorClient)
    client._local_base = 1_000_000
    client._local_stride = LOCAL_LAYOUT.row_stride
    client._head_base = 2_000_000 + 1 * HEAD_LAYOUT.size
    client._head_stride = HEAD_LAYOUT.row_stride

    local = np.array([1_000_000, 1_000_000 + 3 * SLOT + 100], dtype=np.int64)

    assert client.head_addresses(local).tolist() == [
        2_000_000 + SLOT,
        2_000_000 + 3 * HEAD_LAYOUT.row_stride + SLOT + 100,
    ]


@pytest.mark.parametrize(
    ("source", "gpu_direct", "expected"),
    [("auto", True, "gpu"), ("auto", False, "cpu"), ("cpu", True, "cpu")],
)
def test_resolve_mirror_source(monkeypatch, source, gpu_direct, expected):
    monkeypatch.setattr(
        region_mirror, "gpu_direct_rdma_supported", lambda device: gpu_direct
    )
    assert resolve_mirror_source(source, 0) == expected
    with pytest.raises(ValueError, match="mirror_source"):
        resolve_mirror_source("disk", 0)


class _FakeGPUClient(_FakeClient):
    def write(self, gpu, local, sizes) -> None:
        self._copy("write", gpu)

    def read(self, gpu, local, sizes) -> None:
        self._copy("read", gpu)


def _gpu_mirror_worker(monkeypatch, client: _FakeGPUClient):
    events: list[MagicMock] = []

    def new_event():
        events.append(MagicMock())
        return events[-1]

    monkeypatch.setattr(region_mirror.torch, "Event", new_event)
    inner = MagicMock()
    ops = (np.array([10, 11]), np.array([20, 21]), np.array([64, 64]))
    inner.store_copy_ops.return_value = ops
    inner.load_copy_ops.return_value = (ops[1], ops[0], ops[2])
    worker = RegionMirrorWorker(inner, gpu_client=client)  # type: ignore[arg-type]
    return worker, inner, events


def test_gpu_source_store_reads_its_blocks_after_the_compute_stream(monkeypatch):
    client = _FakeGPUClient()
    client.gate.clear()
    worker, inner, events = _gpu_mirror_worker(monkeypatch, client)
    try:
        assert worker.submit_store(1, MagicMock(), MagicMock())
        inner.submit_store.assert_not_called()
        events[0].record.assert_called_once()
        time.sleep(0.05)
        assert worker.get_finished() == []

        client.gate.set()
        [result] = _finished(worker)
        assert result.job_id == 1 and result.success
        assert result.transfer_size == 128
        events[0].synchronize.assert_called_once()
        assert client.calls == [("write", [10, 11])]
    finally:
        worker.shutdown()


def test_gpu_source_wait_covers_stores_too(monkeypatch):
    """The copy reads the GPU blocks, so they cannot be reused before it ends."""
    client = _FakeGPUClient()
    worker, inner, _ = _gpu_mirror_worker(monkeypatch, client)
    try:
        assert worker.submit_store(2, MagicMock(), MagicMock())
        worker.wait({2})
        assert client.calls == [("write", [10, 11])]
        inner.wait.assert_called_once_with({2})
    finally:
        worker.shutdown()


def test_gpu_source_load_is_reported_after_its_read(monkeypatch):
    client = _FakeGPUClient()
    worker, inner, _ = _gpu_mirror_worker(monkeypatch, client)
    try:
        assert worker.submit_load(3, MagicMock(), MagicMock())
        [result] = _finished(worker)
        assert result.job_id == 3 and result.success
        assert client.calls == [("read", [10, 11])]
        inner.submit_load.assert_not_called()
    finally:
        worker.shutdown()
