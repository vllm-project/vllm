# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the multi-node CPU offload region mirror."""

import threading
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.kv_offload.cpu.region_mirror as region_mirror
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.region_mirror import (
    MirrorStuckError,
    MirrorTopology,
    RegionMirrorClient,
    RegionMirrorHead,
    RegionMirrorHeadWorker,
    RegionMirrorWorker,
    SlotLayout,
    attach_region_mirror,
    slot_planners,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

SLOT = 4096
NUM_CHUNKS = 8
HEAD_LAYOUT = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=2 * SLOT, size=SLOT)


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def nixl_mirror():
    """A head region and a remote worker's GPU blocks joined by NIXL."""
    pytest.importorskip("nixl")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    tag = uuid.uuid4().hex[:8]
    head_region = SharedOffloadRegion(
        engine_id=f"mirror-head-{tag}",
        num_chunks=NUM_CHUNKS,
        rank=None,
        kv_bytes_per_chunk=HEAD_LAYOUT.row_stride,
        cpu_page_size=SLOT,
    )
    blocks = torch.zeros((NUM_CHUNKS, SLOT), dtype=torch.uint8, device="cuda")
    head = RegionMirrorHead(head_region, HEAD_LAYOUT, f"head-{tag}")
    client = RegionMirrorClient(
        [(blocks.data_ptr(), blocks.numel(), blocks.device.index)],
        head.metadata,
        f"client-{tag}",
        timeout_s=10,
    )
    head.add_peers([client.metadata])
    assert head_region.mmap_obj is not None
    head_rows = np.frombuffer(head_region.mmap_obj, dtype=np.uint8).reshape(
        NUM_CHUNKS, HEAD_LAYOUT.row_stride
    )
    try:
        yield head_rows, blocks, client, head.base + SLOT
    finally:
        del head_rows
        client.shutdown()
        head.shutdown()
        head_region.cleanup()


def _plan(blocks: torch.Tensor, slot: int, pairs: list[tuple[int, int]]):
    """(GPU address, head address, size) copying GPU block b to chunk c."""
    gpu = np.array([blocks[b].data_ptr() for b, _ in pairs], dtype=np.int64)
    head = np.array(
        [slot + c * HEAD_LAYOUT.row_stride for _, c in pairs], dtype=np.int64
    )
    return gpu, head, np.full(len(pairs), SLOT, dtype=np.int64)


def test_write_fills_only_the_remote_slot(nixl_mirror):
    head_rows, blocks, client, slot = nixl_mirror
    head_rows[:] = 7
    for block in range(NUM_CHUNKS):
        blocks[block] = 100 + block
    torch.cuda.synchronize()

    client.write(*_plan(blocks, slot, [(0, 2), (1, 5)]))

    for chunk in range(NUM_CHUNKS):
        # Rank 1's slot of a written chunk, nothing else.
        expected = {2: 100, 5: 101}.get(chunk, 7)
        assert (head_rows[chunk, SLOT:] == expected).all()
        assert (head_rows[chunk, :SLOT] == 7).all()


def test_read_fills_the_gpu_blocks_from_the_remote_slot(nixl_mirror):
    head_rows, blocks, client, slot = nixl_mirror
    for chunk in range(NUM_CHUNKS):
        head_rows[chunk, :SLOT] = 1
        head_rows[chunk, SLOT:] = 50 + chunk

    client.read(*_plan(blocks, slot, [(0, 3), (4, 6)]))

    expected = [53, 0, 0, 0, 56, 0, 0, 0]
    assert blocks[:, 0].tolist() == expected
    assert (blocks == blocks[:, :1]).all()


def _kv_caches(page_sizes: list[int], num_blocks: int = 16) -> CanonicalKVCaches:
    tensors = [
        CanonicalKVCacheTensor(
            tensor=torch.zeros((num_blocks, size), dtype=torch.int8),
            page_size_bytes=size,
        )
        for size in page_sizes
    ]
    refs = [
        CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=size)
        for i, size in enumerate(page_sizes)
    ]
    return CanonicalKVCaches(tensors=tensors, group_data_refs=[refs])


def test_slot_planners_address_this_workers_slot_of_the_head_region():
    """Each tensor's chunk page sits in turn inside the slot, as on the head."""
    kv_caches = _kv_caches([1024, 512])
    layout = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=4 * SLOT, size=SLOT)
    slot = 1_000_000 + 2 * SLOT
    store, load = slot_planners(kv_caches, 2, layout, slot)

    gpu_spec = GPULoadStoreSpec(np.array([4, 5]), group_sizes=[2], block_indices=[0])
    cpu_spec = BlockIDsLoadStoreSpec(np.array([3]))
    gpu, head, sizes = store.copy_ops(gpu_spec, cpu_spec)

    k = kv_caches.tensors
    assert gpu.tolist() == [
        k[0].tensor[4].data_ptr(),
        k[0].tensor[5].data_ptr(),
        k[1].tensor[4].data_ptr(),
        k[1].tensor[5].data_ptr(),
    ]
    row = slot + 3 * layout.row_stride
    assert head.tolist() == [row, row + 1024, row + 2048, row + 2048 + 512]
    assert sizes.tolist() == [1024, 1024, 512, 512]
    head_back, gpu_back, _ = load.copy_ops(cpu_spec, gpu_spec)
    assert head_back.tolist() == head.tolist()
    assert gpu_back.tolist() == gpu.tolist()


def test_slot_planners_reject_a_slot_too_small_for_the_kv():
    layout = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=2 * SLOT, size=SLOT)
    with pytest.raises(ValueError, match="slot of the head region"):
        slot_planners(_kv_caches([SLOT, 512]), 1, layout, 0)


class _FakeClient:
    """Stands in for RegionMirrorClient: copies run when the test lets them."""

    def __init__(self):
        self.calls: list[tuple[str, list[int]]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail: str | None = None
        self.stuck: str | None = None
        self.stopped = False
        self.shut_down = False

    def _copy(self, op: str, gpu: np.ndarray) -> None:
        self.gate.wait()
        self.calls.append((op, gpu.tolist()))
        if self.stuck == op:
            raise MirrorStuckError(f"{op} stuck")
        if self.fail == op:
            raise RuntimeError(f"{op} failed")

    def write(self, gpu, local, sizes) -> None:
        self._copy("write", gpu)

    def read(self, gpu, local, sizes) -> None:
        self._copy("read", gpu)

    def stop(self, reason: str = "") -> None:
        self.stopped = True
        self.gate.set()

    def shutdown(self) -> None:
        self.shut_down = True


@pytest.fixture
def events(monkeypatch) -> list[MagicMock]:
    recorded: list[MagicMock] = []

    def new_event():
        recorded.append(MagicMock())
        return recorded[-1]

    monkeypatch.setattr(region_mirror.torch, "Event", new_event)
    return recorded


def _mirror_worker(
    client: _FakeClient, max_transfer_bytes: int = 1 << 20
) -> tuple[RegionMirrorWorker, MagicMock, MagicMock]:
    store, load = MagicMock(), MagicMock()
    ops = (np.array([10, 11]), np.array([20, 21]), np.array([64, 64]))
    store.copy_ops.return_value = ops
    load.copy_ops.return_value = (ops[1], ops[0], ops[2])
    worker = RegionMirrorWorker(
        store,
        load,
        client,  # type: ignore[arg-type]
        max_transfer_bytes=max_transfer_bytes,
    )
    return worker, store, load


def _finished(worker: RegionMirrorWorker) -> list[TransferResult]:
    results: list[TransferResult] = []

    def collected() -> bool:
        results.extend(worker.get_finished())
        return bool(results)

    _wait_for(collected)
    return results


def test_store_is_reported_after_its_write_behind_the_compute_stream(events):
    client = _FakeClient()
    client.gate.clear()
    worker, _, _ = _mirror_worker(client)
    try:
        assert worker.submit_store(1, MagicMock(), MagicMock())
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


def test_load_is_reported_after_its_read(events):
    client = _FakeClient()
    worker, _, _ = _mirror_worker(client)
    try:
        assert worker.submit_load(2, MagicMock(), MagicMock())
        [result] = _finished(worker)
        assert result.job_id == 2 and result.success
        assert client.calls == [("read", [10, 11])]
    finally:
        worker.shutdown()


def test_wait_covers_stores_too(events):
    """The copy reads the GPU blocks, so they cannot be reused before it ends."""
    client = _FakeClient()
    worker, _, _ = _mirror_worker(client)
    try:
        assert worker.submit_store(3, MagicMock(), MagicMock())
        worker.wait({3})
        assert client.calls == [("write", [10, 11])]
        assert worker.get_finished()[0].job_id == 3
    finally:
        worker.shutdown()


@pytest.mark.parametrize("op", ["write", "read"])
def test_failed_copy_reports_a_failed_transfer(events, op):
    client = _FakeClient()
    client.fail = op
    worker, _, _ = _mirror_worker(client)
    try:
        if op == "write":
            assert worker.submit_store(4, MagicMock(), MagicMock())
        else:
            assert worker.submit_load(4, MagicMock(), MagicMock())
        assert _finished(worker) == [TransferResult(job_id=4, success=False)]
    finally:
        worker.shutdown()


def test_a_stuck_transfer_is_never_reported_and_stops_the_engine(events):
    """A transfer that may still land must not let the chunk be reused."""
    client = _FakeClient()
    client.stuck = "write"
    worker, _, _ = _mirror_worker(client)
    try:
        assert worker.submit_store(5, MagicMock(), MagicMock())
        _wait_for(lambda: client.stopped)
        with pytest.raises(RuntimeError, match="may still land"):
            worker.get_finished()
    finally:
        worker.shutdown()
    # Its GPU memory may still be written, so it stays registered.
    assert not client.shut_down


def test_loads_overtake_waiting_stores(events):
    client = _FakeClient()
    client.gate.clear()
    worker, _, load = _mirror_worker(client)
    load.copy_ops.return_value = (
        np.array([20]),
        np.array([99]),
        np.array([64]),
    )
    try:
        for job_id in (6, 7, 8):
            assert worker.submit_store(job_id, MagicMock(), MagicMock())
        _wait_for(lambda: len(worker._stores) == 2)
        assert worker.submit_load(9, MagicMock(), MagicMock())
        client.gate.set()
        _wait_for(lambda: len(client.calls) == 4)
        assert [op for op, _ in client.calls] == ["write", "read", "write", "write"]
    finally:
        worker.shutdown()


def test_large_copies_are_split_and_loads_go_between_their_transfers(events):
    client = _FakeClient()
    client.gate.clear()
    worker, store, load = _mirror_worker(client, max_transfer_bytes=128)
    store.copy_ops.return_value = (
        np.arange(10, 15),
        np.arange(20, 25),
        np.full(5, 64),
    )
    load.copy_ops.return_value = (
        np.array([30]),
        np.array([40]),
        np.array([64]),
    )
    try:
        assert worker.submit_store(12, MagicMock(), MagicMock())
        _wait_for(lambda: not worker._stores)
        assert worker.submit_load(13, MagicMock(), MagicMock())
        client.gate.set()
        worker.wait({12, 13})
        results = worker.get_finished()
        assert client.calls == [
            ("write", [10, 11]),
            ("read", [40]),
            ("write", [12, 13]),
            ("write", [14]),
        ]
        stored = next(r for r in results if r.job_id == 12)
        assert stored.success and stored.transfer_size == 320
    finally:
        worker.shutdown()


def test_mismatched_head_layouts_are_rejected(monkeypatch):
    """Uneven pipeline stages would address the wrong bytes of each row."""
    other = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=3 * SLOT, size=SLOT)
    monkeypatch.setattr(
        region_mirror,
        "_all_gather",
        lambda topology, value: [(None, ((b"head", 0), other)), value],
    )
    new_client = MagicMock()
    monkeypatch.setattr(region_mirror, "RegionMirrorClient", new_client)
    topology = MirrorTopology(
        group=MagicMock(), rank=1, on_head=False, remote_ranks=frozenset({1})
    )
    with pytest.raises(ValueError, match="same CPU region geometry"):
        attach_region_mirror(None, None, MagicMock(), 1, HEAD_LAYOUT, topology, "e")
    new_client.assert_not_called()


def test_shutdown_is_prompt_with_copies_waiting(events):
    client = _FakeClient()
    client.gate.clear()
    worker, _, _ = _mirror_worker(client)
    assert worker.submit_store(10, MagicMock(), MagicMock())
    started = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - started < 5
    assert client.shut_down


def test_head_worker_leaves_transfers_to_its_inner_worker():
    inner = MagicMock()
    inner.submit_store.return_value = True
    head = MagicMock()
    worker = RegionMirrorHeadWorker(inner, head)
    assert worker.submit_store(11, MagicMock(), MagicMock())
    worker.wait({11})
    inner.wait.assert_called_once_with({11})
    worker.shutdown()
    head.shutdown.assert_called_once()
    inner.shutdown.assert_called_once()
