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
from vllm.distributed import nixl_utils
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
from vllm.v1.kv_offload.pointer_utils import resolve_device_pointers

SLOT = 4096
NUM_CHUNKS = 8
HEAD_LAYOUT = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=2 * SLOT, size=SLOT)
_CPU_SPEC = MagicMock(spec=BlockIDsLoadStoreSpec)


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def nixl_mirror():
    """A head region and a remote worker's GPU blocks joined by NIXL."""
    if nixl_utils.NixlWrapper is None:  # nixl, or nixl_rocm on ROCm
        pytest.skip("needs NIXL")
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
    torch.accelerator.synchronize()

    client.write(*_plan(blocks, slot, [(0, 2), (1, 5)]))

    for chunk in range(NUM_CHUNKS):
        # Rank 1's slot of a written chunk, nothing else.
        expected = {2: 100, 5: 101}.get(chunk, 7)
        assert (head_rows[chunk, SLOT:] == expected).all()
        assert (head_rows[chunk, :SLOT] == 7).all()


def test_write_after_reregistration(nixl_mirror):
    """Sleep mode remaps the GPU KV caches between these two hooks."""
    head_rows, blocks, client, slot = nixl_mirror
    client.unregister_kv_caches()
    client.unregister_kv_caches()
    client.restore_kv_caches()
    blocks[0] = 42
    torch.accelerator.synchronize()

    client.write(*_plan(blocks, slot, [(0, 1)]))

    assert (head_rows[1, SLOT:] == 42).all()


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
    device_ptrs = resolve_device_pointers(gpu_spec, kv_caches)
    cpu_spec = BlockIDsLoadStoreSpec(np.array([3]))
    gpu, head, sizes = store.copy_ops(device_ptrs, cpu_spec)

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
    head_back, gpu_back, _ = load.copy_ops(device_ptrs, cpu_spec)
    assert head_back.tolist() == head.tolist()
    assert gpu_back.tolist() == gpu.tolist()


def test_slot_planners_reject_a_slot_too_small_for_the_kv():
    layout = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=2 * SLOT, size=SLOT)
    with pytest.raises(ValueError, match="slot of the head region"):
        slot_planners(_kv_caches([SLOT, 512]), 1, layout, 0)


def test_strided_views_register_their_whole_storage():
    """Per-layer views of a block-outermost cache are strided: their blocks
    reach past ``numel() * element_size()`` bytes from the view's start."""
    blocks = torch.zeros((16, 2, 512), dtype=torch.int8)
    caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=blocks[:, layer], page_size_bytes=512)
            for layer in range(2)
        ],
        group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]],
    )

    assert region_mirror._storage_ranges(caches) == [
        (blocks.data_ptr(), blocks.nbytes, 0)
    ]


@pytest.mark.parametrize(
    "remaps_kv_cache, params",
    [(False, {}), (True, {"engine_config": "RCACHE_ENABLE=n,TLS=^cuda_ipc"})],
)
def test_remapping_agents_do_not_cache_vram_registrations(
    monkeypatch, remaps_kv_cache, params
):
    agent = MagicMock()
    monkeypatch.setattr(nixl_utils, "NixlWrapper", MagicMock(return_value=agent))
    monkeypatch.setattr(nixl_utils, "nixl_agent_config", MagicMock())
    assert region_mirror._new_agent("a", remaps_kv_cache=remaps_kv_cache) is agent
    agent.create_backend.assert_called_once_with("UCX", params)


class _FakeClient:
    """Stands in for RegionMirrorClient: copies run when the test lets them."""

    def __init__(self):
        self.calls: list[tuple[str, list[int]]] = []
        self.gate = threading.Event()
        self.gate.set()
        # Set once a copy has started and waits on the gate.
        self.entered = threading.Event()
        self.fail: str | None = None
        self.stuck: str | None = None
        self.stopped = False
        self.shut_down = False

    def _copy(self, op: str, gpu: np.ndarray) -> None:
        self.entered.set()
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

    def unregister_kv_caches(self) -> None:
        self.calls.append(("unregister", []))

    def restore_kv_caches(self) -> None:
        self.calls.append(("restore", []))

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
        assert worker.submit_store(1, MagicMock(), _CPU_SPEC)
        events[0].record.assert_called_once()
        assert client.entered.wait(10)
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
        assert worker.submit_load(2, _CPU_SPEC, MagicMock())
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
        assert worker.submit_store(3, MagicMock(), _CPU_SPEC)
        worker.wait({3})
        assert client.calls == [("write", [10, 11])]
        assert worker.get_finished()[0].job_id == 3
    finally:
        worker.shutdown()


def test_unregister_waits_for_copies_in_flight(events):
    """sleep() does not wait for this thread: a queued store must finish
    while the GPU memory is still registered."""
    client = _FakeClient()
    client.gate.clear()
    worker, _, _ = _mirror_worker(client)
    try:
        assert worker.submit_store(4, MagicMock(), _CPU_SPEC)
        assert client.entered.wait(10)
        unregistered = threading.Event()

        def unregister() -> None:
            worker.unregister_kv_caches()
            unregistered.set()

        thread = threading.Thread(target=unregister)
        thread.start()
        # Still blocked on the copy: the registration must not be dropped yet.
        assert not unregistered.wait(0.2)
        assert client.calls == []
        client.gate.set()
        assert unregistered.wait(10)
        thread.join(10)
        assert client.calls == [("write", [10, 11]), ("unregister", [])]
        worker.restore_kv_caches()
        assert client.calls[-1] == ("restore", [])
        assert [r.job_id for r in worker.get_finished()] == [4]
    finally:
        worker.shutdown()


@pytest.mark.parametrize("op", ["write", "read"])
def test_failed_copy_reports_a_failed_transfer(events, op):
    client = _FakeClient()
    client.fail = op
    worker, _, _ = _mirror_worker(client)
    try:
        if op == "write":
            assert worker.submit_store(4, MagicMock(), _CPU_SPEC)
        else:
            assert worker.submit_load(4, _CPU_SPEC, MagicMock())
        assert _finished(worker) == [TransferResult(job_id=4, success=False)]
    finally:
        worker.shutdown()


def test_a_stuck_transfer_is_never_reported_and_stops_the_engine(events):
    """A transfer that may still land must not let the chunk be reused."""
    client = _FakeClient()
    client.stuck = "write"
    worker, _, _ = _mirror_worker(client)
    try:
        assert worker.submit_store(5, MagicMock(), _CPU_SPEC)
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
            assert worker.submit_store(job_id, MagicMock(), _CPU_SPEC)
        _wait_for(lambda: len(worker._stores) == 2)
        assert worker.submit_load(9, _CPU_SPEC, MagicMock())
        client.gate.set()
        _wait_for(lambda: len(client.calls) == 4)
        assert [op for op, _ in client.calls] == ["write", "read", "write", "write"]
    finally:
        worker.shutdown()


def test_expedited_loads_overtake_waiting_loads(events):
    client = _FakeClient()
    client.gate.clear()
    worker, _, load = _mirror_worker(client)
    load.copy_ops.side_effect = [
        (np.array([40]), np.array([job_id]), np.array([64])) for job_id in (14, 15, 16)
    ]
    try:
        assert worker.submit_load(14, _CPU_SPEC, MagicMock())
        assert client.entered.wait(10)
        assert worker.submit_load(15, _CPU_SPEC, MagicMock())
        assert worker.submit_load(16, _CPU_SPEC, MagicMock(), expedited=True)
        client.gate.set()
        worker.wait({14, 15, 16})
        assert client.calls == [("read", [14]), ("read", [16]), ("read", [15])]
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
        assert worker.submit_store(12, MagicMock(), _CPU_SPEC)
        _wait_for(lambda: not worker._stores)
        assert worker.submit_load(13, _CPU_SPEC, MagicMock())
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
    assert worker.submit_store(10, MagicMock(), _CPU_SPEC)
    started = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - started < 5
    assert client.shut_down


def test_head_worker_leaves_transfers_to_its_inner_worker():
    inner = MagicMock()
    inner.submit_store.return_value = True
    head = MagicMock()
    worker = RegionMirrorHeadWorker(inner, head)
    assert worker.submit_store(11, MagicMock(), _CPU_SPEC)
    worker.wait({11})
    inner.wait.assert_called_once_with({11})
    worker.shutdown()
    head.shutdown.assert_called_once()
    inner.shutdown.assert_called_once()
