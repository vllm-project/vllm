# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the multi-node CPU offload region mirror."""

import os
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
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
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
    reach past `numel() * element_size()` bytes from the view's start."""
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
    """Stands in for RegionMirrorClient, the NIXL boundary: copies run when the
    test lets them, and each one is recorded with its descriptors."""

    def __init__(self):
        self.calls: list[tuple[str, list[int]]] = []
        # (op, GPU addresses, head addresses, sizes) of each copy.
        self.copies: list[tuple[str, list[int], list[int], list[int]]] = []
        self.gate = threading.Event()
        self.gate.set()
        # Set once a copy has started and waits on the gate.
        self.entered = threading.Event()
        self.fail: str | None = None
        self.stuck: str | None = None
        self.stopped = False
        self.shut_down = False

    def _copy(self, op: str, gpu: np.ndarray, local: np.ndarray, sizes) -> None:
        self.entered.set()
        self.gate.wait()
        self.calls.append((op, gpu.tolist()))
        self.copies.append((op, gpu.tolist(), local.tolist(), sizes.tolist()))
        if self.stuck == op:
            raise MirrorStuckError(f"{op} stuck")
        if self.fail == op:
            raise RuntimeError(f"{op} failed")

    def write(self, gpu, local, sizes) -> None:
        self._copy("write", gpu, local, sizes)

    def read(self, gpu, local, sizes) -> None:
        self._copy("read", gpu, local, sizes)

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


# One KV tensor of 64-byte pages and one GPU block per CPU chunk, so each
# block is one 64-byte copy to its chunk's row of this worker's slot.
_PAGE = 64
_SLOT_BASE = 1_000_000 + SLOT


class _Transfers:
    """Real transfer specs for a worker's GPU blocks, and the addresses that
    the planners from `slot_planners` should resolve them to."""

    def __init__(self):
        self.kv_caches = _kv_caches([_PAGE])

    def gpu(self, *blocks: int):
        spec = GPULoadStoreSpec(
            np.array(blocks), group_sizes=[len(blocks)], block_indices=[0]
        )
        return resolve_device_pointers(spec, self.kv_caches)

    @staticmethod
    def cpu(*chunks: int) -> BlockIDsLoadStoreSpec:
        return BlockIDsLoadStoreSpec(np.array(chunks))

    def addresses(self, *blocks: int) -> list[int]:
        return [self.kv_caches.tensors[0].tensor[b].data_ptr() for b in blocks]

    @staticmethod
    def rows(*chunks: int) -> list[int]:
        return [_SLOT_BASE + c * HEAD_LAYOUT.row_stride for c in chunks]


def _mirror_worker(
    client: _FakeClient, max_transfer_bytes: int = 1 << 20
) -> tuple[RegionMirrorWorker, _Transfers]:
    transfers = _Transfers()
    store, load = slot_planners(transfers.kv_caches, 1, HEAD_LAYOUT, _SLOT_BASE)
    worker = RegionMirrorWorker(
        store,
        load,
        client,  # type: ignore[arg-type]
        max_transfer_bytes=max_transfer_bytes,
    )
    return worker, transfers


def _store(worker: RegionMirrorWorker, t: _Transfers, job_id: int, *blocks: int):
    """Store GPU blocks to the CPU chunks of the same index."""
    return worker.submit_store(job_id, t.gpu(*blocks), t.cpu(*blocks))


def _load(
    worker: RegionMirrorWorker,
    t: _Transfers,
    job_id: int,
    *blocks: int,
    expedited: bool = False,
):
    """Load GPU blocks from the CPU chunks of the same index."""
    return worker.submit_load(
        job_id, t.cpu(*blocks), t.gpu(*blocks), expedited=expedited
    )


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
    worker, t = _mirror_worker(client)
    try:
        assert worker.submit_store(1, t.gpu(4, 5), t.cpu(2, 3))
        events[0].record.assert_called_once()
        assert client.entered.wait(10)
        assert worker.get_finished() == []

        client.gate.set()
        [result] = _finished(worker)
        assert result.job_id == 1 and result.success
        assert result.transfer_size == 2 * _PAGE
        events[0].synchronize.assert_called_once()
        assert client.copies == [
            ("write", t.addresses(4, 5), t.rows(2, 3), [_PAGE, _PAGE])
        ]
    finally:
        worker.shutdown()


def test_load_is_reported_after_its_read(events):
    client = _FakeClient()
    worker, t = _mirror_worker(client)
    try:
        assert worker.submit_load(2, t.cpu(6, 1), t.gpu(3, 7))
        [result] = _finished(worker)
        assert result.job_id == 2 and result.success
        assert result.transfer_size == 2 * _PAGE
        assert client.copies == [
            ("read", t.addresses(3, 7), t.rows(6, 1), [_PAGE, _PAGE])
        ]
    finally:
        worker.shutdown()


def test_wait_covers_stores_too(events):
    """The copy reads the GPU blocks, so they cannot be reused before it ends."""
    client = _FakeClient()
    worker, t = _mirror_worker(client)
    try:
        assert _store(worker, t, 3, 0, 1)
        worker.wait({3})
        assert client.calls == [("write", t.addresses(0, 1))]
        assert worker.get_finished()[0].job_id == 3
    finally:
        worker.shutdown()


def test_unregister_waits_for_copies_in_flight(events):
    """sleep() does not wait for this thread: a queued store must finish
    while the GPU memory is still registered."""
    client = _FakeClient()
    client.gate.clear()
    worker, t = _mirror_worker(client)
    try:
        assert _store(worker, t, 4, 0, 1)
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
        assert client.calls == [("write", t.addresses(0, 1)), ("unregister", [])]
        worker.restore_kv_caches()
        assert client.calls[-1] == ("restore", [])
        assert [r.job_id for r in worker.get_finished()] == [4]
    finally:
        worker.shutdown()


@pytest.mark.parametrize("op", ["write", "read"])
def test_failed_copy_reports_a_failed_transfer(events, op):
    client = _FakeClient()
    client.fail = op
    worker, t = _mirror_worker(client)
    try:
        if op == "write":
            assert _store(worker, t, 4, 0, 1)
        else:
            assert _load(worker, t, 4, 0, 1)
        assert _finished(worker) == [TransferResult(job_id=4, success=False)]
    finally:
        worker.shutdown()


def test_a_stuck_transfer_is_never_reported_and_stops_the_engine(events):
    """A transfer that may still land must not let the chunk be reused."""
    client = _FakeClient()
    client.stuck = "write"
    worker, t = _mirror_worker(client)
    try:
        assert _store(worker, t, 5, 0, 1)
        # The worker stops the client before it records the stuck transfer,
        # so poll get_finished() itself: it must raise, never report job 5.
        with pytest.raises(RuntimeError, match="may still land"):
            _finished(worker)
        assert client.stopped
    finally:
        worker.shutdown()
    # Its GPU memory may still be written, so it stays registered.
    assert not client.shut_down


def test_loads_overtake_waiting_stores(events):
    client = _FakeClient()
    client.gate.clear()
    worker, t = _mirror_worker(client)
    try:
        for job_id, block in ((6, 0), (7, 1), (8, 2)):
            assert _store(worker, t, job_id, block)
        assert client.entered.wait(10)
        assert _load(worker, t, 9, 5)
        client.gate.set()
        _wait_for(lambda: len(client.calls) == 4)
        assert client.calls == [
            ("write", t.addresses(0)),
            ("read", t.addresses(5)),
            ("write", t.addresses(1)),
            ("write", t.addresses(2)),
        ]
    finally:
        worker.shutdown()


def test_expedited_loads_overtake_waiting_loads(events):
    client = _FakeClient()
    client.gate.clear()
    worker, t = _mirror_worker(client)
    try:
        assert _load(worker, t, 14, 1)
        assert client.entered.wait(10)
        assert _load(worker, t, 15, 2)
        assert _load(worker, t, 16, 3, expedited=True)
        client.gate.set()
        worker.wait({14, 15, 16})
        assert client.calls == [
            ("read", t.addresses(1)),
            ("read", t.addresses(3)),
            ("read", t.addresses(2)),
        ]
    finally:
        worker.shutdown()


def test_large_copies_are_split_and_loads_go_between_their_transfers(events):
    client = _FakeClient()
    client.gate.clear()
    worker, t = _mirror_worker(client, max_transfer_bytes=2 * _PAGE)
    try:
        assert _store(worker, t, 12, 0, 1, 2, 3, 4)
        assert client.entered.wait(10)
        assert _load(worker, t, 13, 7)
        client.gate.set()
        worker.wait({12, 13})
        results = worker.get_finished()
        assert client.copies == [
            ("write", t.addresses(0, 1), t.rows(0, 1), [_PAGE, _PAGE]),
            ("read", t.addresses(7), t.rows(7), [_PAGE]),
            ("write", t.addresses(2, 3), t.rows(2, 3), [_PAGE, _PAGE]),
            ("write", t.addresses(4), t.rows(4), [_PAGE]),
        ]
        stored = next(r for r in results if r.job_id == 12)
        assert stored.success and stored.transfer_size == 5 * _PAGE
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
    worker, t = _mirror_worker(client)
    assert _store(worker, t, 10, 0, 1)
    started = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - started < 5
    assert client.shut_down


class _FakeNixlAgent:
    """Stands in for `nixl_utils.NixlWrapper`, the NIXL agent: records the
    memory registered and the peers added."""

    def __init__(self, name: str, config):
        self.registered: list[tuple[str, list]] = []
        self.peers: list[str] = []

    def create_backend(self, backend: str, params: dict) -> None:
        pass

    def register_memory(self, descs: list, mem_type: str):
        self.registered.append((mem_type, [(addr, size) for addr, size, *_ in descs]))
        return self.registered[-1]

    def deregister_memory(self, registration) -> None:
        self.registered.remove(registration)

    def get_agent_metadata(self) -> bytes:
        return b"head"

    def add_remote_agent(self, metadata: bytes) -> str:
        self.peers.append(metadata.decode())
        return self.peers[-1]

    def remove_remote_agent(self, peer: str) -> None:
        self.peers.remove(peer)


def test_head_worker_moves_rank_0_slots_and_releases_the_region(monkeypatch):
    """Rank 0 keeps its own CPU worker for its slots of the head region, and
    shutting down drops the region's NIXL registration and the region."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    agents: list[_FakeNixlAgent] = []

    def new_agent(name, config):
        agents.append(_FakeNixlAgent(name, config))
        return agents[-1]

    monkeypatch.setattr(nixl_utils, "NixlWrapper", new_agent)
    monkeypatch.setattr(nixl_utils, "nixl_agent_config", lambda **kwargs: None)
    region = SharedOffloadRegion(
        engine_id=f"mirror-head-worker-{uuid.uuid4().hex[:8]}",
        num_chunks=NUM_CHUNKS,
        rank=0,
        kv_bytes_per_chunk=HEAD_LAYOUT.row_stride,
        cpu_page_size=SLOT,
    )
    blocks = torch.zeros((16, SLOT), dtype=torch.int8, device="cuda")
    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=blocks, page_size_bytes=SLOT)],
        group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=SLOT)]],
    )
    inner = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_chunks=NUM_CHUNKS,
        mmap_region=region,
    )
    head = RegionMirrorHead(region, HEAD_LAYOUT, "head")
    head.add_peers([b"rank1"])
    worker = RegionMirrorHeadWorker(inner, head)
    assert region.mmap_obj is not None
    rows = np.frombuffer(region.mmap_obj, dtype=np.int8).reshape(
        NUM_CHUNKS, HEAD_LAYOUT.row_stride
    )

    def gpu(*ids: int):
        spec = GPULoadStoreSpec(
            np.array(ids), group_sizes=[len(ids)], block_indices=[0]
        )
        return resolve_device_pointers(spec, kv_caches)

    blocks[4], blocks[5] = 40, 50
    torch.accelerator.synchronize()
    assert worker.submit_store(11, gpu(4, 5), BlockIDsLoadStoreSpec(np.array([2, 3])))
    worker.wait({11})
    # Rank 0's slot of each stored chunk, and nothing in the other rank's.
    assert (rows[2, :SLOT] == 40).all() and (rows[3, :SLOT] == 50).all()
    assert (rows[:, SLOT:] == 0).all()

    rows[6, :SLOT] = 60
    assert worker.submit_load(12, BlockIDsLoadStoreSpec(np.array([6])), gpu(7))
    worker.wait({12})
    assert (blocks[7] == 60).all()
    finished = [(r.job_id, r.success) for r in worker.get_finished()]
    assert sorted(finished) == [(11, True), (12, True)]

    (agent,) = agents
    assert agent.registered == [("DRAM", [(head.base, region.total_size_bytes)])]
    assert agent.peers == ["rank1"]
    del rows
    worker.shutdown()
    assert agent.registered == [] and agent.peers == []
    assert region.mmap_obj is None
    assert not os.path.exists(region.mmap_path)
