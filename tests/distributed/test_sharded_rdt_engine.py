# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused receive-pool and scheduler tests for sharded RDT."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_engine as engine_module
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _CallPlan,
    _Chunk,
    _PullHooks,
    _run_lanes,
)


def _chunk(
    nbytes: int = 32,
    *,
    stream: int = 0,
    key: str = "w",
    owner: int = 0,
    seq: int = 0,
    free: list[int] | None = None,
) -> _Chunk:
    return _Chunk(
        scatters=[],
        keys=[(key, ())],
        pack_layout=[(0, torch.uint8, nbytes, (nbytes,))],
        pack_bytes=nbytes,
        materialize=[],
        quant=[],
        free=list(free or []),
        owner=owner,
        seq=seq,
        stream=stream,
    )


def _plan(*, streams: int = 1, chunk_bytes: tuple[int, ...] = (32,)) -> _CallPlan:
    chunks = [
        _chunk(nbytes, stream=index % max(1, streams))
        for index, nbytes in enumerate(chunk_bytes)
    ]
    return _CallPlan(chunks=chunks, pre_free=[], num_streams=streams)


def _stream_plan(*, streams: int = 4, chunks_per_stream: int = 3) -> _CallPlan:
    chunks = []
    group = 0
    for stream in range(streams):
        for seq in range(chunks_per_stream):
            chunks.append(
                _chunk(
                    1,
                    stream=stream,
                    key=f"s{stream}.{seq}",
                    owner=stream,
                    seq=seq,
                    free=[group],
                )
            )
            group += 1
    return _CallPlan(chunks=chunks, pre_free=[group], num_streams=streams)


def _engine(
    *,
    staging_device: str = "gpu",
    stages_in_flight: int = 1,
    ring_depth: int = 2,
    streams: int = 1,
    chunk_bytes: tuple[int, ...] = (32,),
) -> ShardedRDTWeightTransferEngine:
    engine = object.__new__(ShardedRDTWeightTransferEngine)
    engine.device = torch.device("cpu")
    engine._proc_thread = None
    engine._slot_read_done = []
    engine._slot_queued = []
    engine._slot_done = []
    engine._slot_cv = None
    engine._pool_geometry = None
    engine._staging_device = "gpu"
    engine._stages_in_flight = 1
    engine._ring_depth = 1
    engine._lanes = 1
    engine._dest_buffers = [{}]
    engine._lane_cursors = [0]
    engine._targets_cache = {}
    engine._buffer_presize = 0
    engine._cached_plan = _plan(streams=streams, chunk_bytes=chunk_bytes)
    engine._router = SimpleNamespace(
        num_producers=1,
        reserve_serve_buffers=lambda _sizes, _digests: [],
        pull=lambda _owner, _keys, _seq: object(),
    )
    config = engine._configure_ring(
        ShardedRDTWeightTransferInitInfo(
            staging_device=staging_device,
            stages_in_flight=stages_in_flight,
            num_rdt_buffers=ring_depth,
        )
    )
    engine._size_pool(config, engine._cached_plan)
    return engine


@pytest.fixture
def registered(monkeypatch):
    buffers: list[torch.Tensor] = []
    monkeypatch.setattr(engine_module, "register_nixl_memory", buffers.append)
    monkeypatch.setattr(
        engine_module,
        "buffer_alloc_bytes",
        lambda nbytes, presize=0: max(nbytes, presize),
    )
    return buffers


@pytest.fixture
def allocations(monkeypatch):
    real_empty = torch.empty
    state = SimpleNamespace(calls=[], failure=None)

    def empty(*args, **kwargs):
        state.calls.append(dict(kwargs))
        if kwargs.get("pin_memory"):
            if state.failure is not None:
                raise state.failure
            kwargs.pop("pin_memory")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    return state


@pytest.mark.parametrize(
    ("streams", "requested_lanes", "ring_depth", "expected"),
    [(4, 2, 3, (2, 6)), (3, 5, 2, (3, 6)), (0, 5, 2, (1, 2))],
)
def test_pool_geometry_clamps_lanes_and_assigns_disjoint_slot_blocks(
    streams, requested_lanes, ring_depth, expected
):
    engine = _engine(
        stages_in_flight=requested_lanes,
        ring_depth=ring_depth,
        streams=streams,
        chunk_bytes=() if streams == 0 else (32,),
    )
    assert (engine._lanes, engine._num_slots) == expected
    blocks = [set(engine._lane_slots(lane)) for lane in range(engine._lanes)]
    assert all(len(block) == ring_depth for block in blocks)
    assert len(set().union(*blocks)) == engine._num_slots


def test_active_pool_rejects_a_geometry_change_without_mutation():
    engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
    engine._proc_thread = object()
    engine._slot_read_done = [object()] * 4
    engine._slot_queued = [0] * 4
    engine._slot_done = [0] * 4
    before = (
        engine._pool_geometry,
        engine._dest_buffers,
        engine._slot_read_done,
        engine._lane_cursors,
    )
    config = engine._configure_ring(
        ShardedRDTWeightTransferInitInfo(
            stages_in_flight=1,
            num_rdt_buffers=4,
        )
    )

    with pytest.raises(RuntimeError, match=r"old=.*L=2, K=2.*new="):
        engine._size_pool(config, _plan(streams=1))

    assert (
        engine._pool_geometry,
        engine._dest_buffers,
        engine._slot_read_done,
        engine._lane_cursors,
    ) == before


@pytest.mark.parametrize(
    ("staging_device", "expected_device", "pinned"),
    [("gpu", torch.device("cpu"), False), ("host", "cpu", True)],
)
def test_receive_slots_use_the_selected_memory_and_register_once(
    staging_device, expected_device, pinned, registered, allocations
):
    engine = _engine(
        staging_device=staging_device,
        stages_in_flight=2,
        ring_depth=2,
        streams=2,
        chunk_bytes=(48, 16),
    )
    engine._preregister_at_init()

    buffers = [slot[torch.uint8] for slot in engine._dest_buffers]
    slot_calls = [
        call for call in allocations.calls if call.get("dtype") is torch.uint8
    ]
    assert len(slot_calls) == len(buffers) == 4
    assert all(call["device"] == expected_device for call in slot_calls)
    assert all(call.get("pin_memory", False) is pinned for call in slot_calls)
    assert registered == buffers
    assert all(buffer.numel() == 48 for buffer in buffers)

    engine._preregister_at_init()
    assert registered == buffers


def test_slot_reuse_waits_for_the_queued_read_done_event():
    class Event:
        ready = False

        def query(self):
            return self.ready

    engine = _engine(ring_depth=1)
    event = Event()
    engine._slot_read_done = [event]
    engine._slot_cv = threading.Condition()
    engine._slot_queued = [1]
    engine._slot_done = [0]

    assert not engine._slot_free(0)
    engine._slot_done[0] = 1
    assert not engine._slot_free(0)
    event.ready = True
    assert engine._slot_free(0)


class _SharedPacks:
    def __init__(self, replicas):
        self.replicas = replicas
        self.cv = threading.Condition()
        self.arrived: dict[tuple[int, int], set[int]] = {}

    def arrive(self, key, replica):
        with self.cv:
            self.arrived.setdefault(key, set()).add(replica)
            self.cv.notify_all()

    def landed(self, key):
        return len(self.arrived.get(key, ())) == self.replicas


class _ReplicaPulls:
    def __init__(self, packs, replica, copy_seconds):
        self.packs = packs
        self.replica = replica
        self.copy_seconds = copy_seconds
        self.busy_until: dict[int, float] = {}
        self.slots: dict[int, SimpleNamespace] = {}
        self.issued = 0

    def slot_free(self, slot):
        return time.monotonic() >= self.busy_until.get(slot, 0.0)

    def issue(self, chunk, slot):
        while not self.slot_free(slot):
            time.sleep(0.001)
        key = (chunk.owner, chunk.seq)
        self.packs.arrive(key, self.replica)
        pending = SimpleNamespace(key=key)
        self.slots[slot] = pending
        self.issued += 1
        return pending

    def wait_any(self, pulls, timeout=None):
        with self.packs.cv:
            landed = self.packs.cv.wait_for(
                lambda: any(self.packs.landed(pull.key) for pull in pulls),
                timeout,
            )
            if not landed:
                return None
            return next(
                index for index, pull in enumerate(pulls) if self.packs.landed(pull.key)
            )

    def drain(self, pending, chunk):
        slot = next(slot for slot, pull in self.slots.items() if pull is pending)
        self.busy_until[slot] = time.monotonic() + self.copy_seconds[chunk.stream]

    def hooks(self):
        return _PullHooks(
            issue=self.issue,
            slot_free=self.slot_free,
            wait_any=self.wait_any,
            drain=self.drain,
            signal=lambda _group: None,
        )


def test_bounded_slot_recheck_breaks_the_shared_pack_cycle():
    plan = _stream_plan(streams=2, chunks_per_stream=8)
    streams = [
        [chunk for chunk in plan.chunks if chunk.stream == stream]
        for stream in range(2)
    ]
    packs = _SharedPacks(replicas=2)
    replicas = [
        _ReplicaPulls(packs, 0, {0: 0.0, 1: 0.05}),
        _ReplicaPulls(packs, 1, {0: 0.05, 1: 0.0}),
    ]
    threads = [
        threading.Thread(
            target=_run_lanes,
            args=(streams,),
            kwargs={
                "depth": 2,
                "blocks": [range(0, 2), range(2, 4)],
                "cursors": [0, 0],
                "hooks": replica.hooks(),
            },
            daemon=True,
        )
        for replica in replicas
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not any(thread.is_alive() for thread in threads)
    assert [replica.issued for replica in replicas] == [len(plan.chunks)] * 2


def _record_pipeline(engine, monkeypatch):
    events = []
    active = 0
    peak = 0

    def issue(chunk, slot):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        events.append(("issue", chunk.keys[0][0], slot))
        return SimpleNamespace(chunk=chunk, slot=slot)

    def complete(pending):
        nonlocal active
        active -= 1
        events.append(("drain", pending.chunk.keys[0][0]))
        return {}

    monkeypatch.setattr(engine, "_issue_pull", issue)
    monkeypatch.setattr(engine, "_complete_pull", complete)
    monkeypatch.setattr(engine, "_dispatch_item", lambda _item: None)
    monkeypatch.setattr(
        engine,
        "_signal_group_done",
        lambda group: events.append(("signal", group)),
    )
    monkeypatch.setattr(engine, "_wait_any_head", lambda _pulls, timeout=None: 0)
    engine._run_chunk_pipeline(engine._cached_plan)
    return events, peak


def test_sync_zero_is_one_lane_at_depth_one_in_whole_plan_order(monkeypatch):
    engine = _engine(stages_in_flight=3, ring_depth=2, streams=3)
    engine._cached_plan = _stream_plan(streams=3, chunks_per_stream=2)
    engine._completed_syncs = 0

    events, peak = _record_pipeline(engine, monkeypatch)
    issues = [event for event in events if event[0] == "issue"]

    assert peak == 1
    assert [event[1] for event in issues] == [
        chunk.keys[0][0] for chunk in engine._cached_plan.chunks
    ]
    assert {event[2] // engine._ring_depth for event in issues} == {0}
