# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receive-pool and staging tests for the sharded-RDT consumer."""

import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_engine as engine_module
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    _SLOT_POLL_S,
    HostStagingAllocationError,
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
            kwargs = {
                key: value for key, value in kwargs.items() if key != "pin_memory"
            }
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    return state


class TestPoolGeometry:
    def test_configuration_records_settings_without_allocating(self, monkeypatch):
        engine = _engine()
        old_pool = engine._dest_buffers
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *_args, **_kwargs: pytest.fail("configuration allocated"),
        )

        config = engine._configure_ring(
            ShardedRDTWeightTransferInitInfo(
                staging_device="host",
                stages_in_flight=3,
                num_rdt_buffers=4,
                buffer_presize_gb=0.5,
            )
        )

        assert (
            config.staging_device,
            config.stages_in_flight,
            config.ring_depth,
            config.buffer_presize,
        ) == ("host", 3, 4, 1 << 29)
        assert engine._dest_buffers is old_pool

    def test_pool_has_one_k_slot_block_per_effective_lane(self):
        engine = _engine(stages_in_flight=2, ring_depth=3, streams=4)
        assert engine._lanes == 2
        assert engine._num_slots == 6
        assert len(engine._dest_buffers) == 6
        assert list(engine._lane_slots(0)) == [0, 1, 2]
        assert list(engine._lane_slots(1)) == [3, 4, 5]

    def test_requested_stages_are_clamped_to_plan_streams(self):
        engine = _engine(stages_in_flight=5, ring_depth=2, streams=3)
        assert (engine._lanes, engine._num_slots) == (3, 6)

    def test_empty_plan_keeps_one_lane(self):
        engine = _engine(
            stages_in_flight=5,
            ring_depth=2,
            streams=0,
            chunk_bytes=(),
        )
        assert (engine._lanes, engine._num_slots) == (1, 2)


class TestPoolReinitialization:
    @staticmethod
    def _activate(engine):
        engine._proc_thread = object()
        engine._slot_read_done = [object() for _ in range(engine._num_slots)]
        engine._slot_queued = [0] * engine._num_slots
        engine._slot_done = [0] * engine._num_slots

    @staticmethod
    def _references(engine):
        return (
            engine._dest_buffers,
            engine._slot_read_done,
            engine._slot_queued,
            engine._slot_done,
            engine._lane_cursors,
            engine._targets_cache,
        )

    @pytest.mark.parametrize(
        ("info", "plan"),
        [
            pytest.param(
                ShardedRDTWeightTransferInitInfo(
                    staging_device="host",
                    stages_in_flight=2,
                    num_rdt_buffers=2,
                ),
                _plan(streams=2),
                id="staging",
            ),
            pytest.param(
                ShardedRDTWeightTransferInitInfo(
                    stages_in_flight=1,
                    num_rdt_buffers=2,
                ),
                _plan(streams=1),
                id="lanes",
            ),
            pytest.param(
                ShardedRDTWeightTransferInitInfo(
                    stages_in_flight=1,
                    num_rdt_buffers=4,
                ),
                _plan(streams=1),
                id="equal-product-different-k",
            ),
        ],
    )
    def test_geometry_change_is_rejected_atomically(self, info, plan):
        engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
        self._activate(engine)
        references = self._references(engine)
        values = (
            engine._pool_geometry,
            engine._staging_device,
            engine._lanes,
            engine._ring_depth,
            engine._num_slots,
        )

        config = engine._configure_ring(info)
        with pytest.raises(RuntimeError, match=r"old=.*L=2, K=2.*new="):
            engine._size_pool(config, plan)

        assert all(
            actual is expected
            for actual, expected in zip(self._references(engine), references)
        )
        assert (
            engine._pool_geometry,
            engine._staging_device,
            engine._lanes,
            engine._ring_depth,
            engine._num_slots,
        ) == values

    def test_rejected_reinit_restores_the_complete_previous_init(self, monkeypatch):
        engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
        self._activate(engine)
        engine.parallel_config = SimpleNamespace(enable_eplb=False)
        engine._slot_cv = object()
        engine._num_consumers_override = 3
        engine._name_to_plan = {"old": [object()]}
        engine._name_meta = {"old": ("bfloat16", [4])}
        engine._live_names = {"old"}

        old_references = (
            engine._router,
            engine._name_to_plan,
            engine._name_meta,
            engine._live_names,
            engine._cached_plan,
            engine._dest_buffers,
            engine._slot_read_done,
            engine._slot_queued,
            engine._slot_done,
            engine._slot_cv,
            engine._lane_cursors,
            engine._targets_cache,
        )
        old_baked_values = (
            dict(engine._name_to_plan),
            dict(engine._name_meta),
            set(engine._live_names),
        )
        old_pool_values = (
            engine._pool_geometry,
            engine._staging_device,
            engine._stages_in_flight,
            engine._ring_depth,
            engine._lanes,
            engine._buffer_presize,
        )

        candidate_router = object()

        def resolve_producers(_info):
            engine._router = candidate_router

        def bake(_info):
            engine._name_to_plan["candidate"] = [object()]
            engine._name_meta = {"candidate": ("float32", [8])}
            engine._live_names = {"candidate"}

        monkeypatch.setattr(engine_module, "check_ray_rdt_version", lambda _mode: None)
        monkeypatch.setattr(engine, "_resolve_producers", resolve_producers)
        monkeypatch.setattr(engine, "_bake", bake)
        monkeypatch.setattr(
            engine, "_build_static_plan", lambda _info: _plan(streams=1)
        )
        monkeypatch.setattr(
            engine,
            "_preregister_at_init",
            lambda: pytest.fail("rejected re-init preregistered buffers"),
        )

        info = ShardedRDTWeightTransferInitInfo(
            stages_in_flight=1,
            num_rdt_buffers=4,
            num_consumers=9,
        )
        with pytest.raises(RuntimeError, match=r"old=.*L=2, K=2.*new=.*L=1, K=4"):
            engine.init_transfer_engine(info)

        assert all(
            actual is expected
            for actual, expected in zip(
                (
                    engine._router,
                    engine._name_to_plan,
                    engine._name_meta,
                    engine._live_names,
                    engine._cached_plan,
                    engine._dest_buffers,
                    engine._slot_read_done,
                    engine._slot_queued,
                    engine._slot_done,
                    engine._slot_cv,
                    engine._lane_cursors,
                    engine._targets_cache,
                ),
                old_references,
            )
        )
        assert (
            engine._name_to_plan,
            engine._name_meta,
            engine._live_names,
        ) == old_baked_values
        assert (
            engine._pool_geometry,
            engine._staging_device,
            engine._stages_in_flight,
            engine._ring_depth,
            engine._lanes,
            engine._buffer_presize,
        ) == old_pool_values
        assert engine._num_consumers_override == 3

    def test_failed_preregistration_restores_active_init(self, monkeypatch):
        engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
        self._activate(engine)
        engine.parallel_config = SimpleNamespace(enable_eplb=False)
        engine._slot_cv = object()
        engine._num_consumers_override = 3
        engine._name_to_plan = {"old": [object()]}
        engine._name_meta = {"old": ("bfloat16", [4])}
        engine._live_names = {"old"}
        engine._lane_cursors[:] = [4, 5]
        old_target = object()
        engine._targets_cache[("old", 0)] = old_target
        old_buffer_values = []
        for slot in engine._dest_buffers:
            value = object()
            slot[torch.uint8] = value
            old_buffer_values.append(value)

        reference_names = (
            "_router",
            "_name_to_plan",
            "_name_meta",
            "_live_names",
            "_cached_plan",
            "_dest_buffers",
            "_lane_cursors",
            "_targets_cache",
            "_slot_read_done",
            "_slot_queued",
            "_slot_done",
            "_slot_cv",
        )
        old_references = {name: getattr(engine, name) for name in reference_names}
        old_dest_slots = list(engine._dest_buffers)
        old_baked_values = (
            dict(engine._name_to_plan),
            dict(engine._name_meta),
            set(engine._live_names),
        )
        old_pool_values = (
            engine._pool_geometry,
            engine._staging_device,
            engine._stages_in_flight,
            engine._ring_depth,
            engine._lanes,
            engine._buffer_presize,
        )
        old_worker_values = (
            list(engine._slot_read_done),
            list(engine._slot_queued),
            list(engine._slot_done),
        )
        candidate_router = object()
        candidate_plan = _plan(streams=2)
        candidate_buffer = object()

        def resolve_producers(_info):
            engine._router = candidate_router

        def bake(_info):
            engine._name_to_plan["candidate"] = [object()]
            engine._name_meta = {"candidate": ("float32", [8])}
            engine._live_names = {"candidate"}

        def preregister():
            assert engine._cached_plan is candidate_plan
            for slot in engine._dest_buffers:
                slot.clear()
                slot[torch.uint8] = candidate_buffer
            engine._dest_buffers.append({torch.uint8: candidate_buffer})
            engine._lane_cursors[:] = [8, 9]
            engine._lane_cursors = [10]
            engine._targets_cache[("candidate", 0)] = object()
            engine._targets_cache = {("replacement", 0): object()}
            engine._slot_read_done.clear()
            engine._slot_queued[:] = [11]
            engine._slot_done[:] = [12]
            engine._slot_read_done = [object()]
            engine._slot_queued = [13]
            engine._slot_done = [14]
            engine._slot_cv = object()
            engine._pool_geometry = ("host", 1, 7)
            engine._staging_device = "host"
            engine._stages_in_flight = 1
            engine._ring_depth = 7
            engine._lanes = 1
            engine._buffer_presize = 123
            raise RuntimeError("candidate preregistration failed")

        monkeypatch.setattr(engine_module, "check_ray_rdt_version", lambda _mode: None)
        monkeypatch.setattr(engine, "_resolve_producers", resolve_producers)
        monkeypatch.setattr(engine, "_bake", bake)
        monkeypatch.setattr(engine, "_build_static_plan", lambda _info: candidate_plan)
        monkeypatch.setattr(engine, "_preregister_at_init", preregister)
        monkeypatch.setattr(
            engine,
            "_ensure_proc_worker",
            lambda: pytest.fail("failed preregistration started a worker"),
        )

        info = ShardedRDTWeightTransferInitInfo(
            stages_in_flight=5,
            num_rdt_buffers=2,
            buffer_presize_gb=0.5,
            num_consumers=9,
        )
        with pytest.raises(RuntimeError, match="candidate preregistration failed"):
            engine.init_transfer_engine(info)

        assert all(
            getattr(engine, name) is reference
            for name, reference in old_references.items()
        )
        assert len(engine._dest_buffers) == len(old_dest_slots)
        assert all(
            actual is expected
            for actual, expected in zip(engine._dest_buffers, old_dest_slots)
        )
        assert all(
            slot == {torch.uint8: value}
            for slot, value in zip(engine._dest_buffers, old_buffer_values)
        )
        assert (
            engine._name_to_plan,
            engine._name_meta,
            engine._live_names,
        ) == old_baked_values
        assert (
            engine._pool_geometry,
            engine._staging_device,
            engine._stages_in_flight,
            engine._ring_depth,
            engine._lanes,
            engine._buffer_presize,
        ) == old_pool_values
        assert engine._lane_cursors == [4, 5]
        assert engine._targets_cache == {("old", 0): old_target}
        assert (
            engine._slot_read_done,
            engine._slot_queued,
            engine._slot_done,
        ) == old_worker_values
        assert engine._num_consumers_override == 3

    def test_exact_geometry_reuses_every_pool_reference(self):
        engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
        self._activate(engine)
        references = self._references(engine)
        config = engine._configure_ring(
            ShardedRDTWeightTransferInitInfo(
                stages_in_flight=5,
                num_rdt_buffers=2,
            )
        )

        engine._size_pool(config, _plan(streams=2))

        assert engine._pool_geometry == ("gpu", 2, 2)
        assert all(
            actual is expected
            for actual, expected in zip(self._references(engine), references)
        )


class TestReceiveAllocation:
    def test_gpu_slots_use_engine_device_without_pinning(
        self,
        registered,
        allocations,
    ):
        engine = _engine()

        engine._preregister_at_init()

        slot_calls = [
            call for call in allocations.calls if call.get("dtype") is torch.uint8
        ]
        assert len(slot_calls) == 2
        assert all(call["device"] == engine.device for call in slot_calls)
        assert all("pin_memory" not in call for call in slot_calls)
        assert len(registered) == 2

    def test_preregisters_every_l_by_k_slot_once(self, registered):
        engine = _engine(
            stages_in_flight=2,
            ring_depth=2,
            streams=2,
            chunk_bytes=(48, 16),
        )

        engine._preregister_at_init()

        buffers = [slot[torch.uint8] for slot in engine._dest_buffers]
        assert registered == buffers
        assert len({buffer.data_ptr() for buffer in buffers}) == 4
        assert all(buffer.numel() == 48 for buffer in buffers)

        engine._preregister_at_init()
        assert registered == buffers

    def test_defensive_grow_uses_the_same_registered_seam(
        self,
        registered,
        monkeypatch,
    ):
        import ray.experimental

        monkeypatch.setattr(
            ray.experimental,
            "set_target_for_ref",
            lambda _ref, _targets: None,
        )
        engine = _engine(ring_depth=1)
        engine._preregister_at_init()
        engine._dest_buffers[0].clear()

        engine._issue_pull(engine._cached_plan.chunks[0], 0)

        assert registered[-1] is engine._dest_buffers[0][torch.uint8]
        assert len(registered) == 2

    def test_host_slots_are_pinned_cpu_memory(
        self,
        registered,
        allocations,
    ):
        engine = _engine(
            staging_device="host",
            stages_in_flight=2,
            ring_depth=2,
            streams=2,
        )

        engine._preregister_at_init()

        slot_calls = [
            call for call in allocations.calls if call.get("dtype") is torch.uint8
        ]
        assert len(slot_calls) == 4
        assert all(
            call["device"] == "cpu" and call["pin_memory"] is True
            for call in slot_calls
        )
        assert len(registered) == 4

    def test_host_allocation_failure_is_named_and_preserves_cause(
        self,
        registered,
        allocations,
    ):
        del registered
        engine = _engine(
            staging_device="host",
            stages_in_flight=2,
            ring_depth=2,
            streams=2,
        )
        allocations.failure = RuntimeError("pinned allocator exhausted")

        with pytest.raises(HostStagingAllocationError) as raised:
            engine._preregister_at_init()

        assert "4 slots (2 lane(s) x K=2)" in str(raised.value)
        assert "128 bytes per worker" in str(raised.value)
        assert raised.value.__cause__ is allocations.failure

    def test_host_registration_failure_preserves_cause(
        self,
        monkeypatch,
        allocations,
        registered,
    ):
        del registered
        engine = _engine(staging_device="host", ring_depth=1)
        cause = RuntimeError("NIXL registration failed")
        monkeypatch.setattr(
            engine_module,
            "register_nixl_memory",
            lambda _buffer: (_ for _ in ()).throw(cause),
        )

        with pytest.raises(
            HostStagingAllocationError,
            match="NIXL registration failed",
        ) as raised:
            engine._preregister_at_init()

        assert raised.value.__cause__ is cause


def test_slot_reuse_waits_for_queued_read_done_event():
    class Event:
        ready = False
        queries = 0

        def query(self):
            self.queries += 1
            return self.ready

    engine = _engine(ring_depth=1)
    event = Event()
    engine._slot_read_done = [event]
    engine._slot_cv = threading.Condition()
    engine._slot_queued = [1]
    engine._slot_done = [0]

    assert not engine._slot_free(0)
    assert event.queries == 0

    engine._slot_done[0] = 1
    assert not engine._slot_free(0)
    event.ready = True
    assert engine._slot_free(0)


class _LanePulls:
    """Transport-free hooks that check each lane's slot window."""

    def __init__(self, depth: int):
        self.depth = depth
        self.inflight: dict[int, _Chunk] = {}
        self.events: list[tuple] = []
        self.waits: list[tuple[list[SimpleNamespace], float | None]] = []
        self.peak_by_block: Counter[int] = Counter()

    def issue(self, chunk, slot):
        assert slot not in self.inflight
        self.inflight[slot] = chunk
        block = slot // self.depth
        count = sum(other // self.depth == block for other in self.inflight)
        self.peak_by_block[block] = max(self.peak_by_block[block], count)
        self.events.append(("issue", chunk, slot))
        return SimpleNamespace(chunk=chunk, slot=slot)

    def wait_any(self, pulls, timeout=None):
        self.waits.append((list(pulls), timeout))
        return 0

    def drain(self, pending, chunk):
        assert self.inflight.pop(pending.slot) is chunk
        self.events.append(("drain", chunk, pending.slot))

    def signal(self, group):
        self.events.append(("signal", group))

    def hooks(self):
        return _PullHooks(
            issue=self.issue,
            slot_free=lambda _slot: True,
            wait_any=self.wait_any,
            drain=self.drain,
            signal=self.signal,
        )


def _run_lane_plan(plan: _CallPlan, *, lanes: int = 2, depth: int = 2):
    pulls = _LanePulls(depth)
    streams = [
        [chunk for chunk in plan.chunks if chunk.stream == stream]
        for stream in range(plan.num_streams)
    ]
    _run_lanes(
        streams,
        depth=depth,
        blocks=[range(lane * depth, (lane + 1) * depth) for lane in range(lanes)],
        cursors=[0] * lanes,
        hooks=pulls.hooks(),
    )
    return pulls


class TestLaneScheduler:
    def test_each_lane_is_a_k_deep_fifo_on_a_disjoint_slot_block(self):
        depth = 2
        pulls = _run_lane_plan(_stream_plan(), lanes=2, depth=depth)

        issues: dict[int, list[_Chunk]] = {}
        drains: dict[int, list[_Chunk]] = {}
        stream_blocks: dict[int, set[int]] = {}
        for kind, chunk, slot in (
            event for event in pulls.events if event[0] != "signal"
        ):
            block = slot // depth
            target = issues if kind == "issue" else drains
            target.setdefault(block, []).append(chunk)
            if kind == "issue":
                stream_blocks.setdefault(chunk.stream, set()).add(block)

        assert pulls.peak_by_block
        assert all(peak == depth for peak in pulls.peak_by_block.values())
        assert drains == issues
        assert all(len(blocks) == 1 for blocks in stream_blocks.values())

    def test_lane_handoff_spans_the_last_pull_of_the_previous_stream(self):
        pulls = _run_lane_plan(_stream_plan(streams=3), lanes=2, depth=2)
        stream_two_issue = next(
            index
            for index, event in enumerate(pulls.events)
            if event[0] == "issue" and event[1].stream == 2
        )
        before = pulls.events[:stream_two_issue]
        issued_zero = [
            event[1] for event in before if event[0] == "issue" and event[1].stream == 0
        ]
        drained = {id(event[1]) for event in before if event[0] == "drain"}
        blocks = {
            chunk.stream: slot // 2
            for chunk, slot in (
                (event[1], event[2]) for event in pulls.events if event[0] == "issue"
            )
        }

        assert len(issued_zero) == 3
        assert id(issued_zero[-1]) not in drained
        assert blocks[2] == blocks[0]

    def test_waits_on_all_active_lane_heads(self):
        pulls = _run_lane_plan(_stream_plan(streams=2), lanes=2, depth=2)
        heads, timeout = pulls.waits[0]
        assert {pending.slot for pending in heads} == {0, 2}
        assert timeout is None

    def test_slot_recheck_passes_exact_five_millisecond_timeout(self):
        class StopAfterWait(Exception):
            pass

        streams = [
            [_chunk(1, stream=stream, key=f"s{stream}.{seq}") for seq in range(2)]
            for stream in range(2)
        ]
        seen = {}

        def wait_any(pulls, timeout):
            seen["heads"] = pulls
            seen["timeout"] = timeout
            raise StopAfterWait

        hooks = _PullHooks(
            issue=lambda chunk, slot: SimpleNamespace(chunk=chunk, slot=slot),
            # Lane 0 issues slot 0, then waits for slot 1. Lane 1 fills.
            slot_free=lambda slot: slot != 1,
            wait_any=wait_any,
            drain=lambda _pending, _chunk: None,
            signal=lambda _group: None,
        )
        with pytest.raises(StopAfterWait):
            _run_lanes(
                streams,
                depth=2,
                blocks=[range(0, 2), range(2, 4)],
                cursors=[0, 0],
                hooks=hooks,
            )

        assert len(seen["heads"]) == 2
        assert seen["timeout"] == _SLOT_POLL_S == 0.005


class _SharedPacks:
    """A pull lands only after every replica has issued its shared pack."""

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
    monkeypatch.setattr(
        engine,
        "_wait_any_head",
        lambda _pulls, timeout=None: 0,
    )
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


def test_one_lane_matches_the_pre_lane_issue_drain_and_free_order(monkeypatch):
    engine = _engine(stages_in_flight=1, ring_depth=2, streams=3)
    chunks = [
        _chunk(1, stream=0, key="a", free=[0]),
        _chunk(1, stream=0, key="b"),
        _chunk(1, stream=1, key="c", free=[1]),
        _chunk(1, stream=2, key="d", free=[2]),
    ]
    engine._cached_plan = _CallPlan(
        chunks=chunks,
        pre_free=[9],
        num_streams=3,
    )
    engine._completed_syncs = 1

    events, peak = _record_pipeline(engine, monkeypatch)

    assert peak == 2
    assert events == [
        ("signal", 9),
        ("issue", "a", 0),
        ("issue", "b", 1),
        ("drain", "a"),
        ("signal", 0),
        ("issue", "c", 0),
        ("drain", "b"),
        ("issue", "d", 1),
        ("drain", "c"),
        ("signal", 1),
        ("drain", "d"),
        ("signal", 2),
    ]
