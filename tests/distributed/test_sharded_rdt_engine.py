# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receive-pool, scheduler, settings and CUDA-path tests for sharded RDT.

Most tests drive the engine on CPU with a fake CUDA runtime and stubbed NIXL
registration; the cases under ``requires_cuda`` need one GPU (they page-lock
through libcudart and time stream holds), so they skip on CPU and on ROCm.
"""

import logging
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_engine as engine_module
from tests.distributed.test_sharded_rdt_plan import consumer_engine
from tests.distributed.test_weight_transfer import RecordingClient
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    check_ray_rdt_version,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    HostStagingAllocationError,
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _CallPlan,
    _Chunk,
    _ProcItem,
    _PullHooks,
    _run_lanes,
)
from vllm.distributed.weight_transfer.sharded_rdt_fake import _Scatter
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)
from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_cuda()),
    reason="requires CUDA (page-locks through libcudart, times stream holds)",
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
    engine = consumer_engine()
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
def cuda_runtime(monkeypatch):
    """A fake ``CudaRTLibrary``: records host (un)registrations, returns codes."""
    state = SimpleNamespace(
        registered=[],
        unregistered=[],
        events=[],
        register_code=0,
        unregister_code=0,
        load_error=None,
    )

    class FakeCudaRT:
        def __init__(self, so_file=None):
            if state.load_error is not None:
                raise state.load_error

        def cudaHostRegister(self, ptr, size, flags=0):
            state.registered.append((ptr, size, flags))
            return state.register_code

        def cudaHostUnregister(self, ptr):
            state.unregistered.append(ptr)
            state.events.append(("unregister", ptr))
            return state.unregister_code

        def cudaGetErrorString(self, code):
            return f"fake cuda error {code}"

        def cudaGetLastError(self):
            return 0

    # raising=False: a tree without the fix has no such name, and the red run
    # must fail on behaviour, not on a missing attribute.
    monkeypatch.setattr(engine_module, "CudaRTLibrary", FakeCudaRT, raising=False)
    return state


@pytest.fixture
def nixl(monkeypatch, cuda_runtime):
    """Stub NIXL (de)registration; ``events`` interleaves with the CUDA fake."""
    state = SimpleNamespace(
        registered=[],
        deregistered=[],
        failure=None,
        deregister_failure=None,
        events=cuda_runtime.events,
    )

    def register(tensor):
        if state.failure is not None:
            raise state.failure
        state.registered.append(tensor)

    def deregister(tensor):
        if state.deregister_failure is not None:
            raise state.deregister_failure
        state.deregistered.append(tensor)
        state.events.append(("deregister", _base(tensor)))

    monkeypatch.setattr(engine_module, "register_nixl_memory", register)
    monkeypatch.setattr(
        engine_module, "deregister_nixl_memory", deregister, raising=False
    )
    monkeypatch.setattr(
        engine_module,
        "buffer_alloc_bytes",
        lambda nbytes, presize=0: max(nbytes, presize),
    )
    return state


@pytest.fixture
def engine_log(caplog):
    """``caplog`` wired to the engine's logger, which does not propagate to the
    root logger vLLM configures."""
    caplog.set_level(logging.ERROR, logger=engine_module.logger.name)
    engine_module.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        engine_module.logger.removeHandler(caplog.handler)


def _errors(caplog) -> list[str]:
    """The error-level messages, each record once even when the engine's
    logger also propagates to the root handler."""
    seen: dict[int, str] = {}
    for record in caplog.records:
        if record.levelno >= logging.ERROR:
            seen.setdefault(id(record), record.getMessage())
    return list(seen.values())


def _base(buffer: torch.Tensor) -> int:
    """The storage base address, what cudaHostRegister/Unregister take."""
    return buffer.untyped_storage().data_ptr()


def _ids(tensors) -> list[int]:
    return [id(tensor) for tensor in tensors]


def _released(buffers) -> list[tuple[str, int]]:
    """The event sequence a release of ``buffers`` in order must record."""
    return [
        event
        for buffer in buffers
        for event in (("deregister", _base(buffer)), ("unregister", _base(buffer)))
    ]


def _host_pool_engine(staging_device: str = "host") -> ShardedRDTWeightTransferEngine:
    """A 2x2 pool whose largest chunk is 48 bytes."""
    return _engine(
        staging_device=staging_device,
        stages_in_flight=2,
        ring_depth=2,
        streams=2,
        chunk_bytes=(48, 16),
    )


def _stub_init(engine, monkeypatch, plan: _CallPlan, *, fail: Exception | None):
    """Stub bake and routing so ``init_transfer_engine`` exercises the pool,
    pre-registration and (optionally) a failure right after it."""
    engine.parallel_config = SimpleNamespace(enable_eplb=False)
    engine._num_consumers_override = 0
    monkeypatch.setattr(engine_module, "check_ray_rdt_version", lambda _device: None)
    monkeypatch.setattr(engine, "_resolve_producers", lambda _info: None)
    monkeypatch.setattr(engine, "_bake", lambda _info: None)
    monkeypatch.setattr(engine, "_build_static_plan", lambda _info: plan)

    def ensure_proc_worker():
        if fail is not None:
            raise fail

    monkeypatch.setattr(engine, "_ensure_proc_worker", ensure_proc_worker)


_HOST_INIT = ShardedRDTWeightTransferInitInfo(
    staging_device="host", stages_in_flight=2, num_rdt_buffers=2
)


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
    ("staging_device", "expected_device", "host"),
    [("gpu", torch.device("cpu"), False), ("host", "cpu", True)],
)
def test_receive_slots_use_the_selected_memory_and_register_once(
    staging_device, expected_device, host, nixl, cuda_runtime, allocations
):
    engine = _host_pool_engine(staging_device)
    if not host:
        # GPU staging never touches the host CUDA runtime wrapper.
        cuda_runtime.load_error = AssertionError("libcudart is not loaded")
    engine._preregister_at_init()

    buffers = [slot[torch.uint8] for slot in engine._dest_buffers]
    slot_calls = [
        call for call in allocations.calls if call.get("dtype") is torch.uint8
    ]
    assert len(slot_calls) == len(buffers) == 4
    assert all(call["device"] == expected_device for call in slot_calls)
    # Exact size: never torch's pinned allocator, which rounds and caches.
    assert all("pin_memory" not in call for call in slot_calls)
    assert all(buffer.untyped_storage().nbytes() == 48 for buffer in buffers)
    assert _ids(nixl.registered) == _ids(buffers)
    # One cudaHostRegister per host slot, covering its whole storage.
    expected = [(_base(buffer), 48, 0) for buffer in buffers] if host else []
    assert cuda_runtime.registered == expected

    engine._preregister_at_init()
    assert _ids(nixl.registered) == _ids(buffers)
    assert cuda_runtime.registered == expected
    assert cuda_runtime.events == []


def test_shutdown_releases_every_host_slot_before_dropping_it(nixl, cuda_runtime):
    engine = _host_pool_engine()
    engine._preregister_at_init()
    buffers = [slot[torch.uint8] for slot in engine._dest_buffers]

    engine.shutdown()

    assert engine._dest_buffers == [{}, {}, {}, {}]
    assert _ids(nixl.deregistered) == _ids(buffers)
    assert cuda_runtime.events == _released(buffers)


def _regrow_slot_zero(engine, monkeypatch) -> None:
    """Pull a 64-byte chunk into slot 0, which every pool here sized smaller."""
    import ray.experimental

    monkeypatch.setattr(
        ray.experimental, "set_target_for_ref", lambda _ref, _targets: None
    )
    pending = engine._issue_pull(_chunk(64), 0)
    assert pending.blob.numel() == 64


def test_the_regrow_in_issue_pull_releases_the_replaced_host_slot(
    nixl, cuda_runtime, monkeypatch
):
    engine = _engine(staging_device="host", ring_depth=1, chunk_bytes=(32,))
    engine._preregister_at_init()
    old = engine._dest_buffers[0][torch.uint8]
    assert old.numel() == 32

    _regrow_slot_zero(engine, monkeypatch)

    new = engine._dest_buffers[0][torch.uint8]
    assert new is not old and new.numel() == 64
    assert _ids(nixl.registered) == _ids([old, new])
    assert cuda_runtime.events == _released([old])
    assert cuda_runtime.registered[-1] == (_base(new), 64, 0)


def test_gpu_staging_releases_nothing_on_regrow_or_shutdown(
    nixl, cuda_runtime, monkeypatch
):
    # GPU staging never touches the host CUDA runtime wrapper.
    cuda_runtime.load_error = AssertionError("libcudart is not loaded")
    engine = _host_pool_engine("gpu")
    engine._preregister_at_init()
    old = engine._dest_buffers[0][torch.uint8]

    _regrow_slot_zero(engine, monkeypatch)
    new = engine._dest_buffers[0][torch.uint8]
    engine.shutdown()

    # GPU slots keep their NIXL registration for the process lifetime: the
    # regrow and the shutdown only drop the engine's references.
    assert new is not old and new.numel() == 64
    assert _ids(nixl.registered)[-1] == id(new)
    assert engine._dest_buffers == [{}, {}, {}, {}]
    assert nixl.deregistered == [] and cuda_runtime.events == []
    assert engine._parked_host_slots == []


def _shutdown_with_a_live_worker(engine, monkeypatch) -> None:
    """Shut down while the post-processing thread survives its join: it may
    still be scattering out of the slots."""
    engine._proc_thread = SimpleNamespace(
        join=lambda timeout=None: None, is_alive=lambda: True
    )
    engine._proc_queue = SimpleNamespace(put=lambda _item: None)
    engine._proc_stream = None
    monkeypatch.setattr(engine, "drain_pending", lambda: None)
    engine.shutdown()


@pytest.mark.parametrize(
    ("release", "trigger", "reason", "nixl_let_go"),
    [
        ("regrow", "nixl_deregister_raises", "NIXL deregistration failed", False),
        ("regrow", "cuda_host_unregister_nonzero", "cudaHostUnregister failed", True),
        ("shutdown", "worker_thread_alive", "did not stop", False),
        ("shutdown", "nixl_deregister_raises", "NIXL deregistration failed", False),
    ],
)
def test_a_host_slot_whose_release_is_unsure_is_parked_and_reported(
    release, trigger, reason, nixl_let_go, nixl, cuda_runtime, monkeypatch, engine_log
):
    engine = _host_pool_engine()
    engine._preregister_at_init()
    slots = [slot[torch.uint8] for slot in engine._dest_buffers]
    if trigger == "nixl_deregister_raises":
        nixl.deregister_failure = RuntimeError("NIXL refcount already zero")
    elif trigger == "cuda_host_unregister_nonzero":
        cuda_runtime.unregister_code = 1  # cudaErrorInvalidValue

    if release == "regrow":
        _regrow_slot_zero(engine, monkeypatch)
        parked = slots[:1]
        assert engine._dest_buffers[0][torch.uint8] is not slots[0]
    elif trigger == "worker_thread_alive":
        _shutdown_with_a_live_worker(engine, monkeypatch)
        parked = slots
        assert engine._dest_buffers == [{}, {}, {}, {}]
    else:
        engine.shutdown()
        parked = slots
        assert engine._dest_buffers == [{}, {}, {}, {}]

    # A slot that may still be NIXL-registered, page-locked or being read is
    # never handed back to the allocator: it stays referenced for the process
    # lifetime, and nothing that was not proven released is unregistered.
    assert _ids(engine._parked_host_slots) == _ids(parked)
    if nixl_let_go:
        assert _ids(nixl.deregistered) == _ids(parked)
        assert cuda_runtime.events == _released(parked)
    else:
        assert nixl.deregistered == [] and cuda_runtime.events == []
    # Each parked slot is reported once at error level with its base address
    # and byte count, so an operator can account for the leaked memory.
    errors = _errors(engine_log)
    if trigger == "worker_thread_alive":
        assert errors[0].startswith("RDT post-processing thread did not stop")
        errors = errors[1:]
    assert len(errors) == len(parked)
    for buffer, error in zip(parked, errors):
        assert f"{_base(buffer):#x}" in error and "48 bytes" in error
        assert reason in error and "never freed" in error


def test_a_failed_first_init_releases_every_host_slot_it_built(
    nixl, cuda_runtime, monkeypatch
):
    engine = _host_pool_engine()
    _stub_init(
        engine,
        monkeypatch,
        engine._cached_plan,
        fail=RuntimeError("worker start failed"),
    )

    with pytest.raises(RuntimeError, match="worker start failed"):
        engine.init_transfer_engine(_HOST_INIT)

    built = nixl.registered
    assert len(built) == 4
    assert engine._dest_buffers == [{}, {}, {}, {}]
    assert _ids(nixl.deregistered) == _ids(built)
    assert cuda_runtime.events == _released(built)


def test_a_failed_re_init_grow_keeps_the_restored_host_slots_registered(
    nixl, cuda_runtime, monkeypatch
):
    engine = _host_pool_engine()
    engine._preregister_at_init()
    old = [slot[torch.uint8] for slot in engine._dest_buffers]
    grown = _plan(streams=2, chunk_bytes=(64, 16))
    _stub_init(engine, monkeypatch, grown, fail=RuntimeError("worker start failed"))

    with pytest.raises(RuntimeError, match="worker start failed"):
        engine.init_transfer_engine(_HOST_INIT)

    new = nixl.registered[4:]
    assert len(new) == 4 and all(buffer.numel() == 64 for buffer in new)
    # The restored slots are the old ones, still registered and never released.
    assert _ids(slot[torch.uint8] for slot in engine._dest_buffers) == _ids(old)
    assert engine._cached_plan is not grown
    assert _ids(nixl.deregistered) == _ids(new)
    assert cuda_runtime.events == _released(new)


def test_a_committed_re_init_grow_releases_the_replaced_host_slots(
    nixl, cuda_runtime, monkeypatch
):
    engine = _host_pool_engine()
    engine._preregister_at_init()
    old = [slot[torch.uint8] for slot in engine._dest_buffers]
    grown = _plan(streams=2, chunk_bytes=(64, 16))
    _stub_init(engine, monkeypatch, grown, fail=None)

    engine.init_transfer_engine(_HOST_INIT)

    new = [slot[torch.uint8] for slot in engine._dest_buffers]
    assert _ids(new) == _ids(nixl.registered[4:])
    assert all(buffer.numel() == 64 for buffer in new)
    assert engine._cached_plan is grown
    assert _ids(nixl.deregistered) == _ids(old)
    assert cuda_runtime.events == _released(old)


_GEOMETRY = "4 slots (2 lane(s) x K=2) x 48 bytes = 192 bytes per worker"


def test_a_nonzero_cuda_host_register_code_fails_loudly_without_a_leak(
    nixl, cuda_runtime, allocations
):
    engine = _host_pool_engine()
    cuda_runtime.register_code = 712

    with pytest.raises(HostStagingAllocationError) as excinfo:
        engine._preregister_at_init()

    message = str(excinfo.value)
    assert "staging_device='host'" in message and _GEOMETRY in message
    assert "cudaHostRegister" in message and "NIXL" in message
    assert "712" in message and "fake cuda error 712" in message
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError) and "fake cuda error 712" in str(cause)
    assert len(cuda_runtime.registered) == 1
    assert nixl.registered == [] and cuda_runtime.events == []
    assert torch.uint8 not in engine._dest_buffers[0]
    assert all("pin_memory" not in call for call in allocations.calls)


def test_a_nixl_registration_failure_unregisters_the_partial_slot(nixl, cuda_runtime):
    engine = _host_pool_engine()
    nixl.failure = RuntimeError("NIXL refused the buffer")

    with pytest.raises(HostStagingAllocationError) as excinfo:
        engine._preregister_at_init()

    message = str(excinfo.value)
    assert "staging_device='host'" in message and _GEOMETRY in message
    assert "cudaHostRegister" in message and "NIXL" in message
    assert excinfo.value.__cause__ is nixl.failure
    assert len(cuda_runtime.registered) == 1
    base, size, _flags = cuda_runtime.registered[0]
    assert size == 48
    assert nixl.deregistered == [] and cuda_runtime.events == [("unregister", base)]
    assert torch.uint8 not in engine._dest_buffers[0]


def test_an_unloadable_cuda_runtime_fails_loudly_in_host_mode(
    nixl, cuda_runtime, allocations
):
    engine = _host_pool_engine()
    cuda_runtime.load_error = AssertionError("libcudart is not loaded")

    with pytest.raises(HostStagingAllocationError) as excinfo:
        engine._preregister_at_init()

    message = str(excinfo.value)
    assert "staging_device='host'" in message and _GEOMETRY in message
    assert "libcudart is not loaded" in message
    assert excinfo.value.__cause__ is cuda_runtime.load_error
    assert cuda_runtime.registered == [] and nixl.registered == []
    assert cuda_runtime.events == []
    assert torch.uint8 not in engine._dest_buffers[0]
    assert all("pin_memory" not in call for call in allocations.calls)


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


# ---------------------------------------------------------------------------
# Settings shared by trainers and workers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("staging_device", "cpu"),
        ("stages_in_flight", 0),
        ("stages_in_flight", True),
        ("num_rdt_buffers", -1),
        ("buffer_presize_gb", float("nan")),
        ("num_consumers", 0),
    ],
)
def test_trainer_rejects_invalid_settings(field, value):
    with pytest.raises(ValueError, match=field):
        ShardedRDTTrainerInitInfo(rank=0, **{"num_consumers": 1, field: value})


def test_settings_reach_workers_unchanged():
    trainer = ShardedRDTTrainerInitInfo(
        rank=0,
        num_consumers=7,
        staging_device="host",
        stages_in_flight=3,
        num_rdt_buffers=4,
        buffer_presize_gb=0.5,
    )
    engine = ShardedRDTTrainerWeightTransferEngine(
        client=RecordingClient(), source=None, is_sender=True, init_info=trainer
    )
    engine._groups = [["w"]]
    engine._meta = [SimpleNamespace(name="w", dtype=torch.bfloat16, shape=(4,))]
    engine._owner_sets = [[0]]
    engine._name_owner_class = [0]

    worker = ShardedRDTWeightTransferInitInfo(
        **asdict(engine._build_worker_init_info(["server-0"]))
    )

    assert (
        worker.num_consumers,
        worker.staging_device,
        worker.stages_in_flight,
        worker.num_rdt_buffers,
        worker.buffer_presize_gb,
    ) == (7, "host", 3, 4, 0.5)


@pytest.mark.parametrize(
    ("staging_device", "installed", "passes"),
    [
        ("gpu", "2.55.0", False),
        ("gpu", "2.56.0", True),
        ("host", "2.57.0", False),
        ("host", "2.58.0", True),
    ],
)
def test_ray_floor_depends_on_staging_mode(
    monkeypatch, staging_device, installed, passes
):
    monkeypatch.setattr("importlib.metadata.version", lambda _name: installed)
    if passes:
        check_ray_rdt_version(staging_device)
    else:
        with pytest.raises(ValueError, match=staging_device):
            check_ray_rdt_version(staging_device)


# ---------------------------------------------------------------------------
# CUDA: host-staged scatter copies, receive-slot lifetime, quant-stream lifetime
# ---------------------------------------------------------------------------

SLICE = 64 << 20
HOLD_SECONDS = 0.5
NUMEL = 16 << 20


def _hold(
    stream: torch.cuda.Stream,
    seconds: float,
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """Keep ``stream`` busy for approximately ``seconds``."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    probe = 10_000_000
    start.record()
    torch.cuda._sleep(probe)
    end.record()
    end.synchronize()
    cycles = int(probe * seconds * 1000 / start.elapsed_time(end))

    held = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    with torch.cuda.stream(stream):
        held[0].record()
        torch.cuda._sleep(cycles)
        held[1].record()
    return held


@pytest.fixture
def host_scatter(monkeypatch):
    """Build a one-slot host-staging engine without its worker threads."""
    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device)
    monkeypatch.setattr(engine_module, "register_nixl_memory", lambda _tensor: None)
    monkeypatch.setattr(
        engine_module, "deregister_nixl_memory", lambda _tensor: None, raising=False
    )

    engine = consumer_engine(device=device)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(SLICE, dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    scatter = _Scatter(
        layer=layer,
        param_name="weight",
        src=("w", ()),
        offset=0,
        shape=(SLICE,),
        stride=(1,),
        dtype=torch.uint8,
    )
    chunk = _Chunk(
        scatters=[scatter],
        keys=[("w", ())],
        pack_layout=[(0, torch.uint8, SLICE, (SLICE,))],
        pack_bytes=SLICE,
        materialize=[],
        quant=[],
        free=[],
        owner=0,
    )
    plan = _CallPlan(chunks=[chunk], pre_free=[], num_streams=1)
    config = engine._configure_ring(
        ShardedRDTWeightTransferInitInfo(
            num_rdt_buffers=1,
            staging_device="host",
        )
    )
    engine._size_pool(config, plan)

    slot = engine._alloc_receive_buffer(SLICE)
    # The whole storage is page-locked, so the scatter copies at the pinned rate.
    assert slot.is_pinned()
    engine._dest_buffers[0][torch.uint8] = slot
    engine._router = SimpleNamespace(
        pull=lambda _owner, _keys, _seq: object(),
    )
    engine._proc_stream = torch.cuda.Stream(device=device)
    engine._slot_read_done = [torch.cuda.Event()]
    engine._slot_queued = [0]
    engine._slot_done = [0]
    engine._slot_cv = threading.Condition()

    state = SimpleNamespace(
        engine=engine,
        chunk=chunk,
        layer=layer,
        slot=slot,
    )
    # Warm the copy path so first-use costs are not measured as scatter time.
    _scatter(state, hold=False)
    engine._proc_stream.synchronize()
    layer.weight.data.zero_()
    state.pattern = (torch.arange(SLICE, dtype=torch.int32) % 251).to(torch.uint8)
    slot.copy_(state.pattern)
    yield state
    # Shutdown gives the host slot back (NIXL deregistration is stubbed above,
    # cudaHostUnregister is real) once the scatter out of it has run.
    engine.shutdown()


def _scatter(state, hold=True) -> float:
    """Process one landed slot while optionally holding the process stream."""
    item = _ProcItem(
        chunk=state.chunk,
        results={("w", ()): state.slot[:SLICE]},
        slot=0,
    )
    state.engine._slot_queued[0] += 1
    if hold:
        state.held = _hold(state.engine._proc_stream, HOLD_SECONDS)
    start = time.perf_counter()
    state.engine._process_item(item)
    return time.perf_counter() - start


@requires_cuda
def test_the_copy_is_queued_not_waited_for(host_scatter):
    seconds = _scatter(host_scatter)
    queued = not host_scatter.engine._slot_read_done[0].query()
    host_scatter.engine._proc_stream.synchronize()
    held = host_scatter.held[0].elapsed_time(host_scatter.held[1]) / 1000

    assert seconds < held / 5, (
        f"the scatter took {seconds:.3f}s of the stream's {held:.3f}s hold"
    )
    assert queued, "the copy should still be queued behind the hold"
    assert torch.equal(host_scatter.layer.weight.cpu(), host_scatter.pattern)


@requires_cuda
def test_the_slot_is_pulled_into_again_only_after_its_copy_ran(
    host_scatter,
    monkeypatch,
):
    import ray.experimental

    _scatter(host_scatter)
    assert not host_scatter.engine._slot_free(0), (
        "the copy is still queued, so the slot is not free"
    )
    seen = {}

    def set_target_for_ref(_ref, _buffers):
        seen["copy ran"] = host_scatter.engine._slot_read_done[0].query()

    monkeypatch.setattr(
        ray.experimental,
        "set_target_for_ref",
        set_target_for_ref,
    )
    host_scatter.engine._issue_pull(host_scatter.chunk, 0)

    assert seen["copy ran"], (
        "the next pull was pointed at the slot before the copy out of it ran"
    )
    host_scatter.slot.zero_()
    host_scatter.engine._proc_stream.synchronize()
    assert torch.equal(host_scatter.layer.weight.cpu(), host_scatter.pattern)


@requires_cuda
def test_a_layers_kernel_copy_is_not_overwritten_by_the_next_layer():
    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device)
    engine = consumer_engine(device=device)
    engine._proc_stream = torch.cuda.Stream(device=device)
    engine._quant_stream = torch.cuda.Stream(device=device)
    kernel = torch.nn.Parameter(
        torch.zeros(NUMEL, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    torch.accelerator.synchronize()

    # The layer as materialize_layer leaves it and the scatter fills it.
    layer = torch.nn.Module()
    with torch.cuda.stream(engine._proc_stream):
        loaded = torch.full((NUMEL,), 1.0, dtype=torch.bfloat16, device=device)
        layer.weight = torch.nn.Parameter(loaded, requires_grad=False)
        ready = torch.cuda.Event()
        ready.record(engine._proc_stream)
    loaded_ptr = loaded.data_ptr()
    del loaded
    LAYERWISE_INFO[layer] = LayerReloadingInfo(
        restore_metadata=({}, {}),
        restore_device=device,
        kernel_tensors=({"weight": kernel}, {}),
    )
    engine._proc_stream.synchronize()

    _hold(engine._quant_stream, HOLD_SECONDS)
    engine._run_quant([layer], ready)

    # The next layer, materialized and scattered on the process stream.
    with torch.cuda.stream(engine._proc_stream):
        following = torch.empty(NUMEL, dtype=torch.bfloat16, device=device)
        following.fill_(2.0)
    reused = following.data_ptr() == loaded_ptr
    engine._quant_stream.synchronize()
    engine._proc_stream.synchronize()

    assert layer.weight is kernel
    assert torch.equal(kernel, torch.full_like(kernel, 1.0)), (
        "the kernel tensor holds the next layer's bytes: its source was reused "
        f"before the quant stream's copy read it (memory reused: {reused})"
    )
