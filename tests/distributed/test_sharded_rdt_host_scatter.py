# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise host-staged scatter copy and receive-slot lifetime on one GPU."""

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
    _ProcItem,
)
from vllm.distributed.weight_transfer.sharded_rdt_fake import _Scatter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)

SLICE = 64 << 20
HOLD_SECONDS = 0.5


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
def setup(monkeypatch):
    """Build a one-slot host-staging engine without its worker threads."""
    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device)
    monkeypatch.setattr(engine_module, "register_nixl_memory", lambda _tensor: None)
    monkeypatch.setattr(
        engine_module, "deregister_nixl_memory", lambda _tensor: None, raising=False
    )

    engine = object.__new__(ShardedRDTWeightTransferEngine)
    engine.device = device
    engine._proc_thread = None
    engine._quant_thread = None
    engine._slot_read_done = []
    engine._pool_geometry = None
    engine._dest_buffers = [{}]
    engine._name_to_plan = {}
    engine._name_meta = {}
    engine._live_names = set()
    engine._cached_plan = None

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


def test_the_copy_is_queued_not_waited_for(setup):
    seconds = _scatter(setup)
    queued = not setup.engine._slot_read_done[0].query()
    setup.engine._proc_stream.synchronize()
    held = setup.held[0].elapsed_time(setup.held[1]) / 1000

    assert seconds < held / 5, (
        f"the scatter took {seconds:.3f}s of the stream's {held:.3f}s hold"
    )
    assert queued, "the copy should still be queued behind the hold"
    assert torch.equal(setup.layer.weight.cpu(), setup.pattern)


def test_the_slot_is_pulled_into_again_only_after_its_copy_ran(
    setup,
    monkeypatch,
):
    import ray.experimental

    _scatter(setup)
    assert not setup.engine._slot_free(0), (
        "the copy is still queued, so the slot is not free"
    )
    seen = {}

    def set_target_for_ref(_ref, _buffers):
        seen["copy ran"] = setup.engine._slot_read_done[0].query()

    monkeypatch.setattr(
        ray.experimental,
        "set_target_for_ref",
        set_target_for_ref,
    )
    setup.engine._issue_pull(setup.chunk, 0)

    assert seen["copy ran"], (
        "the next pull was pointed at the slot before the copy out of it ran"
    )
    setup.slot.zero_()
    setup.engine._proc_stream.synchronize()
    assert torch.equal(setup.layer.weight.cpu(), setup.pattern)
