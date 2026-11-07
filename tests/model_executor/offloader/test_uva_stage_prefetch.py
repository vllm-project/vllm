# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-stream prefetch of staged UVA-offloaded expert weights.

The device is simulated: every stream is a queue of operations, an operation
runs when the test scheduler picks its stream, and an operation that waits on
an incomplete event blocks its stream. Copies are real tensor copies that land
when their operation runs, so a staged read sees exactly what a device would
have in the buffer at that point of the schedule.
"""

import contextlib
import random

import pytest
import torch
import torch.nn as nn

from vllm.config import OffloadConfig, UVAOffloadConfig
from vllm.model_executor.offloader import uva
from vllm.model_executor.offloader.uva import UVAExpertStager, UVAOffloader

pytestmark = pytest.mark.skip_global_cleanup


class _Event:
    def __init__(self) -> None:
        self.done = False

    def record(self, stream: "_Stream") -> None:
        def complete() -> bool:
            self.done = True
            return True

        stream.ops.append(complete)


class _Stream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: list = []
        self.capturing = False

    def is_capturing(self) -> bool:
        return self.capturing

    def wait_event(self, event: _Event) -> None:
        self.ops.append(lambda: event.done)

    def step(self) -> bool:
        """Run the next operation if it is not blocked; False if blocked."""
        if not self.ops:
            return False
        if self.ops[0]():
            self.ops.pop(0)
            return True
        return False


class _Device:
    """Stands in for torch.xpu / torch.cuda: streams, events, the current one."""

    Event = _Event

    def __init__(self) -> None:
        self.compute = _Stream("compute")
        self.current = self.compute
        self.streams = [self.compute]
        self.copies: list[int] = []  # ids of the weights copied, in issue order

    def Stream(self, device=None) -> _Stream:  # noqa: N802
        s = _Stream(f"side{len(self.streams)}")
        self.streams.append(s)
        return s

    def current_stream(self) -> _Stream:
        return self.current

    @contextlib.contextmanager
    def stream(self, stream: _Stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def run_some(self, rng: random.Random, n: int) -> None:
        for _ in range(n):
            rng.choice(self.streams).step()

    def drain(self) -> None:
        while any(s.ops for s in self.streams):
            if not any(s.step() for s in self.streams):
                raise AssertionError("deadlock: every stream waits on an event")


@pytest.fixture
def dev(monkeypatch) -> _Device:
    device = _Device()
    monkeypatch.setattr(torch, "get_device_module", lambda _d=None: device)
    monkeypatch.setattr(torch.accelerator, "current_stream", device.current_stream)

    def copy(dst: torch.Tensor, src: torch.Tensor) -> None:
        # Lands when the current stream gets to it, as a device copy would.
        device.copies.append(id(src))
        device.current.ops.append(lambda: dst.copy_(src) is not None)

    monkeypatch.setattr(uva._PrefetchRing, "_copy", staticmethod(copy))
    return device


def _layers(n: int, shape=(4, 8)) -> list[nn.Parameter]:
    out = []
    for k in range(n):
        p = nn.Parameter(torch.full(shape, float(k + 1)), requires_grad=False)
        p._vllm_is_uva_offloaded = True
        out.append(p)
    return out


def _step(stager, dev, layers, reads, rng=None, tokens=4096, slot="w1"):
    """One forward over `layers`: stage, then enqueue the kernel's read of the
    staged buffer on the compute stream (it records what it saw)."""
    for k, w in enumerate(layers):
        staged = stager.maybe_stage(w, tokens, slot)

        def read(staged=staged, k=k, w=w):
            reads.append((k, torch.equal(staged, w), staged.data_ptr()))
            return True

        dev.compute.ops.append(read)
        if rng is not None:
            dev.run_some(rng, rng.randint(0, 6))


def test_ring_alternates_buffers(dev):
    stager = UVAExpertStager(min_tokens=1, prefetch=True)
    layers = _layers(6)
    reads: list = []
    _step(stager, dev, layers, reads)
    dev.drain()
    assert [ok for _, ok, _ in reads] == [True] * 6
    ptrs = [p for _, _, p in reads]
    assert len(set(ptrs)) == 2
    assert ptrs[0] != ptrs[1]
    assert ptrs[0::2] == [ptrs[0]] * 3 and ptrs[1::2] == [ptrs[1]] * 3
    assert not stager._buffers  # the single-buffer path is not used
    assert len(stager._rings) == 1


def test_layer_order_is_learned_from_the_calls(dev):
    stager = UVAExpertStager(min_tokens=1, prefetch=True)
    layers = _layers(5)
    reads: list = []
    _step(stager, dev, layers, reads)
    dev.drain()
    ring = next(iter(stager._rings.values()))
    # Learned: each layer's successor.
    assert [ring.next[id(layers[k])] is layers[k + 1] for k in range(4)] == [True] * 4

    # The first step could not prefetch (no order yet); the second does: when a
    # layer stages, the copy of the following one is already issued.
    dev.copies.clear()
    for k in range(5):
        stager.maybe_stage(layers[k], 4096, "w1")
        if k < 4:
            assert id(layers[k + 1]) in dev.copies
    dev.drain()
    # ...and the first layer follows the last, seen when the step repeats.
    assert ring.next[id(layers[4])] is layers[0]


def test_no_prefetch_while_capturing(dev):
    stager = UVAExpertStager(min_tokens=1, prefetch=True)
    layers = _layers(3)
    dev.compute.capturing = True
    for w in layers:
        assert stager.maybe_stage(w, 4096, "w1") is w
    assert not stager._rings and not dev.copies and not dev.streams[1:]

    # Learned order is not disturbed by the captured calls either.
    dev.compute.capturing = False
    reads: list = []
    _step(stager, dev, layers, reads)
    dev.drain()
    assert all(ok for _, ok, _ in reads)


@pytest.mark.parametrize("seed", range(20))
def test_staged_read_sees_its_own_layer(dev, seed):
    """Random landing of copies and kernels over several steps, with
    decode-sized steps (no staging) in between: every staged read holds its
    layer."""
    rng = random.Random(seed)
    stager = UVAExpertStager(min_tokens=1024, prefetch=True)
    layers = _layers(7)
    reads: list = []
    for step in range(6):
        _step(stager, dev, layers, reads, rng)
        dev.run_some(rng, rng.randint(0, 20))
        # A decode-sized gap: nothing stages, nothing is read.
        for w in layers:
            assert stager.maybe_stage(w, 8, "w1") is w
    dev.drain()
    assert len(reads) == 6 * 7
    assert all(ok for _, ok, _ in reads), [r for r in reads if not r[1]]


def test_ring_check_has_a_control(dev, monkeypatch):
    """The check above catches a ring that does not wait for the kernel that
    reads a buffer before it is overwritten."""
    monkeypatch.setattr(
        uva._PrefetchRing, "_load", _load_without_free_wait, raising=True
    )
    caught = 0
    for seed in range(20):
        dev.copies.clear()
        for s in dev.streams:
            s.ops.clear()
        rng = random.Random(seed)
        stager = UVAExpertStager(min_tokens=1024, prefetch=True)
        layers = _layers(7)
        reads: list = []
        for _ in range(4):
            _step(stager, dev, layers, reads, rng)
        dev.drain()
        caught += any(not ok for _, ok, _ in reads)
    assert caught


def _load_without_free_wait(self, i, weight):
    with self._mod.stream(self.stream):
        self._copy(self.bufs[i], weight)
        event = self._mod.Event()
        event.record(self.stream)
    self.ready[i] = event
    self.owner[i] = id(weight)


def _after_two_steps(dev):
    """The order is learned in the first step, so from the second step's end
    on, layer 0 is prefetched for the next step."""
    stager = UVAExpertStager(min_tokens=1, prefetch=True)
    layers = _layers(4)
    for _ in range(2):
        _step(stager, dev, layers, [])
        dev.drain()
    return stager, layers


def test_invalidate_recopies_changed_weights(dev):
    """A weight changed in place after it was prefetched is read again from the
    host once the offloader is told (the swap does, at every map flip)."""
    stager, layers = _after_two_steps(dev)
    layers[0].data.fill_(-5.0)  # rewritten in place, e.g. a row of the swap
    stager.invalidate()
    reads: list = []
    _step(stager, dev, layers, reads)
    dev.drain()
    assert len(reads) == 4 and all(ok for _, ok, _ in reads)


def test_prefetched_copy_is_stale_without_invalidate(dev):
    """Control for the test above: the prefetched copy of layer 0 is read."""
    stager, layers = _after_two_steps(dev)
    layers[0].data.fill_(-7.0)
    reads: list = []
    _step(stager, dev, layers, reads)
    dev.drain()
    assert [k for k, ok, _ in reads if not ok] == [0]


def test_flag_off_is_the_base_behaviour(dev):
    off = UVAExpertStager(min_tokens=1024)
    base = UVAExpertStager(min_tokens=1024, prefetch=False)
    assert off.prefetch is False and not off._rings
    layers = _layers(4)
    outs = [off.maybe_stage(w, 4096, "w1") for w in layers]
    # One buffer shared by every layer, copied in place on the current stream.
    assert len({o.data_ptr() for o in outs}) == 1
    assert torch.equal(outs[-1], layers[-1])
    assert not off._rings and not dev.copies and not dev.streams[1:]
    assert len(off._buffers) == 1 and not base._buffers
    # Nothing to invalidate.
    off.invalidate()
    assert off.maybe_stage(layers[0], 4096, "w1") is outs[0]
    # Below the threshold and for plain tensors it is the weight itself.
    assert off.maybe_stage(layers[0], 1023, "w1") is layers[0]


def test_flag_plumbing_defaults_off():
    assert UVAOffloadConfig().cpu_offload_stage_prefetch is False
    offloader = UVAOffloader(cpu_offload_max_bytes=0, stage_min_tokens=16)
    assert offloader.expert_stager.prefetch is False
    on = UVAOffloader(cpu_offload_max_bytes=0, stage_min_tokens=16, stage_prefetch=True)
    assert on.expert_stager.prefetch is True
    on.invalidate_staged()  # no rings yet: a no-op

    def offload(prefetch):
        return OffloadConfig(
            uva=UVAOffloadConfig(cpu_offload_gb=1, cpu_offload_stage_prefetch=prefetch)
        )

    # Like the threshold, the flag acts only inside the opaque moe_forward op.
    assert offload(False).compile_factors() == offload(True).compile_factors()
