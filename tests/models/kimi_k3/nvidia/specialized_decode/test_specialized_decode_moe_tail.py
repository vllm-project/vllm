# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the MoE tail's front-mode policy tables
(`cutedsl.moe.tail_runtime`: TAIL_GATE and POLL_NS per (TP size, DCP size) and batch).

- The tables reach only the MoeTail front batches they name for their (TP, DCP) pair;
  published batches (WaveMoeTail), other pairs and other batches compile the default
  MoeTail. `PreparedTail` passes the deployment's DCP size to the lookup.
- The policy kernels compile, and their PTX keeps every instruction of the default
  kernel but the poll sleep (and its operand move) and adds one CTA barrier: the same
  loads, stores and arithmetic, so the tail's output bits are the default's.
"""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import tail_runtime
from vllm.platforms import current_platform

BATCHES = tuple(range(1, 65))


def policy_on(tp, dcp, batch):
    """The shipped tables: TP16 + DCP16, MoeTail front batches 5-8."""
    return (tp, dcp) == (16, 16) and 5 <= batch <= 8


@pytest.mark.parametrize("dcp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("tp", [8, 16])
def test_tail_policy_tables(tp, dcp):
    for b in BATCHES:
        on = policy_on(tp, dcp, b)
        assert tail_runtime.front_gate(tp, dcp, b) is on, (tp, dcp, b)
        want = 0 if on else None
        assert tail_runtime.front_poll_ns(tp, dcp, b) == want, (tp, dcp, b)


def test_tail_policy_skips_published_batches(monkeypatch):
    """Table entries on published batches (WaveMoeTail) have no effect."""
    gate = {(8, 1): "1-64:1", (16, 4): "1-64:1"}
    sleep = {(8, 1): "1-64:32", (16, 4): "1-64:0"}
    monkeypatch.setattr(tail_runtime, "TAIL_GATE", gate)
    monkeypatch.setattr(tail_runtime, "POLL_NS", sleep)
    for tp, dcp in gate:
        for b in BATCHES:
            front = not tail_runtime.published(tp, b)
            assert tail_runtime.front_gate(tp, dcp, b) is front, (tp, b)
            want = (32 if tp == 8 else 0) if front else None
            assert tail_runtime.front_poll_ns(tp, dcp, b) == want, (tp, b)


class CompileRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, tp, batch, device, mode="front", trigger=2, **kwargs):
        self.calls.append((tp, batch, mode, kwargs))
        return object()


@pytest.mark.parametrize("tp,dcp", [(8, 1), (8, 8), (16, 1), (16, 8), (16, 16)])
def test_prepared_tail_compiles_the_policy(tp, dcp, monkeypatch):
    """Every batch's compile request through PreparedTail and MoeTailWeights (no
    GPU: the compiles, tensor maps and CuTe views are stubbed)."""
    record = CompileRecorder()
    monkeypatch.setattr(tail_runtime, "compile_tail", record)
    monkeypatch.setattr(tail_runtime, "compile_push", lambda *args: object())
    monkeypatch.setattr(tail_runtime, "tensor_maps", lambda specs, device: object())
    monkeypatch.setattr(tail_runtime, "to_cute", lambda tensor, *args: tensor)
    monkeypatch.setattr(tail_runtime, "resident_clusters", lambda device: 7)
    op = SimpleNamespace(contract=SimpleNamespace(tp_size=tp, rms_eps=1e-6), rank=0)
    up = torch.empty((7168, 3584), dtype=torch.bfloat16)
    norm = torch.empty(3584, dtype=torch.bfloat16)
    box = SimpleNamespace(tail=(None, None, None, None), push_box=(None, None))
    tail_runtime.PreparedTail(op, norm, up, BATCHES, (box, 1, 92), dcp=dcp)
    assert [call[1] for call in record.calls] == list(BATCHES)
    twoshots = tail_runtime.twoshot_batches(tp)
    for _, b, mode, kwargs in record.calls:
        published = tail_runtime.published(tp, b)
        want = "twoshot" if b in twoshots else "publish" if published else "front"
        assert mode == want, (tp, b, mode)
        on = policy_on(tp, dcp, b)
        assert kwargs["gate"] is on, (tp, dcp, b, kwargs)
        assert kwargs["poll_ns"] == (0 if on else None), (tp, dcp, b, kwargs)


def tail_ptx(batch, kwargs, dump, monkeypatch):
    """PTX of the TP16 MoeTail front kernel for `batch` (uncached compile)."""
    import cutlass.cute as cute

    compile_ = cute.compile
    dump.mkdir(parents=True, exist_ok=True)

    def keep_ptx(*args, options="", **kw):
        options = f"{options} --keep-ptx --dump-dir={dump}"
        return compile_(*args, options=options, **kw)

    torch.zeros(1, device="cuda")  # the compile's driver calls need a context
    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", keep_ptx)
        device = torch.accelerator.current_device_index()
        code = tail_runtime.compile_tail.__wrapped__(16, batch, device, **kwargs)
    ptx = code.compiled.__ptx__
    if not ptx:
        pytest.skip("this CuTe DSL does not return the kernel PTX")
    return ptx


def opcodes(ptx):
    """Instruction counts by opcode (predicate guards and operands dropped)."""
    counts: Counter[str] = Counter()
    for line in ptx.splitlines():
        line = line.split("//")[0].strip()
        if not line.endswith(";") or line.startswith((".", "{", "}")):
            continue
        if line.startswith("@"):
            line = line.split(None, 1)[1]
        counts[line.split()[0].rstrip(";")] += 1
    return counts


CTA_BARRIER = ("bar.sync", "barrier.sync", "barrier.sync.aligned", "bar.cta.sync")


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)
@pytest.mark.parametrize("batch", [5, 6, 7, 8])
def test_tail_policy_kernel_changes_only_the_poll_start(batch, monkeypatch, tmp_path):
    """Every instruction count of the default kernel is kept but one more CTA
    barrier, no poll sleep and at most one register move per removed sleep (its
    operand)."""
    default = opcodes(tail_ptx(batch, {}, tmp_path / "default", monkeypatch))
    policy = dict(gate=True, poll_ns=0)
    gated = opcodes(tail_ptx(batch, policy, tmp_path / "policy", monkeypatch))
    delta = {op: gated[op] - default[op] for op in default | gated}
    delta = {op: n for op, n in delta.items() if n}
    assert sum(delta.pop(op, 0) for op in CTA_BARRIER) == 1, delta
    sleeps = [op for op in default if op.startswith("nanosleep")]
    removed = sum(default[op] for op in sleeps)
    assert removed > 0 and sum(-delta.pop(op, 0) for op in sleeps) == removed
    moves = [op for op in delta if op.startswith("mov.")]
    assert 0 <= sum(-delta.pop(op) for op in moves) <= removed, moves
    assert not delta, delta
