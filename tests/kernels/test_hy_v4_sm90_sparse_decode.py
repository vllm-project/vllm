# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 split-K sparse MLA decode on SM90 against FlashMLA's sparse kernel, on
the HY4-Flash TP4 decode shapes (DSA: 8 heads, 2048 indices; SWA: 16 heads, 512
indices), eagerly and inside CUDA graphs, plus its fallbacks and its trap."""

import random
import subprocess
import sys
import textwrap

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(90):
    pytest.skip("needs SM90", allow_module_level=True)

from vllm.models.hy_v4.nvidia import sparse_mla_decode as splitk  # noqa: E402
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd  # noqa: E402

DIM, NUM_KV, SCALE = 576, 1 << 16, 0.0625
DSA, SWA = (8, 2048), (16, 512)
# The ranges cross the block size (T 8 -> 9) and merge-chunk (T * heads 64 ->
# 65, 128 -> 129) boundaries.
SHAPES = [(*DSA, t) for t in (1, 2, 3, 8, 9, 16, 17, 24, 40)] + [
    (*SWA, t) for t in (1, 4, 5, 8, 9, 16)
]
# Two BF16 kernels that round P at different block boundaries. About 2x the
# worst gap to FlashMLA over 40 seeds of every taken shape on H200 (1.4e-3 at
# rtol 2^-7).
ATOL, RTOL = 3e-3, 2**-7


def _inputs(num_tokens, heads, width, seed, num_kv=NUM_KV):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # A head slice of the 64-head query buffer, as the backend passes it.
    q = torch.randn(num_tokens, 64, DIM, generator=gen, device="cuda") * 0.5
    q = q.bfloat16()[:, :heads]
    kv = torch.randn(num_kv, 1, DIM, generator=gen, device="cuda").bfloat16()
    indices = torch.randint(
        0, num_kv, (num_tokens, 1, width), generator=gen, device="cuda"
    ).int()
    indices[:, :, ::7] = -1
    lengths = torch.randint(
        1, width + 1, (num_tokens,), generator=gen, device="cuda"
    ).int()
    lengths[0] = width
    if num_tokens > 2:
        lengths[1] = 0
    sinks = torch.randn(heads, generator=gen, device="cuda")
    return q, kv, indices, sinks, lengths


def _flashmla(q, kv, indices, sinks, lengths):
    return flash_mla_sparse_fwd(
        q, kv, indices, SCALE, attn_sink=sinks, topk_length=lengths
    )[0]


def _run(kernel, q, kv, indices, sinks, lengths):
    return kernel(q, kv, indices, SCALE, sinks, lengths)


def _kernel_launches(fn) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.accelerator.synchronize()
    return sum(e.count for e in prof.key_averages() if "_decode_kernel" in e.key)


@pytest.fixture(scope="module")
def kernel():
    kernel = splitk.SparseMLADecodeKernel()
    assert kernel.prepare(DSA[0]) and kernel.prepare(SWA[0])
    return kernel


@pytest.mark.parametrize("heads,width,num_tokens", SHAPES)
def test_matches_flashmla(kernel, workspace_init, heads, width, num_tokens):
    """Holes, a cut and an empty index row, against FlashMLA; bitwise
    run-to-run."""
    inputs = _inputs(num_tokens, heads, width, seed=num_tokens)
    out = _run(kernel, *inputs)
    assert out is not None
    ref = _flashmla(*inputs)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
    if num_tokens > 2:
        assert out[1].abs().max().item() == 0 == ref[1].abs().max().item()
    assert torch.equal(out, _run(kernel, *inputs))


@pytest.mark.parametrize("fused", [True, False])
def test_repeated_launches_bitwise_stable(kernel, request, fused):
    """Every compiled row count, fused and two-launch, 100 launches each with
    an L2 flush in between, bit for bit. This guards the software-pipelined
    gather loop against shared-memory races (cf. triton-lang/triton#6691),
    which comparing a graph with the eager call cannot catch."""
    if fused:
        request.getfixturevalue("workspace_init")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for (heads, width), max_tokens in ((DSA, 40), (SWA, 16)):
        for num_tokens in range(1, max_tokens + 1):
            inputs = _inputs(num_tokens, heads, width, seed=num_tokens)
            first = _run(kernel, *inputs)
            assert first is not None
            mismatches = torch.zeros((), dtype=torch.int64, device="cuda")
            for _ in range(100):
                flush.zero_()
                mismatches += (_run(kernel, *inputs) != first).sum()
            assert mismatches.item() == 0, (heads, num_tokens, fused)


@pytest.mark.parametrize("heads,width,num_tokens", [(*DSA, 16), (*SWA, 16)])
def test_flashmla_contract_edges(kernel, workspace_init, heads, width, num_tokens):
    """Indices >= the KV row count are unused and +inf / -inf sinks behave as in
    FlashMLA (a +inf sink zeroes its head)."""
    num_kv = 4096
    q, kv, indices, sinks, lengths = _inputs(
        num_tokens, heads, width, seed=7, num_kv=num_kv
    )
    gen = torch.Generator(device="cuda").manual_seed(8)
    beyond = torch.randint(
        num_kv, 3 * num_kv, indices.shape, generator=gen, device="cuda"
    ).int()
    pick = torch.rand(indices.shape, generator=gen, device="cuda") < 0.3
    indices = torch.where(pick, beyond, indices)
    indices[0, 0, :16] = num_kv + 5
    sinks[0], sinks[1] = float("inf"), float("-inf")

    out = _run(kernel, q, kv, indices, sinks, lengths)
    assert out is not None
    ref = _flashmla(q, kv, indices, sinks, lengths)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
    assert out[:, 0].abs().max().item() == 0 == ref[:, 0].abs().max().item()


def test_declines_other_calls(kernel, workspace_init):
    """Calls outside the measured shapes or the warmed layouts return None, so
    the backend runs FlashMLA."""
    q, kv, indices, sinks, lengths = _inputs(4, *DSA, seed=1)
    assert _run(kernel, *_inputs(41, *DSA, seed=1)) is None
    assert _run(kernel, *_inputs(17, *SWA, seed=1)) is None
    assert _run(kernel, *_inputs(4, 8, 512, seed=1)) is None
    assert _run(kernel, *_inputs(4, 32, 2048, seed=1)) is None
    assert kernel(q, kv, indices, SCALE, None, lengths) is None
    strided = torch.stack([lengths, lengths], 1).view(-1)[::2]
    assert _run(kernel, q, kv, indices, sinks, strided) is None
    unaligned = torch.empty(4, 8, DIM + 1, dtype=q.dtype, device="cuda")[..., 1:]
    unaligned.copy_(q)
    assert _run(kernel, unaligned, kv, indices, sinks, lengths) is None
    assert _run(kernel, q.half(), kv.half(), indices, sinks, lengths) is None


def test_cuda_graphs_replay_in_any_order(kernel, workspace_init):
    """Graphs of several sizes share the workspace: replays in random order
    with fresh inputs match FlashMLA and the eager call bit for bit."""
    sizes = [(1, 1), (8, 4), (16, 16), (40, 9)]
    for dsa_t, swa_t in sizes:  # eager calls reserve the workspace
        _run(kernel, *_inputs(dsa_t, *DSA, seed=0))
        _run(kernel, *_inputs(swa_t, *SWA, seed=0))
    graphs = []
    for dsa_t, swa_t in sizes:
        static = [_inputs(dsa_t, *DSA, seed=1), _inputs(swa_t, *SWA, seed=1)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [_run(kernel, *inputs) for inputs in static]
        graphs.append(((dsa_t, swa_t), graph, static, outs))

    rng = random.Random(0)
    for step in range(40):
        (dsa_t, swa_t), graph, static, outs = rng.choice(graphs)
        fresh = [
            _inputs(dsa_t, *DSA, seed=100 + step),
            _inputs(swa_t, *SWA, seed=200 + step),
        ]
        for dst, src in zip(static, fresh):
            for d, s in zip(dst, src):
                d.copy_(s)
        graph.replay()
        for out, inputs in zip(outs, fresh):
            torch.testing.assert_close(out, _flashmla(*inputs), atol=ATOL, rtol=RTOL)
            assert torch.equal(out, _run(kernel, *inputs))


def test_capture_without_workspace(kernel, workspace_init):
    """A workspace slot first used under capture gets no zero-filled buffer (its
    zero fill would exist only inside the graph): the graph runs split and
    merge as two launches on scratch, and stays correct."""
    inputs = _inputs(16, *DSA, seed=3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(kernel, *inputs)
    graph.replay()
    torch.testing.assert_close(out, _flashmla(*inputs), atol=ATOL, rtol=RTOL)
    assert splitk.reserve_workspace() is not None
    assert _kernel_launches(lambda: _run(kernel, *inputs)) == 1


def test_two_launches_without_workspace_manager(kernel):
    """Without a workspace (here: no workspace manager), a call runs split and
    merge as two launches instead of the fused one."""
    inputs = _inputs(16, *DSA, seed=3)
    assert splitk.reserve_workspace() is None
    assert _kernel_launches(lambda: _run(kernel, *inputs)) == 2
    torch.testing.assert_close(
        _run(kernel, *inputs), _flashmla(*inputs), atol=ATOL, rtol=RTOL
    )


def test_kill_switch_and_batch_invariance(monkeypatch):
    """Either variable routes every call to FlashMLA."""
    assert splitk.splitk_decode_supported(DSA[0])
    for name in ("VLLM_DISABLE_HY4_SM90_SPLITK_MLA_DECODE", "VLLM_BATCH_INVARIANT"):
        with monkeypatch.context() as m:
            m.setenv(name, "1")
            assert not splitk.splitk_decode_supported(DSA[0])
            assert not splitk.SparseMLADecodeKernel().prepare(DSA[0])


def test_compile_failure_falls_back(workspace_init, monkeypatch):
    """A Triton compile error (e.g. after a Triton upgrade) disables the path
    instead of raising, once."""
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("injected compile failure")

    monkeypatch.setattr(splitk._decode_kernel, "warmup", fail)
    kernel = splitk.SparseMLADecodeKernel()
    inputs = _inputs(4, *DSA, seed=4)
    assert _run(kernel, *inputs) is None
    assert _run(kernel, *inputs) is None
    assert len(calls) == 1


_TRAP_CHILD = """
import sys, torch
from vllm.models.hy_v4.nvidia import sparse_mla_decode as splitk
from vllm.v1.worker.workspace import init_workspace_manager

init_workspace_manager(torch.device(0))
kernel = splitk.SparseMLADecodeKernel()
assert kernel.prepare(8)
gen = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(4, 8, 576, generator=gen, device="cuda").bfloat16()
kv = torch.randn(4096, 1, 576, generator=gen, device="cuda").bfloat16()
idx = torch.randint(0, 4096, (4, 1, 2048), generator=gen, device="cuda").int()
lengths = torch.full((4,), 2048, dtype=torch.int32, device="cuda")
sinks = torch.zeros(8, device="cuda")
assert kernel(q, kv, idx, 0.0625, sinks, lengths) is not None
counters = splitk.reserve_workspace().view(torch.int32)
if sys.argv[1] == "ticket":
    counters[splitk._TICKET_SLOT] = 5  # another launch taking tickets
else:
    counters[0] = -1000  # an arrival that never comes
torch.accelerator.synchronize()
try:
    kernel(q, kv, idx, 0.0625, sinks, lengths)
    torch.accelerator.synchronize()
except RuntimeError as e:
    print("TRAPPED:", str(e).splitlines()[0], flush=True)
    import os
    os._exit(3)
print("NO ERROR", flush=True)
"""


@pytest.mark.parametrize("corruption", ["ticket", "counter"])
def test_trap_on_corrupted_workspace(corruption):
    """Misuse of the workspace (two launches sharing it) raises a CUDA error
    instead of hanging or returning wrong data. The trap is a sticky error that
    ends the CUDA context, hence the child process."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_TRAP_CHILD), corruption],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "TRAPPED" in result.stdout
