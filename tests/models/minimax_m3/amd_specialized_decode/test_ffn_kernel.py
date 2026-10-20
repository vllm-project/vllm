# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU stage parity of the FFN launch (a TP=1 build) against vLLM's and
AITER's native ops, with MXFP4 weights from vLLM's real post-load path."""

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn  # noqa: E402
from vllm.models.minimax_m3.amd.specialized_decode.mono.runner import (  # noqa: E402
    FfnRunner,
)

E = ffn.EXPERTS + 1
H, INTER = ffn.HIDDEN, ffn.INTER


def _metrics(got, want):
    got, want = got.float().flatten(), want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    rel = ((got - want).norm() / want.norm()).item()
    return cos, rel


@pytest.fixture(scope="module")
def weights():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scales(*shape):
        # e8m0 codes around 2^-7: gate/up pre-activations of a few units
        return torch.randint(
            119, 123, shape, dtype=torch.uint8, device=dev, generator=g
        )

    def param(t):
        return torch.nn.Parameter(t, requires_grad=False)

    w13, w2, s13, s2, _, _ = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        layer=torch.nn.Module(),
        w13_weight=param(codes(E, 2 * INTER, H // 2)),
        w2_weight=param(codes(E, H, INTER // 2)),
        w13_weight_scale=param(scales(E, 2 * INTER, H // 32)),
        w2_weight_scale=param(scales(E, H, INTER // 32)),
    )
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(ffn.EXPERTS, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(ffn.EXPERTS, device=dev, generator=g).mul(0.01),
    )


def _native_experts(x, w, ids, wts):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    return fused_moe(
        x,
        w.w13,
        w.w2,
        wts,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w.s13,
        w2_scale=w.s2,
        gate_mode="separated",
        swiglu_limit=ffn.LIMIT,
    )


def _native_route(logits, bias):
    from aiter import biased_grouped_topk

    s = logits.shape[0]
    ids = torch.empty(s, ffn.TOPK, dtype=torch.int32, device=logits.device)
    wts = torch.empty(s, ffn.TOPK, dtype=torch.float32, device=logits.device)
    biased_grouped_topk(logits, bias, wts, ids, 1, 1, True, ffn.ROUTED_SCALE)
    return ids, wts


def _clear_choices(logits, bias, margin=1e-4):
    """Rows whose 4th and 5th biased scores differ by more than ``margin``."""
    top = (logits.sigmoid() + bias).topk(ffn.TOPK + 1, dim=-1).values
    return (top[:, ffn.TOPK - 1] - top[:, ffn.TOPK]) > margin


def _pairs(region, n):
    return region[: n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _runner(weights, s, tp=1, experts=ffn.Experts.ALL, **build):
    dev = torch.device("cuda")
    bufs = [torch.zeros(ffn.peer_bytes(tp), dtype=torch.uint8, device=dev)]
    bufs += [torch.zeros_like(bufs[0]) for _ in range(tp - 1)]
    peers = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=dev)
    runner = FfnRunner(tp, 0, bufs[0].data_ptr(), peers, dev, (s,), experts, **build)
    runner.warm(weights)
    runner.peer_bufs = bufs
    return runner


def _error(buf):
    return buf[: 4 * ffn.ERROR_WORDS].view(torch.int32).tolist()


def _inputs(s, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    part = torch.randn(s, H, device="cuda", generator=g).bfloat16()
    residual = torch.randn(s, H, device="cuda", generator=g).mul(4).bfloat16()
    return part, residual


def _run(weights, s, experts=ffn.Experts.ALL, seed=1, inputs=None, slots=None):
    part, residual = _inputs(s, seed) if inputs is None else inputs
    if slots is None:
        slots = torch.arange(s, dtype=torch.int64, device="cuda")
    runner = _runner(weights, s, experts=experts)
    out, res_out = runner(part, residual, weights, slots)
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0])[0] == 0
    x = runner.region(s, "xn")[: s * H * 2].view(torch.bfloat16).view(s, H)
    parts = _pairs(runner.region(s, "logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s * ffn.EXPERTS, ffn.ROUTER_PARTS)
    logits = parts[:, 0].clone()
    for p in range(1, ffn.ROUTER_PARTS):
        logits += parts[:, p]
    route = _pairs(runner.region(s, "route"), s * 2 * ffn.TOPK).view(s, 2 * ffn.TOPK)
    return SimpleNamespace(
        part=part,
        residual=residual,
        out=out,
        res_out=res_out,
        x=x,
        logits=logits.view(s, ffn.EXPERTS),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
    )


@pytest.mark.parametrize("s", [1, 3, 8, 16])
def test_ffn_stages_match_native(weights, s):
    r = _run(weights, s)
    want_x, want_res = gemma_fused_add_rmsnorm(
        r.part, r.residual, weights.post_norm, ffn.EPS
    )
    torch.testing.assert_close(r.res_out, want_res, rtol=0, atol=0)
    cos, rel = _metrics(r.x, want_x)
    assert cos > 0.99999 and rel < 4e-3

    want_logits = r.x.float() @ weights.router.T
    torch.testing.assert_close(r.logits, want_logits, rtol=1e-4, atol=1e-4)

    ids, wts = _native_route(r.logits, weights.router_bias)
    clear = _clear_choices(r.logits, weights.router_bias)
    assert clear.any()
    got_ids, order = r.ids.sort(-1)
    want_ids, want_order = ids.sort(-1)
    assert torch.equal(got_ids[clear], want_ids[clear])
    torch.testing.assert_close(
        r.wts.gather(1, order)[clear],
        wts.gather(1, want_order)[clear],
        rtol=1e-5,
        atol=1e-6,
    )

    shared = torch.full((s, 1), ffn.SHARED, dtype=torch.int32, device=r.x.device)
    all_ids = torch.cat([r.ids, shared], 1)
    all_wts = torch.cat([r.wts, torch.ones_like(r.wts[:, :1])], 1)
    want = _native_experts(r.x, weights, all_ids, all_wts)
    cos, rel = _metrics(r.out, want)
    assert cos > 0.9999 and rel < 1e-2


SHORT_WAIT = 10_000_000  # 100 ms of the 100 MHz clock


def test_production_wait_bound_is_ten_seconds():
    assert ffn.FfnBuild(tokens=1).wait_ticks == 1_000_000_000


def test_missing_peer_times_out_and_marks_every_rank(weights):
    """A TP=2 launch whose peer never runs: its waits give up, the error word
    is set on both ranks, and the launch still completes."""
    runner = _runner(weights, 1, tp=2, wait_ticks=SHORT_WAIT)
    part, residual = _inputs(1)
    runner(part, residual, weights, torch.zeros(1, dtype=torch.int64, device="cuda"))
    torch.accelerator.synchronize()
    for buf in runner.peer_bufs:
        failed, rank, epoch, site = _error(buf)
        # stages that start waiting together race to report their site
        assert (failed, rank, epoch) == (1, 0, 0) and site > 0


def test_missing_epoch_mark_fails_without_advancing(weights):
    runner = _runner(weights, 1, drop_mark=7, wait_ticks=SHORT_WAIT)
    part, residual = _inputs(1)
    runner(part, residual, weights, torch.zeros(1, dtype=torch.int64, device="cuda"))
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0]) == [1, 0, 0, ffn.SITE_EPOCH]
    assert runner.epoch[0].item() == 0


def test_padded_rows_are_inert(weights):
    """Rows whose slot is -1 (graph padding) hold NaN: the real rows match a
    launch without them bit for bit, the padded rows pick no expert and return
    zeros."""
    real, pad = 3, 5
    part, residual = _inputs(real)
    nan = torch.full((pad, H), float("nan"), device="cuda").bfloat16()
    slots = torch.tensor([7, 9, 11] + [-1] * pad, dtype=torch.int64, device="cuda")
    padded = _run(
        weights,
        real + pad,
        inputs=(torch.cat([part, nan]), torch.cat([residual, nan])),
        slots=slots,
    )
    alone = _run(weights, real, inputs=(part, residual))
    assert torch.equal(padded.out[:real], alone.out)
    assert torch.equal(padded.res_out[:real], alone.res_out)
    assert torch.equal(padded.ids[:real], alone.ids)
    assert (padded.ids[real:] == -1).all()
    assert (padded.out[real:] == 0).all()


def test_all_padded_launch_completes(weights):
    r = _run(weights, 4, slots=torch.full((4,), -1, dtype=torch.int64, device="cuda"))
    assert (r.ids == -1).all() and (r.out == 0).all()


def test_route_ties_and_near_ties(weights):
    """Experts 0 and 1 share a router row and bias: an exact 4th-place tie,
    which the kernel gives to the lower expert. Elsewhere the selection matches
    AITER's on native FP32 logits wherever the 4th/5th margin exceeds the
    logits' measured difference."""
    w = SimpleNamespace(**vars(weights))
    w.router = weights.router.clone()
    w.router_bias = weights.router_bias.clone()
    w.router[1] = w.router[0]
    w.router_bias[:2] = 0.5
    w.router_bias[[10, 11, 12]] = 2.0
    w.router_bias[2:10] = -1.0
    w.router_bias[13:] = -1.0
    r = _run(w, 16)
    assert (r.ids[:, :3].sort(-1).values == torch.tensor([10, 11, 12]).cuda()).all()
    assert (r.ids[:, 3] == 0).all()

    r = _run(weights, 16, seed=3)
    logits = r.x.float() @ weights.router.T
    bound = (r.logits - logits).abs().max().item() * 0.25 * 2
    ids, _ = _native_route(logits, weights.router_bias)
    top = (logits.sigmoid() + weights.router_bias).topk(ffn.TOPK + 1, dim=-1).values
    clear = (top[:, ffn.TOPK - 1] - top[:, ffn.TOPK]) > bound
    assert clear.float().mean() > 0.5
    assert torch.equal(r.ids.sort(-1).values[clear], ids.sort(-1).values[clear])


@pytest.mark.parametrize("experts", [ffn.Experts.SHARED, ffn.Experts.ROUTED])
@pytest.mark.parametrize("s", [1, 16])
def test_ffn_expert_sides_match_native(weights, s, experts):
    r = _run(weights, s, experts)
    if experts is ffn.Experts.SHARED:
        ids = torch.full((s, 1), ffn.SHARED, dtype=torch.int32, device=r.x.device)
        wts = torch.ones(s, 1, device=r.x.device)
    else:
        ids, wts = r.ids, r.wts
    want = _native_experts(r.x, weights, ids, wts)
    cos, rel = _metrics(r.out, want)
    assert cos > 0.9999 and rel < 1e-2


def test_every_width_is_resident(weights):
    """The driver's occupancy of every production build: one 512-thread CTA per
    CU on each of the 256 CUs."""
    dev = torch.device("cuda")
    peer = torch.zeros(ffn.peer_bytes(4), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()] * 4, dtype=torch.int64, device=dev)
    runner = FfnRunner(4, 0, peer.data_ptr(), peers, dev)
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    runner.warm(weights, compute_units=cus)
    assert sorted(runner.residency) == list(range(1, ffn.MAX_TOKENS + 1))
    assert all(n * cus >= ffn.BLOCKS for n in runner.residency.values())


def test_graph_replay_is_bit_stable(weights):
    """Every width captured once, replayed 300 rounds in alternation: each
    replay matches an eager launch bit for bit, also after the persistent slot
    buffer marks a different set of rows as padding."""
    dev = torch.device("cuda")
    widths = (1, 2, 4, 8, 16)
    peer = torch.zeros(ffn.peer_bytes(1), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()], dtype=torch.int64, device=dev)
    runner = FfnRunner(1, 0, peer.data_ptr(), peers, dev, widths)
    runner.warm(weights)
    inputs = {s: _inputs(s, seed=s) for s in widths}
    slots = {s: torch.arange(s, dtype=torch.int64, device=dev) for s in widths}

    def eager(s):
        out, res = runner(*inputs[s], weights, slots[s])
        return out.clone(), res.clone()

    graphs, outs = {}, {}
    for s in widths:
        eager(s)
        graphs[s] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[s]):
            outs[s] = runner(*inputs[s], weights, slots[s])
    refs = {s: eager(s) for s in widths}
    for _ in range(300):
        for s in widths:
            graphs[s].replay()
        for s in widths:
            assert torch.equal(outs[s][0], refs[s][0])
            assert torch.equal(outs[s][1], refs[s][1])

    for s in widths[1:]:
        slots[s][s // 2 :] = -1
        ref = eager(s)
        graphs[s].replay()
        assert torch.equal(outs[s][0], ref[0])
        assert (outs[s][0][s // 2 :] == 0).all()
    torch.accelerator.synchronize()
    assert _error(peer)[0] == 0
    assert runner.epoch[0].item() > 300 * len(widths)


def _slots(s):
    return torch.arange(s, dtype=torch.int64, device="cuda")


def test_final_launch_handshake_completes(weights):
    runner = _runner(weights, 2)
    inputs = _inputs(2)
    plain = runner(*inputs, weights, _slots(2))[0].clone()
    final = runner(*inputs, weights, _slots(2), final=True)[0]
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0])[0] == 0
    assert torch.equal(plain, final)


def test_injected_late_fault_reaches_the_error_word(weights):
    runner = _runner(weights, 1)
    runner.inject.copy_(torch.tensor([1, 100_000], dtype=torch.int32))
    runner(*_inputs(1), weights, _slots(1), final=True)
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0]) == [1, 0, 0, ffn.SITE_INJECT]


def test_epoch_limit_is_a_fail_stop_backstop(weights):
    runner = _runner(weights, 1)
    runner.epoch[0] = ffn.EPOCH_LIMIT
    runner(*_inputs(1), weights, _slots(1))
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0])[3] == ffn.SITE_WRAP


def test_forced_epoch_reset_clears_and_resumes(weights, monkeypatch):
    """The executor's step hook resets every region and the epoch once the
    launches since the last reset reach its bound; launches then resume from
    tag one with the same results, the error word kept."""
    from vllm.models.minimax_m3.amd.specialized_decode.runtime import executor as ex

    monkeypatch.setattr(ex, "tp_agree", lambda ok: ok)
    runner = _runner(weights, 2)
    inputs = _inputs(2)
    before = runner(*inputs, weights, _slots(2))[0].clone()
    executor = object.__new__(ex.FfnExecutor)
    executor.rank, executor.runner, executor.resets = 0, runner, 0
    executor.peer = SimpleNamespace(bytes=runner.peer_bufs[0])
    executor._layers = dict.fromkeys(range(3, 60))
    executor.outputs, executor._since_reset = 0, 0
    executor._reset_launches = 3 * len(executor._layers)
    executor._epoch_event = None
    executor._epoch_host = torch.zeros(1, dtype=torch.int32)
    runner.peer_bufs[0][4] = 7  # an error-word field the reset must keep
    for _ in range(3):
        runner(*inputs, weights, _slots(2))
        executor.output_step()
    torch.accelerator.synchronize()
    assert executor.resets == 1 and runner.epoch[0].item() == 0
    assert runner.peer_bufs[0][4].item() == 7
    assert not runner.peer_bufs[0][4 * ffn.ERROR_WORDS :].any()
    assert not any(buf.any() for buf in runner.scratch.values())
    runner.peer_bufs[0][4] = 0
    after = runner(*inputs, weights, _slots(2))[0]
    torch.accelerator.synchronize()
    assert torch.equal(before, after) and _error(runner.peer_bufs[0])[0] == 0


OBJDUMP = shutil.which("llvm-objdump") or next(
    iter(
        glob.glob(
            "/usr/local/lib/python3.12/dist-packages/_rocm_sdk_*/lib/llvm/bin/llvm-objdump"
        )
    ),
    None,
)


@pytest.mark.skipif(OBJDUMP is None, reason="needs llvm-objdump")
def test_isa_encodes_the_mailbox_contract(weights):
    """Every system-scope access is one aligned 64-bit pair (or the 32-bit
    error word); every release is a cache write-back then a drain before its
    store, every acquire a drain then an invalidate (LLVM AMDGPU GFX942
    memory-model code sequences)."""
    from vllm.models.minimax_m3.amd.specialized_decode.mono.residency import (
        code_object,
    )

    dev = torch.device("cuda")
    peer = torch.zeros(ffn.peer_bytes(4), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()] * 4, dtype=torch.int64, device=dev)
    runner = FfnRunner(4, 0, peer.data_ptr(), peers, dev, (1,))
    runner.warm(weights)
    elf = Path(tempfile.mkdtemp()) / "ffn.elf"
    elf.write_bytes(code_object(runner._compiled[1]._keepalive._ir_text))
    assert OBJDUMP is not None
    isa = subprocess.run(
        [OBJDUMP, "-d", "--mcpu=gfx950", str(elf)], capture_output=True, text=True
    ).stdout
    ops = [line.split("//")[0].split() for line in isa.splitlines() if "\t" in line]
    ops = [op for op in ops if op]
    system = [op[0] for op in ops if "sc0" in op and "sc1" in op]
    assert {o for o in system if "store" in o} == {"global_store_dwordx2"}
    assert {o for o in system if "load" in o} <= {
        "global_load_dwordx2",
        "global_load_dword",
    }
    for i, op in enumerate(ops):
        if op[0] == "buffer_wbl2":
            assert ops[i + 1][:2] == ["s_waitcnt", "vmcnt(0)"]
        if op[0] == "buffer_inv":
            assert ops[i - 1][0] == "s_waitcnt" and "vmcnt(0)" in ops[i - 1][1]
    kinds = {" ".join(op) for op in ops if op[0] in ("buffer_wbl2", "buffer_inv")}
    assert {"buffer_wbl2 sc1", "buffer_wbl2 sc0 sc1", "buffer_inv sc1"} <= kinds
