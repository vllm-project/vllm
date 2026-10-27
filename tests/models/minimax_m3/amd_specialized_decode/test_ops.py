# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-level tests of the fused FFN launch (TP=1 builds unless noted), with
MXFP4 weights from vLLM's real post-load path: every stage against a float64
reference (TP1 and TP4), routing ties, padding, residency, graph replay,
fail-stop sites, epoch reset, the gfx950 hand-off ISA, and the hand-off
stresses (one-GPU cross-XCD; TP4 over XGMI). ``*_match_vllm_ops``: the stages
against vLLM's and AITER's native ops (nightly)."""

import glob
import os
import shutil
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

from tests.models import mono_reference as R  # noqa: E402
from tests.models.minimax_m3.amd_specialized_decode import (  # noqa: E402
    _xcd_stress as xs,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono import (  # noqa: E402
    ffn,
    sync,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono.residency import (  # noqa: E402
    code_object,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono.runner import (  # noqa: E402
    FfnRunner,
    _writable_flydsl_cache,
)

E = ffn.EXPERTS + 1
H, INTER = ffn.HIDDEN, ffn.INTER
SHORT_WAIT = 10_000_000  # 100 ms of the 100 MHz clock


def _weights(rank=0, seed=0):
    """The checkpoint's MXFP4 experts (``raw``) and vLLM's post-load copies;
    ``rank`` > 0 draws its own experts (a TP shard), the replicated tensors
    alike on every rank."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)

    def experts(g):
        def codes(*shape):
            return torch.randint(
                0, 256, shape, dtype=torch.uint8, device=dev, generator=g
            )

        def scales(*shape):
            # e8m0 codes around 2^-7: gate/up pre-activations of a few units
            return torch.randint(
                119, 123, shape, dtype=torch.uint8, device=dev, generator=g
            )

        return SimpleNamespace(
            w13=codes(E, 2 * INTER, H // 2),
            w2=codes(E, H, INTER // 2),
            s13=scales(E, 2 * INTER, H // 32),
            s2=scales(E, H, INTER // 32),
        )

    def param(t):
        return torch.nn.Parameter(t.clone(), requires_grad=False)

    raw = experts(g)
    if rank:
        raw = experts(torch.Generator(device=dev).manual_seed(seed << 8 | rank))
    w13, w2, s13, s2, _, _ = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        layer=torch.nn.Module(),
        w13_weight=param(raw.w13),
        w2_weight=param(raw.w2),
        w13_weight_scale=param(raw.s13),
        w2_weight_scale=param(raw.s2),
    )
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        raw=raw,
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(ffn.EXPERTS, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(ffn.EXPERTS, device=dev, generator=g).mul(0.01),
    )


@pytest.fixture(scope="module")
def weights():
    return _weights()


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


def _native_router(x, router):
    """The FP32 router GEMM ``GateLinear`` runs on gfx950 up to 32 rows."""
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )

    return rocm_fp32_router_gemm(x, router)


def _clear_choices(logits, bias, err):
    """Rows whose float64 4th and 5th biased scores differ by more than twice
    ``err``, the most any path's scores may be off: both pick the same four."""
    top = (logits.double().sigmoid() + bias.double()).topk(ffn.TOPK + 1, dim=-1)
    return (top.values[:, ffn.TOPK - 1] - top.values[:, ffn.TOPK]) > 2 * err


def _pairs(region, n):
    return region[: n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _runner(weights, s, tp=1, widths=None, **build):
    dev = torch.device("cuda")
    bufs = [torch.zeros(ffn.peer_bytes(tp), dtype=torch.uint8, device=dev)]
    bufs += [torch.zeros_like(bufs[0]) for _ in range(tp - 1)]
    peers = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=dev)
    runner = FfnRunner(tp, 0, bufs[0].data_ptr(), peers, dev, widths or (s,), **build)
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


def _slots(s):
    return torch.arange(s, dtype=torch.int64, device="cuda")


def _run(weights, s, seed=1, inputs=None, slots=None):
    part, residual = _inputs(s, seed) if inputs is None else inputs
    runner = _runner(weights, s)
    out, res_out = runner(
        part, residual, weights, _slots(s) if slots is None else slots
    )
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0])[0] == 0
    x = runner.region(s, "xn")[: s * H * 2].view(torch.bfloat16).view(s, H)
    parts = _pairs(runner.region(s, "logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s, ffn.EXPERTS, ffn.ROUTER_PARTS)
    route = _pairs(runner.region(s, "route"), s * 2 * ffn.TOPK).view(s, 2 * ffn.TOPK)
    parts = parts.view(s, ffn.EXPERTS, 4, ffn.ROUTER_PARTS // 4)
    return SimpleNamespace(
        part=part,
        residual=residual,
        out=out,
        res_out=res_out,
        x=x,
        logits=_f32_sum(_f32_sum(parts.double())).float(),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
    )


@pytest.mark.parametrize("s", [1, 3, 16])
def test_ffn_stages_match_vllm_ops(weights, s):
    """Seam residual (exact), Gemma norm, FP32 router, top-4 and the shared +
    routed A16W4 experts, each against its native op on the kernel's own input
    to that stage, within the two paths' a-priori bounds."""
    r = _run(weights, s)
    want_x, want_res = gemma_fused_add_rmsnorm(
        r.part, r.residual, weights.post_norm, ffn.EPS
    )
    torch.testing.assert_close(r.res_out, want_res, rtol=0, atol=0)
    steps, flips, limit = _norm_flips(r.x, want_x)
    print(f"s={s}: X {steps:.0f} step apart at most, {flips} flips (<= {limit:.0f})")
    assert steps <= 1 and flips <= limit

    want_logits = _native_router(r.x, weights.router)
    err = _ratio(
        (r.logits.double() - want_logits.double()).abs(),
        _router_err(r.x, weights.router),
    )
    print(f"s={s}: logits {err:.2g} of their bound")
    assert err <= 1

    ids, wts = _native_route(r.logits, weights.router_bias)
    # on the same f32 logits each path's scores are within SCORE_ERR, and
    # AITER's weights (rcp(1 + exp2(-LOG2E x)), 3 adds, the division and
    # the product) within the kernel's own bound: twice it apart at most
    clear = _clear_choices(r.logits, weights.router_bias, SCORE_ERR)
    assert clear.any()
    got_ids, order = r.ids.sort(-1)
    want_ids, want_order = ids.sort(-1)
    assert torch.equal(got_ids[clear], want_ids[clear])
    got_w, want_w = r.wts.gather(1, order)[clear], wts.gather(1, want_order)[clear]
    logits = r.logits.gather(1, got_ids.long())[clear].double()
    ek = _weight_err(logits)
    assert _ratio((got_w.double() - want_w.double()).abs(), 2 * ek * want_w) <= 1

    shared = torch.full((s, 1), ffn.SHARED, dtype=torch.int32, device=r.x.device)
    all_ids = torch.cat([r.ids, shared], 1)
    all_wts = torch.cat([r.wts, torch.ones_like(r.wts[:, :1])], 1)
    want = _native_experts(r.x, weights, all_ids, all_wts)
    terms = torch.stack(
        [
            _native_experts(r.x, weights, all_ids[:, [k]], all_wts[:, [k]])
            for k in range(ffn.TOPK + 1)
        ]
    )
    bound, var = _combine_budget(terms.double())
    scale = terms.double().abs().sum(0)
    _assert_combine(_combine_stats(r.out, want, bound, var, scale), f"s={s}")


def test_route_ties(weights):
    """Experts 0 and 1 share a router row and bias: an exact 4th-place tie,
    which the kernel gives to the lower expert."""
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


def test_route_near_ties_match_vllm_ops(weights):
    """The selection matches AITER's on vLLM's FP32 router logits wherever the
    float64 4th/5th margin exceeds twice the most either path's scores may be
    off: the sigmoid's slope (1/4) times its logits' bound, plus SCORE_ERR."""
    r = _run(weights, 16, seed=3)
    logits = _native_router(r.x, weights.router)
    ids, _ = _native_route(logits, weights.router_bias)
    exact = r.x.double() @ weights.router.double().T
    err = _router_err(r.x, weights.router).amax(-1, keepdim=True) / 4 + SCORE_ERR
    clear = _clear_choices(exact, weights.router_bias, err.squeeze(-1))
    assert clear.float().mean() > 0.5
    assert torch.equal(r.ids.sort(-1).values[clear], ids.sort(-1).values[clear])


U = 2.0**-24


def _sigmoid_err(x):
    """``ffn._sigmoid``'s relative error on an f32 ``x``, to first order:
    LOG2E and the product (2 U times the condition number |x| (1 - s)),
    v_exp_f32's 1 ulp (2 U times 1 - s), the add (U), v_rcp_f32's 1 ulp (2 U)."""
    return (2 * x.abs() * (1 - torch.sigmoid(x)) + 5) * U


# a sigmoid + bias score's error: the sigmoid's (5 + 2 c) U s <= 5.45 U (c s <=
# 0.224), the add's half ulp of a score below 4
SCORE_ERR = 8 * U
# vLLM's FP32 router GEMM: each lane sums K / BLOCK_K products in order, then
# tl.sum reduces BLOCK_K lanes, at most BLOCK_K / 64 a thread in order and a
# log2 tree: depth <= 24 + 4 + 8 + the product at BLOCK_K 256, 3 + 32 + 11 + 1
# at 2048, so within 48 U of sum |x w|
NATIVE_ROUTER_DEPTH = 48
# both norms' f32 relative errors before the bf16 rounding, the variance's
# halved by the rsqrt: the kernel's SSQ parts (3 TASK_TRIPLES U) and its I/O
# norm bound; the Triton op's mean square (8 a thread in order, a 10-level
# tree, the products, the division, eps: 20 U), its add, sqrt, division, two
# products and 1 + w
MONO_NORM_ETA = (0.5 * (3 * ffn.TASK_TRIPLES + ffn.SEAM_TASKS + 2) + 6) * U
NORM_ETA = MONO_NORM_ETA + (0.5 * 20 + 6) * U
# Bernstein's tail for the count and mean-square checks: exp(-40) per check
TAIL = 40.0


def _router_err(x, router):
    """Both paths' logits bounds on the same X: the kernel's 16 parts within
    (ROUTER_K + 1) U of their sum |x w| (the I/O bound) summed in two rounds
    of 4 adds, and vLLM's GEMM within its depth."""
    cond = x.double().abs() @ router.double().abs().T
    return (ffn.ROUTER_K + 1 + 6 + NATIVE_ROUTER_DEPTH) * U * cond


def _weight_err(logits):
    """A route weight 2 s_k / sum s's relative error on the picked f32 logits:
    s_k within its sigmoid's error, the sum within its terms' mean error and 3
    adds, the division and the product within U each."""
    sk, ek = torch.sigmoid(logits), _sigmoid_err(logits)
    return ek + (sk * ek).sum(-1, keepdim=True) / sk.sum(-1, keepdim=True) + 5 * U


def _count_limit(mean):
    """Bernstein's bound on a sum of independent Bernoulli whose means sum to
    at most ``mean``, exceeded with probability exp(-TAIL)."""
    return mean + TAIL / 3 + (TAIL**2 / 9 + 2 * TAIL * mean) ** 0.5


def _norm_flips(got, want, eta=NORM_ETA):
    """Two bf16 roundings of f32 norms each within ``eta`` (both paths) of the
    exact one land at most one step apart, and only where a rounding boundary
    falls between them: with a uniform position in its step, at most eta |x| /
    ulp(x) a chance. Returns the largest difference in steps, the flips and
    their Bernstein limit."""
    diff = (got.double() - want.double()).abs()
    step = R.ulp(torch.maximum(got.double().abs(), want.double().abs()))
    expected = (eta * want.double().abs() / R.ulp(want.double())).sum().item()
    return (diff / step).max().item(), int((diff > 0).sum()), _count_limit(expected)


def _combine_budget(terms):
    """The two paths' sums of the same bf16 expert terms ``terms`` [k, ..., H]
    (the native op's, one expert a call: f32 w v rounded once onto a zeroed
    output) -> per element, a bound on |mono - native| and the variance of the
    difference if every rounding is uniform within half an ulp. Both round each
    term's f32 product once; their f32 GEMVs and activations differ by about
    sqrt(K) U of the dot products' scale, under 2^-8 of a bf16 ulp, and the
    rare MID element rounding apart moves a term by |w2| ulp(mid), about 2^-4
    ulp of the term (sqrt(INTER) such products), so the two roundings land at
    most one step apart. Each path then makes k - 1 bf16 adds (mono shared
    first then in top-k order, native's atomics in any order), each within half
    an ulp of a partial sum: a subset sum, at most the larger of the positive
    and the negative terms' sums."""
    k = terms.shape[0]
    mag = terms.abs()
    step = R.ulp(mag + R.ulp(mag))
    partial = torch.maximum(terms.clamp(min=0).sum(0), (-terms).clamp(min=0).sum(0))
    add = R.ulp(partial + step.sum(0) + (k - 1) * R.ulp(mag.sum(0)))
    bound = step.sum(0) + (k - 1) * add
    var = (2 * step.square().sum(0) + 2 * (k - 1) * add.square()) / 12
    return bound, var


def _combine_stats(got, want, bound, var, scale):
    """The largest difference over its bound and in ulps of ``scale``, and the
    sums for the mean-square check: squares, the variance model, and
    Bernstein's terms (each square is at most its bound squared); and the
    signed sums for the bias checks, plain and along ``want``'s signs."""
    signed = got.double() - want.double()
    diff = signed.abs()
    b2 = bound.square()
    return dict(
        ratio=_ratio(diff, bound),
        ulps=(diff / R.ulp(scale)).max().item(),
        n=diff.numel(),
        bias=signed.sum().item(),
        proj=(signed * want.double().sign()).sum().item(),
        sq=diff.square().sum().item(),
        var=var.sum().item(),
        b2var=(b2 * var).sum().item(),
        b2max=b2.max().item(),
    )


def _assert_combine(stats, label):
    """Every element within its bound; the mean square within the variance
    model's plus Bernstein's tail. Every rounding of both paths is to nearest,
    its error symmetric: the differences have mean zero, so their sum, plain
    and along the reference's signs (a scale or a directed rounding), stays
    within Bernstein's two-sided tail."""
    tail = TAIL * stats["b2max"] / 3
    slack = tail + (tail**2 + 2 * TAIL * stats["b2var"]) ** 0.5
    tail = TAIL * stats["b2max"] ** 0.5 / 3
    drift = tail + (tail**2 + 2 * TAIL * stats["var"]) ** 0.5
    bias = max(abs(stats["bias"]), abs(stats["proj"])) / drift
    print(
        f"{label}: MoE {stats['ratio']:.2f} of its bound, max {stats['ulps']:.2f} "
        f"ulp of sum |terms|, rms {(stats['sq'] / stats['var']) ** 0.5:.2f} of "
        f"the model's, bias {bias:.2f} of its tail"
    )
    assert stats["ratio"] <= 1
    assert stats["sq"] <= stats["var"] + slack
    assert bias <= 1


def _mfma_err(x, w, waves):
    """``x @ w.T``'s A16W4 error: a wave's chain of 16x16x32 MFMAs over its
    128-K steps (MFMA d: K 32 g + 8 d .. + 8 of lane groups g), each within
    33 U of its |C| + sum |a b| (test_mono_primitives), then the waves' sums in
    order (U |sum| each)."""
    xk, wk = (t.view(t.shape[0], -1, 4, 4, 8) for t in (x, w))
    p, a = (
        torch.einsum("tsgdj,nsgdj->tnsd", *ops).flatten(2).unflatten(2, (waves, -1))
        for ops in ((xk, wk), (xk.abs(), wk.abs()))
    )
    chain = 33 * U * ((p.cumsum(-1) - p).abs() + a).sum((-2, -1))
    return chain + U * p.sum(-1).cumsum(-1)[..., 1:].abs().sum(-1)


RATIOS = ("ssq", "xn", "logit", "route_w", "mid", "partial")
EXACT = ("sres", "res_out", "route_ids", "out", "pad")


def _io_weights(w):
    """``w`` with norm columns 0 .. 31 at ``1 + w = 0`` (exact zeros in X) and
    experts 0 and 1 tied (one router row and bias) and always picked."""
    w = SimpleNamespace(**vars(w))
    w.post_norm = w.post_norm.clone()
    w.post_norm[:32] = -1
    w.router = w.router.clone()
    w.router[1] = w.router[0]
    w.router_bias = w.router_bias.clone()
    w.router_bias[:2] = 2.0
    return w


@pytest.fixture(scope="module")
def io_weights(weights):
    return _io_weights(weights)


def _io_inputs(s, ranks=1, seed=0):
    """Every rank's partial and the residual; from 3 rows, row 1 is padding
    (NaN, slot -1)."""
    g = torch.Generator(device="cuda")
    parts = []
    for src in range(ranks):
        g.manual_seed(zlib.crc32(f"{seed}/{s}/{src}".encode()))
        parts.append(torch.randn(s, H, device="cuda", generator=g).bfloat16())
    g.manual_seed(zlib.crc32(f"{seed}/{s}".encode()))
    residual = torch.randn(s, H, device="cuda", generator=g).mul(4).bfloat16()
    slots = _slots(s)
    if s >= 3:
        for t in [*parts, residual]:
            t[1] = float("nan")
        slots[1] = -1
    return parts, residual, slots, slots >= 0


def _launch(runner, w, part, residual, slots):
    """One launch: its outputs, every stage's scratch, and every rank's MoE
    partial from this rank's peer region; each pair's tag the launch's."""
    s, tp = part.shape[0], runner.tp
    tag = runner.epoch[0].item() + 1
    out, res_out = runner(part, residual, w, slots)
    torch.accelerator.synchronize()
    peer = runner.peer_bufs[0]
    assert _error(peer)[0] == 0

    def region(name):
        return runner.region(s, name)

    half = ffn.ERROR_BYTES + ((tag - 1) & 1) * ffn.peer_half_bytes(tp)
    moe = peer[half + tp * ffn.MAX_TOKENS * ffn.TRIPLES * 8 :]
    rows = ffn.MAX_TOKENS * H * 4
    partials = [R.pairs(moe[src * rows :], s * H // 2, tag) for src in range(tp)]
    route = R.pairs(region("route"), s * 2 * ffn.TOPK, tag).view(s, -1)
    parts = R.pairs(region("logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS, tag)
    return SimpleNamespace(
        tag=tag,
        out=out,
        res_out=res_out,
        sres=R.words(region("sres"), s * H, torch.float32).view(s, H),
        ssq=R.pairs(region("ssq"), s * ffn.SEAM_TASKS, tag).view(torch.float32),
        xn=R.words(region("xn"), s * H, torch.bfloat16).view(s, H),
        parts=parts.view(torch.float32).view(s, ffn.EXPERTS, ffn.ROUTER_PARTS),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
        upf=region("upf").clone(),
        mid=R.words(region("mid"), ffn.slots(s) * 16 * INTER, torch.bfloat16),
        partials=torch.stack(partials).view(torch.bfloat16).view(tp, s, H),
    )


def _ratio(err, bound):
    return torch.where(err == 0, 0.0, err / bound).max().item()


def _f32_sum(terms):
    """``terms`` [..., n] summed from 0 in order, each add rounded to f32."""
    acc = torch.zeros_like(terms[..., 0])
    for i in range(terms.shape[-1]):
        acc = R.f32_rne(acc + terms[..., i])
    return acc


def _stage_errors(w, r, parts, residual, valid, rank=0, experts=ffn.Experts.ALL):
    """Every stage of launch ``r`` against float64 references of its own
    captured input -> the checks: error over its a-priori bound (``RATIOS``,
    <= 1 passes), exact (``EXACT``), the route weights' f32 ulps, coverage."""
    v, m = valid, {}
    nv = int(v.sum())
    dev = r.sres.device
    norm_w, router, bias = (t.to(dev) for t in (w.post_norm, w.router, w.router_bias))
    ulpb = lambda x: R.ulp(x, torch.bfloat16)  # noqa: E731

    # seam: the ranks' partials in rank order in f32, bf16, + residual in f32
    acc = _f32_sum(torch.stack([p.double() for p in parts], -1))
    sres = R.f32_rne(R.bf16_rne(acc) + residual.double())[v]
    m["sres"] = torch.equal(r.sres[v].double(), sres)
    m["res_out"] = torch.equal(r.res_out[v].double(), R.bf16_rne(sres))
    x = r.sres[v].double()
    sq = x.square().view(nv, ffn.SEAM_TASKS, -1).sum(-1)
    width = 3 * ffn.TASK_TRIPLES
    m["ssq"] = _ratio((r.ssq.view(-1, ffn.SEAM_TASKS)[v] - sq).abs(), width * U * sq)

    # norm on the kernel's SRES and SSQ: the 128 parts' sum, two divisions,
    # sqrt, eps, two products and 1 + w, each rounded to nearest (U; LLVM's
    # IEEE division and sqrt)
    eta = (0.5 * (ffn.SEAM_TASKS + 2) + 6) * U
    var = r.ssq.view(-1, ffn.SEAM_TASKS)[v].double().sum(-1, keepdim=True) / H
    xn = x * torch.rsqrt(var + ffn.EPS) * (1 + norm_w.double())
    bound = 0.5 * ulpb(xn.abs() * (1 + eta)) + eta * xn.abs()
    m["xn"] = _ratio((r.xn[v].double() - xn).abs(), bound)
    m["zero_columns"] = bool((r.xn[v][:, :32] == 0).all())

    # router on the kernel's X: each part a 384-term f32 dot product
    xs = r.xn[v].double()
    xp = xs.view(nv, ffn.ROUTER_PARTS, -1)
    wp = router.double().view(ffn.EXPERTS, ffn.ROUTER_PARTS, -1)
    logit = torch.einsum("tpk,epk->tep", xp, wp)
    cond = torch.einsum("tpk,epk->tep", xp.abs(), wp.abs())
    err = (r.parts[v].double() - logit).abs()
    m["logit"] = _ratio(err, (ffn.ROUTER_K + 1) * U * cond)

    # route on the kernel's parts: quarters of 4 parts, then the 4 in order
    parts_k = r.parts[v].double().view(nv, ffn.EXPERTS, 4, ffn.ROUTER_PARTS // 4)
    logits = _f32_sum(_f32_sum(parts_k))
    score = torch.sigmoid(logits) + bias.double()
    top = score.sort(dim=-1, descending=True, stable=True)
    gaps = top.values[:, : ffn.TOPK] - top.values[:, 1 : ffn.TOPK + 1]
    clear = ((gaps == 0) | (gaps > 2 * SCORE_ERR)).all(-1)
    ids = r.ids[v].long()
    m["route_ids"] = torch.equal(ids[clear], top.indices[clear, : ffn.TOPK])
    m["route_rows"] = int(clear.sum())
    m["tie_rows"] = int(((gaps == 0).any(-1) & clear).sum())
    _, wts, _ = R.route(logits, None, ffn.TOPK, "sigmoid", ffn.ROUTED_SCALE, ids)
    ek = _weight_err(logits.gather(1, ids))
    m["route_w"] = _ratio((r.wts[v].double() - wts).abs(), ek * wts)
    m["pad"] = bool(
        (r.ids[~v] == -1).all()
        and (r.partials[:, ~v] == 0).all()
        and (r.out[~v] == 0).all()
    )

    # table and up on the kernel's X and route: slot 0 the shared expert over
    # the real rows, then the picked experts ascending, columns rows ascending
    rows = v.nonzero().flatten().tolist()
    picked = sorted(set(ids.flatten().tolist()))
    table = [(ffn.SHARED, list(range(nv)))]
    table += [(e, [i for i in range(nv) if e in ids[i].tolist()]) for e in picked]
    mid = r.mid.view(ffn.slots(len(v)), 16, INTER)
    upf = R.pairs(r.upf, len(table) * ffn.UP_TILES, r.tag)
    m["upf"] = bool((upf == 1).all())
    sat = total = 0
    m["mid"] = m["partial"] = 0.0
    contrib = {}
    for u, (e, cols) in enumerate(table):
        wd = R.dequant_mxfp4(w.raw.w13[e].to(dev), w.raw.s13[e].to(dev))
        g, up = (xs[cols] @ wd.T).split(INTER, -1)
        dg, du = _mfma_err(xs[cols], wd, ffn.WAVES).split(INTER, -1)
        sat += int((g > ffn.LIMIT).sum() + (up.abs() > ffn.LIMIT).sum())
        total += 2 * g.numel()
        act = R.swiglu_oai(g, up, ffn.LIMIT, ffn.ALPHA, ffn.BETA)
        gc = g.clamp(max=ffn.LIMIT).abs() + dg
        uc = up.clamp(-ffn.LIMIT, ffn.LIMIT).abs() + du + ffn.BETA
        # |d act / d g| / |u + BETA| = s + ALPHA |g| s (1 - s), s at most at g + dg
        sh = torch.sigmoid(ffn.ALPHA * (g + dg).clamp(max=ffn.LIMIT))
        slope = torch.minimum(1 + ffn.ALPHA / 4 * gc, sh * (1 + ffn.ALPHA * gc))
        lip = dg * slope * uc + du * gc * sh
        z = ffn.ALPHA * g.clamp(max=ffn.LIMIT)
        c = z.abs() * (1 - torch.sigmoid(z))
        # ALPHA and g ALPHA (2 c U), then g s, u + BETA and their product
        d = lip + (act.abs() + lip) * (_sigmoid_err(z) + (2 * c + 3) * U)
        got = mid[u, : len(cols)].double()
        m["mid"] = max(
            m["mid"], _ratio((got - act).abs(), 0.5 * ulpb(act.abs() + d) + d)
        )
        w2 = R.dequant_mxfp4(w.raw.w2[e].to(dev), w.raw.s2[e].to(dev))
        down = got @ w2.T
        cond = _mfma_err(got, w2, 1)
        for j, i in enumerate(cols):
            contrib[(e, i)] = down[j], cond[j]
    m["saturated"] = sat / total

    # down and combine on the kernel's MID and weights: bf16(shared), then
    # bf16(tot + bf16(w v)) in top-k order; the kernel's terms are each within
    # their dot product's bound, and every rounding moves at most 1 ulp more
    for i, t in enumerate(rows):
        tot = d = torch.zeros(H, dtype=torch.float64, device=dev)
        if experts != ffn.Experts.ROUTED:
            val, err = contrib[(ffn.SHARED, i)]
            tot, d = R.bf16_rne(val), err + ulpb(val.abs() + err)
        for k in range(ffn.TOPK if experts != ffn.Experts.SHARED else 0):
            wk = r.wts[t, k].double()
            val, err = contrib[(int(ids[i, k]), i)]
            prod = wk * val
            err = wk.abs() * err
            err = err + U * (prod.abs() + err)
            p = R.bf16_rne(prod)
            d = d + err + ulpb(prod.abs() + err)
            tot = tot + p
            d = d + U * (tot.abs() + d)
            d = d + ulpb(tot.abs() + d)
            tot = R.bf16_rne(tot)
        got = r.partials[rank, t].double()
        m["partial"] = max(m["partial"], _ratio((got - tot).abs(), d))

    # final: the ranks' partials in rank order in f32 -> bf16
    acc = _f32_sum(torch.stack([p.double() for p in r.partials], -1))
    m["out"] = torch.equal(r.out[v].double(), R.bf16_rne(acc[v]))
    m["slots"] = [len(cols) for _, cols in table]
    return m


def _assert_stages(m, valid):
    for name in EXACT:
        assert m[name], name
    for name in RATIOS:
        assert m[name] <= 1, f"{name}: {m[name]:.3g} of its bound"
    nv = int(valid.sum())
    assert m["upf"] and m["zero_columns"]
    assert m["route_rows"] >= nv / 2 and m["tie_rows"] == nv
    assert 0.01 <= m["saturated"] <= 0.2, m["saturated"]
    assert bool((~valid).any()) == (len(valid) >= 3)


def _bits(r):
    return [
        t.contiguous().view(torch.uint8)
        for t in (r.out, r.res_out, r.sres, r.ssq, r.xn, r.parts, r.wts, r.partials)
    ] + [r.ids]


def _to(r, dev):
    return SimpleNamespace(
        **{k: v.to(dev) if torch.is_tensor(v) else v for k, v in vars(r).items()}
    )


@pytest.mark.parametrize(
    "s, experts",
    [(s, ffn.Experts.ALL) for s in range(1, ffn.MAX_TOKENS + 1)]
    + [(6, ffn.Experts.SHARED), (6, ffn.Experts.ROUTED)],
    ids=lambda p: p.value if isinstance(p, ffn.Experts) else str(p),
)
def test_ffn_stages_match_reference(io_weights, s, experts):
    """Every stage of one TP=1 launch against plain float64 on that stage's
    own captured input, within the bound the stage's arithmetic allows; the
    one-sided builds sum the shared or routed terms alone. A relaunch over
    poisoned SRES / X / MID writes every element the stages read, bit for bit
    the same, and no MID column past the table's. Width 1 checks on the CPU."""
    w = io_weights
    runner = _runner(w, s, experts=experts)
    parts, residual, slots, valid = _io_inputs(s)
    clean = _launch(runner, w, parts[0], residual, slots)
    for name in ("sres", "xn", "mid"):
        runner.region(s, name).fill_(0xFF)
    r = _launch(runner, w, parts[0], residual, slots)
    assert all(torch.equal(a, b) for a, b in zip(_bits(clean), _bits(r)))

    dev = "cpu" if s == 1 else "cuda"
    m = _stage_errors(
        w,
        _to(r, dev),
        [p.to(dev) for p in parts],
        residual.to(dev),
        valid.to(dev),
        experts=experts,
    )
    _assert_stages(m, valid)
    want, mid = (x.mid.view(torch.int16).view(-1, 16, INTER) for x in (clean, r))
    for u, n in enumerate(m["slots"]):
        assert torch.equal(mid[u, :n], want[u, :n]) and (mid[u, n:] == -1).all()
    assert (mid[len(m["slots"]) :] == -1).all()


IO_WIDTHS = (1, 4, 16)


def _io_rank(rank, port, results):
    from vllm.models.deepseek_v41.amd.mono.common import peer_memory

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    dev = torch.device("cuda", rank)
    w = _io_weights(_weights(rank))
    peer = peer_memory.PeerBuffer(ffn.peer_bytes(4), dist.group.WORLD, rank, 4, dev)
    peer.bytes.zero_()
    runner = FfnRunner(4, rank, peer.local, peer.addresses, dev, IO_WIDTHS)
    runner.warm(w)
    runner.peer_bufs = [peer.bytes]
    torch.accelerator.synchronize()
    dist.barrier()
    out = {}
    for s in IO_WIDTHS:
        parts, residual, slots, valid = _io_inputs(s, ranks=4)
        r = _launch(runner, w, parts[rank], residual, slots)
        out[s] = _stage_errors(w, r, parts, residual, valid, rank), valid.cpu()
    results[rank] = out
    dist.barrier()
    peer.close()
    dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
def test_tp4_stages_match_reference():
    """TP4 over XGMI, widths 1, 4 and 16: the seam sums the four ranks'
    partials and the final step their MoE partials, both in rank order, and
    every rank's partial (its own expert shard) holds the stage bounds."""
    from vllm.utils.network_utils import get_open_port

    results = mp.Manager().dict()
    mp.start_processes(_io_rank, args=(get_open_port(), results), nprocs=4)
    for rank in range(4):
        for m, valid in results[rank].values():
            _assert_stages(m, valid)


@pytest.mark.parametrize("real, pad", [(3, 5), (1, 15), (15, 1), (8, 8)])
def test_padded_rows_are_inert(weights, real, pad):
    """Rows whose slot is -1 (graph padding) hold NaN: the real rows match a
    launch without them bit for bit, the padded rows pick no expert and return
    zeros; an all-padded launch completes."""
    part, residual = _inputs(real)
    nan = torch.full((pad, H), float("nan"), device="cuda").bfloat16()
    slots = torch.tensor(
        [7 + 2 * i for i in range(real)] + [-1] * pad, dtype=torch.int64, device="cuda"
    )
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
    assert (padded.ids[real:] == -1).all() and (padded.out[real:] == 0).all()

    r = _run(weights, 4, slots=torch.full((4,), -1, dtype=torch.int64, device="cuda"))
    assert (r.ids == -1).all() and (r.out == 0).all()


def test_every_width_is_resident(weights):
    """The driver's occupancy of every production (TP4) build: one 512-thread
    CTA per CU on each of the 256 CUs."""
    dev = torch.device("cuda")
    peer = torch.zeros(ffn.peer_bytes(4), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()] * 4, dtype=torch.int64, device=dev)
    runner = FfnRunner(4, 0, peer.data_ptr(), peers, dev)
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    runner.warm(weights, compute_units=cus)
    assert sorted(runner.residency) == list(range(1, ffn.MAX_TOKENS + 1))
    assert all(n * cus >= ffn.BLOCKS for n in runner.residency.values())


def test_graph_replay_is_bit_stable(weights):
    """Widths captured once and replayed 300 rounds in alternation: each replay
    matches an eager launch bit for bit, also after the persistent slot buffer
    marks a different set of rows as padding."""
    widths = (1, 4, 16)
    runner = _runner(weights, 1, widths=widths)
    inputs = {s: _inputs(s, seed=s) for s in widths}
    slots = {s: _slots(s) for s in widths}

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
    assert _error(runner.peer_bufs[0])[0] == 0
    assert runner.epoch[0].item() > 300 * len(widths)


@pytest.mark.parametrize(
    "build, arm, final, want",
    [
        ({}, None, True, [0, 0, 0, 0]),
        ({"tp": 2}, None, False, [1, 0, 0, None]),
        ({"drop_mark": 7}, None, False, [1, 0, 0, ffn.SITE_EPOCH]),
        ({}, "inject", True, [1, 0, 0, ffn.SITE_INJECT]),
        ({}, "wrap", False, [1, 0, None, ffn.SITE_WRAP]),
    ],
    ids=["handshake", "missing_peer", "missing_epoch_mark", "late_fault", "wrap"],
)
def test_fail_stop_sites(weights, build, arm, final, want):
    """Every bounded wait and backstop sets the sticky error word [failed,
    rank, epoch, site] on every rank and the launch still completes; a clean
    step-end handshake sets nothing. A missing epoch mark does not advance the
    epoch."""
    runner = _runner(weights, 1, wait_ticks=SHORT_WAIT, **build)
    if arm == "inject":
        runner.inject.copy_(torch.tensor([1, 100_000], dtype=torch.int32))
    if arm == "wrap":
        runner.epoch[0] = ffn.EPOCH_LIMIT
    runner(*_inputs(1), weights, _slots(1), final=final)
    torch.accelerator.synchronize()
    for buf in runner.peer_bufs:
        got = _error(buf)
        assert [g if w is not None else None for g, w in zip(got, want)] == want
        # stages that start waiting together race to report their site
        assert got[3] > 0 or not got[0]
    if "drop_mark" in build:
        assert runner.epoch[0].item() == 0


def test_launches_up_to_the_epoch_limit(weights):
    """From two launches short of ``EPOCH_LIMIT`` (both seam parities, the
    16-bit tags at their top) a launch matches one from epoch 0 bit for bit;
    the launch at the limit refuses to run."""
    runner = _runner(weights, 2, wait_ticks=SHORT_WAIT)
    inputs = _inputs(2)
    want = [t.clone() for t in runner(*inputs, weights, _slots(2))]
    runner.reset()
    runner.epoch[0] = ffn.EPOCH_LIMIT - 2
    for _ in range(2):
        got = runner(*inputs, weights, _slots(2))
        torch.accelerator.synchronize()
        assert _error(runner.peer_bufs[0])[0] == 0
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    assert runner.epoch[0].item() == ffn.EPOCH_LIMIT
    runner(*inputs, weights, _slots(2))
    torch.accelerator.synchronize()
    assert _error(runner.peer_bufs[0]) == [1, 0, ffn.EPOCH_LIMIT, ffn.SITE_WRAP]


def _reset_executor(runner, monkeypatch, agree):
    """A TP1 FfnExecutor over ``runner`` that resets every 3 output steps, its
    TP agreement ``agree``."""
    from vllm.models.minimax_m3.amd.specialized_decode.runtime import executor as ex

    monkeypatch.setattr(ex, "tp_agree", agree)
    executor = object.__new__(ex.FfnExecutor)
    executor.rank, executor.runner, executor.resets = 0, runner, 0
    executor.peer = SimpleNamespace(bytes=runner.peer_bufs[0], addresses=runner.peers)
    executor._layers = dict.fromkeys(range(3, 60))
    executor.outputs, executor._since_reset = 0, 0
    executor._reset_launches = 3 * len(executor._layers)
    executor._epoch_event = None
    executor._epoch_host = torch.zeros(1, dtype=torch.int32)
    return executor


def _broken(*_):
    raise RuntimeError("injected")


@pytest.mark.parametrize(
    "where, msg, marked",
    [
        ("clear", r"\(clear\) failed on this TP rank", True),
        ("agreement", r"\(drain\) failed on this TP rank", True),
        ("peer", r"\(drain\) failed on another TP rank", False),
    ],
)
def test_failed_epoch_reset_fails_every_rank(weights, monkeypatch, where, msg, marked):
    """A reset phase failing here (a clear or an agreement that raises) or on
    another rank raises on this rank and does not resume; the failing rank
    marks every rank's error word."""
    runner = _runner(weights, 1)
    runner(*_inputs(1), weights, _slots(1))
    torch.accelerator.synchronize()
    agree = {"agreement": _broken, "peer": lambda ok: False}.get(where, lambda ok: ok)
    executor = _reset_executor(runner, monkeypatch, agree)
    if where == "clear":
        monkeypatch.setattr(runner, "reset", _broken)
    epoch = runner.epoch[0].item()
    executor.output_step()
    executor.output_step()
    with pytest.raises(RuntimeError, match=msg):
        executor.output_step()
    torch.accelerator.synchronize()
    assert executor.resets == 0 and executor._since_reset == 3
    assert runner.epoch[0].item() == epoch
    assert _error(runner.peer_bufs[0]) == ([1, 0, -1, 0] if marked else [0] * 4)


def test_forced_epoch_reset_clears_and_resumes(weights, monkeypatch):
    """The executor's step hook resets every region and the epoch once the
    launches since the last reset reach its bound; launches then resume from
    tag one with the same results, the error word kept."""
    runner = _runner(weights, 2)
    inputs = _inputs(2)
    before = runner(*inputs, weights, _slots(2))[0].clone()
    executor = _reset_executor(runner, monkeypatch, lambda ok: ok)
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


def _disassemble(compiled):
    """The linked code object's instructions, each split into words."""
    if OBJDUMP is None:
        pytest.skip("needs llvm-objdump")
    assert OBJDUMP is not None
    elf = Path(tempfile.mkdtemp()) / "kernel.elf"
    elf.write_bytes(code_object(compiled._keepalive._ir_text))
    isa = subprocess.run(
        [OBJDUMP, "-d", "--mcpu=gfx950", str(elf)], capture_output=True, text=True
    ).stdout
    ops = [line.split("//")[0].split() for line in isa.splitlines() if "\t" in line]
    return [op for op in ops if op]


def test_isa_encodes_the_handoff_contract(weights):
    """The linked TP4 S=1 kernel has the shapes ``mono/sync.py`` argues from:
    every system-scope access one aligned 64-bit pair or the 32-bit error
    word; no buffer op carries cache bits; every global access (only hand-offs
    use them) carries ``sc1``, alone for agent scope, and every agent store /
    load form ``mono/sync.py`` uses is lowered that way; the handshake's
    system release is a write-back then a drain, its acquire a drain then an
    invalidate, and (performance policy) no other L2 maintenance remains."""
    dev = torch.device("cuda")
    peer = torch.zeros(ffn.peer_bytes(4), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()] * 4, dtype=torch.int64, device=dev)
    runner = FfnRunner(4, 0, peer.data_ptr(), peers, dev, (1,))
    runner.warm(weights)
    ops = _disassemble(runner._compiled[1])

    system = [op[0] for op in ops if "sc0" in op and "sc1" in op]
    assert {o for o in system if "store" in o} == {"global_store_dwordx2"}
    assert {o for o in system if "load" in o} <= {
        "global_load_dwordx2",
        "global_load_dword",
    }
    data = [op for op in ops if op[0].startswith(("buffer_load", "buffer_store"))]
    assert data and not [op for op in data if "sc0" in op or "sc1" in op]

    accesses = [op for op in ops if op[0].startswith(("global_load", "global_store"))]
    assert accesses and all("sc1" in op for op in accesses)
    assert {
        "global_store_dword",
        "global_store_dwordx2",
        "global_store_short",
        "global_load_dword",
        "global_load_dwordx2",
        "global_load_dwordx4",
    } <= {op[0] for op in accesses if "sc0" not in op}

    for i, op in enumerate(ops):
        if op[0] == "buffer_wbl2":
            assert ops[i + 1][:2] == ["s_waitcnt", "vmcnt(0)"]
        if op[0] == "buffer_inv":
            assert ops[i - 1][0] == "s_waitcnt" and "vmcnt(0)" in ops[i - 1][1]
    kinds = {" ".join(op) for op in ops if op[0] in ("buffer_wbl2", "buffer_inv")}
    assert kinds == {"buffer_wbl2 sc0 sc1", "buffer_inv sc0 sc1"}


_STREAM = fx.Stream(None)


@flyc.kernel(name="handoff_probe", known_block_size=[64, 1, 1])
def _handoff_probe(data: fx.Int64, pair: fx.Int64, out: fx.Int64, v: fx.Int32):
    tid = fx.Int32(fx.thread_idx.x)
    at = data + fx.Int64(tid) * 64
    sync.store_dev(at, 0, [v])
    sync.store_dev(at, 2, [v, tid])
    sync.store16(at + 16, 0, fx.Float32(v).to(fx.BFloat16), "agent")
    sync.sync_release("agent")
    sync.store64(pair, tid, sync.pack(v, tid), "agent")
    got = sync.load64(pair, 63 - tid, "agent")
    sync.sync_acquire("agent")
    back = data + fx.Int64(63 - tid) * 64
    a = sync.load_dev(back, 0)
    b = sync.load_dev(back, 2, 2)
    c = sync.load_dev(back, 4, 4)
    sync.store32(out, tid, a + b[1] + c[3] + fx.Int32(got), "agent")


@flyc.jit
def _launch_handoff_probe(
    data: fx.Int64,
    pair: fx.Int64,
    out: fx.Int64,
    v: fx.Int32,
    stream: fx.Stream = _STREAM,
):
    _handoff_probe(data, pair, out, v).launch(grid=(1,), block=(64,), stream=stream)


def test_isa_orders_handoff_data_around_av_none_fences():
    """The agent ``av none`` release / acquire on gfx950: a completion wait,
    not an L2 write-back / invalidate, between the data stores and the pair
    store, and between the pair load and the data loads (the data's addresses
    do not depend on the pair, so only the acquire orders them)."""
    buf = torch.zeros(1 << 14, dtype=torch.uint8, device="cuda")
    base = buf.data_ptr()
    stream = torch.cuda.current_stream()
    ops = _disassemble(
        flyc.compile(_launch_handoff_probe, base, base + 8192, base + 12288, 7, stream)
    )
    stores = [i for i, op in enumerate(ops) if op[0].startswith("global_store")]
    loads = [i for i, op in enumerate(ops) if op[0].startswith("global_load")]
    assert all("sc1" in ops[i] and "sc0" not in ops[i] for i in stores + loads)
    *data_stores, pair_store, _out = stores
    pair_load, *data_loads = loads
    assert ops[pair_store][0] == "global_store_dwordx2"
    assert ops[pair_load][0] == "global_load_dwordx2"
    assert {ops[i][0] for i in data_loads} == {
        "global_load_dword",
        "global_load_dwordx2",
        "global_load_dwordx4",
    }

    def drained(lo, hi):
        return any(
            ops[i][0] == "s_waitcnt" and "vmcnt(0)" in ops[i][1] for i in range(lo, hi)
        )

    assert max(data_stores) < pair_store < pair_load < min(data_loads)
    assert drained(max(data_stores) + 1, pair_store)
    assert drained(pair_load + 1, min(data_loads))
    assert not [op for op in ops if op[0] in ("buffer_wbl2", "buffer_inv")]


XCD_LAUNCHES = int(os.environ.get("MINIMAX_M3_XCD_STRESS_LAUNCHES", "200"))


def test_handoffs_between_xcds():
    """The same-GPU hand-off protocol under the adversarial one-GPU stress
    (``_xcd_stress.py``): every hand-off between CTAs on different XCDs
    delivers the producer's current payload, and no wait times out."""
    _writable_flydsl_cache()
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    assert cus >= xs.BLOCKS, "every CTA must be resident"
    err, out = xs.run(XCD_LAUNCHES)
    assert err[0].item() == 0, err.tolist()
    bad = out[:, 2]
    assert bad.sum().item() == 0, out[bad.nonzero().flatten()][:8].tolist()

    xcc = out[:, 0]
    assert sorted(set(xcc.tolist())) == list(range(8))
    # every round's producer / consumer pair on two XCDs
    blocks = torch.arange(xs.BLOCKS)
    for k in range(1, 128, 2):
        assert (xcc != xcc[(blocks + xs.BLOCKS - k) % xs.BLOCKS]).all(), k


@pytest.mark.parametrize("stores, loads", [(False, True), (True, False)])
def test_plain_payload_accesses_are_caught(stores, loads):
    """The stress has teeth: with either side of the payload a plain cached
    access, consumers read stale data."""
    _writable_flydsl_cache()
    err, out = xs.run(4, stores, loads, stop_on_error=False)
    assert err[0].item() == 0, err.tolist()
    assert (out[:, 2] > 0).any()


STRESS_LAUNCHES = int(os.environ.get("MINIMAX_M3_STRESS_LAUNCHES", "120000"))
STRESS_WIDTHS = (1, 4, 16)
MAX_SKEW_CYCLES = 200_000  # ~0.1 ms at the shader clock
RESET_EVERY = 997  # launches between collective epoch resets


def _stress_rank(rank, port, results):
    from vllm.models.deepseek_v41.amd.mono.common import peer_memory
    from vllm.models.minimax_m3.amd.specialized_decode.runtime import executor as ex

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    dev = torch.device("cuda", rank)
    w = _weights()
    peer = peer_memory.PeerBuffer(ffn.peer_bytes(4), dist.group.WORLD, rank, 4, dev)
    peer.bytes.zero_()
    runner = FfnRunner(4, rank, peer.local, peer.addresses, dev, STRESS_WIDTHS)
    runner.warm(w)
    torch.accelerator.synchronize()
    dist.barrier()

    def inputs(s, which):
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{s}/{which}".encode()))
        residual = torch.randn(s, H, device=dev, generator=g).bfloat16()
        g.manual_seed(zlib.crc32(f"{s}/{which}/{rank}".encode()))
        part = torch.randn(s, H, device=dev, generator=g).bfloat16()
        return part, residual, torch.arange(s, dtype=torch.int64, device=dev)

    cases, graphs, outs, refs, keep = [], {}, {}, {}, {}
    for s in STRESS_WIDTHS:
        for which in (0, 1):
            case = (s, which)
            cases.append(case)
            # graphs read their inputs by address: they must outlive them
            keep[case] = part, residual, slots = inputs(s, which)
            final = which == 1
            refs[case] = runner(part, residual, w, slots, final)[0].clone()
            graphs[case] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[case]):
                outs[case] = runner(part, residual, w, slots, final)[0]
    torch.accelerator.synchronize()
    dist.barrier()

    # the executor's step hook, its epoch reset agreed over this group
    ex.tp_agree = lambda ok: peer_memory.tp_agree(ok, dist.group.WORLD)
    executor = object.__new__(ex.FfnExecutor)
    executor.rank, executor.runner, executor.peer = rank, runner, peer
    executor._layers = {0: None}
    executor.outputs = executor._since_reset = executor.resets = 0
    executor._reset_launches = RESET_EVERY
    executor._epoch_event = None
    executor._epoch_host = torch.zeros(1, dtype=torch.int32)

    order = torch.Generator().manual_seed(1234)
    skew = torch.Generator().manual_seed(rank)
    mismatches = torch.zeros((), dtype=torch.int64, device=dev)
    for n in range(STRESS_LAUNCHES):
        case = cases[int(torch.randint(len(cases), (1,), generator=order))]
        torch.cuda._sleep(int(torch.randint(MAX_SKEW_CYCLES, (1,), generator=skew)))
        graphs[case].replay()
        mismatches += (outs[case] != refs[case]).any()
        executor.output_step()
        if n % 4096 == 4095:
            torch.accelerator.synchronize()
    torch.accelerator.synchronize()

    outs_cpu = torch.cat([refs[c].flatten() for c in cases]).view(torch.int32).cpu()
    gathered = [torch.zeros_like(outs_cpu) for _ in range(4)]
    dist.all_gather(gathered, outs_cpu)
    results[rank] = SimpleNamespace(
        mismatches=int(mismatches.item()),
        error=_error(peer.bytes),
        epoch=int(runner.epoch[0].item()),
        resets=executor.resets,
        same_across_ranks=all(torch.equal(gathered[0], g) for g in gathered),
    )
    dist.barrier()
    graphs.clear()
    peer.close()
    dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
def test_peer_handoffs_under_rank_skew():
    """TP4 over XGMI: four ranks replay graphs of three widths, alternating two
    payload sets (half ending in the step-end fault handshake), with random
    per-rank device delays before every launch and the executor's collective
    epoch reset every ``RESET_EVERY`` launches. Every launch reproduces its
    first result bit for bit, the reduced output matches across ranks, and no
    wait times out."""
    from vllm.utils.network_utils import get_open_port

    timeout = 600 + STRESS_LAUNCHES // 1000
    results = mp.Manager().dict()
    ranks = mp.start_processes(
        _stress_rank, args=(get_open_port(), results), nprocs=4, join=False
    )
    deadline = time.monotonic() + timeout
    while not ranks.join(timeout=10):
        if time.monotonic() > deadline:
            for p in ranks.processes:
                p.kill()
            pytest.fail(f"TP4 stress ranks still running after {timeout} s")
    for rank in range(4):
        r = results[rank]
        assert r.error[0] == 0, r.error
        assert r.mismatches == 0
        assert r.same_across_ranks
        # every replayed launch since the last reset moved the epoch on
        assert r.resets == STRESS_LAUNCHES // RESET_EVERY
        eager = 0 if r.resets else 2 * len(STRESS_WIDTHS)
        assert r.epoch == STRESS_LAUNCHES % RESET_EVERY + eager
