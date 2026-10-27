# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's MonoKernels on MI355X, random weights in the checkpoint's formats
at TP8 shapes:

- ``*_matches_slow_reference``: every fused stage of K1 (one GPU) and K2 (eight
  GPUs) read from its mailbox / scratch / peer tap and checked against a
  float64 reference (``tests/models/mono_reference.py``) applied to the
  kernel's own input to that stage, with the kernel's rounding points
  reproduced. Every bound is a priori (error over its bound <= 1), derived
  from the stage's fp32 operations: U = 2^-24 a rounding, v_exp_f32 / v_rcp_f32
  / v_rsq_f32 1 ulp (2 U), the division IEEE, bf16 x bf16 products exact.
- ``*_matches_vllm_ops`` (nightly): every K1 / K2 stage and the op vLLM runs
  for it, on the kernel's own input to that stage, must both round the stage's
  float64 value c +- r: r from both paths' fp32 roundings (Hoeffding) plus each
  bf16 / MXFP4 rounding the two may take apart, so they agree but within an
  ulp of c's rounding points.
- The protocol: a CUDA graph of K2 replayed 300 times, bit-identical, and a
  stalled rank (the fault hook): every rank's launch finishes and the output
  rank's flags fail the step.
"""

import math

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from tests.models import mono_reference as R
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
K1_STEPS = 6
U = 2.0**-24


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the K3 mono kernels need gfx950 (MI355X)"
)


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _tap(region: torch.Tensor, n: int, tag: int, dtype=torch.float32):
    """The ``n`` 32-bit mailbox values of ``region`` (tags checked) as
    ``dtype``: float32 / int32 words, or bf16 pairs (2 n values)."""
    return R.pairs(region, n, tag).view(dtype)


def _dot_err(got, a, b, add=None, out=bf, n=None):
    """|got - (a @ b^T + add)| over its a-priori bound: an fp32-accumulated dot
    of exact products (n, by default K, roundings: n 2^-24 sum |a||b|; an fp32
    add of ``add``), then rounded to ``out``."""
    a, b = a.double(), b.double()
    ref = a @ b.T
    d = (a.size(-1) if n is None else n) * U * (a.abs() @ b.abs().T)
    if add is not None:
        ref = ref + add.double()
        d = d + U * ref.abs()
    return _err(got, ref, d, out)


def _err(got, ref, d, out=bf):
    """|got - ref| over its bound: ``ref`` computed to +-d, rounded to ``out``."""
    ref = ref.double()
    bound = 0.5 * R.ulp(ref.abs() + d, out) + d
    return ((got.double() - ref).abs() / bound).max().item()


def _exp2_rel(z):
    """e^z's relative error as v_exp_f32 (2 U) of RN(RN(z) RN(log2 e)) (2 U);
    exact at z = 0 (the ISA's V_EXP_F32(-0.0) = 1)."""
    return torch.where(z == 0, 0, 2 * U * (1 + z.abs()))


def _sigmoid_rel(z, div=U):
    """1 / (1 + exp2(-z log2 e)): exp2's error through 1 + e, the add, the
    division (IEEE U, v_rcp 2 U)."""
    return torch.sigmoid(-z) * _exp2_rel(z) + U + div


def _silu_err(y, dy):
    """Y sigmoid(y) on y known to +-dy, the product rounded."""
    s = torch.sigmoid(y)
    return (s * (1 + y * (1 - s))).abs() * dy + (y * s).abs() * (_sigmoid_rel(y) + U)


def _tanh_err(x, rel_x=0.0, div=U):
    """(1 - e) / (1 + e) at e = exp2(-2|x| log2 e), x known to rel_x: e's
    error through 2 / (1 + e)^2, then 1 - e, 1 + e and the division (IEEE U;
    v_rcp and a product 3 U)."""
    x = x.abs()
    e, t = torch.exp(-2 * x), torch.tanh(x)
    de = 2 * e * _exp2_rel(2 * x) / (1 + e) ** 2
    return de + (2 * U + div) * t + (1 - t**2) * x * rel_x


def _attn_res_err(prefix, blocks, w_norm, w_qk, w_out, eps, out_eps):
    """``R.attn_res``'s fp32 error in stage_slice / gate / mix: exact bf16
    products, the sources' sums of squares and qk dots 21 adds deep, rsq to
    14 U; log2 logits to +-dl; v_exp_f32 of logit - max (the shift cancels);
    the mix ns deep over its denominator (6 deep, IEEE division); its sum of
    squares 22 deep, rsq; two products."""
    src = torch.cat([blocks.double(), prefix.double()[:, None]], 1)
    ns, wq = src.size(1), w_norm.double() * w_qk.double()
    rstd = torch.rsqrt(src.square().mean(-1) + eps) / math.log(2)
    lg = (src @ wq) * rstd
    dl = 17 * U * lg.abs() + 21 * U * rstd * (src.abs() @ wq.abs())
    top = lg.amax(-1, keepdim=True)
    rel_ex = math.log(2) * (dl + U * (top - lg)) + 2 * U
    p = torch.softmax(lg * math.log(2), -1)[..., None]
    mixed = (p * src).sum(1)
    d_mix = (p * src.abs() * (rel_ex[..., None] + ns * U)).sum(1)
    d_mix = d_mix + mixed.abs() * (rel_ex.amax(-1, True) + 7 * U)
    ss = mixed.square().sum(-1, keepdim=True)
    rel_r = ((2 * mixed.abs() * d_mix).sum(-1, True) / ss + 25 * U) / 2 + 2 * U
    rw = torch.rsqrt(ss / src.size(-1) + out_eps) * w_out.double()
    return rw.abs() * d_mix + (mixed * rw).abs() * (rel_r + 2 * U)


def _kda_state_err(S, k, v, g, beta, a_log, dt_bias, lower_bound):
    """``R.kda_step``'s state in fp32 (kda_gates / kda_recur) on exact inputs:
    k l2-normed to 7 U; exp(a_log), the gate's sigmoid and the decay's exp2;
    the decayed state; its dot with k 9 deep; the delta; the update."""
    S, k, v, a_log = S.double(), k.double(), v.double(), a_log.double()
    kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    z = torch.exp(a_log)[:, None] * (g.double() + dt_bias.double())
    rel_z = _exp2_rel(a_log)[:, None] + 2 * U
    sg = torch.sigmoid(z)
    gate = lower_bound * sg
    d_sg = sg * (1 - sg) * z.abs() * rel_z + sg * _sigmoid_rel(z)
    rel_dec = abs(lower_bound) * d_sg + U * gate.abs() + _exp2_rel(gate)
    S1 = S * torch.exp(gate)[:, None]
    d1 = S1.abs() * (rel_dec[:, None] + U)
    p = torch.einsum("hvk,hk->hv", S1, kn)
    d_p = torch.einsum("hvk,hk->hv", d1 + 16 * U * S1.abs(), kn.abs())
    b = torch.sigmoid(beta.double())
    vn = (v - p) * b[:, None]
    d_vn = b[:, None] * (d_p + U * (v - p).abs())
    d_vn = d_vn + vn.abs() * (_sigmoid_rel(beta.double())[:, None] + U)
    upd = vn[:, :, None] * kn[:, None]
    d = d1 + d_vn[:, :, None] * kn.abs()[:, None] + 8 * U * upd.abs()
    return d + U * (S1 + upd).abs()


def _situ_vllm_err(g, u):
    """situ_vllm on exact g, u: 2 tanh(g / 4) (1 + tanh(g / 2)) 25 tanh(u / 25),
    u / 25 to 2 U, then 1 + t and three products (4 U)."""
    t1, t2, t3 = torch.tanh(g / 4), torch.tanh(g / 2), torch.tanh(u / 25)
    a, b = 2 * t1 * (1 + t2), 25 * t3
    return (
        2 * (1 + t2) * b.abs() * _tanh_err(g / 4)
        + 2 * (t1 * b).abs() * _tanh_err(g / 2)
        + 25 * a.abs() * _tanh_err(u / 25, 2 * U)
        + 4 * U * (a * b).abs()
    )


def _situ_aiter_err(g, u):
    """situ_aiter on exact g, u: 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25), each
    through v_rcp, u / 25 to 2 U, then three products."""
    sg, t1, t3 = torch.sigmoid(g), torch.tanh(g / 4), torch.tanh(u / 25)
    v = 100 * t1 * sg * t3
    return (
        100 * (sg * t3).abs() * _tanh_err(g / 4, div=3 * U)
        + v.abs() * (_sigmoid_rel(g, 2 * U) + 3 * U)
        + 100 * (t1 * sg).abs() * _tanh_err(u / 25, 2 * U, 3 * U)
    )


def _rank_sum(parts: torch.Tensor) -> torch.Tensor:
    """bf16 partials [ranks, ...] summed in fp32 in rank order, -> bf16."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(bf)


def _bf16_ends(x, d):
    """X's bf16 rounding (float64) and, where x +- d rounds otherwise, the
    other end (0 elsewhere)."""
    near, lo, hi = R.boundary(R.bf16_rne, x, d)
    base = R.bf16_rne(x)
    return base, torch.where(near, lo + hi - base, 0)


def _merge_max(m: dict, new: dict) -> None:
    for k, v in new.items():
        m[k] = max(m.get(k, 0.0), v)


# ------------------------------------------------- vLLM parity (nightly)
# A stage's two paths (the kernel; vLLM's op on the kernel's input to it)
# compute the same float64 value c, each with its own fp32 roundings, then round
# to the stage's format. A rounding is a zero-mean error within +-h (U |partial|
# an add, k U |y| a function good to k U); taken independent (Higham & Mary's
# probabilistic model), Hoeffding bounds their sum by LAM sqrt(sum h^2) but for
# 1e-12 of the values (a run checks ~1e7). Variances below are sum h^2 / 3,
# over both paths.
LAM = math.sqrt(2 * math.log(2e12))


def _sum_var(t2, t, depth):
    """One fp32 sum of terms t through at most ``depth`` adds (t2 = sum t^2,
    t = sum t): each add's rounding within +-U |partial|, the partials' sum of
    squares taken as depth (t2 + t^2), above its mean for terms of random sign
    (a chain of n: n / 2 t2 + n / 3 t^2; a tree: depth t2 + 2 t^2)."""
    return U * U / 3 * depth * (t2 + t * t)


def _fn_var(y, k):
    return (k * U * y) ** 2 / 3


def _r(var):
    """The Hoeffding radius of a sum of roundings of variance ``var``."""
    return LAM * (3 * var).sqrt()


def _rounds(c, r, fmt=bf):
    """[lo, hi]: what a value in c +- r rounds to in ``fmt`` (monotone)."""
    fn = R.bf16_rne if fmt == bf else R.f32_rne
    return fn(c - r), fn(c + r)


def _outside(got, c, r, fmt=bf) -> float:
    """How far ``got`` lies outside [lo, hi] of c +- r, in ulps of c."""
    lo, hi = _rounds(c.double(), r, fmt)
    g = got.double()
    out = (lo - g).clamp(min=0) + (g - hi).clamp(min=0)
    out = out / R.ulp(c.double().abs().clamp(min=2.0**-126), fmt)
    return out.max().item() if out.numel() else 0.0


def _ulps(a, b, c) -> float:
    """|a - b| in bf16 ulps of |c|, floored at its row's rms (a value from
    cancellation judged on its row's scale)."""
    a, b, c = a.double(), b.double(), c.double()
    floor = c.square().mean(-1, keepdim=True).sqrt()
    return ((a - b).abs() / R.ulp(torch.maximum(c.abs(), floor))).max().item()


def _both(m, name, got, want, c, r, fmt=bf, r_v=None) -> None:
    """Records how far the kernel's ``got`` (m[name]) and vLLM's ``want``
    (m[name_v], against r_v if given) lie outside c +- r (ulps of c, 0
    inside), and their largest gap (m[name_ulps], _ulps)."""
    for key, val, rr in (
        (name, got, r),
        (name + "_v", want, r if r_v is None else r_v),
    ):
        m[key] = max(m.get(key, 0.0), _outside(val, c, rr, fmt))
    m[name + "_ulps"] = max(m.get(name + "_ulps", 0.0), _ulps(got, want, c))


def _split_var(x, w, parts: int):
    """The extra variance of a bf16 x @ w^T whose K is split into ``parts``
    partials each rounded to bf16 and summed in bf16 (AITER's FlyDSL hgemm,
    vLLM's linear on gfx950: slice-K waves meet in a bf16 LDS tile, split-K
    CTAs add with bf16 atomics): each rounding within half an ulp (<= 2^-8 of
    the value), a partial's square taken at its mean for terms of random sign
    (t2 / parts + (t / parts)^2), a running sum's at t2 + t^2."""
    if parts == 1:
        return 0.0
    x, w = x.double(), w.double()
    t2, t = (x * x) @ (w * w).T, x @ w.T
    sq = parts * (t2 / parts + (t / parts) ** 2) + (parts - 1) * (t2 + t * t)
    return 2.0**-16 / 3 * sq


def _attn_res_var(prefix, blocks, w_norm, w_qk, w_out, eps, out_eps):
    """R.attn_res and both paths' fp32 error variance: every sum over D 63
    deep in all (the kernel's 21; vLLM's Triton kernel 42: 8192 / (64 lanes x
    4 warps) = 32 a thread, a 6-step butterfly, 4 warps), exp2 / rsq / rcp 2 U,
    first order through the softmax and both norms."""
    dd = 21 + 42
    src = torch.cat([blocks.double(), prefix.double()[:, None]], 1)
    D = src.size(-1)
    wq = w_norm.double() * w_qk.double()
    ss = src.square().sum(-1)
    rstd = torch.rsqrt(ss / D + eps)
    v_rstd = (rstd**3 / 2 / D) ** 2 * _sum_var((src**4).sum(-1), ss, dd)
    v_rstd = v_rstd + 2 * _fn_var(rstd, 4)
    dq = src @ wq
    lg = dq * rstd
    v_lg = rstd**2 * _sum_var((src * src) @ (wq * wq), dq, dd) + dq**2 * v_rstd
    v_lg = v_lg + 2 * _fn_var(lg, 2)
    p = torch.softmax(lg, -1)
    v_p = p**2 * (v_lg + (p**2 * v_lg).sum(-1, keepdim=True)) + 2 * _fn_var(p, 6)
    mixed = (p[..., None] * src).sum(1)
    v_mixed = (v_p[..., None] * src**2).sum(1)
    v_mixed = v_mixed + _sum_var((p[..., None] * src).square().sum(1), mixed, 18)
    ms = mixed.square().sum(-1, keepdim=True)
    ro = torch.rsqrt(ms / D + out_eps)
    proj = torch.einsum("md,mid->mi", mixed, src)
    v_ms = 4 * (v_p * proj**2).sum(-1, keepdim=True)
    v_ms = v_ms + _sum_var((mixed**4).sum(-1, keepdim=True), ms, dd)
    v_ro = (ro**3 / 2 / D) ** 2 * v_ms + 2 * _fn_var(ro, 4)
    w = w_out.double()
    y = mixed * ro * w
    v_y = (ro * w) ** 2 * v_mixed + (mixed * w) ** 2 * v_ro + 2 * _fn_var(y, 2)
    return y, v_y


def _dot_var(x, w, depth=None):
    """The float64 x @ w^T and both paths' fp32 error variance over its exact
    products, ``depth`` deep (default K: vLLM's GEMMs document no order)."""
    x, w = x.double(), w.double()
    c = x @ w.T
    depth = x.size(-1) if depth is None else depth
    return c, 2 * _sum_var((x * x) @ (w * w).T, c, depth)


def _l2n_var(c, v, scale=1.0):
    """The l2-normed x / sqrt(sum x^2 + 1e-6) scale, x to variance v (first
    order), plus both paths' 128-deep sum and rsq."""
    n2 = c.square().sum(-1, keepdim=True) + 1e-6
    y = c / n2.sqrt()
    y2 = y * y
    vy = (v * (1 - y2) ** 2 + y2 * ((y2 * v).sum(-1, keepdim=True) - y2 * v)) / n2
    vs = _sum_var((c**4).sum(-1, keepdim=True), n2, c.size(-1)) / (4 * n2 * n2)
    return y * scale, (vy + 2 * vs * y2) * scale**2 + 2 * _fn_var(y * scale, 4)


def _kda_step_var(S, comps, q, k, v, g, beta, a_log, dt_bias, lb, scale, vin):
    """R.kda_step on S = (value, variance) and q, k, v to variances ``vin``:
    both paths' fp32 error to first order. The delta rule's rank-1 errors
    (xi_v kvec_k, one xi a value row) stay ``comps`` so that o, a dot over k,
    keeps their correlation. -> (S', comps', (o, variance))."""

    def mv(a, b):
        return torch.einsum("hvk,hk->hv", a, b)

    Sc, Sv = S
    K = k.size(-1)
    vq, vk, vv = vin
    qn, v_qn = _l2n_var(q.double(), vq, scale)
    kn, v_kn = _l2n_var(k.double(), vk)
    z = torch.exp(a_log.double())[:, None] * (g.double() + dt_bias.double())
    s = torch.sigmoid(z)
    gate = lb * s
    # exp(a_log) and the sum to 4 U, the sigmoid, the product
    v_gate = 2 * (lb * s * (1 - s) * z) ** 2 * (4 * U) ** 2 / 3 + 2 * _fn_var(gate, 6)
    dec = torch.exp(gate)
    v_dec = dec * dec * v_gate + 2 * _fn_var(dec, 4)
    S1 = Sc * dec[:, None]
    v_S1 = Sv * (dec * dec)[:, None] + Sc * Sc * v_dec[:, None] + 2 * _fn_var(S1, 1)
    comps = [(vx, kv * dec) for vx, kv in comps]
    p = mv(S1, kn)
    v_p = mv(v_S1, kn * kn) + mv(S1 * S1, v_kn)
    v_p = v_p + 2 * _sum_var(mv(S1 * S1, kn * kn), p, K)
    b = torch.sigmoid(beta.double())
    e = v.double() - p
    vn = e * b[:, None]
    v_new = (b * b)[:, None] * (vv + v_p + 2 * _fn_var(e, 1)) + 2 * _fn_var(vn, 6)
    comps = [
        (vx, kv - b[:, None] * (kv * kn).sum(-1, keepdim=True) * kn) for vx, kv in comps
    ]
    comps.append((v_new, kn))
    S2 = S1 + vn[:, :, None] * kn[:, None]
    v_S2 = v_S1 + (vn * vn)[:, :, None] * v_kn[:, None] + 2 * _fn_var(S2, 2)
    o = mv(S2, qn)
    v_o = mv(v_S2, qn * qn) + mv(S2 * S2, v_qn)
    v_o = v_o + 2 * _sum_var(mv(S2 * S2, qn * qn), o, K)
    for vx, kv in comps:
        v_o = v_o + vx * ((kv * qn).sum(-1) ** 2)[:, None]
    return (S2, v_S2), comps, (o, v_o)


def _gated_rms_var(o, v_o, g2, w, eps):
    """The gated norm o rsqrt(mean o^2 + eps) w sigmoid(g2), o to variance
    v_o (first order), plus both paths' 128-deep sum, rsq, sigmoid and three
    products."""
    o, g2, w = o.double(), g2.double(), w.double()
    n = o.size(-1)
    ms = o.square().sum(-1, keepdim=True)
    rstd = torch.rsqrt(ms / n + eps)
    v_ms = 2 * _sum_var((o**4).sum(-1, keepdim=True), ms, n)
    v_ms = v_ms + 4 * (o * o * v_o).sum(-1, keepdim=True)
    v_rstd = (rstd**3 / 2 / n) ** 2 * v_ms
    ws = w * torch.sigmoid(g2)
    y = o * rstd * ws
    v_y = ws**2 * (rstd**2 * v_o + o**2 * v_rstd) + 2 * _fn_var(y, 12)
    return y, v_y


# ---------------------------------------------------------------- K1


def _kda_inputs(
    nacc: list[int],
    dim_first: bool,
    q: int,
    slots: int = 16,
    seed: int = 0,
    pads: int = 0,
):
    """``len(nacc)`` requests of ``q`` tokens, request r's last step having
    accepted ``nacc[r]`` of them; the last ``pads`` requests pad a graph
    (NULL_BLOCK_ID 0 state, one accepted token)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        PROJ,
        QKV,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    S = q * len(nacc)
    sl = 3 + (q - 1)
    w = {
        "prefix": _randn(g, S, HIDDEN, scale=2.0),
        "delta": _randn(g, S, HIDDEN, scale=0.5),
        "blocks": _randn(g, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(g, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_in": _randn(g, NPROJ, HIDDEN, scale=0.02),
        "w_fb": _randn(g, PROJ, HD, scale=0.08),
        "conv_w": _randn(g, QKV, 4, scale=0.4, dtype=torch.float32),
        "a_log": _randn(g, NH, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, PROJ, scale=0.5, dtype=torch.float32),
        "on_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    w["w_in"][NPROJ - 4 :].zero_()
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    pad = _randn(g, slots, NH * HD * HD + 24576, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, : NH * HD * HD].view(slots, NH, HD, HD)
    conv = _randn(g, slots, QKV * sl + 4096)[:, : QKV * sl]
    if dim_first:
        w["conv_state"] = conv.view(slots, QKV, sl)
    else:
        w["conv_state"] = conv.view(slots, sl, QKV).transpose(-1, -2)
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(-1, q)
    w["num_acc"] = torch.tensor(nacc, dtype=torch.int32, device="cuda")
    if pads:
        w["st_idx"][-pads:] = 0
        w["num_acc"][-pads:] = 1
    return w


def _strided_copy(t: torch.Tensor) -> torch.Tensor:
    """A copy with ``t``'s strides: vLLM's padded state pages stay padded."""
    out = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device)
    return out.copy_(t)


def _k1_key(w, nb: int, write_idx: int = -1):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        KdaPreBuild,
    )

    return KdaPreBuild(
        tokens=w["st_idx"].numel(),
        qlen=w["st_idx"].size(1),
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
    )


def _k1(w, nb: int, write_idx: int = -1, poison: bool = False):
    """K1 on copies of ``w``'s prefix, blocks and states: them updated, the core
    output and the scratch (its plain x rows NaN-filled first if ``poison``)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        PROJ,
        kda_pre,
        scratch_bytes,
        scratch_layout,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    S = w["st_idx"].numel()
    key = _k1_key(w, nb, write_idx)
    out = {k: _strided_copy(w[k]) for k in ("prefix", "blocks", "conv_state", "rstate")}
    out["core"] = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    out["scratch"] = torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda")
    if poison:
        o, n = scratch_layout(key)["x"]
        out["scratch"][o : o + n] = 0xFF
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    kda_pre(
        key,
        prefix=out["prefix"],
        delta=w["delta"],
        blocks=out["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_in=w["w_in"],
        w_fb=w["w_fb"],
        conv_w=w["conv_w"],
        conv_state=out["conv_state"],
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        on_w=w["on_w"],
        rstate=out["rstate"],
        st_idx=w["st_idx"],
        num_acc=w["num_acc"],
        core_out=out["core"],
        scratch=out["scratch"],
        layer=0,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    return out


def _k1_taps(w, out, nb: int, write_idx: int, tag: int = 257) -> dict:
    """K1's x rows, in_proj output and q, k (each [S, ...]) from ``out``'s
    scratch."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        scratch_layout,
    )

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    lay = scratch_layout(_k1_key(w, nb, write_idx))
    sc = out["scratch"]
    qk = _tap(sc[lay["qk"][0] :], S * NH * 2 * HD, tag).view(-1, NH, q, 2, HD)
    return {
        "x": R.words(sc[lay["x"][0] :], S * HIDDEN, bf).view(S, HIDDEN),
        "proj": _tap(sc[lay["proj"][0] :], S * NPROJ, tag).view(S, NPROJ),
        "qk": qk.permute(0, 2, 3, 1, 4).reshape(S, 2, NH, HD),
    }


def _kda_vllm_stages(w, taps, nb: int, write_idx: int) -> dict:
    """VLLM's ops for K1's stages, each on the kernel's input to it: attn_res
    (the layer's inputs), in_proj (its x), causal_conv1d_update and f_b (its
    proj), fused_recurrent_kda (its q, k; vLLM's v and g1) and
    FusedRMSNormGated (vLLM's o)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        BETA_0,
        FA_0,
        G2_0,
        HD,
        NH,
        PROJ,
        QKV,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    v = {"prefix": w["prefix"].clone(), "blocks": w["blocks"].clone()}
    v["x"] = attn_res(
        v["prefix"],
        w["delta"],
        v["blocks"],
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
        nb,
        write_idx,
        1e-5,
        1e-5,
    )
    v["proj"] = F.linear(taps["x"], w["w_in"])
    proj = taps["proj"].to(bf)
    v["conv_state"] = _strided_copy(w["conv_state"])
    v["rstate"] = _strided_copy(w["rstate"])
    qsl = torch.arange(0, S + 1, q, dtype=torch.int32, device="cuda")
    v["conv"] = causal_conv1d_update(
        proj[:, :QKV].contiguous(),
        v["conv_state"],
        w["conv_w"],
        None,
        activation="silu",
        conv_state_indices=w["st_idx"][:, 0].contiguous(),
        num_accepted_tokens=w["num_acc"],
        query_start_loc=qsl,
        max_query_len=q,
        validate_data=False,
        out=torch.empty(S, QKV, dtype=bf, device="cuda"),
    )
    g1 = F.linear(proj[:, FA_0 : FA_0 + HD], w["w_fb"])
    qk = taps["qk"].to(bf)
    o = torch.empty(1, S, NH, HD, dtype=bf, device="cuda")
    fused_recurrent_kda(
        q=qk[None, :, 0].contiguous(),
        k=qk[None, :, 1].contiguous(),
        v=rearrange(v["conv"][:, 2 * PROJ :], "n (h d) -> 1 n h d", d=HD).contiguous(),
        raw_g=rearrange(g1, "n (h d) -> 1 n h d", d=HD),
        raw_beta=proj[None, :, BETA_0 : BETA_0 + NH],
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        lower_bound=-5.0,
        initial_state=v["rstate"],
        cu_seqlens=qsl,
        ssm_state_indices=w["st_idx"],
        num_accepted_tokens=w["num_acc"],
        uniform_sequence_length=q,
        out=o,
    )
    with set_current_vllm_config(VllmConfig()):
        norm = FusedRMSNormGated(HD, activation="sigmoid").cuda()
    norm.weight.data = w["on_w"].clone()
    g2 = rearrange(proj[:, G2_0 : G2_0 + PROJ], "n (h d) -> 1 n h d", d=HD)
    v["core"] = norm.forward_cuda(o, g2).reshape(S, PROJ)
    return v


def _k1_path(w, taps, rounded: bool, g1=None) -> dict:
    """One path of K1's conv, recurrence and gated norm from the kernel's
    proj (and, ``rounded``, its q, k), as intervals c +- r per stage: "qk" [S,
    2, NH, HD], "state" slot -> [NH, HD, HD], "core" row -> [NH, HD].
    ``rounded``: the kernel's (and vLLM's Triton ops'): q, k, v, g1 and o in
    bf16; v, g1 and o never leave the kernel's LDS, so each the two paths may
    round apart is allowed its exact effect (a float64 rerun). Else AITER's
    fused decode: conv, q, k, v and o in fp32, g1 the given vLLM f_b output."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        BETA_0,
        FA_0,
        G2_0,
        HD,
        NH,
        PROJ,
        QKV,
    )

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    proj = taps["proj"].double()
    cw = w["conv_w"].double()
    lb, eps = w.get("lower_bound", -5.0), w.get("onorm_eps", 1e-5)
    rest = (w["a_log"], w["dt_bias"].view(NH, HD), lb, HD**-0.5)
    res: dict = {"qk": [], "state": {}, "core": {}}
    for rq in range(S // q):
        rows = slice(rq * q, (rq + 1) * q)
        slots = w["st_idx"][rq].tolist()
        acc = int(w["num_acc"][rq]) - 1
        old = w["conv_state"][max(slots[0], 0)].double()
        seq = torch.cat([old[:, acc : acc + 3], proj[rows, :QKV].T], 1)
        t = torch.stack([seq[:, j : j + 4] * cw for j in range(q)])
        pre = t.sum(-1)
        # 4 products rounded, 3 adds: 5 deep; SiLU's exp2, rcp, two products
        v_pre = 2 * _sum_var(t.square().sum(-1), pre, 5)
        sg = torch.sigmoid(pre)
        conv = pre * sg
        v_conv = (sg * (1 + pre * (1 - sg))) ** 2 * v_pre + 2 * _fn_var(conv, 6)
        conv, v_conv = conv.view(q, 3, NH, HD), v_conv.view(q, 3, NH, HD)
        res["qk"].append((conv[:, :2], _r(v_conv[:, :2])))
        slot0 = slots[acc]
        if slot0 <= 0:
            continue
        beta = proj[rows, BETA_0 : BETA_0 + NH]
        flips = []
        if rounded:
            qk = taps["qk"][rows].double()
            c_g1, v_g1 = _dot_var(proj[rows, FA_0 : FA_0 + HD], w["w_fb"])
            vg = []
            for i, (c, var) in enumerate(((conv[:, 2], v_conv[:, 2]), (c_g1, v_g1))):
                c = c.view(q, NH, HD)
                vg.append(R.bf16_rne(c))
                flips += [(i, end) for end in _rounds(c, _r(var.view(q, NH, HD)))]
            vin: list = [None] * q
        else:
            qk = conv[:, :2]
            vg = [conv[:, 2], g1[rows].double().view(q, NH, HD)]
            vin = [(v_conv[j, 0], v_conv[j, 1], v_conv[j, 2]) for j in range(q)]
        S0 = w["rstate"][slot0].double()
        base_S, base_o, St = [], [], S0
        for j in range(q):
            St, oj = R.kda_step(
                St, qk[j, 0], qk[j, 1], vg[0][j], vg[1][j], beta[j], *rest
            )
            base_S.append(St)
            base_o.append(oj)
        allow_S = [torch.zeros_like(S0) for _ in range(q)]
        allow_o = [torch.zeros_like(base_o[0]) for _ in range(q)]
        for i, alt in flips:
            for jj, h, x in (alt != vg[i]).nonzero().tolist():
                hs = slice(h, h + 1)
                Sb = (S0 if jj == 0 else base_S[jj - 1])[hs]
                for t_ in range(jj, q):
                    vgb = [vg[0][t_][hs].clone(), vg[1][t_][hs].clone()]
                    if t_ == jj:
                        vgb[i][0, x] = alt[jj, h, x]
                    hr = (rest[0][hs], rest[1][hs], *rest[2:])
                    Sb, ob = R.kda_step(
                        Sb, qk[t_, 0][hs], qk[t_, 1][hs], *vgb, beta[t_][hs], *hr
                    )
                    allow_S[t_][hs] += (Sb - base_S[t_][hs]).abs()
                    allow_o[t_][hs] += (ob - base_o[t_][hs]).abs()
        St, Sv = S0, torch.zeros_like(S0)
        comps: list = []
        zero = torch.zeros_like(base_o[0])
        for j in range(q):
            (St, Sv), comps, (oc, v_o) = _kda_step_var(
                (St, Sv),
                comps,
                qk[j, 0],
                qk[j, 1],
                vg[0][j],
                vg[1][j],
                beta[j],
                *rest,
                vin=vin[j] or (zero, zero, zero),
            )
            v_S = Sv + sum(vx[:, :, None] * (kv * kv)[:, None] for vx, kv in comps)
            res["state"][slots[j]] = (St, _r(v_S) + allow_S[j])
            g2 = proj[rq * q + j, G2_0 : G2_0 + PROJ].view(NH, HD)
            if not rounded:
                yc, v_y = _gated_rms_var(oc, v_o, g2, w["on_w"], eps)
                res["core"][rq * q + j] = (yc, _r(v_y))
                continue
            lo, hi = _rounds(oc, _r(v_o) + allow_o[j])
            base = R.bf16_rne(oc)
            yc, v_y = _gated_rms_var(base, zero, g2, w["on_w"], eps)
            r_y = _r(v_y)
            for end in (lo, hi):
                for h, x in (end != base).nonzero().tolist():
                    ob = base[h : h + 1].clone()
                    ob[0, x] = end[h, x]
                    yb, _ = _gated_rms_var(ob, zero[:1], g2[h : h + 1], w["on_w"], eps)
                    r_y[h] += (yb[0] - yc[h]).abs()
            res["core"][rq * q + j] = (yc, r_y)
    return res


def _k1_vllm_check(w, out, nb: int, write_idx: int) -> dict:
    """Every K1 stage and vLLM's op on the kernel's input to it against
    their shared interval (_both)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        NH,
        PROJ,
    )

    taps = _k1_taps(w, out, nb, write_idx)
    v = _kda_vllm_stages(w, taps, nb, write_idx)
    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    m: dict = {}

    def both(name, got, want, c, r, fmt=bf):
        _both(m, name, got, want, c, r, fmt)

    assert torch.equal(v["prefix"], out["prefix"]), "prefix"
    assert torch.equal(v["blocks"], out["blocks"]), "blocks"
    assert torch.equal(v["conv_state"], out["conv_state"]), "conv state"
    ares = (
        out["prefix"],
        out["blocks"][:, :nb],
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
    )
    c, var = _attn_res_var(*ares, 1e-5, 1e-5)
    both("x", taps["x"], v["x"], c, _r(var))
    c, var = _dot_var(taps["x"], w["w_in"])
    both("proj", taps["proj"], v["proj"], c, _r(var))
    path = _k1_path(w, taps, rounded=True)
    conv_qk = v["conv"][:, : 2 * PROJ].view(S, 2, NH, HD)
    for rq, (c, r) in enumerate(path["qk"]):
        rows = slice(rq * q, (rq + 1) * q)
        both("qk", taps["qk"][rows], conv_qk[rows], c, r)
    for slot, (c, r) in path["state"].items():
        both("state", out["rstate"][slot], v["rstate"][slot], c, r, torch.float32)
    for row, (c, r) in path["core"].items():
        both("core", out["core"][row].view(NH, HD), v["core"][row].view(NH, HD), c, r)
    return m


@needs_gfx950
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("nacc", [1, 3])
@pytest.mark.parametrize("write", [False, True])
def test_k1_matches_vllm_ops(dim_first: bool, nacc: int, write: bool) -> None:
    """K1 stage by stage against attn_res, in_proj, causal_conv1d_update,
    f_b, fused_recurrent_kda and FusedRMSNormGated, each on the kernel's input
    to it (one request of 8 tokens), the states included."""
    nb = 4
    # a block-write layer appends block nb (block_write_idx == prev_valid_blocks)
    write_idx = nb if write else -1
    w = _kda_inputs([nacc], dim_first, q=8)
    m = _k1_vllm_check(w, _k1(w, nb, write_idx), nb, write_idx)
    stages = ("x", "proj", "qk", "state", "core")
    assert all(m[s + p] == 0 for s in stages for p in ("", "_v")), m


def _k1_check(w, out, nb: int, write_idx: int, tag: int = 257) -> dict:
    """Every K1 stage of one launch (inputs ``w``, results ``out``) against the
    float64 reference on the kernel's own input to it -> metric a stage."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        BETA_0,
        FA_0,
        G2_0,
        HD,
        HIDDEN,
        NH,
        NPROJ,
        NVCH,
        PROJ,
        QKV,
        scratch_layout,
    )

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    lay = scratch_layout(_k1_key(w, nb, write_idx))
    sc = out["scratch"]
    m = {}

    prefix = R.bf16_rne(w["prefix"].double() + w["delta"].double())
    assert torch.equal(out["prefix"].double(), prefix), "prefix"
    blocks = w["blocks"].clone()
    if write_idx >= 0:
        blocks[:, write_idx] = out["prefix"]
    assert torch.equal(out["blocks"], blocks), "blocks"

    x = R.words(sc[lay["x"][0] :], S * HIDDEN, bf).view(S, HIDDEN)
    ares = (out["prefix"], blocks[:, :nb], w["ares_nw"], w["ares_qk"], w["in_nw"])
    m["x"] = _err(x, R.attn_res(*ares, 1e-5, 1e-5), _attn_res_err(*ares, 1e-5, 1e-5))

    proj = _tap(sc[lay["proj"][0] :], S * NPROJ, tag).view(S, NPROJ)
    assert torch.equal(proj.double(), R.bf16_rne(proj)), "proj is not bf16"
    m["proj"] = _dot_err(proj, x, w["w_in"])

    qk = _tap(sc[lay["qk"][0] :], S * NH * 2 * HD, tag).view(-1, NH, q, 2, HD)
    onrm = _tap(sc[lay["onrm"][0] :], S * NH * NVCH, tag).view(-1, NH, q, NVCH)
    written: dict[str, set[int]] = {"conv_state": set(), "rstate": set()}
    pads = 0
    for r in range(S // q):
        rows = slice(r * q, (r + 1) * q)
        slots = w["st_idx"][r].tolist()
        acc = int(w["num_acc"][r]) - 1
        old = w["conv_state"][max(slots[0], 0)].double()
        win = old[:, acc : acc + 3]
        pre, _ = R.causal_conv1d(win, proj[rows, :QKV], w["conv_w"], act=False)
        cond, _ = R.causal_conv1d(
            win.abs(), proj[rows, :QKV].abs(), w["conv_w"].abs(), act=False
        )
        # 4 products summed from 0: 4 U cond
        conv_d = _silu_err(pre, 4 * U * cond).view(q, 3, NH, HD)
        conv = R.silu(pre).view(q, 3, NH, HD)
        got = qk[r].permute(1, 2, 0, 3)
        m["qk"] = max(m.get("qk", 0), _err(got, conv[:, :2], conv_d[:, :2]))
        if slots[0] > 0:
            kept = old[:, acc + 1 : acc + 1 + old.size(1) - q]
            new = torch.cat([kept, proj[rows, :QKV].double().T], 1)
            assert torch.equal(out["conv_state"][slots[0]].double(), new), "conv state"
            written["conv_state"].add(slots[0])
        slot0 = slots[acc]
        if slot0 <= 0:
            assert not out["core"][rows].any(), "a pad request's core"
            pads += 1
            continue
        # bf16 hand-offs only in LDS (v, g1): an input within its error bound
        # of a rounding boundary may round either way; each such flip's effect
        # on the state is allowed
        fa = proj[rows, FA_0 : FA_0 + HD].double()
        g1 = fa @ w["w_fb"].double().T
        g_d = HD * U * (fa.abs() @ w["w_fb"].double().abs().T)
        hand = [
            _bf16_ends(conv[:, 2], conv_d[:, 2]),
            _bf16_ends(g1.view(q, NH, HD), g_d.view(q, NH, HD)),
        ]
        beta = proj[rows, BETA_0 : BETA_0 + NH]
        st = w["rstate"][slot0]
        rest = (w["a_log"], w["dt_bias"].view(NH, HD), -5.0, HD**-0.5)
        for j in range(q):
            qkj = (st, got[j, 0], got[j, 1])
            vg = [x[j] for x, _ in hand]
            ref = R.kda_step(*qkj, vg[0], vg[1], beta[j], *rest)[0]
            d = _kda_state_err(*qkj[::2], vg[0], vg[1], beta[j], *rest[:3])
            allow = torch.zeros_like(ref)
            for i, (_, alt) in enumerate(hand):
                for at in alt[j].nonzero():
                    flip = [x.clone() for x in vg]
                    flip[i][tuple(at)] = alt[j][tuple(at)]
                    allow += (
                        R.kda_step(*qkj, flip[0], flip[1], beta[j], *rest)[0] - ref
                    ).abs()
                m["n_flips"] = m.get("n_flips", 0) + int(alt[j].count_nonzero())
            st = out["rstate"][slots[j]]
            written["rstate"].add(slots[j])
            err = ((st.double() - ref).abs() - allow).clamp(min=0)
            err = torch.where(err > 0, err / d, 0).max().item()
            m["state"] = max(m.get("state", 0), err)
            qn = got[j, 0].double()
            qn = qn * torch.rsqrt(qn.square().sum(-1, keepdim=True) + 1e-6) * HD**-0.5
            o = torch.einsum("hvk,hk->hv", st.double(), qn)
            # q l2-normed and scaled to 9 U, the dot 9 deep
            o_d = 18 * U * torch.einsum("hvk,hk->hv", st.double().abs(), qn.abs())
            o_lo, o_hi = R.bf16_rne(o - o_d), R.bf16_rne(o + o_d)
            hi = torch.maximum(o_lo.square(), o_hi.square()).view(NH, NVCH, -1).sum(-1)
            lo = torch.minimum(o_lo.square(), o_hi.square()) * (o_lo * o_hi > 0)
            lo = lo.view(NH, NVCH, -1).sum(-1)
            ss = onrm[r, :, j].double()
            # 16 exact squares summed from 0
            out_of = torch.maximum(ss - hi, lo - ss).clamp(min=0) / (15 * U * hi)
            m["onrm"] = max(m.get("onrm", 0), out_of.max().item())
            rstd = torch.rsqrt(onrm[r, :, j].double().sum(-1, keepdim=True) / HD + 1e-5)
            g2 = proj[r * q + j, G2_0 : G2_0 + PROJ].double().view(NH, HD)
            ends = torch.stack([o_lo, o_hi]) * rstd * w["on_w"] * torch.sigmoid(g2)
            got_core = out["core"][r * q + j].view(NH, HD).double()
            core = torch.maximum(torch.minimum(got_core, ends.amax(0)), ends.amin(0))
            # rstd: 8 partials and eps to 8 U, rsq 6 U; sigmoid; three products
            top = ends.abs().amax(0)
            d = top * (9 * U + _sigmoid_rel(g2))
            err = (got_core - core).abs() / (0.5 * R.ulp(top + d) + d)
            m["core"] = max(m.get("core", 0), err.max().item())
    for name, done in written.items():
        free = torch.ones(w[name].size(0), dtype=torch.bool)
        free[list(done)] = False
        assert torch.equal(out[name][free], w[name][free]), f"{name}: a free slot"
    return {k: float(v) for k, v in m.items()} | {"pad_requests": pads}


def _k1_io(q, requests, nb, steps, write, dim_first, pad, seed=0) -> dict:
    """``steps`` verify steps, each request accepting a random count a step, each
    step from the states the kernel left: every stage of each against the slow
    reference, a rerun without the poisoned scratch bit-identical."""
    gen = torch.Generator().manual_seed(seed * 64 + q * 16 + requests)
    write_idx = nb if write else -1
    w = _kda_inputs([1] * requests, dim_first, q=q, seed=seed * 64, pads=pad)
    m: dict = {}
    for step in range(steps):
        nacc = torch.randint(1, q + 1, (requests,), generator=gen).tolist()
        x = _kda_inputs(nacc, dim_first, q, seed=seed * 64 + 1 + step, pads=pad)
        w.update({k: x[k] for k in ("prefix", "delta", "blocks", "num_acc")})
        out = _k1(w, nb, write_idx, poison=True)
        _merge_max(m, _k1_check(w, out, nb, write_idx))
        again = _k1(w, nb, write_idx)
        assert all(torch.equal(out[k], again[k]) for k in out), "rerun differs"
        w.update({k: out[k] for k in ("conv_state", "rstate")})
    return m


@needs_gfx950
@pytest.mark.parametrize(
    "q, requests, nb, steps, write, dim_first, pad",
    [
        (8, 1, 4, 1, False, True, 0),
        (8, 1, 4, 1, True, False, 0),
        (2, 3, 3, 1, True, True, 1),
        (2, 4, 1, K1_STEPS, False, False, 1),
        (3, 2, 2, K1_STEPS, False, True, 0),
        (4, 2, 7, K1_STEPS, False, False, 0),
        (8, 1, 8, K1_STEPS, False, False, 0),
    ],
)
def test_k1_matches_slow_reference(q, requests, nb, steps, write, dim_first, pad):
    """K1 stage by stage at widths 2-8 (whole requests of 2-8 tokens), 1-8
    AttnRes blocks, both conv state layouts, block-write layers and graph pad
    requests."""
    m = _k1_io(q, requests, nb, steps, write, dim_first, pad)
    assert m["pad_requests"] == pad and m["n_flips"] > 0, m
    assert all(m[k] <= 1 for k in ("x", "proj", "qk", "state", "onrm", "core")), m


# ---------------------------------------------------------------- K2 / MoE


def _situ(g, u):
    g, u = g.float(), u.float()
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return (a * (25.0 * torch.tanh(u * 0.04))).to(bf)


def _all_reduce_fp32(t: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in fp32, as vLLM's custom all-reduce."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(bf)


def _moe_raw(rank: int, device, seed: int = 0, w13_codes=(118, 123)) -> dict:
    """The MoE weights as the checkpoint holds them: MXFP4 experts as fp4x2
    bytes (gate rows, then up) and E8M0 codes."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        UP_N,
        E,
    )

    gc = torch.Generator(device=device).manual_seed(7 + 1000 * seed)  # replicated
    gr = torch.Generator(device=device).manual_seed(100 + rank + 1000 * seed)  # TP

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    w_up = _randn(gc, HIDDEN, LAT, scale=0.02)
    return {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "bias": _randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": _randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + _randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(bf),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
        "w13": u8(gr, E, 2 * RI, LAT // 2),
        "w2": u8(gr, E, LAT, RI // 2),
        "w13s": u8(gr, E, 2 * RI, LAT // 32, lo=w13_codes[0], hi=w13_codes[1]),
        "w2s": u8(gr, E, LAT, RI // 32, lo=118, hi=123),
    }


def _moe_weights(rank: int, device, raw: dict | None = None) -> dict:
    """``raw`` (``_moe_raw``'s, also kept as "raw") with the experts in aiter's
    shuffled layout, the layout the kernel reads."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import LAT, RI, E

    raw = _moe_raw(rank, device) if raw is None else raw
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    return raw | {
        "raw": raw,
        "w13": rocm_aiter_ops.shuffle_weight_a16w4(raw["w13"].view(fp4), 16, False),
        "w2": rocm_aiter_ops.shuffle_weight_a16w4(raw["w2"].view(fp4), 16, False),
        "w13s": rocm_aiter_ops.shuffle_scale_a16w4(
            raw["w13s"].view(e8).view(-1, LAT // 32), E, False
        ),
        "w2s": e8m0_shuffle(raw["w2s"].view(e8).view(-1, RI // 32)),
    }


def _moe_args(mw):
    return {
        "w_gate": mw["w_gate"],
        "bias": mw["bias"],
        "w_ld": mw["w_ld"],
        "w_sgu": mw["w_sgu"],
        "w_sd": mw["w_sd"],
        "w13": mw["w13"],
        "w13s": mw["w13s"],
        "w2": mw["w2"],
        "w2s": mw["w2s"],
        "ln_w": mw["ln_w"],
        "w_up": mw["w_up_shard"],
    }


def _k2_inputs(rank: int, device, seed: int = 0):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HIDDEN,
        PROJ,
    )

    gc = torch.Generator(device=device).manual_seed(11 + 1000 * seed)
    gr = torch.Generator(device=device).manual_seed(200 + rank + 1000 * seed)
    return {
        "core": _randn(gr, 8, PROJ),
        "w_o": _randn(gr, HIDDEN, PROJ, scale=0.02),
        "prefix": _randn(gc, 8, HIDDEN, scale=2.0),
        "blocks": _randn(gc, 8, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(gc, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
    }


def _k2_launch(runner, w, mw, prefix, out, reset: bool, nb: int = 4):
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import compile_build

    key = k2.K2Build(
        tokens=w["core"].size(0), nblocks=nb + int(reset), reset=reset, **runner.fault
    )
    if not torch.cuda.is_current_stream_capturing():
        # every rank compiles before any launches, as the engine does: a rank
        # still compiling would outlast the in-kernel waits (WAIT_SECONDS)
        compile_build(key)
        dist.barrier()
    k2.k2_launch(
        key,
        core=w["core"],
        w_o=w["w_o"],
        prefix=prefix,
        blocks=w["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        out=out,
        layer=3,
        **_moe_args(mw),
        **runner.launch_args(),
        **runner.peer_args(),
    )
    return key


def _k2_x(runner, key) -> torch.Tensor:
    """The MLP AttnRes rows K2 left in scratch: the MoE's input."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN

    lay = k2.scratch_layout(key)["xrow"][0]
    s = key.tokens
    return runner.scratch[lay : lay + s * HIDDEN * 2].view(bf).view(s, HIDDEN)


def _assert_ranks_agree(out: torch.Tensor, tp_size: int) -> None:
    outs = [torch.empty_like(out) for _ in range(tp_size)]
    dist.all_gather(outs, out)
    assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"


def _assert_replays_bit_identical(runner, w, mw) -> None:
    """A graph of one step (the epoch bump captured with it), replayed: a
    block-write launch writes its prefix, so every replay must match the
    first."""
    prefix = torch.empty_like(w["prefix"])
    out = torch.empty_like(w["prefix"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.step_begin()
        _k2_launch(runner, w, mw, prefix, out, reset=True)
    graph.replay()
    torch.accelerator.synchronize()
    want_prefix, want_out = prefix.clone(), out.clone()
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(prefix, want_prefix) and torch.equal(out, want_out)
    assert not runner.health.any() and not runner.flags.any()


def _assert_mxfp4_equal(got, codes, v, d) -> float:
    """MXFP4 output (E2M1 values [..., K], E8M0 codes [..., K / 32]) of aiter's
    inline quant (``e8m0_roundup6``, RNE) against the reference values ``v``
    known to +-``d``: exact outside the boundary set, one of its two ends
    inside, the set under 1 % -> its fraction."""
    vb, db = v.unflatten(-1, (-1, 32)), d.unflatten(-1, (-1, 32))
    bc, clo, chi = R.boundary(R.e8m0_roundup6, vb.abs().amax(-1), db.amax(-1))
    codes = codes.long()
    assert torch.equal(codes[~bc], clo[~bc]), "codes differ"
    assert ((codes == clo) | (codes == chi))[bc].all(), "codes off by > 1"
    inv = torch.exp2(127 - codes.double())[..., None]
    bv, vlo, vhi = R.boundary(R.to_e2m1_rne, vb * inv, db * inv)
    gb = got.unflatten(-1, (-1, 32))
    assert torch.equal(gb[~bv], vlo[~bv]), "values differ"
    assert ((gb == vlo) | (gb == vhi))[bv].all(), "values off by > 1 step"
    frac = (bv.sum() + bc.sum() * 32).item() / v.numel()
    assert frac < 0.01, f"boundary set {frac:.2%} of the elements"
    return frac


def _peer(runner, s: int, name: str) -> torch.Tensor:
    """This rank's peer region ``name`` of the current step's parity."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import PEER_HEAD

    half = k2.peer_bytes(s) // 2
    at = PEER_HEAD + int(runner.epoch.item()) % 2 * half + k2.peer_layout(s)[name][0]
    return runner.peers.bytes[at:]


def _k2_check(runner, key, w, mw, prefix, out, seed: int) -> dict:
    """Every K2 stage of one launch (inputs ``w``, ``mw``; the prefix ``w``'s
    updated to ``prefix``; the output ``out``) against the float64 reference on
    the kernel's own input to it -> metric a stage."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        TOPK,
        TP,
        UP_N,
        E,
    )

    s, rank, device = key.tokens, runner.rank, w["core"].device
    tag = (int(runner.epoch.item()) % (1 << 22)) * 256 + 3 + 1
    lay = k2.scratch_layout(key)
    sc = runner.scratch

    def scratch(name, n, dtype=torch.float32):
        return _tap(sc[lay[name][0] :], n, tag, dtype)

    def peer(name, n, dtype=torch.float32):
        return _tap(_peer(runner, s, name), n, tag, dtype)

    def plain(name, n):
        return R.words(sc[lay[name][0] :], n, bf)

    m: dict = {}
    ar = peer("ar", TP * s * HIDDEN // 2, bf).view(TP, s, HIDDEN)
    for src in range(TP):
        x = _k2_inputs(src, device, seed)
        _merge_max(m, {"ar": _dot_err(ar[src], x["core"][:s], x["w_o"])})
    delta = _rank_sum(ar)
    want = delta if key.reset else R.bf16_rne(w["prefix"].double() + delta.double())
    assert torch.equal(prefix.double(), want.double()), "prefix"

    x = plain("xrow", s * HIDDEN).view(s, HIDDEN)
    blocks = w["blocks"][:, : key.nblocks]
    ares = (w["ares_nw"], w["ares_qk"], w["in_nw"], key.eps, key.out_eps)
    m["xrow"] = _err(
        x, R.attn_res(prefix, blocks, *ares), _attn_res_err(prefix, blocks, *ares)
    )

    rlog = peer("rlog", TP * s * (E // TP)).view(TP, s, E // TP)
    logits = rlog.transpose(0, 1).reshape(s, E)
    # a wave's 8 chained bf16 MFMAs (33 U each, test_mono_primitives), the 8
    # waves' sum, the 4 K parts'
    n = 33 * 8 + 8 + 3
    m["rlog"] = _dot_err(logits, x, mw["w_gate"], out=torch.float32, n=n)
    lat = peer("lat", s * LAT // 2, bf).view(s, LAT)
    m["lat"] = _dot_err(lat, x, mw["w_ld"])
    sgu = scratch("sgu", s * SI, bf).view(s, 2 * SI)
    m["sgu"] = _dot_err(sgu, x, mw["w_sgu"])

    route = scratch("route", s * 2 * TOPK, torch.int32).view(s, 2, TOPK)
    ids, wts = route[:, 0].long(), route[:, 1].view(torch.float32)
    ref_ids, _, margin = R.route(logits, mw["bias"], TOPK, "sigmoid", 1.0)
    sig = torch.sigmoid(logits.double())
    # sigmoid by v_rcp, + bias rounded
    rel_sig = _sigmoid_rel(logits.double(), 2 * U)
    sel = sig + mw["bias"].double()
    top, at = sel.sort(-1, descending=True)
    sel_d = (sig * rel_sig + U * sel.abs()).gather(1, at)[:, : TOPK + 1]
    gap = top[:, :TOPK] - top[:, 1 : TOPK + 1]
    clear = (gap == 0) | (gap > sel_d[:, :-1] + sel_d[:, 1:])
    rows = clear.all(-1)
    assert torch.equal(ids[rows], ref_ids[rows]), "top-k ids or their order"
    assert torch.equal(ids.sort(-1)[0][clear[:, -1]], ref_ids.sort(-1)[0][clear[:, -1]])
    m["n_route_rows"] = int(rows.sum())
    m["n_boundary_ties"] = int(((margin == 0) & rows).sum())
    ws, rel_ws = sig.gather(1, ids), rel_sig.gather(1, ids)
    # summed serially in rank order, an IEEE 1 / sum, a product
    tot = ws.sum(-1, keepdim=True)
    d_tot = (ws * rel_ws).sum(-1, True) + U * ws.cumsum(-1)[:, 1:].sum(-1, True)
    w_ref = ws / tot
    m["route_w"] = (
        (wts - w_ref).abs() / (w_ref * (rel_ws + d_tot / tot + 2 * U))
    ).max()

    h = scratch("h", s * SI // 2, bf).view(s, SI)
    g, u = sgu[:, :SI].double(), sgu[:, SI:].double()
    m["h"] = _err(h, R.situ(g, u), _situ_vllm_err(g, u))

    lat_codes = R.e8m0_roundup6(lat.double().abs().unflatten(-1, (-1, 32)).amax(-1))
    scale = R.e8m0(lat_codes).repeat_interleave(32, -1)
    latv = R.to_e2m1_rne(lat.double() / scale) * scale
    m["n_zero_lat_groups"] = int((lat_codes == 0).sum())
    union, slot = ids.unique(return_inverse=True)
    m["n_union"] = union.numel()
    raw = mw["raw"]
    w13 = R.dequant_mxfp4(raw["w13"][union], raw["w13s"][union])
    gu = torch.einsum("urk,tk->tur", w13, latv)
    # a wave chains 14 FP4 MFMAs of K 128, each within 129 U of its |C| + sum |a b|
    # (test_mono_primitives); then the two K halves are added
    w13, latv = w13.unflatten(-1, (-1, 128)), latv.unflatten(-1, (-1, 128))
    steps = torch.einsum("urnk,tnk->turn", w13, latv).unflatten(-1, (2, -1))
    chain = steps.cumsum(-1)[..., :-1].abs().sum((-2, -1))
    cond = torch.einsum("urnk,tnk->tur", w13.abs(), latv.abs())
    d_gu = 129 * U * (cond + chain) + U * gu.abs()
    del w13, steps
    picks = torch.arange(s, device=device)[:, None], slot
    g, u = gu[picks].requires_grad_().split(RI, -1)
    v = R.situ(g, u)
    dg, du = torch.autograd.grad(v.sum(), (g, u))
    g, u, v = g.detach(), u.detach(), v.detach()
    d_g, d_u = d_gu[picks].split(RI, -1)
    d = dg.abs() * d_g + du.abs() * d_u
    d = d + _situ_aiter_err(g, u)
    sat = (g.abs() > 4 * 2.65) | (u.abs() > 25 * 2.65)
    m["situ_saturated"] = sat.double().mean().item()
    inter = scratch("inter", s * TOPK * 48, torch.int32).view(s * TOPK, 48)
    inter = inter.view(torch.uint8)
    isc = scratch("isc", s * TOPK * RI // 32, torch.int32).view(s * TOPK, -1)
    vals = R.dequant_mxfp4(inter, torch.full_like(isc, 127))
    m["inter_boundary"] = _assert_mxfp4_equal(
        vals, isc, v.flatten(0, 1), d.flatten(0, 1)
    )
    m["n_zero_inter_groups"] = int((isc == 0).sum())

    iv = R.dequant_mxfp4(inter, isc).view(s, TOPK, RI)
    w2 = R.dequant_mxfp4(raw["w2"][union], raw["w2s"][union])
    y = e = torch.zeros(s, LAT, dtype=torch.float64, device=device)
    for k in range(TOPK):
        wt = wts[:, k : k + 1].double()
        wk = w2[slot[:, k]].unflatten(-1, (-1, 128))
        ik = iv[:, k].unflatten(-1, (-1, 128))
        steps = torch.einsum("tlnr,tnr->tln", wk, ik)
        acc = steps.sum(-1)
        cond = torch.einsum("tlnr,tnr->tl", wk.abs(), ik.abs())
        # 3 chained FP4 MFMAs, as gu's
        chain = steps.cumsum(-1)[..., :-1].abs().sum(-1)
        dk = wt * 129 * U * (cond + chain) + U * (wt * acc).abs()
        c = R.bf16_rne(wt * acc)
        ec = dk + R.ulp(c.abs() + dk, bf)
        y = R.bf16_rne(y + c)
        e = e + ec + R.ulp(y.abs() + e + ec, bf)
    del w2
    latr = peer("latr", TP * s * LAT // 2, bf).view(TP, s, LAT)
    m["latr"] = ((latr[rank].double() - y).abs() / e).max()

    ln = plain("ln", s * LAT).view(s, LAT)
    ln_ref = R.rmsnorm(_rank_sum(latr), mw["ln_w"], key.ln_eps)
    # sum of squares of exact squares 34 deep, / LAT and eps: 37 U; rsq; two products
    m["ln"] = _err(ln, ln_ref, (37 / 2 + 4) * U * ln_ref.abs())
    shp = plain("shp", s * UP_N).view(s, UP_N)
    final = peer("final", TP * s * HIDDEN // 2, bf).view(TP, s, HIDDEN)
    mine = torch.zeros(HIDDEN, dtype=torch.bool, device=device)
    mine[rank * UP_N : (rank + 1) * UP_N] = True
    m["up"] = _dot_err(final[rank][:, mine], ln, mw["w_up_shard"], add=shp)
    m["sdown"] = max(
        _dot_err(shp, h, mw["w_sd"][mine]),
        _dot_err(final[rank][:, ~mine], h, mw["w_sd"][~mine]),
    )
    assert torch.equal(out, _rank_sum(final)), "out"
    return {k: float(v) for k, v in m.items()}


def _k2_case(runner, w, mw, nb, reset, seed, poison=True):
    """One K2 step of ``w``'s rows (scratch's plain rows NaN-filled first if
    ``poison``) -> (metrics, prefix, out, MLP input rows)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2

    s = w["core"].size(0)
    runner.step_begin()
    if poison:
        lay = k2.scratch_layout(k2.K2Build(tokens=s, nblocks=nb + int(reset)))
        for name in ("xrow", "ln", "shp"):
            runner.scratch[lay[name][0] : sum(lay[name])] = 0xFF
    prefix = w["prefix"].clone()
    out = torch.empty_like(prefix)
    key = _k2_launch(runner, w, mw, prefix, out, reset, nb)
    torch.accelerator.synchronize()
    m = _k2_check(runner, key, w, mw, prefix, out, seed)
    x = _k2_x(runner, key).clone()
    _assert_ranks_agree(out, TP_SIZE)
    dist.barrier()
    return m, prefix, out, x


TP_SIZE = 8
# (rows, AttnRes blocks, block-write layer)
K2_CASES = [(8, 4, False), (8, 4, True), (2, 1, False), (4, 7, True)]
TIES = ((5, 21), (84, 100))  # equal router rows 16 apart: equal logits
W13_CODES = (118, 123)


def _k2_io(runner, seed: int = 0) -> dict:
    """K2 stage by stage over ``K2_CASES``, then (8 rows) every token on the same
    16 experts with exact routing ties (the 16th pick between two equal
    experts, two equal experts inside the top 16) and every token on 16 of its
    own (the largest union); a rerun without the poisoned scratch
    bit-identical. A zero 32-column group of the latent down and of one picked
    expert's up rows makes all-zero MXFP4 groups."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import RI, TOPK

    device = runner.scratch.device
    raw = _moe_raw(runner.rank, device, seed, w13_codes=W13_CODES)
    raw["w_ld"][:32] = 0
    raw["w13"][TIES[1][0], RI : RI + 32] = 0
    mw = _moe_weights(runner.rank, device, raw)
    w = _k2_inputs(runner.rank, device, seed)
    m: dict = {}
    for s, nb, reset in K2_CASES:
        ws = w | {k: w[k][:s] for k in ("core", "prefix", "blocks")}
        got, prefix, out, x = _k2_case(runner, ws, mw, nb, reset, seed)
        _merge_max(m, got)
        if (s, nb, reset) == K2_CASES[0]:
            _, prefix2, out2, x2 = _k2_case(runner, ws, mw, nb, reset, seed, False)
            assert torch.equal(prefix, prefix2) and torch.equal(out, out2)
            assert torch.equal(x, x2), "rerun differs"
            own = 0.2 * x.float() / x.float().norm(dim=-1, keepdim=True)

    tie = mw | {"w_gate": mw["w_gate"].clone(), "bias": mw["bias"].clone()}
    tie["bias"][84:98] += 10.0
    for lo, hi in TIES:
        tie["w_gate"][hi] = tie["w_gate"][lo]
    tie["bias"][[5, 21]] = 5.0
    tie["bias"][100] = tie["bias"][84]
    disjoint = mw | {"w_gate": mw["w_gate"].clone()}
    disjoint["bias"] = torch.zeros_like(mw["bias"])
    disjoint["w_gate"][: 8 * TOPK] = own.repeat_interleave(TOPK, dim=0).to(bf)
    dist.broadcast(disjoint["w_gate"], src=0)
    for case, union in ((tie, TOPK), (disjoint, 8 * TOPK)):
        got, *_ = _k2_case(runner, w, case, 4, False, seed)
        _merge_max(m, got)
        assert got["n_union"] == union, got
    return m


def _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    return device


def _k2_taps(runner, key, tag: int) -> dict:
    """K2's stage outputs of one launch (``tag``'s mailbox pairs, plain rows)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        SI,
        TOPK,
        TP,
        UP_N,
        E,
    )

    s = key.tokens
    lay = k2.scratch_layout(key)
    sc = runner.scratch

    def scratch(name, n, dtype=torch.float32):
        return _tap(sc[lay[name][0] :], n, tag, dtype)

    def peer(name, n, dtype=torch.float32):
        return _tap(_peer(runner, s, name), n, tag, dtype)

    def plain(name, n):
        return R.words(sc[lay[name][0] :], n, bf)

    route = scratch("route", s * 2 * TOPK, torch.int32).view(s, 2, TOPK)
    rlog = peer("rlog", TP * s * (E // TP)).view(TP, s, E // TP)
    return {
        "ar": peer("ar", TP * s * HIDDEN // 2, bf).view(TP, s, HIDDEN),
        "x": plain("xrow", s * HIDDEN).view(s, HIDDEN),
        "rlog": rlog.transpose(0, 1).reshape(s, E),
        "lat": peer("lat", s * LAT // 2, bf).view(s, LAT),
        "sgu": scratch("sgu", s * SI, bf).view(s, 2 * SI),
        "h": scratch("h", s * SI // 2, bf).view(s, SI),
        "ids": route[:, 0].long(),
        "wts": route[:, 1].contiguous().view(torch.float32),
        "latr": peer("latr", TP * s * LAT // 2, bf).view(TP, s, LAT),
        "ln": plain("ln", s * LAT).view(s, LAT),
        "shp": plain("shp", s * UP_N).view(s, UP_N),
        "final": peer("final", TP * s * HIDDEN // 2, bf).view(TP, s, HIDDEN),
    }


def _rank_sum_var(parts):
    """The ranks' bf16 parts summed in fp32 in any order (both paths):
    (float64 sum, variance)."""
    p = parts.double()
    c = p.sum(0)
    return c, 2 * _sum_var(p.square().sum(0), c, p.size(0))


def _mx_alts(v, d) -> tuple:
    """MXFP4 (AITER's inline rule: e8m0_roundup6 codes, E2M1 RNE) of rows v
    [..., K] known to +-d: (quantized v, [(row, delta over K)]), a delta for
    each value whose other neighbour, or group whose other code, v +- d may
    round to."""
    vb, db = v.unflatten(-1, (-1, 32)), d.unflatten(-1, (-1, 32))
    code = R.e8m0_roundup6(vb.abs().amax(-1))
    sc = R.e8m0(code)[..., None]
    q = R.to_e2m1_rne(vb / sc) * sc
    alts = []
    for end in R.boundary(R.to_e2m1_rne, vb / sc, db / sc)[1:]:
        for *row, g, j in ((end * sc) != q).nonzero().tolist():
            delta = torch.zeros_like(v[tuple(row)])
            delta[g * 32 + j] = end[(*row, g, j)] * sc[(*row, g, 0)] - q[(*row, g, j)]
            alts.append((tuple(row), delta))
    for end in R.boundary(R.e8m0_roundup6, vb.abs().amax(-1), db.amax(-1))[1:]:
        for *row, g in (end != code).nonzero().tolist():
            s2 = R.e8m0(end[(*row, g)])
            delta = torch.zeros_like(v[tuple(row)])
            delta[g * 32 : g * 32 + 32] = (
                R.to_e2m1_rne(vb[(*row, g)] / s2) * s2 - q[(*row, g)]
            )
            alts.append((tuple(row), delta))
    return q.flatten(-2), alts


def _k2_vllm_check(runner, key, w, mw, prefix, out, ops, tag: int) -> dict:
    """Every K2 stage of one launch and vLLM's op (``ops``) on the kernel's
    input to it against their shared interval, as _k1_vllm_check; both paths'
    top-16 picks among those the scores' intervals allow, the kernel's then
    feeding both (rows with a choice counted)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        TOPK,
        UP_N,
    )

    t = _k2_taps(runner, key, tag)
    s, rank = key.tokens, runner.rank
    m: dict = {}

    def both(name, got, want, c, r, fmt=bf, r_v=None):
        _both(m, name, got, want, c, r, fmt, r_v)

    parts = ops.get("parts", {})

    def gemm(name, got, x, wt, op):
        c, v = _dot_var(x, wt)
        r_v = _r(v + _split_var(x, wt, parts.get(name, 1)))
        both(name, got, op(x), c, _r(v), r_v=r_v)

    gemm("ar", t["ar"][rank], w["core"], w["w_o"], ops["partial"])
    # the attention all-reduce (+ the prefix), on the kernel's partials
    c, v = _rank_sum_var(t["ar"])
    lo, hi = _rounds(c, _r(v))
    delta = ops["all_reduce"](t["ar"][rank].clone())
    want = delta.double()
    if not key.reset:
        lo, hi = (R.bf16_rne(w["prefix"].double() + e) for e in (lo, hi))
        want = R.bf16_rne(w["prefix"].double() + want)
    for name, g in (("prefix", prefix.double()), ("prefix_v", want)):
        m[name] = ((lo - g).clamp(min=0) + (g - hi).clamp(min=0)).max().item()
    blocks = w["blocks"][:, : key.nblocks]
    ares = (w["ares_nw"], w["ares_qk"], w["in_nw"], key.eps, key.out_eps)
    c, v = _attn_res_var(prefix, blocks, *ares)
    both("x", t["x"], ops["attn_res"](prefix, blocks), c, _r(v))
    x = t["x"].double()
    # router logits: the kernel's 8 chained bf16 MFMAs a wave (33 U each), 8
    # waves, 4 K parts (275 deep); vLLM's gate up to three K-deep fp32 sums
    # (its bf16x3 split) and a product rounding
    wg = mw["w_gate"].double()
    c = x @ wg.T
    t2 = (x * x) @ (wg * wg).T
    v_lg = _sum_var(t2, c, 275) + _sum_var(t2, c, 3 * x.size(-1) + 1)
    logits_v = ops["router"](t["x"])
    both("rlog", t["rlog"], logits_v, c, _r(v_lg), torch.float32)
    sig = torch.sigmoid(c)
    sel = sig + mw["bias"].double()
    r_sel = _r(
        (sig * (1 - sig)) ** 2 * v_lg + 2 * _fn_var(sig, 8) + 2 * _fn_var(sel, 1)
    )
    # an expert must be picked when fewer than TOPK others can score above it,
    # may be when fewer than TOPK others surely do
    lo_s, hi_s = sel - r_sel, sel + r_sel
    eye = torch.eye(sel.size(-1), dtype=torch.bool, device=sel.device)
    above = (hi_s[:, None, :] >= lo_s[:, :, None]) & ~eye
    must = above.sum(-1) < TOPK
    may = ((lo_s[:, None, :] > hi_s[:, :, None]) & ~eye).sum(-1) < TOPK
    ids_v, wts_v = ops["select"](t["x"], logits_v)
    for name, got in (("route_ids", t["ids"]), ("route_ids_v", ids_v)):
        picked = torch.zeros_like(must).scatter_(1, got.long(), True)
        bad = (picked.sum(-1) != TOPK) | (picked < must).any(-1)
        m[name] = int((bad | (picked > may).any(-1)).sum())
    m["forced_rows"] = int((must != may).any(-1).sum())
    ids = t["ids"]
    # weights: the picked sigmoids over their 16-deep sum, first order
    ws = sig.gather(1, ids)
    v_ws = ((sig * (1 - sig)) ** 2 * v_lg).gather(1, ids) + 2 * _fn_var(ws, 8)
    tot = ws.sum(-1, keepdim=True)
    c_w = ws / tot
    v_w = (v_ws + c_w**2 * v_ws.sum(-1, keepdim=True)) / tot**2
    v_w = (
        v_w
        + 2 * _sum_var(ws.square().sum(-1, keepdim=True), tot, TOPK) * c_w**2 / tot**2
    )
    v_w = v_w + 2 * _fn_var(c_w, 2)
    m["route_w"] = _outside(t["wts"], c_w, _r(v_w), torch.float32)
    same = (ids.sort(-1).values == ids_v.sort(-1).values).all(-1)
    wv = torch.empty_like(wts_v)
    wv.scatter_(1, ids.argsort(-1), wts_v.gather(1, ids_v.argsort(-1)))
    m["route_w_v"] = _outside(wv[same], c_w[same], _r(v_w)[same], torch.float32)
    for name, wt in (("lat", mw["w_ld"]), ("sgu", mw["w_sgu"])):
        gemm(name, t[name], t["x"], wt, ops[name])
    g, u = t["sgu"][:, :SI].double(), t["sgu"][:, SI:].double()
    both(
        "h",
        t["h"],
        ops["situ"](t["sgu"]),
        R.situ(g, u),
        math.sqrt(2) * LAM * _situ_vllm_err(g, u),
    )

    # routed experts on the kernel's latent and picks. The MXFP4 latent is one
    # rule's in both (exact); gate / up are FP4 MFMA chains; SiTUv2 in fp32
    # (AITER's epilogue); MXFP4 again,
    # each value or code the two may round apart allowed its exact effect;
    # down (3 MFMAs); bf16(w acc) a pick; their bf16 sum in top-k order (the
    # kernel) or any order (AITER's atomic adds): 15 roundings of half an ulp
    # of the partial, any order's partial within the picks' negative and
    # positive sums (and their roundings, 15 2^-9); at most the roundings' sum
    # (below Hoeffding's LAM sqrt(15) at 15)
    raw = mw["raw"]
    lat = t["lat"].double()
    codes = R.e8m0_roundup6(lat.abs().unflatten(-1, (-1, 32)).amax(-1))
    sc = R.e8m0(codes).repeat_interleave(32, -1)
    latq = R.to_e2m1_rne(lat / sc).unflatten(-1, (-1, 128)) * sc.unflatten(
        -1, (-1, 128)
    )
    union, slot = ids.unique(return_inverse=True)
    w13 = R.dequant_mxfp4(raw["w13"][union], raw["w13s"][union])
    w2 = R.dequant_mxfp4(raw["w2"][union], raw["w2s"][union])
    wts = t["wts"].double()
    cs, allow = [], torch.zeros(s, LAT, dtype=torch.float64, device=lat.device)
    m["mx_alts"] = 0

    def mfma_var(wm, a):
        """An FP4 MFMA chain over 128-blocks: an fp32 sum, each product through
        its MFMA's 128 adds (within 129 U, test_mono_primitives) and the chain's
        (both paths)."""
        c = torch.einsum("srnk,snk->sr", wm, a)
        t2 = torch.einsum("srnk,snk->sr", wm * wm, a * a)
        return c, 2 * _sum_var(t2, c, wm.size(-2) + 129)

    for k in range(TOPK):
        e = slot[:, k]
        gu, v_gu = mfma_var(w13[e].unflatten(-1, (-1, 128)), latq)
        g, u = gu.split(RI, -1)
        # SiTU = 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25), its partials
        tg, sg, tu = torch.tanh(g / 4), torch.sigmoid(g), torch.tanh(u / 25)
        dg = 25 * tu * ((1 - tg**2) * sg + 4 * tg * sg * (1 - sg))
        du = 4 * tg * sg * (1 - tu**2)
        vg_, vu_ = v_gu.split(RI, -1)
        d = _r(dg**2 * vg_ + du**2 * vu_) + math.sqrt(2) * LAM * _situ_aiter_err(g, u)
        inter, alts = _mx_alts(R.situ(g, u), d)
        w2e = w2[e]
        acc, v_acc = mfma_var(
            w2e.unflatten(-1, (-1, 128)), inter.unflatten(-1, (-1, 128))
        )
        wk = wts[:, k : k + 1]
        for (row,), dlt in alts:
            allow[row] += (wk[row] * (w2e[row] @ dlt)).abs()
        m["mx_alts"] += len(alts)
        lo, hi = _rounds(wk * acc, wk * _r(v_acc))
        cs.append(R.bf16_rne(wk * acc))
        allow += torch.maximum(hi - cs[-1], cs[-1] - lo)
    y, run = cs[0].clone(), cs[0].clone()
    h1, h2 = torch.zeros_like(allow), torch.zeros_like(allow)
    for ck in cs[1:]:
        y, run = y + ck, run + ck
        h = R.ulp(run.abs().clamp(min=2.0**-126)) / 2
        h1, h2 = h1 + h, h2 + h * h
        run = R.bf16_rne(run)
    picks = torch.stack(cs)
    big = torch.maximum(picks.clamp(min=0).sum(0), (-picks).clamp(min=0).sum(0))
    big = (big * (1 + TOPK * 2.0**-9)).clamp(min=2.0**-126)
    any_order = (TOPK - 1) * R.ulp(big) / 2
    y_v = ops["experts"](t["lat"], t["wts"], ids)
    m["latr"] = _outside(t["latr"][rank], y, torch.minimum(LAM * h2.sqrt(), h1) + allow)
    m["latr_v"] = _outside(y_v, y, any_order + allow)
    m["latr_ulps"] = _ulps(t["latr"][rank], y_v, y)
    # the latent all-reduce (both on the kernel's y), RMSNorm: the sum's
    # roundings the two may take apart allowed their exact effect; the
    # kernel's sum of squares 34 deep, vLLM's LAT
    c, v = _rank_sum_var(t["latr"])
    lo, hi = _rounds(c, _r(v))
    base = R.bf16_rne(c)
    ln = R.rmsnorm(base, mw["ln_w"], key.ln_eps)
    r_ln = torch.zeros_like(ln)
    for end in (lo, hi):
        for row, col in (end != base).nonzero().tolist():
            b2 = base[row].clone()
            b2[col] = end[row, col]
            r_ln[row] += (R.rmsnorm(b2, mw["ln_w"], key.ln_eps) - ln[row]).abs()
    ms = base.square().sum(-1, keepdim=True)
    rstd = torch.rsqrt(ms / LAT + key.ln_eps)
    v_ms = _sum_var((base**4).sum(-1, keepdim=True), ms, 34 + LAT)
    v_ln = (base * mw["ln_w"].double()) ** 2 * (rstd**3 / 2 / LAT) ** 2 * v_ms
    v_ln = v_ln + 2 * _fn_var(ln, 4)
    lnorm_v = ops["lnorm"](ops["all_reduce"](t["latr"][rank].clone()))
    both("ln", t["ln"], lnorm_v, ln, _r(v_ln) + r_ln)
    # up (this rank's columns, onto the shared partial) and the shared down
    mine = torch.zeros(HIDDEN, dtype=torch.bool, device=lat.device)
    mine[rank * UP_N : (rank + 1) * UP_N] = True
    c, v = _dot_var(t["ln"], mw["w_up_shard"])
    c = c + t["shp"].double()
    both(
        "up",
        t["final"][rank][:, mine],
        ops["up"](t["ln"], t["shp"]),
        c,
        _r(v + 2 * _fn_var(c, 1)),
    )
    c, v = _dot_var(t["h"], mw["w_sd"])
    sd = ops["sdown"](t["h"])
    both("sdown", t["shp"], sd[:, mine], c[:, mine], _r(v[:, mine]))
    both(
        "sdown", t["final"][rank][:, ~mine], sd[:, ~mine], c[:, ~mine], _r(v[:, ~mine])
    )
    c, v = _rank_sum_var(t["final"])
    both("out", out, ops["all_reduce"](t["final"][rank].clone()), c, _r(v))
    return m


def _vllm_ops(w, mw, key) -> dict:
    """The ops vLLM's latent-MoE layer runs, as plain calls on these weights."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import SI, TOPK

    def select(x, logits):
        sig = torch.sigmoid(logits)
        ids = torch.topk(sig + mw["bias"], TOPK, dim=-1, sorted=True).indices
        wts = sig.gather(-1, ids)
        return ids, wts / wts.sum(-1, keepdim=True)

    def experts(lat, wts, ids):
        return fused_moe(
            lat,
            mw["w13"],
            mw["w2"],
            wts.contiguous(),
            ids.to(torch.int32).contiguous(),
            None,
            ActivationType.Situv2,
            QuantType.per_1x32,
            False,
            mw["w13s"],
            mw["w2s"],
            None,
            None,
            dtype=bf,
            gate_mode="separated",
            beta=4.0,
            linear_beta=25.0,
        )

    def lnorm(lat):
        lat = lat.float()
        rstd = torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + key.ln_eps)
        return (lat * rstd * mw["ln_w"]).to(bf)

    def ares(prefix, blocks):
        nb = blocks.size(1)
        b = blocks.contiguous()
        return attn_res(
            prefix.clone(),
            None,
            b,
            w["ares_nw"],
            w["ares_qk"],
            w["in_nw"],
            nb,
            -1,
            key.eps,
            key.out_eps,
        )

    return {
        "partial": lambda core: F.linear(core, w["w_o"]),
        "all_reduce": _all_reduce_fp32,
        "attn_res": ares,
        "router": lambda x: torch.mm(x.float(), mw["w_gate"].float().t()),
        "select": select,
        "lat": lambda x: F.linear(x, mw["w_ld"]),
        "sgu": lambda x: F.linear(x, mw["w_sgu"]),
        "situ": lambda sgu: _situ(sgu[:, :SI], sgu[:, SI:]),
        "experts": experts,
        "lnorm": lnorm,
        "up": lambda ln, shp: (
            shp.float() + ln.float() @ mw["w_up_shard"].float().t()
        ).to(bf),
        "sdown": lambda h: F.linear(h, mw["w_sd"]),
    }


K2_STAGES = tuple(
    f"{s}{p}"
    for s in ("ar", "prefix", "x", "rlog", "route_ids", "route_w", "lat", "sgu")
    + ("h", "latr")
    + ("ln", "up", "sdown", "out")
    for p in ("", "_v")
)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    w = _k2_inputs(rank, device)
    for reset in (True, False):
        runner.step_begin()
        prefix = w["prefix"].clone()
        out = torch.empty_like(prefix)
        key = _k2_launch(runner, w, mw, prefix, out, reset)
        torch.accelerator.synchronize()
        tag = (int(runner.epoch.item()) % (1 << 22)) * 256 + 3 + 1
        ops = _vllm_ops(w, mw, key)
        m = _k2_vllm_check(runner, key, w, mw, prefix, out, ops, tag)
        assert all(m[k] == 0 for k in K2_STAGES), m
        _assert_ranks_agree(out, tp_size)

    _assert_replays_bit_identical(runner, w, mw)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_io_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    m = _k2_io(runner)
    assert m["n_boundary_ties"] > 0 and m["n_zero_lat_groups"] > 0, m
    assert m["n_zero_inter_groups"] > 0 and 0.01 <= m["situ_saturated"] <= 0.2, m
    stages = ("ar", "xrow", "rlog", "lat", "sgu", "route_w", "h", "latr", "ln", "up")
    assert all(m[k] <= 1 for k in (*stages, "sdown")), m
    _assert_replays_bit_identical(
        runner, _k2_inputs(rank, device), _moe_weights(rank, device)
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_slow_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """K2 stage by stage at 2, 4 and 8 rows, 1-8 AttnRes blocks, block-write
    layers, exact routing ties and the largest expert union; then a CUDA graph
    of it replayed."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_io_worker)


STALLED = 3


@ray.remote(num_gpus=1, max_calls=1)
def _fault_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import WAIT_TICKS
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # rank STALLED's CTA 0 stalls past every wait's bound in the epoch 2 launch
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_layer": 3,
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    mw = _moe_weights(rank, device)
    w = _k2_inputs(rank, device)
    out = torch.empty(8, HIDDEN, dtype=bf, device=device)
    runner.step_begin()  # epoch 1: no stall
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    assert not runner.flags.any()

    runner.step_begin()  # epoch 2: rank STALLED stalls; every launch ends
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), k2.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_rank_fails_the_step_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _fault_worker)


@needs_gfx950
def test_warmup_compiles_every_k3_build_without_launching() -> None:
    """The JIT warm-up compiles every build Kimi-K3 (93 layers, AttnRes blocks
    of 12) launches at 7 speculative tokens before capture, without running
    them: K1 and K2 at 1-8 blocks, and the 7 block-write KDA layers' K1 (writing
    block nb) and K2 resets. A launch on the null arguments would fault."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        CONV_W,
        KdaPreBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.layer import K2Build
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import compile_build

    k1 = {"tokens": 8, "qlen": 8, "delta": True, "state_len": CONV_W - 1 + 7}
    for nb in range(1, 9):
        compile_build(KdaPreBuild(nblocks=nb, **k1))
        compile_build(K2Build(tokens=8, nblocks=nb))
        if nb < 8:
            compile_build(KdaPreBuild(nblocks=nb, write_idx=nb, **k1))
            compile_build(K2Build(tokens=8, nblocks=nb + 1, reset=True))
    torch.accelerator.synchronize()
