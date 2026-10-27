# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel I/O of the DeepSeek-V4-Pro mono decode layer at TP8 on CDNA4: each
fused stage's output, read from the scratch, peer or runner buffer it hands off
through, against a float64 reference (``mono_reference``) of the stage applied
to the kernels' own captured input, so a failure names the stage. Nothing is
compared with another kernel.

Weights are random in the checkpoint's formats; the references read them raw,
before vLLM's shuffles. Every bound is a priori: the quantization rules and data
movement exact, each rounding of the kernels' fp32 chains at 2^-24 (v_exp /
v_rcp / v_rsq / v_log at 1 ulp, CDNA4 ISA 12.8), the scaled FP8 / FP4 MFMAs as
``_gemv_err`` models them. Plain outputs are poisoned before the checked launch
and must match an unpoisoned one bit for bit; every case then replays in one
graph, bit for bit."""

import pytest
import ray
import torch

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from . import mono_reference as R
from .test_dsv4_mono_numerics import (
    HC,
    HEAD_DIM,
    HIDDEN,
    MODEL,
    NOPE,
    O_RANK,
    Q_RANK,
    SWA_BLOCK,
    TP,
    _decode_step,
    _ExactMoE,
    _fp8_block,
    _on_cdna4,
    _random_moe,
)

LOG2E = 1.4426950408889634
U = 2.0**-24
NEG = -3.4028234663852886e38
REPLAYS = 100

# (compress ratio, requests, tokens a request, pad rows, key table, MoE, fold):
# CSA / HCA, one and two hops (from 42 rows), hash and bias routing, the
# first layer's settled residual, and adversarial inputs ("adv")
CASES = (
    (4, 1, 7, 0, "csa", "bias", True),
    (128, 8, 7, 0, "long", "hash", True),
    (128, 2, 7, 0, "dep", "bias", False),
    (4, 4, 7, 0, "csa", "hash", True),
    (4, 41, 1, 0, "csa", "bias", True),
    (128, 42, 1, 0, "dep", "hash", True),
    (4, 8, 7, 8, "csa", "bias", True),
    (4, 2, 7, 0, "csa", "adv", True),
)
# adversarial MoE: five experts always picked, then an exact tie for the
# sixth (one 64-expert chunk: the lower id wins), router rows past
# softplus' linear branch, gate scales that saturate the SiLU clamps
TOP5, TIE, OUTLIERS = (5, 133, 261, 300, 377), (70, 100), (200, 201, 202, 203)


def _apriori(name: str, ratio: torch.Tensor, ctx) -> None:
    """``ratio``: each element's error over its a-priori bound."""
    worst = ratio.max().item() if ratio.numel() else 0.0
    assert worst <= 1.0, (ctx, name, worst)


def _half_ulp(x):
    return 0.5 * R.ulp(x.double().abs())


def _gemv_err(a, b, waves=8):
    """The error bound of a scaled-MFMA GEMV a @ b^T (f64 operands, as
    dequantized) whose K steps of 128 are split over ``waves`` chains, then
    summed: gfx950 drops a product 2^-14 below its 32-block's largest (32 2^-14
    blockmax a block), plus each MFMA's fp32 accumulation, (128 + 1) 2^-24 (|C|
    + |A| |B|) as test_mono_primitives' ``_mfma_error`` bounds it, along a
    chain, and the chains' sum."""
    steps = -(-a.shape[-1] // (128 * waves))
    return 2.0**-9 * R.blockmax_cond(a, b) + (129 * steps + waves) * U * (
        a.abs() @ b.abs().T
    )


def _deq128(w, s):
    """An FP8 weight [N, K] with fp32 128 x 128 block scales -> f64."""
    return w.double() * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)


def _mm(a, b):
    """(a @ b^T, |a| @ |b|^T) in f64."""
    return a @ b.T, a.abs() @ b.abs().T


def _f32(v: float) -> float:
    return torch.tensor(v, dtype=torch.float32).item()


def _rule_ceil(amax):
    """AITER's MXFP8 code: the ceil code of RN(max(amax, 1e-10) RN(1 / 448))."""
    a = amax.double().clamp(min=_f32(1e-10))
    return R.ceil_code(R.f32_rne(a * _f32(1 / 448))).clamp(1, 254)


def _rule_record(amax):
    """The KV record's UE8M0 a 64 dims: the ceil code of RN(max(amax, 1e-4) / 448)."""
    return R.ceil_code(R.f32_rne(amax.double().clamp(min=_f32(1e-4)) / 448))


def _e4m3(q8):
    return q8.contiguous().view(torch.float8_e4m3fn).double()


def _q8_equal(name, q8, codes, x, rule, ctx):
    """MXFP8 bytes [.., K] and codes [.., K / 32] of exactly known values."""
    ref_q, ref_c = R.mx_quantize(x, rule)
    assert torch.equal(codes.long(), ref_c), (ctx, name, "codes")
    assert torch.equal(_e4m3(q8), ref_q), (ctx, name, "values")


def _q8_within(name, q8, codes, x, d, rule, ctx, sharp=True):
    """MXFP8 of values x known to +-d (``R.assert_quant_equal`` where d may
    span more than one step): each code and value between the rule's at the
    two ends, so exact wherever they agree (``sharp``: the rest under 1 %)."""
    xb, db = x.unflatten(-1, (-1, 32)), d.unflatten(-1, (-1, 32))
    amax, dmax = xb.abs().amax(-1), db.amax(-1)
    lo_c, hi_c = rule((amax - dmax).clamp(min=0)), rule(amax + dmax)
    codes = codes.long()
    assert ((codes >= lo_c) & (codes <= hi_c)).all(), (ctx, name, "codes")
    inv = torch.exp2(127 - codes.double())[..., None]
    lo, hi = R.to_e4m3_rne((xb - db) * inv), R.to_e4m3_rne((xb + db) * inv)
    got = _e4m3(q8).unflatten(-1, (-1, 32))
    assert ((got >= lo) & (got <= hi)).all(), (ctx, name, "values")
    frac = ((lo != hi).sum() + 32 * (lo_c != hi_c).sum()).item() / x.numel()
    assert not sharp or frac < 0.01, (ctx, name, "boundary set", frac)


# ---------------------------------------------------------------- inputs


def _pad(s: dict, pad: int) -> dict:
    """The step with ``pad`` rows past its tokens (slot -1: no keys)."""
    if not pad:
        return s
    out = dict(s, M=s["M"] + pad)
    for name, fill in (
        ("positions", 0),
        ("slot_mapping", -1),
        ("token_to_req", 0),
        ("swa_indices", -1),
        ("swa_lens", 0),
        ("topk", -1),
    ):
        t = s[name]
        extra = torch.full((pad, *t.shape[1:]), fill, dtype=t.dtype, device=t.device)
        out[name] = torch.cat([t, extra]).contiguous()
    return out


def _key_table(s: dict, ratio: int, glob: bool, width: int, keys: int):
    """K1's key table [M, keys] and lengths [2 M] from the step's metadata."""
    M = s["M"]
    kt = torch.full((M, keys), -1, dtype=torch.int64)
    klen = torch.zeros(2 * M, dtype=torch.int64)
    top, e = s["topk"].cpu().long(), s["comp_entries"]
    bt = s["comp_block_table"].cpu().long()
    pos, slot = s["positions"].cpu(), s["slot_mapping"].cpu()
    for t in range(M):
        if slot[t] < 0:
            continue
        nswa = int(s["swa_lens"][t])
        n = min((int(pos[t]) + 1) // ratio, width)
        local = top[t, :n]
        if glob:
            cs = local
        else:
            r = int(s["token_to_req"][t])
            cs = bt[r, local.clamp(min=0) // e] * e + local.clamp(min=0) % e
        kt[t, :n] = torch.where(local >= 0, cs | (1 << 31), -1)
        kt[t, n : n + nswa] = s["swa_indices"][t, :nswa].cpu().long()
        klen[t], klen[M + t] = n + nswa, n
    kt = torch.where(kt >= 1 << 31, kt - (1 << 32), kt)
    return kt.int(), klen.int()


def _record_spans(cache, block, slots):
    """Byte offsets of the records at ``slots`` in the cache's storage: each
    576 data bytes and its 8 scale bytes."""
    slots = slots.long().cpu()
    page = cache.storage_offset() + slots // block * cache.stride(0)
    d0 = page + slots % block * 576
    s0 = page + block * 576 + slots % block * 8
    return [(int(a), 576) for a in d0] + [(int(a), 8) for a in s0]


def _storage(cache):
    flat = torch.empty(0, dtype=torch.uint8, device=cache.device)
    return flat.set_(cache.untyped_storage())


# ---------------------------------------------------------------- the layer


def _attn_weights(seed, rank, H, G, device):
    g = torch.Generator().manual_seed(1000 + seed)
    gs = torch.Generator().manual_seed(1001 + 16 * seed + rank)
    w = {}
    w["wqkv"], w["wqkv_s"] = _fp8_block(Q_RANK + HEAD_DIM, HIDDEN, g, device)
    w["wq_b"], w["wq_b_s"] = _fp8_block(H * HEAD_DIM, Q_RANK, gs, device)
    w["wo_a"], w["wo_a_s"] = _fp8_block(G * O_RANK, H // G * HEAD_DIM, gs, device)
    w["wo_b"], w["wo_b_s"] = _fp8_block(HIDDEN, G * O_RANK, gs, device)
    w["q_norm"] = (1 + torch.randn(Q_RANK, generator=g) * 0.1).to(torch.bfloat16)
    w["kv_norm"] = (1 + torch.randn(HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16)
    w["sink"] = torch.randn(H, generator=gs)
    mix = (2 + HC) * HC
    for sub in ("attn", "ffn"):
        w[f"{sub}_fn"] = torch.randn(mix, HC * HIDDEN, generator=g) * 0.01
        w[f"{sub}_scale"] = (torch.randn(3, generator=g) * 0.5).abs() + 0.5
        w[f"{sub}_base"] = torch.randn(mix, generator=g) * 0.1
        norm = 1 + torch.randn(HIDDEN, generator=g) * 0.1
        w[f"{sub}_norm"] = norm.to(torch.bfloat16)
    return {k: v.to(device) for k, v in w.items()}


def _moe(vc, kind: str, seed: int, rank: int, device):
    """A random MoE layer (layer 0's hash-routed, 3's bias-routed; "adv": layer
    4's with the adversarial router and scales) -> (layer, raw weights)."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE
    from vllm.utils.torch_utils import set_default_torch_dtype

    layer = {"hash": 0, "bias": 3, "adv": 4}[kind]
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), device:
        moe = DeepseekV4MoE(vc, prefix=f"model.layers.{layer}.ffn")
    seed = 100 * seed + 10 * layer
    _random_moe(
        moe,
        torch.Generator(device=device).manual_seed(2001 + rank + seed),
        torch.Generator(device=device).manual_seed(2000 + seed),
        vc.model_config.hf_config.vocab_size,
        # 2^-11 .. 2^-7: the adversarial shared gate stays mostly inside its clamp
        shared_codes=(116, 121),
    )
    if kind == "adv":
        sh, e = moe.shared_experts, moe.experts.routed_experts
        sh.gate_up_proj.weight_scale_inv.data.view(torch.uint8).add_(4)
        e.w13_weight_scale.data.view(torch.uint8).add_(2)
        gw, bias = moe.gate.weight.data, moe.gate.e_score_correction_bias.data
        gw[TIE[1]] = gw[TIE[0]]
        gw[list(OUTLIERS)] *= 15
        bias[list(TOP5)] = 30.0
        bias[list(TIE)] = 12.0
    raw = _ExactMoE(moe)
    process_weights_after_loading(moe, vc.model_config, device)
    raw.shared(moe.shared_experts)
    moe.shared_experts.prepare_gateup_preshuffle()
    return moe, raw


def _seam_ref(x, res, post, comb):
    """The fold of a seam's slice and its a-priori error: fp32 products
    summed in order, then one bf16 rounding (``R.mhc_post``)."""
    terms = (post.double()[..., None] * x.double()[:, None]).abs()
    terms = terms + torch.einsum(
        "mij,mid->mjd", comb.double().abs(), res.double().abs()
    )
    return R.mhc_post(x, res, post, comb), 5 * U * terms


def _check_fold(name, got, x, res, post, comb, ctx):
    ref, eps = _seam_ref(x, res, post, comb)
    _apriori(name, (got.double() - ref).abs() / (_half_ulp(got) + eps), ctx)


def _fn_as_read(fn):
    """The mix weights as the seam applies them: bf16 hi + bf16(fn - hi)."""
    hi = R.bf16_rne(fn)
    return hi + R.bf16_rne(fn.double() - hi)


def _gate_err(x, fn, scale, base, eps, hc_eps, iters):
    """First-order bound on ``stage_gate``'s (pre, post, comb): sum |d out / d e|
    |e| over its roundings e, along its op order -- 2^-24 an fp32 op or the
    IEEE division, 2^-23 a 1-ulp v_exp / v_rcp / v_rsq or an op with a rounded
    f32 constant; the mix projection 58 2^-24 sum |x fn| (4 MFMAs 16x16x4 at
    (4 + 1) roundings, 8 waves, 14 + 16 slices), the sum of squares 37 2^-24
    (a pair, 6 lanes, 14 + 16 slices)."""
    errs = []

    def r(v, rel, extra=0):
        z = torch.zeros_like(v, requires_grad=True)
        errs.append((z, rel * v.detach().abs() + extra))
        return v + z

    def sigmoid(z):
        e = r(torch.exp2(r(z * -LOG2E, 2 * U)), 2 * U)
        return r(1 / r(1 + e, U), U)

    def rsum(c, dim):
        a, b, c_, d = c.unbind(dim)
        return r(r(a + b, U) + r(c_ + d, U), U).unsqueeze(dim)

    x, fn, base = x.double(), fn.double(), base.double()
    with torch.enable_grad():
        mix = r(x @ fn.T, 0, 58 * U * (x.abs() @ fn.abs().T))
        sq = r(x.square().sum(-1, True), 37 * U)
        rs = r(torch.rsqrt(r(sq / x.shape[-1] + eps, 2 * U)), 2 * U)
        v = r(mix * rs, U)
        sl = (slice(0, HC), slice(HC, 2 * HC), slice(2 * HC, None))
        g = [r(v[:, s] * scale[i].item() + base[s], U) for i, s in enumerate(sl)]
        pre = r(sigmoid(g[0]) + hc_eps, 2 * U)
        post = sigmoid(g[1]) * 2
        cv = g[2].view(-1, HC, HC)
        cv = r(cv - cv.detach().amax(-1, True), U)
        cv = r(torch.exp2(r(cv * LOG2E, 2 * U)), 2 * U)
        cv = r(cv * r(1 / rsum(cv, -1), 2 * U) + hc_eps, 2 * U)
        cv = r(cv * r(1 / r(rsum(cv, -2) + hc_eps, 2 * U), 2 * U), U)
        for _ in range(iters - 1):
            for dim in (-1, -2):
                cv = r(cv * r(1 / r(rsum(cv, dim) + hc_eps, 2 * U), 2 * U), U)
        outs = torch.cat([pre, post, cv.flatten(1)], 1)
        bound = torch.zeros_like(outs)
        zs = [z for z, _ in errs]
        for j in range(outs.shape[1]):
            grads = torch.autograd.grad(
                outs[:, j].sum(), zs, retain_graph=True, allow_unused=True
            )
            for gz, (_, e) in zip(grads, errs):
                if gz is not None:
                    bound[:, j] += (gz.abs() * e).flatten(1).sum(1)
    return bound.split((HC, HC, HC * HC), 1)


def _check_mixes(tag, w, sub, res, post, comb, pre, eps, ctx):
    from vllm.models.deepseek_v4.amd.mono.dims import HC_EPS, HC_SINKHORN

    M = res.shape[0]
    x, fn = res.reshape(M, -1), _fn_as_read(w[f"{sub}_fn"])
    scale, base = w[f"{sub}_scale"], w[f"{sub}_base"]
    refs = R.mhc_mixes(x, fn, scale, base, HC, eps, HC_EPS, HC_EPS, 2.0, HC_SINKHORN)
    bounds = _gate_err(x, fn, scale, base, eps, HC_EPS, HC_SINKHORN)
    names = ("pre", "post", "comb")
    for name, got, ref, b in zip(names, (pre, post, comb), refs, bounds):
        err = (got.double().reshape(M, -1) - ref.reshape(M, -1)).abs()
        _apriori(name, err / b, (ctx, tag))


def _bf16_known(x, d):
    """RN-to-bf16 of values x known to +-d: (RN(x), how far another bf16 value
    in [x - d, x + d] lies from it; 0 where the rounding is certain)."""
    lo, mid, hi = R.bf16_rne(x - d), R.bf16_rne(x), R.bf16_rne(x + d)
    return mid, torch.maximum(hi - mid, mid - lo)


def _check_norm(name, got, res, pre, w, eps, serial, ctx):
    """The seam's norm: the collapse (fp32, then bf16) RMSNormed -> bf16."""
    x = R.mhc_collapse(res, pre)
    dx = 4 * U * (pre.double().abs()[..., None] * res.double().abs()).sum(1)
    xs, dxs = _bf16_known(x, dx)
    assert (dxs > 0).double().mean() < 0.01, (ctx, name, "ambiguous")
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    eta = (0.5 * serial + 6) * U + (2 * dx * x.abs()).sum(-1, True) / x.square().sum(
        -1, True
    ).clamp(min=1e-300)
    ref = xs * rs * w.double()
    bound = _half_ulp(got) + eta * ref.abs() + (rs * w.double()).abs() * dxs
    _apriori(name, (got.double() - ref).abs() / bound, ctx)


def _gather_peer(region, rank, M, tag, two_hop, ctx):
    """Every rank's bf16 partial [tp, M, hidden] of a peer all-reduce region
    (PART [src][token][pair], then SUM), gathered across the ranks; the
    region's tags checked; the reduced sum (rank order, fp32 -> bf16), whose
    two-hop copies at every rank must equal it."""
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v4.amd.mono.stages.allreduce import region_bytes

    n = region_bytes(M, _dims()) // 8
    pairs = region[: n * 8].view(torch.int32).view(TP + 1, M, HIDDEN // 2, 2)
    owner = torch.arange(HIDDEN // 2, device=pairs.device) // (HIDDEN // 2 // TP)
    mine = owner == rank
    if two_hop:
        assert (pairs[:TP, :, mine, 1] == tag).all(), (ctx, "PART tags")
        assert (pairs[TP, :, ~mine, 1] == tag).all(), (ctx, "SUM tags")
    else:
        assert (pairs[:TP, ..., 1] == tag).all(), (ctx, "PART tags")
    every = get_tp_group().all_gather(pairs[None].contiguous(), dim=0)
    if two_hop:
        part = every[owner, :TP, :, torch.arange(HIDDEN // 2), 0]
        part = part.permute(1, 2, 0)
    else:
        part = every[rank, :TP, :, :, 0]
    part = part.contiguous().view(torch.bfloat16).view(TP, M, HIDDEN)
    acc = part[0].float()
    for src in range(1, TP):
        acc = acc + part[src].float()
    total = acc.to(torch.bfloat16)
    if two_hop:
        got = pairs[TP, :, :, 0].contiguous().view(torch.bfloat16).view(M, -1, 2)
        assert torch.equal(got[:, ~mine], total.view(M, -1, 2)[:, ~mine]), ctx
    return part, total


def _dims(width: int = 0):
    from vllm.models.deepseek_v4.amd.mono.dims import Dims

    return Dims(TP, width)


@ray.remote(num_gpus=1, max_calls=1)
def _mono_io(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.workspace import init_workspace_manager

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        m.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0")
        rocm_aiter_ops.refresh_env_variables()
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        _check_io(rank, device, seeds=(0,))


def _check_io(rank: int, device, seeds) -> None:
    from vllm._aiter_ops import rocm_aiter_ops as ops
    from vllm.distributed import get_tp_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.models.deepseek_v4.amd.mono.dims import INDEX_TOPK
    from vllm.models.deepseek_v4.amd.mono.runner import MonoRunner
    from vllm.models.deepseek_v4.amd.mono_decode import _hca_width

    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=65536,
        trust_remote_code=True,
    ).create_engine_config()
    widths = {"csa": 0, "dep": _hca_width(vc), "long": 2048}
    assert widths["dep"] <= INDEX_TOPK < widths["long"]
    runner = MonoRunner(
        TP, rank, get_tp_group().cpu_group, device, max_topk=widths["long"]
    )
    steps = []
    for seed in seeds:
        w = _attn_weights(seed, rank, _dims().heads, _dims().groups, device)
        w["wqkv_sh"] = ops.shuffle_weight(w["wqkv"], layout=(16, 16))
        w["wo_b_sh"] = ops.shuffle_weight(w["wo_b"], layout=(16, 16))
        moes = {k: _moe(vc, k, seed, rank, device) for k in ("bias", "hash", "adv")}
        for i, case in enumerate(CASES):
            steps.append(_check_case(runner, vc, w, moes, widths, case, seed, i))
    _check_replays(steps)


def _layer_weights(w, moe, ratio, cos_sin, adv):
    from vllm.models.deepseek_v4.amd.mono.runner import AttnWeights, MonoLayerWeights
    from vllm.models.deepseek_v4.amd.mono_decode import _mx_scales

    sh, e = moe.shared_experts, moe.experts.routed_experts
    attn_norm, ffn_norm = w["attn_norm"].clone(), w["ffn_norm"].clone()
    if adv:
        attn_norm[32:64] = 0
        ffn_norm[64:96] = 0
    return MonoLayerWeights(
        attn=AttnWeights(
            layer_id=3, wqkv=w["wqkv_sh"], wqkv_scale=_mx_scales(w["wqkv_s"]),
            q_norm=w["q_norm"], kv_norm=w["kv_norm"], wq_b=w["wq_b"],
            wq_b_scale=_mx_scales(w["wq_b_s"]), wo_a=w["wo_a"],
            wo_a_scale=_mx_scales(w["wo_a_s"]), wo_b=w["wo_b_sh"],
            wo_b_scale=_mx_scales(w["wo_b_s"]), attn_sink=w["sink"],
            cos_sin=cos_sin, ratio=ratio,
        ),
        hc_attn_fn=w["attn_fn"], hc_attn_scale=w["attn_scale"],
        hc_attn_base=w["attn_base"], attn_norm=attn_norm,
        hc_ffn_fn=w["ffn_fn"], hc_ffn_scale=w["ffn_scale"],
        hc_ffn_base=w["ffn_base"], ffn_norm=ffn_norm,
        gate_w=moe.gate.weight, bias=moe.gate.e_score_correction_bias,
        w13=e.w13_weight, w13_s=e.w13_weight_scale, w2=e.w2_weight,
        w2_s=e.w2_weight_scale, sgu=sh.gate_up_proj.weight,
        sgu_s=_mx_scales(sh._gateup_scale), sw2=sh.down_proj.weight,
        sw2_s=_mx_scales(sh.down_proj.weight_scale_inv),
    )  # fmt: skip


# plain regions: poisoned (0xFF: NaN, an impossible code) before the checked
# launch, so an element a stage failed to write shows
PLAIN = ("x8", "x8s", "qx8", "qx8s", "part", "pstat", "xo", "xos", "x8b", "x8bs",
         "x8r", "x8rs", "mid", "mids", "normed")  # fmt: skip
K1_TAPS = ("x8", "x8s", "qkvp", "qx8", "qx8s")


def _check_case(runner, vc, w, moes, widths, case, seed, index):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4.amd.mono.layer import peer_half_bytes, scratch_layout
    from vllm.models.deepseek_v4.amd.mono.stages.moe import ffn_offset
    from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

    ratio, nreq, T, pad, table, kind, fold = case
    cfg = vc.model_config.hf_config
    eps = cfg.rms_norm_eps
    rank, dev = runner.rank, runner.device
    bf = torch.bfloat16
    width, hca = widths[table], ratio == 128
    with set_current_vllm_config(VllmConfig()):
        rope = build_deepseek_v4_rope(
            cfg, head_dim=HEAD_DIM, rope_head_dim=64,
            max_position_embeddings=cfg.max_position_embeddings,
            compress_ratio=ratio,
        )  # fmt: skip
    cos_sin = rope.cos_sin_cache.to(device=dev, dtype=torch.float32)
    s = _pad(
        _decode_step(
            ratio, nreq, T, w["kv_norm"], cos_sin, seed=7 + ratio + nreq + seed
        ),
        pad,
    )
    M = s["M"]
    ctx = (seed, case)
    moe, raw = moes[kind]
    mw = _layer_weights(w, moe, ratio, cos_sin, kind == "adv")
    mw.attn.check(runner.d)
    mw.check(runner.d)

    gi = torch.Generator().manual_seed(11 + 100 * seed + index)
    res0 = torch.randn(M, HC, HIDDEN, generator=gi)
    res0 = (res0 * torch.rsqrt(res0.pow(2).mean(-1, keepdim=True))).to(bf).to(dev)
    x0 = (torch.randn(M, HIDDEN, generator=gi) * 0.5).to(bf).to(dev)
    comb0 = torch.rand(M, HC, HC, generator=gi) + 0.1
    for _ in range(20):
        comb0 = comb0 / comb0.sum(-1, keepdim=True)
        comb0 = comb0 / comb0.sum(-2, keepdim=True)
    comb0 = comb0.to(dev)
    post0 = (torch.sigmoid(torch.randn(M, HC, 1, generator=gi)) * 2).to(dev)
    ids = torch.randint(0, cfg.vocab_size, (M,), generator=gi).to(dev)
    hashed = moe.gate.tid2eid is not None
    route = moe.gate.tid2eid[ids].contiguous() if hashed else None
    meta = dict(
        topk_indices=s["topk"], comp_cache=s["comp_cache"],
        comp_block_table=s["comp_block_table"], route_ids=route, topk=width,
        global_keys=hca, fold=fold,
    )  # fmt: skip
    args = (s["positions"], s["slot_mapping"], s["swa_cache"], s["swa_indices"],
            s["swa_lens"], s["token_to_req"])  # fmt: skip

    # ---- the checked launch: plain outputs poisoned, K1's taps kept at the
    # hand-off to K2 (which reuses x8 / x8s)
    layout = scratch_layout(M, TP, width)
    scratch = runner.scratch(M, width)

    def region(name, buf=scratch):
        off, n = layout[name]
        return buf[off : off + n]

    for name in PLAIN:
        region(name).fill_(255)
    for t in (runner.q, runner.normed, runner.qr, runner.res_mid, runner.post_a,
              runner.comb_a, runner.pre_a, runner.kt, runner.klen):  # fmt: skip
        t.view(torch.uint8).fill_(255)
    outs = runner._outs(M, res0)
    for t in outs:
        t.view(torch.uint8).fill_(255)
    swa_before = _storage(s["swa_cache"]).clone()
    k1 = {}

    def select(normed, qr):
        k1.update({n: region(n).clone() for n in K1_TAPS})

    torch.accelerator.synchronize()
    e = int(runner.epoch[0])
    runner.forward(mw, x0, res0, post0, comb0, *args, select=select, outs=outs,
                   **meta)  # fmt: skip
    torch.accelerator.synchronize()
    half = peer_half_bytes(TP)
    base = (e & 1) * half
    peer = runner.peer.bytes[base : base + half].clone()
    tag1, tag2 = 2 * e + 1, 2 * e + 2
    o = [t.clone() for t in outs]

    # ---- K1: the attention seam
    res_mid, pre_a = runner.res_mid[:M], runner.pre_a[:M]
    post_a, comb_a = runner.post_a[:M], runner.comb_a[:M]
    if fold:
        _check_fold("res_mid", res_mid, x0, res0, post0.squeeze(-1), comb0, ctx)
    else:
        assert torch.equal(res_mid, res0), ctx
    _check_mixes("attn", w, "attn", res_mid, post_a, comb_a, pre_a, eps, ctx)
    normed = runner.normed[:M]
    _check_norm("normed", normed, res_mid, pre_a, mw.attn_norm, eps, 40, ctx)
    x8 = k1["x8"][: M * HIDDEN].view(M, HIDDEN)
    x8s = k1["x8s"][: M * HIDDEN // 8].view(torch.int32).view(M, -1)
    _q8_equal("x8", x8, x8s, normed.double(), R.mx_code_vllm, ctx)
    if kind == "adv":
        assert (x8s[:, 1] == 0).all(), (ctx, "zero groups")

    # ---- K1: the attention front
    xd = R.dequant_fp8(x8, x8s, (1, 32))
    wqkv = _deq128(w["wqkv"], w["wqkv_s"])
    qkvp = R.pairs(k1["qkvp"], 2 * M * 2048, tag1).view(torch.float32)
    qkvp = qkvp.view(2, M, 2048)
    for half_k in range(2):
        ks = slice(half_k * HIDDEN // 2, (half_k + 1) * HIDDEN // 2)
        err = (qkvp[half_k].double() - xd[:, ks] @ wqkv[:, ks].T).abs()
        _apriori("qkvp", err / _gemv_err(xd[:, ks], wqkv[:, ks]), ctx)
    h = (qkvp[0] + qkvp[1]).to(bf)
    qr = runner.qr[:M]
    ref = R.rmsnorm(h[:, :Q_RANK], w["q_norm"], eps)
    _apriori(
        "qr", (qr.double() - ref).abs() / (_half_ulp(qr) + 15 * U * ref.abs()), ctx
    )
    qx8 = k1["qx8"][: M * Q_RANK].view(M, Q_RANK)
    qx8s = k1["qx8s"][: M * Q_RANK // 8].view(torch.int32).view(M, -1)
    _q8_equal("qx8", qx8, qx8s, qr.double(), R.mx_code_vllm, ctx)
    _check_records(s, h[:, Q_RANK:], w["kv_norm"], eps, cos_sin, swa_before, ctx)
    _check_q(runner.q[:M], s, qx8, qx8s, w, cos_sin, ctx)
    keys = _dims(width).keys
    table_w = s["topk"].stride(0) if hca else _dims(width).topk_width
    kt, klen = _key_table(s, ratio, hca, table_w, keys)
    got_kt = runner.kt[: M * keys].view(M, keys).cpu()
    got_klen = runner.klen[: 2 * M].cpu()
    assert torch.equal(got_klen, klen), (ctx, "klen")
    live = torch.arange(keys)[None] < klen[:M, None]
    if hca:
        assert torch.equal(got_kt[live], kt[live]), (ctx, "kt")
    else:
        assert torch.equal(got_kt, kt), (ctx, "kt")
    if pad:
        assert (klen[M - pad : M] == 0).all(), (ctx, "pad rows")

    # ---- K2: the attention back
    two_hop = runner.two_hop(M)
    _check_attention(runner, s, region, w, cos_sin, width, eps, ctx)
    xo = region("xo")[: M * runner.d.q_rows].view(M, -1)
    xos = region("xos")[: M * runner.d.q_rows // 8].view(torch.int32).view(M, -1)
    xod = R.dequant_fp8(xo, xos, (1, 32))
    o_rows, gk = runner.d.o_rows, runner.d.group_k
    woa = _deq128(w["wo_a"], w["wo_a_s"])
    # wo_a: a 16-row tile's K over 4 waves (``WOA_SPLIT_K``), then bf16
    zs = [
        (a @ b.T, _gemv_err(a, b, 4))
        for a, b in (
            (xod[:, g * gk : (g + 1) * gk], woa[g * O_RANK : (g + 1) * O_RANK])
            for g in range(runner.d.groups)
        )
    ]
    z, ze = (torch.cat([v[i] for v in zs], 1) for i in (0, 1))
    x8b = region("x8b")[: M * o_rows].view(M, o_rows)
    x8bs = region("x8bs")[: M * o_rows // 8].view(torch.int32).view(M, -1)
    d = ze + _half_ulp(z.abs() + ze)
    _q8_within("woa", x8b, x8bs, z, d, R.mx_code_vllm, ctx, sharp=False)
    part, attn = _gather_peer(peer, rank, M, tag2, two_hop, ctx)
    x8bd, wob = R.dequant_fp8(x8b, x8bs, (1, 32)), _deq128(w["wo_b"], w["wo_b_s"])
    err = (part[rank].double() - x8bd @ wob.T).abs()
    _apriori("wob", err / (_half_ulp(part[rank]) + _gemv_err(x8bd, wob)), ctx)

    # ---- K2: the FFN seam
    _check_fold("res_out", o[1], attn, res_mid, post_a, comb_a, ctx)
    _check_mixes("ffn", w, "ffn", o[1], o[2], o[3], o[4], eps, ctx)
    xk = region("normed")[: M * HIDDEN * 2].view(bf).view(M, HIDDEN)
    _check_norm("normed_ffn", xk, o[1], o[4], mw.ffn_norm, eps, 40, ctx)

    # ---- K2: the MoE and its all-reduce
    ffn = peer[ffn_offset(M, _dims()) :]
    total = _check_moe(region, moe, raw, xk, ffn, route, kind, rank, tag2, two_hop, ctx)
    assert torch.equal(o[0], total), (ctx, "out")

    step_outs = runner._outs(M, res0)

    def step():
        runner.forward(mw, x0, res0, post0, comb0, *args, select=lambda n, q: None,
                       outs=step_outs, **meta)  # fmt: skip
        return list(step_outs)

    again = step()
    torch.accelerator.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(again, o)), (ctx, "poisoned run")
    if table == "long":
        # a table wider than every token's keys (``bounded``): as one cut to them
        cut = runner._outs(M, res0)
        runner.forward(mw, x0, res0, post0, comb0, *args, select=lambda n, q: None,
                       outs=cut, **dict(meta, topk=widths["dep"]))  # fmt: skip
        torch.accelerator.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(cut, o)), (ctx, "bounded")
    return step


def _cos_sin(cos_sin, pos):
    cs = cos_sin[pos.long()].double()
    return cs[..., :32], cs[..., 32:]


def _check_q(got, s, qx8, qx8s, w, cos_sin, ctx):
    """wq_b: the GEMV -> bf16 -> GPT-J RoPE of each head's dims 448.. -> bf16."""
    M, H = got.shape[:2]
    qd, wqb = R.dequant_fp8(qx8, qx8s, (1, 32)), _deq128(w["wq_b"], w["wq_b_s"])
    v, ge = (t.view(M, H, HEAD_DIM) for t in (qd @ wqb.T, _gemv_err(qd, wqb)))
    cos, sin = (c[:, None] for c in _cos_sin(cos_sin, s["positions"]))
    ref, rc = v.clone(), ge.clone()
    ref[..., NOPE:] = R.rope_gptj(v[..., NOPE:], cos, sin)
    ce, co = ge[..., NOPE::2], ge[..., NOPE + 1 :: 2]
    rc[..., NOPE::2] = cos.abs() * ce + sin.abs() * co
    rc[..., NOPE + 1 :: 2] = sin.abs() * ce + cos.abs() * co
    rnd = _half_ulp(got)
    rv = v[..., NOPE:].abs()
    pair = (rv[..., 0::2] + rv[..., 1::2]).repeat_interleave(2, -1)
    trig = (cos.abs() + sin.abs()).repeat_interleave(2, -1)
    hu = _half_ulp(rv + ge[..., NOPE:])
    hu = torch.maximum(hu[..., 0::2], hu[..., 1::2]).repeat_interleave(2, -1)
    rnd[..., NOPE:] += trig * (hu + 2 * U * pair)
    _apriori("q", (got.double() - ref).abs() / (rnd + rc), ctx)


def _record_bytes(cache, block, slots):
    """The raw records at ``slots``: (data [n, 576] u8, codes [n, 7])."""
    flat = _storage(cache)
    slots = slots.long()
    page = cache.storage_offset() + slots // block * cache.stride(0)
    d0 = page + slots % block * 576
    s0 = page + block * 576 + slots % block * 8
    ar = torch.arange(576, device=cache.device)
    return flat[d0[:, None] + ar], flat[s0[:, None] + ar[:7]]


def _check_records(s, kv, kv_norm, eps, cos_sin, before, ctx):
    """The step's KV records (kv_norm -> bf16 -> RoPE -> bf16 -> fp8_ds_mla)
    from the kernel's own kv, either rounding of an ambiguous element; every
    other cache byte untouched."""
    cache = s["swa_cache"]
    live = s["slot_mapping"] >= 0
    slots, pos = s["slot_mapping"][live], s["positions"][live]
    kn = R.rmsnorm(kv[live], kv_norm, eps)
    amb, lo, hi = R.boundary(R.bf16_rne, kn, 12 * U * kn.abs())
    assert amb.double().mean() < 0.01, (ctx, "records ambiguous")
    data, codes = _record_bytes(cache, SWA_BLOCK, slots)
    got_q = _e4m3(data[:, :NOPE]).view(-1, 7, 64)
    codes = codes.long()
    cos, sin = _cos_sin(cos_sin, pos)
    trig = cos.abs() + sin.abs()
    got_rope = data[:, NOPE:].contiguous().view(torch.bfloat16).double()
    ok_c = torch.zeros_like(codes, dtype=torch.bool)
    ok_q = torch.zeros_like(got_q, dtype=torch.bool)
    ok_r = torch.zeros_like(got_rope, dtype=torch.bool).view(-1, 32, 2)
    variants = (lo, R.bf16_rne(kn), hi)
    for v in variants:
        nope = v[:, :NOPE].view(-1, 7, 64)
        ok_c |= codes == _rule_record(nope.abs().amax(-1))
        q = R.to_e4m3_rne(nope * torch.exp2(127 - codes.double())[..., None])
        ok_q |= got_q == q
    # a rope pair from either rounding of each of its elements, its fp32
    # rotation within 2 ulps of the f64 one, then bf16
    for ve in variants:
        for vo in variants:
            e, o = ve[:, NOPE::2], vo[:, NOPE + 1 :: 2]
            d = 2 * U * (e.abs() + o.abs()) * trig
            for i, r in enumerate((e * cos - o * sin, e * sin + o * cos)):
                g = got_rope.view(-1, 32, 2)[..., i]
                ok_r[..., i] |= (g >= R.bf16_rne(r - d)) & (g <= R.bf16_rne(r + d))
    assert ok_c.all() and ok_q.all() and ok_r.all(), (ctx, "records")
    after = _storage(cache)
    keep = torch.ones_like(after, dtype=torch.bool)
    for a, n in _record_spans(cache, SWA_BLOCK, slots):
        keep[a : a + n] = False
    assert torch.equal(after[keep], before[keep]), (ctx, "other records")


def _keys_of(s, kt_row):
    """f64 key rows [n, 512] of a key table row: compressed slots (bit 31),
    window slots, -1 none (zeros) -> (keys, valid)."""
    comp = (kt_row < 0) & (kt_row != -1)
    swa = kt_row >= 0
    keys = torch.zeros(
        kt_row.numel(), HEAD_DIM, dtype=R.F64, device=s["swa_cache"].device
    )
    if comp.any():
        slots = kt_row[comp] & 0x7FFFFFFF
        keys[comp.to(keys.device)] = R.read_records(
            s["comp_cache"], s["comp_entries"], slots
        )
    if swa.any():
        keys[swa.to(keys.device)] = R.read_records(
            s["swa_cache"], SWA_BLOCK, kt_row[swa]
        )
    return keys, (kt_row != -1).to(keys.device)


def _check_attention(runner, s, region, w, cos_sin, width, eps, ctx):
    """The split (each chunk's (max, sum) and bf16 partial against f64 over
    the keys of K1's table, with each head's q RMSNorm on its scores) and the
    combine (the kernel's chunks merged with the sink -> bf16 -> inverse RoPE
    -> vLLM's MXFP8)."""
    from vllm.models.deepseek_v4.amd.mono.attention.back import chunk_keys

    M, d = s["M"], _dims(width)
    H, ck = d.heads, chunk_keys(d, M)
    nchunk = -(-d.keys // ck)
    bf = torch.bfloat16
    part = region("part")[: M * nchunk * H * HEAD_DIM * 2].view(bf)
    part = part.view(M, nchunk, H, HEAD_DIM).double()
    pstat = region("pstat")[: M * nchunk * H * 8].view(torch.float32)
    pstat = pstat.view(M, nchunk, H, 2)
    kt = runner.kt[: M * d.keys].view(M, d.keys).long().cpu()
    klen = runner.klen[:M].cpu()
    q = runner.q[:M].double()
    scale = HEAD_DIM**-0.5 * LOG2E * torch.rsqrt(q.square().mean(-1) + eps)
    eta = (0.5 * 23 + 6) * U
    sink = w["sink"][:H].float() * _f32(LOG2E)
    cos, sin = _cos_sin(cos_sin, s["positions"])
    ys, dys = [], []
    for t in range(M):
        L = int(klen[t])
        nl = -(-L // ck)
        keys, valid = _keys_of(s, kt[t, :L])
        sc = (q[t] @ keys.T) * scale[t, :, None]
        ds = scale[t, :, None] * HEAD_DIM * U * (q[t].abs() @ keys.abs().T)
        ds = ds + eta * sc.abs()
        for ch in range(nl):
            cs = slice(ch * ck, min((ch + 1) * ck, L))
            vm = valid[cs]
            gm, gl, gp = (
                pstat[t, ch, :, 0].double(),
                pstat[t, ch, :, 1].double(),
                part[t, ch],
            )
            c2 = (ctx, "chunk", t, ch)
            if not vm.any():
                assert (gm == NEG).all() and (gl == 0).all() and (gp == 0).all(), c2
                continue
            sv, dv, kv = sc[:, cs][:, vm], ds[:, cs][:, vm], keys[cs][vm]
            _apriori("pmax", (gm - sv.amax(-1)).abs() / dv.amax(-1), c2)
            p = torch.exp2(sv - gm[:, None])
            dv = dv + U * (sv - gm[:, None]).abs()
            pe = p * (0.6931471805599453 * dv + 2.0**-22)
            lb = pe.sum(-1) + 16 * U * p.sum(-1)
            _apriori("psum", (gl - p.sum(-1)).abs() / lb, c2)
            pb = _half_ulp(gp) + (p @ kv.abs()) * (2.0**-8 + ck * U) + pe @ kv.abs()
            _apriori("part", (gp - p @ kv).abs() / pb, c2)
        m_c = pstat[t, :nl, :, 0]
        m_fin = torch.maximum(m_c.amax(0), sink) if nl else sink
        wc = torch.exp2((m_c - m_fin).double())
        ls = torch.exp2((sink - m_fin).double()) + (pstat[t, :nl, :, 1] * wc).sum(0)
        tot = (part[t, :nl] * wc[..., None]).sum(0)
        o = tot / ls[:, None]
        do = (part[t, :nl].abs() * wc[..., None]).sum(0) / ls[:, None]
        do = do * ((nl + 1) * U + 2.0**-23) + o.abs() * ((nl + 4) * U + 2.0**-21)
        ob, hu = _bf16_known(o, do)
        y = ob.clone()
        y[:, NOPE:] = R.rope_gptj(ob[:, NOPE:], cos[t], sin[t], inverse=True)
        dy = hu.clone()
        trig = (cos[t].abs() + sin[t].abs()).repeat_interleave(2, -1)
        rv, hr = ob[:, NOPE:].abs(), hu[:, NOPE:]
        pair = (rv[:, 0::2] + rv[:, 1::2]).repeat_interleave(2, -1)
        hpair = torch.maximum(hr[:, 0::2], hr[:, 1::2]).repeat_interleave(2, -1)
        dy[:, NOPE:] = trig * (hpair + 2 * U * pair)
        ys.append(y.reshape(-1))
        dys.append(dy.reshape(-1))
    xo = region("xo")[: M * d.q_rows].view(M, -1)
    xos = region("xos")[: M * d.q_rows // 8].view(torch.int32).view(M, -1)
    _q8_within("xo", xo, xos, torch.stack(ys), torch.stack(dys), R.mx_code_vllm, ctx)


def _swiglu(g, u, eg, eu):
    """DeepSeek's clamped SwiGLU in f64 and its partials (d/dg, d/du), each
    kept on a clamp within the inputs' errors (eg, eu) of its edge."""
    gc, uc = g.clamp(max=10), u.clamp(-10, 10)
    sg = torch.sigmoid(gc)
    act = gc * sg
    dg = (sg + gc * sg * (1 - sg)) * uc * (g - eg < 10)
    du = act * (u.abs() - eu < 10)
    return act, act * uc, dg, du


def _check_moe(region, moe, raw, xk, ffn, route_ids, kind, rank, tag, two_hop, ctx):
    """The MoE's stages from its input row ``xk`` (the FFN seam's): its two
    MXFP8 quants, the router's logits, the route, the shared expert, each
    pick's MXFP8 intermediate, this rank's partial output -> the reduced
    output (``_gather_peer``)."""
    from vllm.models.deepseek_v4.amd.mono.dims import EXPERTS, ROUTED_SCALE, TOPK
    from vllm.models.deepseek_v4.amd.mono.stages.moe import ROUTE_COPIES

    M, d = xk.shape[0], _dims()
    x = xk.double()
    i32 = torch.int32
    x8r = region("x8r")[: M * HIDDEN].view(M, HIDDEN)
    x8rs = region("x8rs")[: M * HIDDEN // 8].view(i32).view(M, -1)
    _q8_equal("x8r", x8r, x8rs, x, _rule_ceil, ctx)
    x8 = region("x8")[: M * HIDDEN].view(M, HIDDEN)
    x8s = region("x8s")[: M * HIDDEN // 8].view(i32).view(M, -1)
    _q8_equal("x8_ffn", x8, x8s, x, R.mx_code_vllm, ctx)
    if kind == "adv":
        assert (x8s[:, 2] == 0).all(), (ctx, "zero groups")

    lg = R.pairs(region("logit"), M * EXPERTS * 4, tag).view(torch.float32)
    lg = lg.view(M, EXPERTS, 4)
    tot = ((lg[..., 0] + lg[..., 1]) + lg[..., 2]) + lg[..., 3]
    ref, cond = _mm(x, moe.gate.weight.double())
    # router_tile: a wave's chain of HIDDEN / 1024 bf16 16x16x32 MFMAs at (32 +
    # 1) u each, 8 wave partials summed from 0, the 4 parts' 3 adds
    eta = (33 * HIDDEN // 1024 + 8 + 3) * U
    _apriori("logit", (tot.double() - ref).abs() / (eta * cond), ctx)

    rt = R.pairs(region("route"), ROUTE_COPIES * M * 2 * TOPK, tag)
    rt = rt.view(ROUTE_COPIES, M, 2 * TOPK)
    assert (rt == rt[0]).all(), (ctx, "route copies")
    ids, wts = rt[0, :, :TOPK], rt[0, :, TOPK:].view(torch.float32).double()
    scores = R.router_scores(tot, "sqrt_softplus")
    # ``_score``: x log2e (2 u), exp2 (2 u + 2 u |x|), 1 + e (u), log (2 u),
    # x ln2 (2 u) -> sp; sqrt (IEEE, u): each score's relative error
    lx = tot.double()
    ex = lx.clamp(max=20).exp()
    sp = torch.where(lx > 20, lx, torch.log1p(ex))
    dsp = 4 * U * sp + U * (1 + ex / (1 + ex) * (2 + 2 * lx.abs()))
    dsc = torch.where(lx > 20, 0, dsp) / (2 * sp) + U
    if route_ids is not None:
        assert torch.equal(ids, route_ids), (ctx, "hash ids")
    else:
        bias = moe.gate.e_score_correction_bias.double()
        ref_ids, _, margin = R.route(tot, bias, TOPK, "sqrt_softplus", ROUTED_SCALE)
        tie = margin == 0
        # the picks are certain where every reference pick's score + bias (the
        # add: u) clears every other's by their errors
        sel = scores + bias
        es = dsc * scores + U * sel.abs()
        top = torch.zeros_like(sel, dtype=torch.bool).scatter(1, ref_ids.long(), True)
        gap = torch.where(top, sel - es, torch.inf).amin(-1)
        gap = gap - torch.where(top, -torch.inf, sel + es).amax(-1)
        rows = R.assert_topk(ids, ref_ids, torch.where(tie, torch.inf, gap), 0)
        assert rows >= M - M // 8, (ctx, "routes checked", rows)
        if kind == "adv":
            assert tie.all() and (tot[:, TIE[0]] == tot[:, TIE[1]]).all(), ctx
            assert (tot > 20).any(), (ctx, "softplus' linear branch")
    sw, dsc = scores.gather(1, ids.long()), dsc.gather(1, ids.long())
    wref = sw * ROUTED_SCALE / sw.sum(-1, keepdim=True)
    # the weights: 5 adds, a division (IEEE), a product
    rel = dsc + (sw * dsc).sum(-1, True) / sw.sum(-1, True) + 7 * U
    _apriori("route_w", (wts - wref).abs() / (rel * wref), ctx)

    sh = d.sh_inter
    xd = R.dequant_fp8(x8, x8s, (1, 32))
    sg, su = raw.sgu[:sh].double(), raw.sgu[sh:].double()
    g, u, eg, eu = xd @ sg.T, xd @ su.T, _gemv_err(xd, sg), _gemv_err(xd, su)
    act, ref, dg, du = _swiglu(g, u, eg, eu)
    smid = R.pairs(region("smid"), M * sh, tag).view(torch.float32).view(M, sh)
    # gate and up -> bf16; silu: expf (3 u: v_exp's 1 ulp, its reduced
    # argument's rounding ln 2 u / 2), 1 + e (u), the division (IEEE, u), ->
    # bf16; x up (u) -> bf16
    rnd = (
        _half_ulp(smid)
        + u.clamp(-10, 10).abs() * _half_ulp(act)
        + dg.abs() * _half_ulp(g.abs() + eg)
        + act.abs() * _half_ulp(u.abs() + eu)
        + 6 * U * ref.abs()
    )
    err = (smid.double() - ref).abs()
    _apriori("smid", err / (rnd + dg.abs() * eg + du.abs() * eu), ctx)
    if kind == "adv":
        sat = (g > 10).double().mean().item()
        assert 0.01 <= sat <= 0.2, (ctx, "shared gate clamp", sat)
    smq = R.pairs(region("smq"), M * sh // 4, tag).view(torch.uint8).view(M, sh)
    smc = R.pairs(region("smc"), M * sh // 32, tag).view(M, -1)
    _q8_equal("smq", smq, smc, smid.double(), R.mx_code_vllm, ctx)

    # the picks in slot order: experts ascending, then tokens
    inter = d.inter
    groups = inter // 32
    picks = sorted((int(ids[t, k]), t, k) for t in range(M) for k in range(TOPK))
    row = {(t, k): r for r, (_, t, k) in enumerate(picks)}
    mid = region("mid")[: len(picks) * inter].view(len(picks), inter)
    mids = region("mids")[: len(picks) * groups * 4].view(i32).view(-1, groups)
    gi = torch.arange(groups)
    mids = mids[:, gi % 4 * (inter // 128) + gi // 4]
    x8rd = R.dequant_fp8(x8r, x8rs, (1, 32))
    vs, ds, rows_ = [], [], []
    out = torch.zeros(M, TOPK, HIDDEN, dtype=R.F64, device=x.device)
    eout = torch.zeros_like(out)
    sat = []
    for e in sorted({p[0] for p in picks}):
        tk = [(t, k) for p_e, t, k in picks if p_e == e]
        toks, ks = [t for t, _ in tk], [k for _, k in tk]
        rr = [row[p] for p in tk]
        w13 = R.dequant_mxfp4(raw.w13[0][e], raw.w13[1][e])
        # ug: K in 2 or 4 slices (``ug_rounds``), summed pairwise
        h, eh = x8rd[toks] @ w13.T, _gemv_err(x8rd[toks], w13, 2)
        hg, hu, eg, eu = h[:, :inter], h[:, inter:], eh[:, :inter], eh[:, inter:]
        _, v, dg, du = _swiglu(hg, hu, eg, eu)
        sat.append((hg > 10).double().mean())
        # sigmoid: x log2e (2 u), exp2 (2 u + 2 u |g|), 1 + e (u), rcp (2 u);
        # x g, x up (2 u); then bf16
        dv = dg.abs() * eg + du.abs() * eu
        dv = dv + (7 + 2 * hg.clamp(max=10).abs()) * U * v.abs()
        vs.append(v)
        ds.append(dv + _half_ulp(v.abs() + dv))
        rows_ += rr
        w2 = R.dequant_mxfp4(raw.w2[0][e], raw.w2[1][e][..., :groups])
        md = R.dequant_fp8(mid[rr], mids[rr], (1, 32))
        wk = wts[toks, ks][:, None]
        out[toks, ks] = wk * (md @ w2.T)
        # down: K on DOWN_CHAINS = 2 accumulators
        eout[toks, ks] = wk * _gemv_err(md, w2, 2)
    _q8_within(
        "mid", mid[rows_], mids[rows_], torch.cat(vs), torch.cat(ds), _rule_ceil,
        ctx, sharp=False,
    )  # fmt: skip
    if kind == "adv":
        sat = torch.stack(sat).mean().item()
        assert 0.01 <= sat <= 0.2, (ctx, "routed gate clamp", sat)
    # ``down_combine``: routed = bf16(routed + bf16(v w)) in top-k order, then
    # bf16(routed + bf16(shared)); an fp32 add (u) before each bf16
    acc = torch.zeros_like(out[:, 0])
    bound = torch.zeros_like(acc)
    for k in range(TOPK):
        c = out[:, k]
        ec = eout[:, k] + U * c.abs()
        acc, bound = acc + c, bound + ec + _half_ulp(c.abs() + ec)
        bound = bound + U * acc.abs() + _half_ulp(acc.abs() + bound)
    smd, sd = R.dequant_fp8(smq, smc, (1, 32)), raw.sd.double()
    y, ey = smd @ sd.T, _gemv_err(smd, sd)
    acc, bound = acc + y, bound + ey + _half_ulp(y.abs() + ey)
    bound = bound + U * acc.abs() + _half_ulp(acc.abs() + bound)
    part, total = _gather_peer(ffn, rank, M, tag, two_hop, ctx)
    _apriori("ffn", (part[rank].double() - acc).abs() / bound, ctx)
    return total


def _check_replays(steps) -> None:
    """Every case back to back in one graph, each replay bit for bit the
    eager run's."""
    outs = [t for step in steps for t in step()]
    torch.accelerator.synchronize()
    ref = [t.clone() for t in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in steps:
            step()
    for i in range(REPLAYS):
        graph.replay()
        if i % 50 == 49:
            torch.accelerator.synchronize()
            bad = [
                j for j, (a, b) in enumerate(zip(outs, ref)) if not torch.equal(a, b)
            ]
            assert not bad, f"replay {i}: outputs {bad} differ from the eager run"


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
def test_mono_decode_stages_match_float64(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, TP, 1, _mono_io)
