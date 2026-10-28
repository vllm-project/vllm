# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""References for the kernel tests: random weights in the checkpoint's formats
at the TP8 rank shapes as vLLM holds them after loading, and the ops vLLM runs
for the same modules (the Gemma residual add + norm, AITER's GDN decode, the
MXFP4 MoE)."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.platforms import current_platform

bf = torch.bfloat16
COS_MIN = 0.9999


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _u8(g, *shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device=g.device).to(torch.uint8)


def _all_reduce_fp32(t: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in fp32, as vLLM's custom all-reduce."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(bf)


def _gemma_norm(x, residual, w, eps=1e-6):
    """``GemmaRMSNorm.forward_native(x, residual)`` -> (x, the residual)."""
    t = x.float() if residual is None else x.float() + residual.float()
    res = t.to(bf)
    t = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (t * (w.float() + 1.0)).to(bf), res


def _block_input(g, s: int, first: bool) -> dict:
    """A layer's input: the previous layer's output and residual (bf16, the
    residual stream larger), the Gemma norm weight (1 + w)."""
    from vllm.models.qwen3_5.amd.specialized_decode.mono.layout import HIDDEN

    return {
        "hidden": _randn(g, s, HIDDEN, scale=0.5),
        "residual": None if first else _randn(g, s, HIDDEN, scale=4.0),
        "ln_w": _randn(g, HIDDEN, scale=0.2),
    }


def _mxfp4_linear(gr, n: int, k: int, shuffled: bool = True):
    """An online-MXFP4 linear (``Mxfp4OnlineLinearMethod``) as the AITER kernel
    holds it: random fp4x2 (n, k / 2) and e8m0 (n, k / 32); on the ASM FP4 GEMM
    (``shuffled``) the weight through ``shuffle_weight(layout=(16, 16))`` and
    the scales through ``AiterMxfp4LinearKernel``'s swizzle, on the Triton FP4
    GEMM the weight as is and the scales transposed. Returns (raw q, raw
    scales, w, ws)."""
    from aiter.ops.shuffle import shuffle_weight

    q = _u8(gr, n, k // 2)
    sc = _u8(gr, n, k // 32, lo=116, hi=121)
    if not shuffled:
        return q, sc, q.clone(), sc.T.contiguous()
    w = shuffle_weight(q.clone().view(torch.float4_e2m1fn_x2), layout=(16, 16))
    sm, sn = sc.shape
    ws = sc.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6)
    return q, sc, w, ws.contiguous().view(sm, sn)


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mxfp4_dequant(q, scales):
    """Packed e2m1 bytes (low nibble first) times their e8m0 scales -> fp32."""
    lut = torch.tensor(_FP4 + tuple(-v for v in _FP4), device=q.device)
    v = lut[torch.stack([q & 0xF, q >> 4], -1).flatten(-2).long()]
    return v * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _mxfp4_qdq(v, rceil: bool = True):
    """Dynamic MXFP4 of the last dim as AITER's activation quant: 1 x 32 blocks,
    e8m0 scale ceil_pow2(amax / 6) (``rceil``: the ASM FP4 GEMM's
    ``per_1x32_f4_quant_hip``) or 2^(floor(log2(amax rounded at its top two
    mantissa bits)) - 2) (the Triton FP4 GEMM's ``dynamic_mxfp4_quant``), e2m1
    nearest even saturating at 6."""
    b = v.float().unflatten(-1, (-1, 32))
    amax = b.abs().amax(-1, keepdim=True)
    if rceil:
        bits = (amax / 6.0).view(torch.int32)
        e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
        e = e.clamp(1, 253)
    else:
        e = (((amax.view(torch.int32) + 0x200000) >> 23) & 0xFF) - 2
        e = e.clamp(1, 254)
    sc = torch.exp2(e.float() - 127)
    a = (b / sc).abs().clamp(max=6.0)
    q = torch.where(
        a < 2,
        torch.round(a * 2) / 2,
        torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
    )
    return (torch.sign(b) * q * sc).flatten(-2)


# ---------------------------------------------------------------- gdn
def _gdn_inputs(
    s: int, n_real: int, slots: int = 24, shuffled: bool = True, first: bool = False
):
    from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

    # live rows on slots 3 .. 3 + n_real
    slots = max(slots, n_real + 3)
    g = torch.Generator(device="cuda").manual_seed(0)
    w = {
        **_block_input(g, s, first),
        "w_ba": _randn(g, L.BA, L.HIDDEN, scale=0.02),
        "conv_w": _randn(g, L.CONV, 1, 4, scale=0.4),
        "a_log": _randn(g, L.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, L.NV, scale=0.5),
        "norm_w": (1 + _randn(g, L.HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    q, sc, w["w_qkvz"], w["w_qkvzs"] = _mxfp4_linear(g, L.QKVZ, L.HIDDEN, shuffled)
    w["w_qkvz_ref"] = _mxfp4_dequant(q, sc).to(bf)
    w["shuffled"] = shuffled
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    n = L.NV * L.HD * L.HD
    pad = _randn(g, slots, n + 8192, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, :n].view(slots, L.NV, L.HD, L.HD)
    conv = _randn(g, slots, L.CONV * 3 + 1024)[:, : L.CONV * 3]
    w["conv_state"] = conv.view(slots, 3, L.CONV).transpose(-1, -2)
    # live rows on slots 3 ..; pad rows on the null block 0
    idx = torch.zeros(s, dtype=torch.int32, device="cuda")
    idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device="cuda")
    w["st_idx"] = idx
    return w


def _gdn_core_ref(w, n: int, qkvz, ba):
    """The GDN decode path on ROCm from in_proj's outputs (AITER's conv update
    and fused gated delta rule, then RMSNormGated with the SiLU output gate) on
    the first ``n`` rows; the rest are the cudagraph pad. -> (conv state, SSM
    state, core [S, NV HD])."""
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token,
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

    s, nk, nv, hd = qkvz.size(0), L.NK, L.NV, L.HD
    with set_current_vllm_config(VllmConfig()):
        norm = RMSNormGated(
            hd, eps=1e-6, norm_before_gate=True, activation="silu", device="cuda"
        )
    norm.weight.data = w["norm_w"].clone()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    z = torch.empty(s, nv, hd, dtype=bf, device="cuda")
    core = torch.empty(s, nv, hd, dtype=bf, device="cuda")
    mixed, b, a = fused_reshape_causal_conv1d_update_single_token(
        qkvz.contiguous(),
        n,
        nk,
        nv,
        hd,
        hd,
        ba.contiguous(),
        z,
        core,
        conv_state,
        w["conv_w"].view(-1, 4),
        None,
        "silu",
        conv_state_indices=w["st_idx"][:n],
        validate_data=True,
        qkvz_layout="flat",
    )
    fused_rearrange_sigmoid_gated_delta_rule(
        A_log=w["a_log"],
        a=a,
        b=b,
        dt_bias=w["dt_bias"],
        qkv=mixed,
        key_dim=nk * hd,
        value_dim=nv * hd,
        head_k_dim=hd,
        head_v_dim=hd,
        initial_state=rstate,
        inplace_final_state=True,
        cu_seqlens=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
        ssm_state_indices=w["st_idx"][:n],
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core.reshape(-1),
    )
    return conv_state, rstate, norm.forward_native(core, z).reshape(s, nv * hd)


def _gdn_reference(w, n: int):
    """The input norm, then the GDN decode path on ROCm (MXFP4 in_proj_qkvz,
    AITER's conv update and fused gated delta rule, RMSNormGated) on the first
    ``n`` rows; the rest are the cudagraph pad."""
    x, res = _gemma_norm(w["hidden"], w["residual"], w["ln_w"])
    qkvz = F.linear(_mxfp4_qdq(x, w["shuffled"]).to(bf), w["w_qkvz_ref"])
    ba = F.linear(x, w["w_ba"])
    conv_state, rstate, core = _gdn_core_ref(w, n, qkvz, ba)
    return {
        "x": x,
        "res": res,
        "qkvz": qkvz,
        "ba": ba,
        "conv_state": conv_state,
        "rstate": rstate,
        "core": core,
    }


# ---------------------------------------------------------------- moe
def _moe_weights(rank: int, device):
    """One layer's MoE block at TP8 as vLLM holds it after loading the fused
    shared expert as expert 512: bf16 router and shared-expert gate; the
    experts' MXFP4 (RI = RIP: no intermediate pad) through ``shuffle_weights``
    and ``e8m0_shuffle``."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

    h, ri, ne = L.HIDDEN, L.RI, L.E + 1
    assert ri == L.RIP
    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w13q, w2q = _u8(gr, ne, 2 * ri, h // 2), _u8(gr, ne, h, ri // 2)
    w13s = _u8(gr, ne, 2 * ri, h // 32, lo=116, hi=121)
    w2s = _u8(gr, ne, h, ri // 32, lo=118, hi=123)
    raw = {"w13": w13q, "w2": w2q, "w13s": w13s, "w2s": w2s}
    w13, w2 = rocm_aiter_ops.shuffle_weights(w13q.view(fp4), w2q.view(fp4))
    return {
        "raw": raw,
        "w_gate": _randn(gc, L.E, h, scale=0.02),
        "w_sg": _randn(gc, 1, h, scale=0.02),
        "w13": w13,
        "w2": w2,
        "w13s": e8m0_shuffle(w13s.view(ne * 2 * ri, -1).view(e8)).view(ne, 2 * ri, -1),
        "w2s": e8m0_shuffle(w2s.view(ne * h, -1).view(e8)).view(ne, h, -1),
    }


def _route_and_shared(x, mw, ids=None):
    """The bf16 router's softmax top-10 (or the experts ``ids``) renormalized,
    then the fused shared expert as pick 10, expert 512, weighted by the
    sigmoid (fp32) of the bf16 shared-expert gate logit. -> (weights, ids)."""
    from vllm.models.qwen3_5.amd.specialized_decode.mono.layout import TOPK, E

    logits = F.linear(x, mw["w_gate"])
    probs = torch.softmax(logits.float(), dim=-1)
    if ids is None:
        ids = torch.topk(probs, TOPK, dim=-1, sorted=True)[1]
    wts = probs.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    sg = torch.sigmoid(F.linear(x, mw["w_sg"]).float())
    wts = torch.cat([wts, sg], dim=-1)
    ids = torch.cat([ids, torch.full_like(ids[:, :1], E)], dim=-1)
    return wts, ids


def _check_route(scratch, region, x, mw):
    """The kernel's ROUTE: each token's picks a top-10 of the bf16 logits up
    to ties (the kernel's logit sums round to bf16 in another order, and equal
    keys go to the lower expert), their weights the renormalized softmax.
    Returns the picks, so the numerics compare on the same experts."""
    from vllm.models.qwen3_5.amd.specialized_decode.mono.layout import TOPK

    s = x.shape[0]
    off = region[0]
    words = scratch[off : off + s * 2 * TOPK * 8].view(torch.int32)
    words = words.view(s, 2 * TOPK, 2)[:, :, 0]
    ids, wts = words[:, :TOPK].long(), words[:, TOPK:].contiguous().view(torch.float32)
    logits = F.linear(x, mw["w_gate"]).float()
    tenth = torch.topk(logits, TOPK, dim=-1)[0][:, -1:]
    slack = 2 * 2.0**-7 * tenth.abs()
    assert (logits.gather(-1, ids) >= tenth - slack).all(), "picks not a top-10"
    assert all(len(set(r.tolist())) == TOPK for r in ids), "repeated pick"
    want = _route_and_shared(x, mw, ids)[0][:, :TOPK]
    # a renormalized weight moves by its logit's rounding minus the picks'
    # weighted mean of theirs: up to two bf16 ulps of the largest pick
    ulp = 2.0**-7 * logits.gather(-1, ids).abs().amax(-1, keepdim=True)
    assert ((wts - want).abs() <= want * (2 * ulp + 1e-3) + 1e-5).all(), (
        (wts - want).abs().max().item()
    )
    return ids


def _moe_exact(x, mw, ids):
    """The routed experts in fp32 on the dequantized weights, x and the
    intermediate unquantized, then the all-reduce."""
    raw = mw["raw"]
    ri = raw["w2"].shape[-1] * 2
    wts, ids = _route_and_shared(x, mw, ids)
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ x[t].float()
            h = F.silu(gu[:ri]) * gu[ri:]
            routed[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _all_reduce_fp32(routed.to(bf))


def _moe_emulation(x, mw, ids=None):
    """The MoE block with the routed experts as the stock decode kernel computes
    them (x and silu(g) u through dynamic MXFP4, fp32 accumulation, the route
    weight after w2), in plain torch; then the all-reduce."""
    raw = mw["raw"]
    ri = raw["w2"].shape[-1] * 2
    wts, ids = _route_and_shared(x, mw, ids)
    xq = _mxfp4_qdq(x)
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ xq[t]
            h = _mxfp4_qdq(F.silu(gu[:ri]) * gu[ri:])
            routed[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _all_reduce_fp32(routed.to(bf))


def _moe_reference(x, mw, ids=None):
    """Qwen3NextSparseMoeBlock's decode path on ``x`` through vLLM's ops: the
    router and shared expert of ``_route_and_shared``, aiter's ``fused_moe`` for
    the MXFP4 experts, the all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    wts, ids = _route_and_shared(x, mw, ids)
    routed = fused_moe(
        x,
        mw["w13"],
        mw["w2"],
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Silu,
        QuantType.per_1x32,
        False,
        mw["w13s"],
        mw["w2s"],
        None,
        None,
        dtype=bf,
    )
    return _all_reduce_fp32(routed)
