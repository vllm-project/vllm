# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the DeepSeek-V4-Pro mono decode kernels at TP8 on CDNA4 against
the ops vLLM runs for the same decode step: random weights in the checkpoint's
formats (FP8 128 x 128 blocks, MXFP4 experts), processed as vLLM processes
them (aiter's preshuffled shared gate_up, its routed experts' layout).

The whole layer (K1 + K2) against aiter's fused mHC seams, the ROCm attention
op chain (block-scaled FP8 GEMMs, the fused q norm / RoPE / KV insert, the
sparse MLA, the inverse-RoPE MXFP8 wo_a BMM), the TP all-reduce and
``DeepseekV4MoE``: on CSA layers (their top-k) and HCA ones (every compressed
row), bias-routed and hash-routed, at one- and two-hop widths.

Each of vLLM's stages runs on the kernels' own input of that stage, so the two
outputs differ by the stage's arithmetic only, and every bound is derived from
it (``seam_parity``, the units below): where both paths round one fp32 chain
to bf16, to a bf16 ulp; where they quantize onto different E4M3 grids, in
units of the expected rms of those roundings."""

import math

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

MODEL = "deepseek-ai/DeepSeek-V4-Pro"
TP = 8
HIDDEN, HC = 7168, 4
Q_RANK, HEAD_DIM, NOPE, ROPE = 1536, 512, 448, 64
N_HEADS, O_GROUPS, O_RANK = 128, 16, 1024
WINDOW, TOPK, SWA_BLOCK, CACHE_BLOCK = 128, 1024, 64, 256
RECORD, DATA, ALIGN = 584, 576, 576
REPLAYS = 200


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


# ---------------------------------------------------------------- parity bounds
# Shared with ``test_dsv4_mono_layers.py``. The kernels and vLLM's ops compute
# the same functions in different formats; each comparison feeds both the same
# stage input, so a difference is the two paths' arithmetic only:
# - where both round the same fp32 chain once to bf16 (the seams' folds and
#   norms), elementwise to one bf16 ulp;
# - where both quantize the same values onto different E4M3 grids (the
#   kernels' MXFP8, E8M0 a 32, against vLLM's FP8 with an fp32 scale a 128 or
#   its 128-wide MXFP8), in units of the two independent roundings' rms.
U = 2.0**-24


def _rn_rms(bits: int) -> float:
    """A row's relative error ||RN(x) - x|| / ||x|| (as ``_rows_rel``) after
    one round-to-nearest onto a grid of ``bits`` mantissa bits: the error
    uniform over +-half a step, the step 2^-bits of the value's binade, so
    E[step^2 / 12] / E[x^2], summed over the binades of a Gaussian row (the
    rows here are sums of thousands of random products; the scale and a
    SwiGLU product's distribution move it by < 0.3 %)."""

    def mass(e):
        return math.erfc(2.0**e / 2**0.5) - math.erfc(2.0 ** (e + 1) / 2**0.5)

    return math.sqrt(sum(mass(e) * 4.0 ** (e - bits) / 12 for e in range(-40, 8)))


# one rounding onto E4M3 (3 mantissa bits): 0.0266 (an MXFP8 a 32 or 128 scale
# is a power of two, which leaves it; a 128 block's fp32 scale, 0.0257)
E4M3_RN = _rn_rms(3)
# two such roundings onto unrelated grids: independent, sqrt(2) of one
TWO_GRIDS = math.sqrt(2) * E4M3_RN
# the attention's activation quantizations, on different grids on the two
# paths: its input (wqkv), qr (wq_b), the output heads (wo_a), z (wo_b). Each
# relative error reaches the output with gain 1: RMSNorm, RoPE and the random
# FP8 GEMMs keep a relative error in expectation, and a score error from q
# moves the softmax output by the scores' spread, 1 (q is RMS-normed per head,
# the keys kv-normed).
ATTN_GRIDS = 4
# the same for bf16 (7 mantissa bits)
BF16_RN = _rn_rms(7)
# a step's KV record: the input's grid (wqkv; kv_norm and RoPE keep a
# relative error), then the record's own E4M3 rounding of the 448 nope dims
# (the 64 RoPE dims stay bf16)
KV_GRIDS = math.sqrt(1 + 448 / 512)
# The routed MoE: both paths quantize its input by the same rule (AITER's
# MXFP8, bit for bit) but its SwiGLU output from different values: AITER's
# tuned FlyDSL stage 1 quantizes the fp32 SwiGLU in its epilogue, the kernels
# (as AITER's unfused stage 1, which writes bf16) its bf16 rounding. E4M3's
# rounding midpoints are bf16 values (5 significant bits), so the bf16 value
# ties at a midpoint -- and rounds to even -- wherever the fp32 one is within
# half a bf16 ulp of it: a 2^-7 / 2^-3 / 2 = 2^-5 chance an element, one step
# s there. E4M3_RN^2 = E[s^2] / 12 / E[y^2], so the rms relative difference
# sqrt(2^-5 E[s^2] / E[y^2]) = sqrt(12 2^-5) E4M3_RN: 1.6 %.
MID_FLIP = math.sqrt(12 * 2.0**-5) * E4M3_RN
# The bf16 roundings of a MoE output on either path, in roundings of the whole
# output (each at its operand's share of the energy, the picks' outputs
# independent, alike in size): the six weighted picks (1), the adds of the 2nd
# .. 6th (sum k / 6 = 20 / 6; AITER's stage 2 adds bf16 atomically, the
# kernels in top-k order), routed + shared (1), the TP sum (fp32, one rounding;
# a rank's partial roundings carry 1 / TP of the energy each)
MOE_BF16 = 1 + 20 / 6 + 1 + 1
# a "unit" is the derived expectation. Within two is good (its gain model is
# an expectation, not a bound); a wiring or semantic fault shows as tens.
UNITS_OK = 2.0


def _rows_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double().flatten(1), b.double().flatten(1)
    return (a - b).norm(dim=1) / b.norm(dim=1)


def seam_parity(p: dict, sub: str, x, res, post, comb, first: bool, m, v):
    """A seam's two paths on the same inputs (x, res, post, comb: the fold's;
    ``first``: the first layer's seam, no fold) -> each check's worst error
    over its bound (<= 1 holds). ``m`` / ``v``: (residual, post, comb, normed)
    of the kernels / vLLM. ``p``: the layer's seam parameters.

    - residual: one fp32 fold (5 products: 5 u |terms| each) then bf16 on
      both, so the paths differ by at most a bf16 ulp;
    - post / comb: vLLM's fused seam mixes the fp32 fold, the kernels (as
      vLLM's unfused seam and its torch reference) the bf16 residual, and read
      fn as bf16 hi + lo: that shift is exact in f64; the rest, each path's
      fp32 chain, is ``_gate_err``'s a-priori bound;
    - normed: both collapse their own bf16 residual by their own pre mix,
      round it to bf16, RMSNorm it (the RMS of the unrounded collapse) and
      round to bf16: each within one bf16 ulp of the f64 value, either
      rounding of a collapse within its a-priori error of a tie."""
    from . import mono_reference as R
    from .test_dsv4_mono_io import _bf16_known, _fn_as_read, _gate_err, _seam_ref

    eps, hc_eps, iters, norm_eps = p["eps"]
    fn = p[f"hc_{sub}_fn"].double()
    scale, base = p[f"hc_{sub}_scale"], p[f"hc_{sub}_base"]
    w = p[f"{sub}_norm"].double()
    M = res.shape[0]
    out = {}
    if first:
        res_b = res_f = res.double()
    else:
        res_f, eps_f = _seam_ref(x, res, post.reshape(M, -1), comb)
        err = (m[0].double() - v[0].double()).abs()
        bound = 0.5 * (R.ulp(m[0]) + R.ulp(v[0])) + 2 * eps_f
        out["residual"] = (err / bound).max().item()
        res_b = m[0].double()
    fn_m = _fn_as_read(fn)
    mix_m = R.mhc_mixes(res_b.flatten(1), fn_m, scale, base, HC, eps, hc_eps,
                        hc_eps, 2.0, iters)  # fmt: skip
    mix_v = R.mhc_mixes(res_f.flatten(1), fn, scale, base, HC, eps, hc_eps, hc_eps,
                        2.0, iters)  # fmt: skip
    b = _gate_err(res_b.flatten(1), fn_m, scale, base, eps, hc_eps, iters)
    for k, name in ((1, "post"), (2, "comb")):
        d = v[k].double().reshape(M, -1) - m[k].double().reshape(M, -1)
        shift = (mix_v[k] - mix_m[k]).reshape(M, -1)
        out[name] = ((d - shift).abs() / (2 * b[k])).max().item()
    paths = (("normed", m[3], mix_m[0], res_b),
             ("normed_vllm", v[3], mix_v[0], v[0].double()))  # fmt: skip
    for who, got, pre, rows in paths:
        rows = rows.view(M, HC, -1)
        col = R.mhc_collapse(rows, pre)
        dcol = ((4 * U * pre.abs() + b[0])[..., None] * rows.abs()).sum(1)
        col_b, amb = _bf16_known(col, dcol)
        gain = torch.rsqrt(col.square().mean(-1, keepdim=True) + norm_eps) * w
        ref = col_b * gain
        bound = R.ulp(ref) + gain.abs() * amb
        out[who] = ((got.double() - ref).abs() / bound).max().item()
    return out


def _random_moe(moe, gd, gr, vocab: int, shared_codes=(118, 123)) -> None:
    """Random parameters in the loaded layout, before post-load processing: this
    rank's shards (``gd``, a device generator) of the MXFP4 routed experts
    (random nibbles, E8M0 scales) and the FP8 shared expert (one E8M0 a
    128 x 128 block, 2^-9 .. 2^-5: half the routed experts' weight rms, FP4
    rms 2.9 at 2^-9 .. 2^-6, so the shared output is about a fifth of the
    routed one -- a fault in either shows over the other's roundings);
    the replicated router from ``gr``, seeded alike on every rank (a hash
    layer's table: distinct experts a token). ``shared_codes`` bounds the
    shared expert's E8M0 codes."""
    u8 = torch.uint8
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        w = torch.randn(lin.weight.shape, generator=gd, device=gd.device)
        lin.weight.data.copy_(w.clamp(-6, 6).to(lin.weight.dtype))
        sc = lin.weight_scale_inv.data.view(u8)
        sc.copy_(
            torch.randint(
                *shared_codes, sc.shape, generator=gd, device=gd.device, dtype=u8
            )
        )
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 122)):
        for w in ("w13", "w2"):
            t = getattr(e, f"{w}_{name}").data.view(u8)
            t.copy_(
                torch.randint(lo, hi, t.shape, generator=gd, device=gd.device, dtype=u8)
            )
    moe.gate.weight.data.copy_(
        torch.randn(moe.gate.weight.shape, generator=gr, device=gr.device) * 0.02
    )
    if moe.gate.e_score_correction_bias is not None:
        b = moe.gate.e_score_correction_bias
        b.data.copy_(torch.randn(b.shape, generator=gr, device=gr.device) * 0.02)
    if moe.gate.tid2eid is not None:
        experts, topk = moe.n_routed_experts, moe.n_activated_experts
        ids = torch.rand(vocab, experts, generator=gr, device=gr.device)
        moe.gate.tid2eid.data.copy_(ids.argsort(1)[:, :topk].to(torch.int32))


# ---------------------------------------------------------------- caches


def _paged_cache(num_blocks: int, block: int, device) -> torch.Tensor:
    """[blocks, block, 584] u8 over pages of round_up(block * 584, 576) bytes,
    the way vLLM binds a packed V4 cache."""
    page = -(-block * RECORD // ALIGN) * ALIGN
    raw = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)
    return torch.as_strided(raw, (num_blocks, block, RECORD), (page, RECORD, 1))


def _clone_cache(cache: torch.Tensor) -> torch.Tensor:
    raw = torch.as_strided(cache, (cache.shape[0] * cache.stride(0),), (1,)).clone()
    return torch.as_strided(raw, cache.shape, cache.stride())


def _rope(kv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """GPT-J RoPE of the last 64 dims (fp32 math) of rows [n, 512] or heads
    [n, h, 512]."""
    cs = cos_sin[positions.long()]
    if kv.dim() == 3:
        cs = cs[:, None]
    cos, sin = cs[..., :32], cs[..., 32:]
    x = kv.float()
    e, o = x[..., NOPE::2].clone(), x[..., NOPE + 1 :: 2].clone()
    x[..., NOPE::2] = e * cos - o * sin
    x[..., NOPE + 1 :: 2] = e * sin + o * cos
    return x


def _write_records(cache, block: int, slots, kv):
    """fp8_ds_mla records of roped bf16 kv rows at ``slots``: the 448 nope dims
    in e4m3 with a UE8M0 per 64, the rope dims in bf16, as vLLM's insert."""
    nope = kv.float()[:, :NOPE].view(-1, 7, 64)
    exp = torch.ceil(torch.log2(nope.abs().amax(-1).clamp_min(1e-4) / 448.0))
    q = (nope * torch.exp2(-exp)[..., None]).clamp(-448, 448).reshape(-1, NOPE)
    data = torch.cat(
        [
            q.to(torch.float8_e4m3fn).view(torch.uint8),
            kv[:, NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8),
        ],
        1,
    )
    scales = torch.zeros(kv.shape[0], 8, dtype=torch.uint8, device=kv.device)
    scales[:, :7] = (exp + 127).clamp(0, 255).to(torch.uint8)
    flat, d0, s0 = _record_offsets(cache, block, slots)
    ar = torch.arange(DATA, device=kv.device)
    flat[(d0[:, None] + ar).reshape(-1)] = data.reshape(-1)
    ar = torch.arange(8, device=kv.device)
    flat[(s0[:, None] + ar).reshape(-1)] = scales.reshape(-1)


def _record_offsets(cache, block: int, slots):
    n = cache.untyped_storage().nbytes() - cache.storage_offset()
    flat = torch.as_strided(cache, (n,), (1,))
    slots = slots.to(cache.device).long()
    b, o = slots // block, slots % block
    return (
        flat,
        b * cache.stride(0) + o * DATA,
        b * cache.stride(0) + block * DATA + o * 8,
    )


def _read_records(cache, block: int, slots) -> torch.Tensor:
    """The fp32 key rows of the records at ``slots``."""
    flat, d0, s0 = _record_offsets(cache, block, slots)
    data = flat[(d0[:, None] + torch.arange(DATA, device=cache.device)).reshape(-1)]
    data = data.view(-1, DATA)
    sc = flat[(s0[:, None] + torch.arange(7, device=cache.device)).reshape(-1)]
    nope = data[:, :NOPE].contiguous().view(torch.float8_e4m3fn).float().view(-1, 7, 64)
    nope = nope * torch.exp2(sc.view(-1, 7).float() - 127)[..., None]
    rope = data[:, NOPE:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope.reshape(-1, NOPE), rope], 1)


def _normed_kv(n: int, kv_norm, g, device):
    x = torch.randn(n, HEAD_DIM, generator=g).to(device)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * kv_norm.float()
    return x.to(torch.bfloat16)


def _decode_step(ratio: int, R: int, T: int, kv_norm, cos_sin, seed: int):
    """R requests x T tokens past random contexts: the step's metadata as
    vLLM's builders lay it out, its window rows and the compressed cache
    filled; a CSA layer's random top-k a token (request-local indices), an HCA
    layer's every compressed row (slots, through the block table: vLLM's
    c128a table)."""
    g = torch.Generator().manual_seed(seed)
    dev = kv_norm.device
    ctx = [int(c) for c in torch.randint(3000, 40000, (R,), generator=g)]
    M = R * T
    spans = [
        (max(0, c - WINDOW - T) // SWA_BLOCK, (c + T - 1) // SWA_BLOCK) for c in ctx
    ]
    n_swa = 1 + sum(hi - lo + 1 for lo, hi in spans)
    perm = (torch.randperm(n_swa - 1, generator=g) + 1).tolist()
    bt = torch.zeros(R, max(hi for _, hi in spans) + 1, dtype=torch.int64)
    for r, (lo, hi) in enumerate(spans):
        for lb in range(lo, hi + 1):
            bt[r, lb] = perm.pop()
    swa = _paged_cache(n_swa, SWA_BLOCK, dev)

    def slot(r, p):
        return int(bt[r, p // SWA_BLOCK]) * SWA_BLOCK + p % SWA_BLOCK

    pos = [ctx[r] + t for r in range(R) for t in range(T)]
    t2r = [r for r in range(R) for _ in range(T)]
    swa_idx = torch.full((M, WINDOW), -1, dtype=torch.int32)
    swa_len = torch.zeros(M, dtype=torch.int32)
    for i in range(M):
        ps = range(max(pos[i] - WINDOW + 1, 0), pos[i] + 1)
        swa_idx[i, : len(ps)] = torch.tensor([slot(t2r[i], p) for p in ps])
        swa_len[i] = len(ps)
    hist = [
        (r, p) for r in range(R) for p in range(max(0, ctx[r] - WINDOW - T), ctx[r])
    ]
    hp = torch.tensor([p for _, p in hist], device=dev)
    kv = _rope(_normed_kv(len(hist), kv_norm, g, dev), hp, cos_sin)
    _write_records(swa, SWA_BLOCK, torch.tensor([slot(r, p) for r, p in hist]), kv)

    entries = CACHE_BLOCK // ratio
    lens = [(c + T) // ratio for c in ctx]
    nblk = [-(-n // entries) for n in lens]
    perm = (torch.randperm(sum(nblk), generator=g) + 1).tolist()
    cbt = torch.zeros(R, max(nblk), dtype=torch.int32)
    for r in range(R):
        for b in range(nblk[r]):
            cbt[r, b] = perm.pop()
    comp = _paged_cache(1 + sum(nblk), entries, dev)
    for r in range(R):
        idx = torch.arange(lens[r])
        slots = cbt[r, idx // entries].long() * entries + idx % entries
        kv = _rope(_normed_kv(lens[r], kv_norm, g, dev), (idx * ratio).to(dev), cos_sin)
        _write_records(comp, entries, slots, kv)
    if ratio == 4:
        topk = torch.full((M, TOPK), -1, dtype=torch.int32)
        for i in range(M):
            cand = (pos[i] + 1) // ratio
            n = min(cand, TOPK)
            topk[i, :n] = torch.randperm(cand, generator=g)[:n].to(torch.int32)
    else:
        n_max = max((p + 1) // ratio for p in pos)
        width = max(128, 1 << (n_max - 1).bit_length())
        topk = torch.full((M, width), -1, dtype=torch.int32)
        for i in range(M):
            j = torch.arange((pos[i] + 1) // ratio)
            topk[i, : len(j)] = cbt[t2r[i], j // entries] * entries + j % entries
    return dict(
        M=M,
        positions=torch.tensor(pos, dtype=torch.int64, device=dev),
        slot_mapping=torch.tensor(
            [slot(t2r[i], pos[i]) for i in range(M)], dtype=torch.int64, device=dev
        ),
        token_to_req=torch.tensor(t2r, dtype=torch.int32, device=dev),
        swa_indices=swa_idx.to(dev),
        swa_lens=swa_len.to(dev),
        swa_cache=swa,
        topk=topk.to(dev),
        comp_cache=comp,
        comp_block_table=cbt.to(dev),
        comp_entries=entries,
    )


def _keys(s: dict, i: int, ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Token i's (compressed slots, window slots) of the step."""
    top = s["topk"][i]
    top = top[top >= 0].long()
    if ratio == 4:
        e, r = s["comp_entries"], s["token_to_req"][i]
        top = s["comp_block_table"][r, top // e].long() * e + top % e
    return top, s["swa_indices"][i, : s["swa_lens"][i]].long()


def _fp8_block(n: int, k: int, g, device):
    """A random FP8 weight [n, k] e4m3 and its 128 x 128 block scales as vLLM
    holds them after loading: the rows padded as on ROCm, fp32 powers of two."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _maybe_pad_fp8_weight,
    )

    w = torch.randn(n, k, generator=g).clamp(-6, 6).to(torch.float8_e4m3fn)
    s = torch.randint(116, 121, (n // 128, k // 128), generator=g).float()
    return _maybe_pad_fp8_weight(w.to(device)), torch.exp2(s - 127).to(device)


def _deq(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


class _ExactMoE:
    """A MoE layer's output on this rank in fp32 from its weights as loaded
    (before post-load processing reshuffles them): MXFP4 / FP8 dequantized,
    activations unquantized, vLLM's routing."""

    FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)

    def __init__(self, moe):
        e = moe.experts.routed_experts
        u8 = torch.uint8
        self.moe = moe
        self.w13 = (
            e.w13_weight.data.view(u8).clone(),
            e.w13_weight_scale.data.view(u8).clone(),
        )
        self.w2 = (
            e.w2_weight.data.view(u8).clone(),
            e.w2_weight_scale.data.view(u8).clone(),
        )
        self.lut = torch.tensor(self.FP4, device=e.w13_weight.device)

    def shared(self, sh) -> None:
        """The shared expert's weights, after their scales' upcast."""

        def deq(lin):
            w, sc = lin.weight.float(), lin.weight_scale_inv.float()
            return w * sc.repeat_interleave(128, 0).repeat_interleave(128, 1)

        self.sgu, self.sd = deq(sh.gate_up_proj), deq(sh.down_proj)

    def _experts(self, raw, ids):
        w, sc = raw[0][ids], raw[1][ids]
        lo, hi = self.lut[(w & 15).long()], self.lut[(w >> 4).long()]
        v = torch.stack([lo, hi], -1).flatten(-2)
        return v * torch.exp2(sc.float() - 127).repeat_interleave(32, -1)

    @staticmethod
    def _swiglu(gu):
        n = gu.shape[-1] // 2
        g, u = gu[..., :n].clamp(max=10), gu[..., n:].clamp(-10, 10)
        return torch.nn.functional.silu(g) * u

    def rules(self, x: torch.Tensor, ids: torch.Tensor, fp32_mid: bool):
        """The routed experts as the A8W4 rules compute them, in f64 from the
        loaded weights, the shared expert exact: x as MXFP8 (the ceil code of
        RN(amax RN(1 / 448)), AITER's activation quant and the kernels'), each
        pick's gate / up GEMM and clamped SwiGLU, MXFP8 again -- of the SwiGLU
        rounded to bf16 (the kernels; AITER's unfused stage 1, which writes
        bf16) or of its fp32 value (``fp32_mid``: AITER's tuned FlyDSL stage 1,
        which quantizes in its epilogue) -- and the down GEMM, f64 routing
        weights. ``ids``: [M, topk] picks -> (output, shared) on this rank."""
        from . import mono_reference as R
        from .test_dsv4_mono_io import _rule_ceil

        def mxfp8(t):
            q, codes = R.mx_quantize(t, _rule_ceil)
            return (q.unflatten(-1, (-1, 32)) * R.e8m0(codes)[..., None]).flatten(-2)

        moe, x = self.moe, x.double()
        scores = torch.nn.functional.softplus(x @ moe.gate.weight.double().T).sqrt()
        w = scores.gather(1, ids.long())
        w = w / w.sum(1, keepdim=True) * moe.routed_scaling_factor
        shared = self._swiglu(x @ self.sgu.double().T) @ self.sd.double().T
        out = shared.clone()
        xq = mxfp8(x)
        for t in range(x.shape[0]):
            w13 = self._experts(self.w13, ids[t].long()).double()
            act = self._swiglu(torch.einsum("knh,h->kn", w13, xq[t]))
            act = mxfp8(act if fp32_mid else R.bf16_rne(act))
            w2 = self._experts(self.w2, ids[t].long()).double()
            out[t] += (w[t, :, None] * torch.einsum("khn,kn->kh", w2, act)).sum(0)
        return out, shared

    def __call__(self, x: torch.Tensor, route: torch.Tensor | None) -> torch.Tensor:
        moe, x = self.moe, x.float()
        scores = torch.nn.functional.softplus(x @ moe.gate.weight.float().T).sqrt()
        if route is None:
            bias = moe.gate.e_score_correction_bias
            ids = (scores + bias).topk(moe.n_activated_experts, dim=1).indices
        else:
            ids = route.long()
        w = scores.gather(1, ids)
        w = w / w.sum(1, keepdim=True) * moe.routed_scaling_factor
        out = self._swiglu(x @ self.sgu.T) @ self.sd.T
        for t in range(x.shape[0]):
            w13, w2 = self._experts(self.w13, ids[t]), self._experts(self.w2, ids[t])
            act = self._swiglu(torch.einsum("knh,h->kn", w13, x[t]))
            out[t] += (w[t, :, None] * torch.einsum("khn,kn->kh", w2, act)).sum(0)
        return out


class _ExactAttention:
    """A layer's attention on this rank in fp32 from its normed input: the
    weights dequantized, the keys as a cache holds them (the window's rows of
    the step written by the path under test), wo_b's partial out."""

    def __init__(self, w: dict, eps: float, sink: torch.Tensor):
        self.wqkv = _deq(w["wqkv"], w["wqkv_s"])
        self.wq_b = _deq(w["wq_b"], w["wq_b_s"])
        self.wo_a = _deq(w["wo_a"], w["wo_a_s"])
        self.wo_b = _deq(w["wo_b"], w["wo_b_s"])
        self.q_norm, self.kv_norm = w["q_norm"].float(), w["kv_norm"].float()
        self.eps, self.sink = eps, sink.float()

    def _rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def kv(self, x, positions, cos_sin) -> torch.Tensor:
        """The KV rows of a step's tokens: kv_norm, then RoPE."""
        kv = (x.float() @ self.wqkv.T)[:, Q_RANK:]
        return _rope(self._rms(kv, self.kv_norm), positions, cos_sin)

    def __call__(self, x, s: dict, ratio: int, swa_cache, cos_sin) -> torch.Tensor:
        M, pos = x.shape[0], s["positions"]
        qr_kv = x.float() @ self.wqkv.T
        q = (self._rms(qr_kv[:, :Q_RANK], self.q_norm) @ self.wq_b.T).view(M, -1, 512)
        q = _rope(self._rms(q, 1.0), pos, cos_sin)
        heads = q.shape[1]
        o = torch.empty(M, heads, HEAD_DIM, device=x.device)
        for i in range(M):
            top, win = _keys(s, i, ratio)
            k = torch.cat(
                [
                    _read_records(s["comp_cache"], s["comp_entries"], top),
                    _read_records(swa_cache, SWA_BLOCK, win),
                ]
            )
            sc = (q[i] @ k.T) * HEAD_DIM**-0.5
            m = torch.maximum(sc.amax(1), self.sink[:heads])
            p = torch.exp(sc - m[:, None])
            den = p.sum(1) + torch.exp(self.sink[:heads] - m)
            o[i] = (p @ k) / den[:, None]
        # inverse RoPE of each head's last 64 dims
        cs = cos_sin[pos.long()][:, None]
        cos, sin = cs[..., :32], cs[..., 32:]
        e, od = o[..., NOPE::2].clone(), o[..., NOPE + 1 :: 2].clone()
        o[..., NOPE::2] = e * cos + od * sin
        o[..., NOPE + 1 :: 2] = od * cos - e * sin
        groups = self.wo_a.shape[0] // O_RANK
        og = o.view(M, groups, -1)
        z = torch.einsum("mgk,gnk->mgn", og, self.wo_a.view(groups, O_RANK, -1))
        return z.reshape(M, -1) @ self.wo_b.T


def _route_check(moe, x, route, *picks, tag=None):
    """The f64 route of the MoE input rows ``x`` (a hash layer's ``route``)
    against each path's ``picks`` [M, topk]: as sets, equal on every row
    whose 6th / 7th margin of score + bias clears both paths' score errors --
    the kernels' logits (``test_dsv4_mono_io``'s router bound, 242 u |x| |w|),
    vLLM's fp32 router GEMM in any order (HIDDEN u |x| |w|), through
    sqrt(softplus)'s slope, and the bias add. -> (f64 picks, rows both paths
    route as them)."""
    k = moe.n_activated_experts
    xd, gw = x.double(), moe.gate.weight.double()
    if route is not None:
        ids = route.long()
    else:
        logits = xd @ gw.T
        sel = torch.nn.functional.softplus(logits).sqrt()
        sel = sel + moe.gate.e_score_correction_bias.double()
        top = sel.topk(k + 1, dim=1)
        ids = top.indices[:, :k]
        dl = (33 * HIDDEN // 1024 + 11 + HIDDEN) * U * (xd.abs() @ gw.abs().T)
        slope = torch.sigmoid(logits) / (
            2 * torch.nn.functional.softplus(logits).sqrt()
        )
        tie = 2 * (slope * dl + 4 * U * sel.abs()).amax(1)
        margin = top.values[:, k - 1] - top.values[:, k]
    keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for got in picks:
        same = (got.long().sort(1).values == ids.sort(1).values).all(1)
        if route is not None:
            assert same.all(), (tag, "hash route")
        else:
            assert (margin[~same] < tie[~same]).all(), (tag, "route", margin, tie)
        keep &= same
    return ids, keep


def _attn_output(runner, epoch: int, M: int, rank: int) -> torch.Tensor:
    """The wo_b partials every rank pushed to this rank's ATTN region in the
    launch pair of ``epoch``, summed: [M, hidden] fp32. On two hops a rank
    holds its own columns' partials only: the ranks' columns gathered."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.models.deepseek_v4.amd.mono.layer import peer_half_bytes

    half = peer_half_bytes(runner.tp)
    n = runner.tp * M * HIDDEN // 2
    base = (epoch & 1) * half
    pairs = runner.peer.bytes[base : base + n * 8].view(torch.int32).view(n, 2)
    parts = pairs[:, 0].contiguous().view(torch.bfloat16).view(runner.tp, M, HIDDEN)
    out = parts.float().sum(0)
    if runner.two_hop(M):
        cols = slice(rank * HIDDEN // runner.tp, (rank + 1) * HIDDEN // runner.tp)
        own = torch.zeros_like(out)
        own[:, cols] = out[:, cols]
        out = tensor_model_parallel_all_reduce(own)
    return out


@ray.remote(num_gpus=1, max_calls=1)
def _mono_layer_numerics(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
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
        _check_layer(rank, tp_size, device)


def _check_layer(rank: int, tp: int, device) -> None:
    from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale
    from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant

    from vllm._aiter_ops import rocm_aiter_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v4.amd.mono.dims import INDEX_TOPK
    from vllm.models.deepseek_v4.amd.mono.layer import scratch_layout
    from vllm.models.deepseek_v4.amd.mono.runner import (
        AttnWeights,
        MonoLayerWeights,
        MonoRunner,
    )
    from vllm.models.deepseek_v4.amd.mono_decode import _hca_width, _mx_scales
    from vllm.models.deepseek_v4.amd.rocm import (
        compute_global_topk_ragged_indices_and_indptr,
    )
    from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tp,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=65536,
        trust_remote_code=True,
    ).create_engine_config()
    cfg = vc.model_config.hf_config
    eps = cfg.rms_norm_eps
    bf = torch.bfloat16
    g = torch.Generator().manual_seed(1000)
    gs = torch.Generator().manual_seed(1001 + rank)
    H, G = N_HEADS // tp, O_GROUPS // tp
    # the HCA key tables: the deployment's (64K context: 512, within the top-k)
    # and a long context's (wider than the top-k: ``bounded``)
    hca_dep, hca_long = _hca_width(vc), 2048
    assert hca_dep <= INDEX_TOPK < hca_long
    runner = MonoRunner(tp, rank, get_tp_group().cpu_group, device, max_topk=hca_long)
    with set_current_vllm_config(vc):
        fused = MHCFusedPostPreOp()

    def random_moe(layer):
        """A layer's MoE (layer 0's routes by hash) and its exact reference."""
        with set_current_vllm_config(vc), set_default_torch_dtype(bf), device:
            moe = DeepseekV4MoE(vc, prefix=f"model.layers.{layer}.ffn")
        _random_moe(
            moe,
            torch.Generator(device=device).manual_seed(2001 + rank + 10 * layer),
            torch.Generator(device=device).manual_seed(2000 + layer),
            cfg.vocab_size,
        )
        exact = _ExactMoE(moe)  # the weights as loaded, before processing
        process_weights_after_loading(moe, vc.model_config, device)
        exact.shared(moe.shared_experts)
        moe.shared_experts.prepare_gateup_preshuffle()
        assert moe.shared_experts._gateup_scale is not None
        return moe, exact

    moes = {layer: random_moe(layer) for layer in (3, 0)}

    def f32(*shape, s=1.0, gen=g):
        return (torch.randn(*shape, generator=gen) * s).to(device)

    mix = (2 + HC) * HC
    seams = {
        sub: dict(
            fn=f32(mix, HC * HIDDEN, s=0.01),
            scale=f32(3, s=0.5).abs() + 0.5,
            base=f32(mix, s=0.1),
            norm=(1 + f32(HIDDEN, s=0.1)).to(bf),
        )
        for sub in ("attn", "ffn")
    }

    def seam(sub, x, res, post, comb):
        sm = seams[sub]
        with set_current_vllm_config(vc):
            return fused(
                x, res, post, comb, sm["fn"], sm["scale"], sm["base"], eps,
                cfg.hc_eps, cfg.hc_eps, 2.0, cfg.hc_sinkhorn_iters,
                norm_weight=sm["norm"], norm_eps=eps,
            )  # fmt: skip

    def moe_out(moe, x, ids):
        with set_forward_context(None, vc):
            return moe(x.clone(), ids)

    # the attention's weights, as vLLM holds them after loading and its
    # preshuffle (wqkv and wo_b shuffled in place, scales fp32)
    w = {}
    w["wqkv"], w["wqkv_s"] = _fp8_block(Q_RANK + HEAD_DIM, HIDDEN, g, device)
    w["wq_b"], w["wq_b_s"] = _fp8_block(H * HEAD_DIM, Q_RANK, gs, device)
    w["wo_a"], w["wo_a_s"] = _fp8_block(G * O_RANK, H // G * HEAD_DIM, gs, device)
    w["wo_b"], w["wo_b_s"] = _fp8_block(HIDDEN, G * O_RANK, gs, device)
    w["q_norm"] = (1 + f32(Q_RANK, s=0.1)).to(bf)
    w["kv_norm"] = (1 + f32(HEAD_DIM, s=0.1)).to(bf)
    sink = f32(H, gen=gs)
    wqkv_sh = ops.shuffle_weight(w["wqkv"], layout=(16, 16))
    wo_b_sh = ops.shuffle_weight(w["wo_b"], layout=(16, 16))
    wo_a_e8m0 = _mx_scales(w["wo_a_s"])[::4, ::4].contiguous()
    exact_attn = _ExactAttention(w, eps, sink)

    steps = []
    # (compress ratio, requests, tokens a request, key table width, MoE layer)
    for ratio, R, T, width, layer in (
        (4, 1, 7, 0, 3),
        (128, 8, 7, hca_long, 0),
        (128, 2, 7, hca_dep, 3),
        (4, 4, 7, 0, 0),
    ):
        with set_current_vllm_config(VllmConfig()):
            rope = build_deepseek_v4_rope(
                cfg, head_dim=HEAD_DIM, rope_head_dim=ROPE,
                max_position_embeddings=cfg.max_position_embeddings,
                compress_ratio=ratio,
            )  # fmt: skip
        cos_sin = rope.cos_sin_cache.to(device=device, dtype=torch.float32)
        cos_bf, sin_bf = (c.contiguous() for c in cos_sin.to(bf).chunk(2, dim=-1))
        s = _decode_step(ratio, R, T, w["kv_norm"], cos_sin, seed=7 + ratio + R)
        M = s["M"]
        tag = f"ratio {ratio}, {R} x {T}, width {width}, MoE layer {layer}"
        hca = ratio == 128
        moe, exact_moe = moes[layer]
        sh, e = moe.shared_experts, moe.experts.routed_experts
        mw = MonoLayerWeights(
            attn=AttnWeights(
                layer_id=3, wqkv=wqkv_sh, wqkv_scale=_mx_scales(w["wqkv_s"]),
                q_norm=w["q_norm"], kv_norm=w["kv_norm"], wq_b=w["wq_b"],
                wq_b_scale=_mx_scales(w["wq_b_s"]), wo_a=w["wo_a"],
                wo_a_scale=_mx_scales(w["wo_a_s"]), wo_b=wo_b_sh,
                wo_b_scale=_mx_scales(w["wo_b_s"]), attn_sink=sink,
                cos_sin=cos_sin, ratio=ratio,
            ),
            hc_attn_fn=seams["attn"]["fn"], hc_attn_scale=seams["attn"]["scale"],
            hc_attn_base=seams["attn"]["base"], attn_norm=seams["attn"]["norm"],
            hc_ffn_fn=seams["ffn"]["fn"], hc_ffn_scale=seams["ffn"]["scale"],
            hc_ffn_base=seams["ffn"]["base"], ffn_norm=seams["ffn"]["norm"],
            gate_w=moe.gate.weight, bias=moe.gate.e_score_correction_bias,
            w13=e.w13_weight, w13_s=e.w13_weight_scale, w2=e.w2_weight,
            w2_s=e.w2_weight_scale, sgu=sh.gate_up_proj.weight,
            sgu_s=_mx_scales(sh._gateup_scale), sw2=sh.down_proj.weight,
            sw2_s=_mx_scales(sh.down_proj.weight_scale_inv),
        )  # fmt: skip
        mw.attn.check(runner.d)
        mw.check(runner.d)

        # the layer's inputs at the attention seam (the same on every rank)
        gi = torch.Generator().manual_seed(11 + ratio)
        res0 = torch.randn(M, HC, HIDDEN, generator=gi)
        res0 = (res0 * torch.rsqrt(res0.pow(2).mean(-1, keepdim=True))).to(bf)
        res0 = res0.to(device)
        x0 = (torch.randn(M, HIDDEN, generator=gi) * 0.5).to(bf).to(device)
        comb0 = torch.rand(M, HC, HC, generator=gi) + 0.1
        for _ in range(20):
            comb0 = comb0 / comb0.sum(-1, keepdim=True)
            comb0 = comb0 / comb0.sum(-2, keepdim=True)
        comb0 = comb0.to(device)
        post0 = (torch.sigmoid(torch.randn(M, HC, 1, generator=gi)) * 2).to(device)
        ids = torch.randint(0, cfg.vocab_size, (M,), generator=gi).to(device)
        hashed = moe.gate.tid2eid is not None
        route = moe.gate.tid2eid[ids].contiguous() if hashed else None
        swa_v, swa_m = s["swa_cache"], _clone_cache(s["swa_cache"])

        # ---- the whole layer by the kernels (K1 + K2); then each of vLLM's
        # stages on the kernels' input of that stage
        epoch = int(runner.epoch[0])
        o = runner.forward(
            mw, x0, res0, post0, comb0, s["positions"], s["slot_mapping"],
            swa_m, s["swa_indices"], s["swa_lens"], s["token_to_req"],
            topk_indices=s["topk"], comp_cache=s["comp_cache"],
            select=lambda normed, qr: None,
            comp_block_table=s["comp_block_table"], route_ids=route,
            topk=width, global_keys=hca,
        )  # fmt: skip
        torch.accelerator.synchronize()
        res_mid, normed = runner.res_mid[:M].clone(), runner.normed[:M].clone()
        post_a, comb_a = runner.post_a[:M].clone(), runner.comb_a[:M].clone()
        a_m = _attn_output(runner, epoch, M, rank)
        layout, scratch = scratch_layout(M, tp, width), runner.scratch(M, width)
        off = layout["normed"][0]
        xk = scratch[off : off + M * HIDDEN * 2].view(bf).view(M, HIDDEN).clone()
        off = layout["route"][0]
        n = moe.n_activated_experts
        picks = scratch[off : off + M * 2 * n * 8].view(torch.int32).view(M, 2 * n, 2)
        assert (picks[..., 1] == 2 * epoch + 2).all(), tag
        route_m = picks[:, :n, 0].long()
        p = {
            f"hc_{sub}_{name}": seams[sub][name]
            for sub in seams
            for name in ("fn", "scale", "base")
        }
        p.update({f"{sub}_norm": seams[sub]["norm"] for sub in seams})
        p["eps"] = (eps, cfg.hc_eps, cfg.hc_sinkhorn_iters, eps)
        within, units = {}, {}

        # the attention seam
        res, post, comb, xa = seam("attn", x0, res0, post0, comb0)
        for k, r in seam_parity(p, "attn", x0, res0, post0, comb0, False,
                                (res_mid, post_a, comb_a, normed),
                                (res, post, comb, xa)).items():  # fmt: skip
            within[f"attention seam {k}"] = r

        # vLLM's attention on the kernels' normed rows, into its own cache copy
        x8, xs = ops.group_fp8_quant(normed, transpose_scale=True)
        qr_kv = ops.gemm_a8w8_blockscale_bpreshuffle(x8, wqkv_sh, xs, w["wqkv_s"], bf)
        qr8, qrs, kvn = ops.fused_qk_rmsnorm_group_quant(
            qr_kv[:, :Q_RANK], w["q_norm"], eps, qr_kv[:, Q_RANK:], w["kv_norm"], eps,
            group_size=128, transpose_scale=False,
        )  # fmt: skip
        q = ops.gemm_a8w8_blockscale(qr8, w["wq_b"], qrs, w["wq_b_s"], [128, 128], bf)
        swa2d = swa_v.view(swa_v.shape[0], -1)
        q_roped = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.view(M, H, HEAD_DIM), kvn, swa2d, s["slot_mapping"], s["positions"],
            cos_sin, H, eps, SWA_BLOCK, True, False, True, False,
        )  # fmt: skip
        swa_ind, swa_ptr = build_ragged_indices_from_dense(
            s["swa_indices"].reshape(M, -1), s["swa_lens"]
        )
        if hca:
            lens = (s["topk"] >= 0).sum(1).to(torch.int32)
            top_ind, top_ptr = build_ragged_indices_from_dense(s["topk"], lens)
        else:
            valid = torch.ones(M, dtype=torch.bool, device=device)
            top_ind, top_ptr, _ = compute_global_topk_ragged_indices_and_indptr(
                s["topk"], s["token_to_req"], s["comp_block_table"],
                s["comp_entries"], valid,
            )  # fmt: skip
        attn = torch.empty(M, H, HEAD_DIM, dtype=bf, device=device)
        ops.triton_sparse_mla_fwd(
            q_roped, swa_v, attn, HEAD_DIM**-0.5, swa_ptr, swa_ind,
            attn_sink=sink, extra_kv_buffer=s["comp_cache"],
            extra_kv_indptr=top_ptr, extra_kv_indices=top_ind,
        )  # fmt: skip
        o8, os_ = inverse_rope_group_quant(
            attn, s["positions"], cos_bf, sin_bf, num_groups=G, quant_group_size=128
        )
        z = batched_gemm_a8w8_mxscale(
            o8, w["wo_a"].view(G, O_RANK, -1), os_,
            wo_a_e8m0.view(G, O_RANK // 128, -1), dtype=bf,
        ).flatten(1)  # fmt: skip
        z8, zs = ops.group_fp8_quant(z.contiguous(), transpose_scale=True)
        part = ops.gemm_a8w8_blockscale_bpreshuffle(z8, wo_b_sh, zs, w["wo_b_s"], bf)
        a = tensor_model_parallel_all_reduce(part.clone())

        # each path's KV records of the step and attention output against the
        # exact ones of the shared input (each over its own cache), and the two
        # outputs against each other
        kv = exact_attn.kv(normed, s["positions"], cos_sin)
        for who, cache in (("kernels", swa_m), ("vllm", swa_v)):
            err = _rows_rel(_read_records(cache, SWA_BLOCK, s["slot_mapping"]), kv)
            units[f"KV records, {who}"] = err / (KV_GRIDS * E4M3_RN)
        expect = math.sqrt(ATTN_GRIDS) * E4M3_RN
        for who, got, cache in (("kernels", a_m, swa_m), ("vllm", a, swa_v)):
            ref = tensor_model_parallel_all_reduce(
                exact_attn(normed, s, ratio, cache, cos_sin)
            )
            units[f"attention, {who}"] = _rows_rel(got, ref) / expect
        units["attention parity"] = _rows_rel(a_m, a) / (expect * math.sqrt(2))

        # the FFN seam on the kernels' attention output (bf16, as it reads it)
        a_m = a_m.to(bf)
        res2, post2, comb2, xf = seam("ffn", a_m, res_mid, post_a.view(M, HC, 1),
                                      comb_a)  # fmt: skip
        for k, r in seam_parity(p, "ffn", a_m, res_mid, post_a, comb_a, False,
                                (o[1], o[2], o[3], xk),
                                (res2, post2, comb2, xf)).items():  # fmt: skip
            within[f"FFN seam {k}"] = r

        # the MoE on the kernels' FFN input
        router, select = moe.experts.router, moe.experts.router.select_experts
        picked: dict = {}

        def routed(*args, select=select, picked=picked, **kw):
            picked["w"], picked["ids"] = select(*args, **kw)
            return picked["w"], picked["ids"]

        router.select_experts = routed
        try:
            mo = moe_out(moe, xk, ids)
        finally:
            del router.select_experts
        route_v = picked["ids"].long()
        rids, keep = _route_check(moe, xk, route, route_m, route_v, tag=tag)
        want = {}
        for who, fp32_mid in (("kernels", False), ("vllm", True)):
            y, shared = exact_moe.rules(xk, rids, fp32_mid)
            want[who] = tensor_model_parallel_all_reduce(y.float()).double()
            shared = tensor_model_parallel_all_reduce(shared.float()).double()
        # each path against its own rules: what is left is the shared expert's
        # two MXFP8 / FP8 roundings (the input's reaching both SwiGLU operands:
        # sqrt(3) roundings' worth) and the bf16 roundings of the partial sums
        # (``MOE_BF16``)
        size = want["kernels"].norm(dim=1)
        rest = (3 * (E4M3_RN * shared.norm(dim=1)) ** 2
                + MOE_BF16 * (BF16_RN * size) ** 2).sqrt() / size  # fmt: skip
        for who, got in (("kernels", o[0]), ("vllm", mo)):
            units[f"MoE, {who}"] = (_rows_rel(got, want[who]) / rest)[keep]
        # the two rules' difference: the SwiGLU's MXFP8 flips, relative to the
        # routed output, aggregated over the step's tokens (each token's few
        # hundred flips are heavy-tailed)
        d = (want["kernels"] - want["vllm"])[keep]
        y_routed = (want["vllm"] - shared)[keep]
        units["MoE SwiGLU flips"] = (d.norm() / y_routed.norm() / MID_FLIP).reshape(1)

        for k, r in within.items():
            assert r <= 1.0, (tag, rank, k, r)
        for k, r in units.items():
            assert r.max().item() <= UNITS_OK, (tag, rank, k, r.max().item())
        if rank == 0:
            print(tag, {k: round(r, 3) for k, r in within.items()},
                  {k: round(r.max().item(), 3) for k, r in units.items()})  # fmt: skip

        outs = runner._outs(M, res0)
        meta = dict(
            topk_indices=s["topk"], comp_cache=s["comp_cache"],
            select=lambda n, q: None, comp_block_table=s["comp_block_table"],
            route_ids=route, topk=width, global_keys=hca,
        )  # fmt: skip
        args = (s["positions"], s["slot_mapping"], swa_m, s["swa_indices"],
                s["swa_lens"], s["token_to_req"])  # fmt: skip

        def step(mw=mw, x0=x0, res0=res0, post0=post0, comb0=comb0,
                 args=args, meta=meta, outs=outs):  # fmt: skip
            runner.forward(mw, x0, res0, post0, comb0, *args, **meta, outs=outs)
            return list(outs)

        steps.append(step)

    # the hand-off protocol under back-to-back launches: the layers' widths and
    # key tables alternating in one graph, every replay bit for bit the first's
    outs = [t for step in steps for t in step()]
    torch.accelerator.synchronize()
    ref_outs = [t.clone() for t in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in steps:
            step()
    for i in range(REPLAYS):
        graph.replay()
        if i % 50 == 49 or i == REPLAYS - 1:
            torch.accelerator.synchronize()
            bad = [
                j
                for j, (t, r) in enumerate(zip(outs, ref_outs))
                if not torch.equal(t, r)
            ]
            assert not bad, f"replay {i}: outputs {bad} differ from the first run"


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
def test_mono_decode_layer_matches_vllm(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, TP, 1, _mono_layer_numerics)
