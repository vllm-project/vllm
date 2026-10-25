# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the DeepSeek-V4.1 mono decode kernels at TP2 and TP4 on CDNA4:
every fused stage against a float64 reference in plain torch
(``mono_reference``), on the stage's own input as the kernels left it in their
scratch and peer memory, so a failure names the stage. Random weights in the
checkpoint's formats, which the reference reads as loaded and the kernels after
vLLM's post-load processing.

- K1: the attention seam (fold, mixes, collapse, norm, MXFP8), the key table,
  wqkv, the q / kv norms with the q quant and the KV record, wq_b with RoPE.
- K2: the sparse attention with its sink, inverse RoPE and MXFP8, wo_a, wo_b,
  the attention all-reduce, the FFN seam, the MoE's input quants, router,
  routing, shared expert, routed gate / up and down, the MoE all-reduce.
- The FFN launch from a given unreduced wo_b output; the DSpark draft layers'
  runner (128 / 3 routing, the block's non-causal SWA rows).

Exact where the arithmetic is (data movement, the rank-order sums, the
quantization rules on the kernel's own input); the documented rounding chain
otherwise, with KAPPA for the hardware transcendentals and the scaled MFMA's
accumulation. The steps of the first test also run the ops vLLM runs for the
same step and compare against them.

On gfx942 at TP4, ``test_mono_decode_942_replays`` runs the composed runner on
the copies of the weights that the model makes at load, at steps of 6, 8 and 42
rows, checks the arch-independent stages exactly and replays the steps in one
CUDA graph. The real model's numerics on gfx942 are checked by
``test_dsv41_mono_layers.py``."""

from contextlib import contextmanager
from typing import NamedTuple

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from . import mono_reference as R

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
HIDDEN, Q_RANK, HEAD_DIM, NOPE, ROPE = 5120, 1280, 512, 448, 64
N_HEADS, O_GROUPS, O_RANK = 64, 8, 1024
WINDOW, TOPK, SWA_BLOCK, CACHE_BLOCK = 128, 512, 32, 128
RECORD, DATA, ALIGN = 584, 576, 576
REPLAYS = 300
HC = 4
DRAFTS = 5  # DSpark's num_speculative_tokens: a request's rows in a draft step
# contexts under the window and under a ratio's top-k
SHORT_CONTEXTS = (1, 7, 64, 127, 129, 300)
# the hard case's router: five experts always picked, then an exact tie for
# the sixth (one 64-expert chunk: the lower id wins)
TOP5, TIE = (5, 20, 37, 115, 127), (70, 100)


class Case(NamedTuple):
    """A decode step of R requests x T tokens on a layer of compress ratio
    ``ratio``. ``entry``: the residual K1 takes, "fold" (the previous layer's
    output folded in), "settled" (after an Engram write) or "embedding" (the
    first layer's). ``pad``: rows a FULL graph pads the step with (slot -1).
    ``short``: SHORT_CONTEXTS among the long contexts. ``index``: an index
    source, its top-k selected between K1 and K2. ``vllm``: also against the
    ops vLLM runs for the step. ``hard``: a zeroed 32-column group in each norm
    weight and the FFN norm's x6 (the gate / up clamps, router logits past 20),
    and the router's TOP5 and TIE."""

    ratio: int
    R: int
    T: int
    entry: str = "fold"
    pad: int = 0
    short: bool = False
    index: bool = False
    vllm: bool = False
    hard: bool = False


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of ``a`` against ``b``: unlike the cosine, it sees a
    wrong scale."""
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------- slow references

# Measured error constants (2x the maximum over 20 seeds, every case, TP2 and
# TP4), each relative to its stage's conditioning scale: the scaled-MFMA GEMVs
# (FP8 x FP8, the routed FP4 x FP8 gate / up and down) over R.blockmax_cond,
# "rstd" and "mix" relative, "act" of the f32 SiLU, "route" of the routing
# weights. The attention's is over sum_n p_n |v_n|.
KAPPA = dict(
    fp8=0.00011,
    fp4_ug=9.9e-06,
    fp4_down=0.014,
    rstd=0,
    mix=7.9e-06,
    act=0,
    route=5.6e-07,
)
# a-priori: a bf16 MFMA GEMV of K = 5120 accumulates exact products in fp32
KAPPA["logit"] = HIDDEN * 2.0**-24
# a-priori: p and each chunk's partial rounded to bf16 (2 u_bf16 of sum_n p_n |v_n|)
# and the f32 scores' rounding
KAPPA["attn"] = 2 * 2.0**-8 + 2.0**-16
F32_2 = 2.0**-22  # two f32 roundings, relative to the sum of the terms' sizes
FLOOR_1E10 = torch.tensor(1e-10).item()
FLOOR_1E4 = torch.tensor(1e-4).item()
RCP448 = torch.tensor(1 / 448).item()


def _code_aiter(amax):
    """AITER's MXFP8 code: ceil code of RN_f32(max(amax, 1e-10) RN_f32(1/448))."""
    return R.ceil_code(R.f32_rne(amax.clamp(min=FLOOR_1E10) * RCP448)).clamp(1, 254)


def _code_record(amax):
    """The KV record's UE8M0: ceil code of RN_f32(max(amax, 1e-4) / 448)."""
    return R.ceil_code(R.f32_rne(amax.clamp(min=FLOOR_1E4) / 448))


def _assert_stage(name: str, check) -> None:
    check(KAPPA[name])


def _le(tag: str, err: torch.Tensor, bound: float) -> None:
    worst = err.max().item()
    assert worst <= bound, f"{tag}: {worst:.4g} > {bound:.4g}"


def _first_bad(bad: torch.Tensor, *ts) -> str:
    i = tuple(bad.nonzero()[0].tolist())
    return f"{int(bad.sum())} of {bad.numel()}, first at {i}: " + ", ".join(
        f"{t[i].item():.9g}" for t in ts
    )


def _within(tag: str, got, lo, hi) -> None:
    got = got.double()
    bad = (got < lo) | (got > hi)
    assert not bad.any(), f"{tag}: " + _first_bad(bad, got, lo, hi)


def _span(ref, d):
    """The bf16 values a computation within ref +- d rounds to: [lo, hi]."""
    return R.bf16_rne(ref - d), R.bf16_rne(ref + d)


def _widen(lo, hi, rel):
    m = torch.maximum(lo.abs(), hi.abs()) * rel
    return lo - m, hi + m


def _mul(a_lo, a_hi, b_lo, b_hi):
    c = torch.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi])
    return c.amin(0), c.amax(0)


def _rot(lo, hi, cos, sin, inverse=False, rnd=R.bf16_rne):
    """GPT-J rotation of each (even, odd) pair of [..., D] value spans in f32
    (products and the sum rounded), then ``rnd`` -> spans."""
    sin = -sin if inverse else sin
    e, o = (lo[..., 0::2], hi[..., 0::2]), (lo[..., 1::2], hi[..., 1::2])
    ec, es, oc, os_ = (
        _mul(*e, cos, cos),
        _mul(*e, sin, sin),
        _mul(*o, cos, cos),
        _mul(*o, sin, sin),
    )
    m = F32_2 * (torch.maximum(*map(torch.abs, e)) + torch.maximum(*map(torch.abs, o)))
    elo, ehi = rnd(ec[0] - os_[1] - m), rnd(ec[1] - os_[0] + m)
    olo, ohi = rnd(es[0] + oc[0] - m), rnd(es[1] + oc[1] + m)
    return (
        torch.stack([elo, olo], -1).flatten(-2),
        torch.stack([ehi, ohi], -1).flatten(-2),
    )


def _silu(lo, hi, kappa):
    """SiLU over [lo, hi] (its minimum at -1.2785 inside), widened by kappa."""
    a, b = R.silu(lo), R.silu(hi)
    mn = torch.minimum(a, b)
    inside = (lo < -1.2785) & (hi > -1.2785)
    mn = torch.where(inside, R.silu(torch.full_like(lo, -1.27846454)), mn)
    return _widen(mn, torch.maximum(a, b), kappa + F32_2)


def _norm_span(x, w, eps, kappa):
    """bf16((x rstd) w) of rows x: rstd within kappa (relative) of the f64."""
    x = x.double()
    z = x * w.double() * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    lo, hi = _widen(z, z, kappa + F32_2)
    return R.bf16_rne(lo), R.bf16_rne(hi)


def _mx_within(tag, q, codes, lo, hi, rule, block=32) -> float:
    """MXFP8 output (E4M3 bytes [..., K], codes [..., K / block]) of values
    known to lie in [lo, hi]: each code one the rule gives some value in the
    spans, each byte the RNE of a value in its span at that code. -> the
    fraction of bytes that could take two values."""
    lb, hb = lo.unflatten(-1, (-1, block)), hi.unflatten(-1, (-1, block))
    least = torch.where(lb * hb > 0, torch.minimum(lb.abs(), hb.abs()), 0)
    clo, chi = rule(least.amax(-1)), rule(torch.maximum(lb.abs(), hb.abs()).amax(-1))
    c = codes.long()
    bad = (c < clo) | (c > chi)
    assert not bad.any(), f"{tag} codes: " + _first_bad(bad, c, clo, chi)
    inv = torch.exp2(127 - c.double())[..., None]
    qlo, qhi = R.to_e4m3_rne(lb * inv), R.to_e4m3_rne(hb * inv)
    got = q.view(torch.float8_e4m3fn).double().unflatten(-1, (-1, block))
    bad = (got < qlo) | (got > qhi)
    assert not bad.any(), f"{tag} values: " + _first_bad(bad, got, qlo, qhi)
    return (qlo != qhi).double().mean().item()


def _fold(x, residual, post, comb):
    """The seam's new residual in f32 as the kernel computes it: ((post x +
    c0 r0) + c1 r1) + c2 r2, each product rounded, then fma(c3, r3, ..)."""
    r, x = residual.float(), x.float()
    c = comb.float()
    v = post.reshape(-1, HC, 1) * x[:, None] + r[:, :1] * c[:, 0, :, None]
    for h in (1, 2):
        v = v + c[:, h, :, None] * r[:, h : h + 1]
    v = (c[:, 3, :, None].double() * r[:, 3:4].double() + v.double()).float()
    return v.to(torch.bfloat16)


def _collapse(residual, pre):
    """The layer input: products rounded, summed in stream order, f32 -> bf16."""
    r, p = residual.float(), pre.float()
    acc = p[:, :1, None] * r[:, 0:1]
    for h in range(1, HC):
        acc = acc + p[:, h : h + 1, None] * r[:, h : h + 1]
    return acc[:, 0].to(torch.bfloat16)


def _rank_sum(parts):
    """The TP partials summed in rank order in f32 -> bf16."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


def _peer(runner, M: int, ep: int):
    """Every rank's attention and FFN partials of launch pair ``ep`` (its K2 or
    FFN launch), as pushed to this rank's peer memory."""
    from vllm.models.deepseek_v41.amd.mono.layer import peer_half_bytes

    tp = runner.tp
    n = tp * M * HIDDEN // 2
    pb = runner.peer.bytes[(ep & 1) * peer_half_bytes(tp) :]
    attn = R.pairs(pb, n, 2 * ep + 2).view(torch.bfloat16).view(tp, M, HIDDEN)
    ffn = R.pairs(pb[n * 8 :], n, 2 * ep + 2).view(torch.bfloat16)
    return attn, ffn.view(tp, M, HIDDEN)


def _check_reduces(tag, runner, M, ep, outs, res, post, comb):
    """The launch's two all-reduces from the partials every rank pushed: the
    FFN seam's residual folds in their rank-order sum, and the output is the
    MoE's. -> (attention, FFN) partials."""
    attn, ffn = _peer(runner, M, ep)
    assert torch.equal(outs[1], _fold(_rank_sum(attn), res, post, comb)), tag
    assert torch.equal(outs[0], _rank_sum(ffn)), tag
    return attn, ffn


# ---------------------------------------------------------------- caches


def _paged_cache(num_blocks: int, block: int, device) -> torch.Tensor:
    """[blocks, block, 584] u8 over pages of round_up(block * 584, 576) bytes,
    the way vLLM binds a packed V4 cache."""
    page = -(-block * RECORD // ALIGN) * ALIGN
    raw = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)
    return torch.as_strided(raw, (num_blocks, block, RECORD), (page, RECORD, 1))


def _flat(cache: torch.Tensor) -> torch.Tensor:
    """Every byte of the cache's pages."""
    n = cache.untyped_storage().nbytes() - cache.storage_offset()
    return torch.as_strided(cache, (n,), (1,))


def _record_bytes(cache: torch.Tensor, block: int, slots: torch.Tensor):
    """The offsets in ``_flat(cache)`` of the records at ``slots``: their data
    bytes [n, 576] and their scale bytes [n, 8]."""
    slots = slots.to(cache.device).long()
    b, o = slots // block, slots % block
    d0 = b * cache.stride(0) + o * DATA
    s0 = b * cache.stride(0) + block * DATA + o * 8
    ar = torch.arange(DATA, device=cache.device)
    return d0[:, None] + ar, s0[:, None] + ar[:8]


def _rope(kv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """GPT-J RoPE of the last 64 dims (fp32 math, bf16 out)."""
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, :32], cs[:, 32:]
    x = kv.float()
    e, o = x[:, NOPE::2].clone(), x[:, NOPE + 1 :: 2].clone()
    x[:, NOPE::2] = e * cos - o * sin
    x[:, NOPE + 1 :: 2] = e * sin + o * cos
    return x.to(torch.bfloat16)


def _write_records(cache, block: int, slots, kv, fnuz: bool = False):
    """fp8_ds_mla records of roped bf16 kv rows at ``slots``: the 448 nope dims
    in e4m3 with a UE8M0 per 64, the rope dims in bf16, as vLLM's insert.
    ``fnuz``: the e4m3 bytes in FNUZ with a maximum of 224, as vLLM's gfx942
    window cache holds them. In FNUZ the OCP -0 byte would be NaN."""
    fp8, fp8_max = torch.float8_e4m3fn, 448.0
    if fnuz:
        fp8, fp8_max = torch.float8_e4m3fnuz, 224.0
    nope = kv.float()[:, :NOPE].view(-1, 7, 64)
    exp = torch.ceil(torch.log2(nope.abs().amax(-1).clamp_min(1e-4) / fp8_max))
    q = nope * torch.exp2(-exp)[..., None]
    q = q.clamp(-fp8_max, fp8_max).reshape(-1, NOPE)
    data = torch.cat(
        [
            q.to(fp8).view(torch.uint8),
            kv[:, NOPE:].contiguous().view(torch.uint8),
        ],
        1,
    )
    scales = torch.zeros(kv.shape[0], 8, dtype=torch.uint8, device=kv.device)
    scales[:, :7] = (exp + 127).clamp(0, 255).to(torch.uint8)
    flat = _flat(cache)
    d, s = _record_bytes(cache, block, slots)
    flat[d.reshape(-1)] = data.reshape(-1)
    flat[s.reshape(-1)] = scales.reshape(-1)


def _normed_kv(n: int, kv_norm, g, device):
    x = torch.randn(n, HEAD_DIM, generator=g).to(device)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * kv_norm.float()
    return x.to(torch.bfloat16)


def _decode_step(
    ratio: int,
    R: int,
    T: int,
    kv_norm,
    cos_sin,
    seed: int,
    fnuz: bool = False,
    short: bool = False,
    pad: int = 0,
    causal: bool = True,
):
    """R requests x T tokens past random contexts: the step's metadata as
    vLLM's builders lay it out, its window rows and (ratio > 0) the compressed
    cache filled, a random top-512 a token. ``fnuz``: the window records in
    FNUZ, as on gfx942. The compressed records stay OCP, as vLLM's Triton
    compressor writes them on gfx942 too. ``short``: SHORT_CONTEXTS among
    them. ``pad``: rows after the step's, as a FULL graph pads it (slot -1, no
    window). ``causal=False``: the DSpark draft block's rows, each the window
    before the block and the whole block."""
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )

    g = torch.Generator().manual_seed(seed)
    dev = kv_norm.device
    ctx = [int(c) for c in torch.randint(2000, 40000, (R,), generator=g)]
    if short:
        ctx[: len(SHORT_CONTEXTS)] = SHORT_CONTEXTS
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
    width = WINDOW if causal else get_dspark_swa_index_width(WINDOW, T)
    swa_idx = torch.full((M + pad, width), -1, dtype=torch.int32)
    swa_len = torch.zeros(M + pad, dtype=torch.int32)
    for i in range(M):
        r = t2r[i]
        lo, hi = pos[i] - WINDOW + 1, pos[i] + 1
        if not causal:
            lo, hi = ctx[r] - WINDOW, ctx[r] + T
        ps = range(max(lo, 0), hi)
        swa_idx[i, : len(ps)] = torch.tensor([slot(r, p) for p in ps])
        swa_len[i] = len(ps)
    hist = [
        (r, p) for r in range(R) for p in range(max(0, ctx[r] - WINDOW - T), ctx[r])
    ]
    hp = torch.tensor([p for _, p in hist], device=dev)
    kv = _rope(_normed_kv(len(hist), kv_norm, g, dev), hp, cos_sin)
    _write_records(
        swa, SWA_BLOCK, torch.tensor([slot(r, p) for r, p in hist]), kv, fnuz
    )

    step = dict(
        M=M + pad,
        positions=torch.tensor(pos + [0] * pad, dtype=torch.int64, device=dev),
        slot_mapping=torch.tensor(
            [slot(t2r[i], pos[i]) for i in range(M)] + [-1] * pad,
            dtype=torch.int64,
            device=dev,
        ),
        token_to_req=torch.tensor(t2r + [0] * pad, dtype=torch.int32, device=dev),
        swa_indices=swa_idx.to(dev),
        swa_lens=swa_len.to(dev),
        swa_cache=swa,
        topk=None,
        comp_cache=None,
        comp_block_table=None,
        comp_entries=0,
    )
    if ratio:
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
            kv = _rope(
                _normed_kv(lens[r], kv_norm, g, dev), (idx * ratio).to(dev), cos_sin
            )
            _write_records(comp, entries, slots, kv)
        topk = torch.full((M + pad, TOPK), -1, dtype=torch.int32)
        for i in range(M):
            cand = (pos[i] + 1) // ratio
            n = min(cand, TOPK)
            topk[i, :n] = torch.randperm(cand, generator=g)[:n].to(torch.int32)
        step.update(
            topk=topk.to(dev),
            comp_cache=comp,
            comp_block_table=cbt.to(dev),
            comp_entries=entries,
        )
    return step


def _rows(s: dict, a: int, b: int) -> dict:
    """The decode step ``s`` cut to its rows a .. b - 1, as a step of its own
    on the same caches. The rows keep their request numbers, so the block
    tables of ``s`` stay valid."""
    cut = dict(s, M=b - a)
    for k in ("positions", "slot_mapping", "token_to_req", "swa_indices", "swa_lens"):
        cut[k] = s[k][a:b]
    if s["topk"] is not None:
        cut["topk"] = s["topk"][a:b]
    return cut


def _key_table(s: dict, ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys K2 attends over, as K1 lists them from the step's metadata:
    a row's top-k compressed slots (with bit 31 set) then its window slots,
    -1 past them and for a pad row, [M, 640]; and each row's (keys, top-k
    keys), [2, M]."""
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS

    M = s["M"]
    slots, pos = s["slot_mapping"].tolist(), s["positions"].tolist()
    swa_idx, swa_len = s["swa_indices"].cpu(), s["swa_lens"].tolist()
    keys = torch.full((M, KEYS), -1, dtype=torch.int64)
    lens = torch.zeros(2, M, dtype=torch.int32)
    for t in range(M):
        if slots[t] < 0:
            continue
        ntopk = min((pos[t] + 1) // ratio, TOPK) if ratio else 0
        if ntopk:
            local = s["topk"][t, :ntopk].long().cpu()
            e = s["comp_entries"]
            blk = s["comp_block_table"][s["token_to_req"][t]].long().cpu()
            comp = blk[local.clamp_min(0) // e] * e + local.clamp_min(0) % e
            keys[t, :ntopk] = torch.where(local >= 0, comp - 2**31, -1)
        keys[t, ntopk : ntopk + swa_len[t]] = swa_idx[t, : swa_len[t]]
        lens[:, t] = torch.tensor([ntopk + swa_len[t], ntopk])
    return keys.to(torch.int32), lens


# ---------------------------------------------------------------- weights


def _mxfp8(n: int, k: int, g, device):
    """A random MXFP8 weight [n, k] e4m3 and its E8M0 32 x 32 block scales."""
    w = torch.randn(n, k, generator=g).clamp(-6, 6).to(torch.float8_e4m3fn)
    s = torch.randint(118, 123, (n // 32, k // 32), generator=g, dtype=torch.uint8)
    return w.to(device), s.to(device)


def _random_moe(moe, gd, gr, inter: int) -> None:
    """Random parameters in the loaded layout, before post-load processing: this
    rank's shards (``gd``, a device generator) of the MXFP4 routed experts
    (random nibbles, E8M0 scales; the intermediate past ``inter`` zero, as the
    loader pads it) and the MXFP8 shared expert (one E8M0 a 32 x 32 block); the
    replicated router from ``gr``, seeded alike on every rank."""
    u8 = torch.uint8
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        n = lin.weight.shape[0]
        w = torch.randn(lin.weight.shape, generator=gd, device=gd.device)
        lin.weight.data.copy_(w.clamp(-6, 6).to(lin.weight.dtype))
        sc = lin.weight_scale.data.view(u8)
        rows = -(-n // 32) if sc.shape[0] == n else sc.shape[0]
        blocks = torch.randint(
            118, 123, (rows, sc.shape[1]), generator=gd, device=gd.device, dtype=u8
        )
        sc.copy_(blocks.repeat_interleave(32, 0)[:n] if sc.shape[0] == n else blocks)
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 122)):
        for w in ("w13", "w2"):
            t = getattr(e, f"{w}_{name}").data.view(u8)
            t.copy_(
                torch.randint(lo, hi, t.shape, generator=gd, device=gd.device, dtype=u8)
            )
    padded = e.w13_weight.shape[1] // 2
    if padded > inter:
        w13, w2 = e.w13_weight.data.view(u8), e.w2_weight.data.view(u8)
        for half in (0, padded):
            w13[:, half + inter : half + padded] = 0
        w2[..., inter // 2 :] = 0
    for t in (moe.gate.weight, moe.gate.e_score_correction_bias):
        t.data.copy_(torch.randn(t.shape, generator=gr, device=gr.device) * 0.02)


# ---------------------------------------------------------------- the worker


def _in_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, check):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.workspace import init_workspace_manager

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        # the checkpoint's MXFP4 experts run on AITER's MoE; its flags are read
        # at import, so refresh them
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        rocm_aiter_ops.refresh_env_variables()
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        check(rank, tp_size, device)


@ray.remote(num_gpus=1, max_calls=1)
def _mono_numerics(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _in_worker(
        monkeypatch, tp_size, pp_size, rank, distributed_init_port, _check_layers
    )


@ray.remote(num_gpus=1, max_calls=1)
def _mono_variants(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _in_worker(
        monkeypatch, tp_size, pp_size, rank, distributed_init_port, _check_variants
    )


@ray.remote(num_gpus=1, max_calls=1)
def _mono_draft(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _in_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, _check_draft)


@ray.remote(num_gpus=1, max_calls=1)
def _mono_index(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _in_worker(
        monkeypatch, tp_size, pp_size, rank, distributed_init_port, _check_index_source
    )


@contextmanager
def _k1_taps(runner, taps: dict):
    """Each K1 launch followed by a copy of its step's scratch into ``taps``:
    K2 reuses K1's X8 regions."""
    kernels = runner.kernels

    def tapped(tokens, *key):
        k1, k2 = kernels(tokens, *key)

        def k1_tap(*args, **kwargs):
            k1(*args, **kwargs)
            taps["k1"] = runner.scratch(tokens).clone()

        return k1_tap, k2

    runner.kernels = tapped
    try:
        yield
    finally:
        del runner.kernels


def _bench(rank: int, tp: int, device, draft: bool = False, seed: int = 0, stage=None):
    """The process's mono runner of the target's layers, or of the DSpark
    draft's (``draft``), on random weights of one layer (``seed``), and
    ``check(case)``: every stage of the case's step against a float64
    reference on the stage's own input (``stage(name, check)`` runs a check
    with its error constant), which returns the step's launches on fixed
    buffers for the replay stress."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (
        rocm_mxfp8_block32_gemm,
    )
    from vllm.model_executor.layers.mhc import MHCPreDelayedOp
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.models.deepseek_v41.amd import mono_decode
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS, Dims
    from vllm.models.deepseek_v41.amd.mono.layer import scratch_layout
    from vllm.models.deepseek_v41.amd.mono.runner import AttnWeights, MonoLayerWeights
    from vllm.models.deepseek_v41.amd.mono.stages.dims import Dims as MoeDims
    from vllm.models.deepseek_v41.amd.rocm import (
        compute_global_topk_ragged_indices_and_indptr,
    )
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
        rocm_inverse_rope_mxfp8_rows,
    )

    stage = stage or _assert_stage
    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tp,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=65536,
        language_model_only=True,
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
    ).create_engine_config()
    cfg = vc.model_config.hf_config
    cfg = getattr(cfg, "text_config", cfg)
    # replicated tensors alike on every rank; each rank's own TP shards
    sd = 7919 * seed
    g = torch.Generator().manual_seed(1000 + sd)
    gs = torch.Generator().manual_seed(1001 + rank + sd)
    H, G = N_HEADS // tp, O_GROUPS // tp
    bf, f64, i32 = torch.bfloat16, torch.float64, torch.int32
    eps, hc_eps = cfg.rms_norm_eps, cfg.hc_eps

    # a layer past the backbone's is a draft layer, with the draft's routing
    layer_id = cfg.num_hidden_layers if draft else 21
    with set_current_vllm_config(vc), set_default_torch_dtype(bf), device:
        moe = DeepseekV4MoE(vc, prefix=f"model.layers.{layer_id}.ffn")
        mhc = MHCPreDelayedOp()
    _random_moe(
        moe,
        torch.Generator(device=device).manual_seed(2001 + rank + sd),
        torch.Generator(device=device).manual_seed(2000 + sd),
        inter=2304 // tp,
    )
    e, sh = moe.experts.routed_experts, moe.shared_experts
    # the checkpoint-format tensors the reference reads, before post-load
    # processing shuffles them for the kernels
    raw = [
        t.data.view(torch.uint8).clone()
        for t in (e.w13_weight, e.w13_weight_scale, e.w2_weight, e.w2_weight_scale)
    ]
    SGU, SW2 = (
        R.dequant_fp8(
            lin.weight.data,
            lin.weight_scale.data.view(torch.uint8),
            (1 if lin.weight_scale.shape[0] == lin.weight.shape[0] else 32, 32),
        )
        for lin in (sh.gate_up_proj, sh.down_proj)
    )
    process_weights_after_loading(moe, vc.model_config, device)
    experts, topk_k = moe.n_routed_experts, moe.n_activated_experts
    if draft:
        assert (experts, topk_k) == (128, 3), (experts, topk_k)
        width = get_dspark_swa_index_width(cfg.sliding_window, DRAFTS)
        runner = mono_decode._mono_runner(
            device,
            mono_decode.MonoDecodeLayer(
                draft=True, experts=experts, topk=topk_k, swa_width=width
            ),
        )
    else:
        runner = mono_decode._mono_runner(device)
    md = MoeDims(tp)
    inter, real, shi = md.inter, md.inter_real, md.sh_inter

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

    def seam(sub, residual, x, post, comb, pre):
        s = seams[sub]
        return mhc(
            residual, s["fn"], s["scale"], s["base"], cfg.rms_norm_eps,
            cfg.hc_eps, cfg.hc_eps, 2.0, cfg.hc_sinkhorn_iters,
            pre_mix=pre, sublayer_out=x, post_layer_mix=post, comb_res_mix=comb,
            norm_weight=s["norm"], norm_eps=cfg.rms_norm_eps,
        )  # fmt: skip

    def moe_out(x):
        # the rows tiled to a full step: at some small token counts aiter picks
        # fused-FP8 stage-1 kernels with another MXFP8 scale rule (1.5-1.9%,
        # https://github.com/ROCm/aiter/issues/6548)
        rows = torch.arange(max(x.shape[0], mono_decode.MAX_ROWS), device=device)
        rows %= x.shape[0]
        with set_forward_context(None, vc):
            ids = torch.zeros(rows.numel(), dtype=torch.int64, device=device)
            return moe(x[rows], ids)[: x.shape[0]]

    wqkv, wqkv_s = _mxfp8(1792, HIDDEN, g, device)
    wq_b, wq_b_s = _mxfp8(H * HEAD_DIM, Q_RANK, gs, device)
    wo_a, wo_a_s = _mxfp8(G * O_RANK, H // G * HEAD_DIM, gs, device)
    wo_b, wo_b_s = _mxfp8(HIDDEN, G * O_RANK, gs, device)
    q_norm, kv_norm = (1 + f32(Q_RANK, s=0.1)).to(bf), (1 + f32(HEAD_DIM, s=0.1)).to(bf)
    sink = f32(H, gen=gs)
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    gw, bias = moe.gate.weight.float(), moe.gate.e_score_correction_bias.float()
    router = {False: (moe.gate.weight, moe.gate.e_score_correction_bias)}
    router[True] = tuple(t.clone() for t in router[False])
    router[True][0][TIE[1]] = router[True][0][TIE[0]]
    router[True][1][list(TOP5)] = 30.0
    router[True][1][list(TIE)] = 12.0
    WQKV, WQB, WOA, WOB = (
        R.dequant_fp8(w, s)
        for w, s in ((wqkv, wqkv_s), (wq_b, wq_b_s), (wo_a, wo_a_s), (wo_b, wo_b_s))
    )
    norm_w: dict = {}
    fns = {}
    for sub, sm in seams.items():
        hi = sm["fn"].to(bf).float()
        fns[sub] = hi.double() + (sm["fn"] - hi).to(bf).double()

    def top(x):
        """The experts vLLM routes to: sqrt(softplus(logits)) + bias."""
        sc = torch.nn.functional.softplus(x.float() @ gw.T).sqrt() + bias
        return sc.topk(topk_k, dim=1).indices.sort(1).values

    # ---- the stage checks, on the kernels' own inputs (taps)
    def mx_rows(buf, words, codes, rows, k):
        """MXFP8 rows of a scratch region pair -> (e4m3 bytes, codes)."""
        q = buf[words[0] : words[0] + rows * k].view(rows, k)
        c = buf[codes[0] : codes[0] + rows * k // 8].view(i32).view(rows, k // 32)
        return q, c

    def gemv(tag, got, a, w, kappa_name, quant=None):
        """``got`` (f32 or bf16, or ``quant``'s MXFP8 of the bf16) = a @ w^T,
        a scaled-MFMA GEMV: within kappa of R.blockmax_cond."""
        ref, cond = a @ w.T, R.blockmax_cond(a, w)

        def check(kappa):
            if quant is not None:
                lo, hi = _span(ref, kappa * cond)
                _mx_within(tag, quant[0], quant[1], lo, hi, R.mx_code_vllm)
            elif got.dtype == torch.float32:
                _le(tag, (got.double() - ref).abs() / cond, kappa)
            else:
                _within(tag, got, *_span(ref, kappa * cond))

        stage(kappa_name, check)
        return ref, cond

    def mixes(tag, sub, res, post, comb, pre):
        sm = seams[sub]
        want = R.mhc_mixes(
            res.flatten(1), fns[sub], sm["scale"], sm["base"], HC, eps, hc_eps,
            hc_eps, 2.0, cfg.hc_sinkhorn_iters,
        )  # fmt: skip
        got = (pre, post.view(-1, HC), comb)

        def check(kappa):
            for name, a, b in zip(("pre", "post", "comb"), got, want):
                err = ((a.double() - b) / b).abs().max().item()
                assert err <= kappa, f"{tag} {name}: {err:.3g} > {kappa:.3g}"

        stage("mix", check)

    def norm(tag, lin, w, got=None, quant=None):
        """bf16 RMSNorm rows of the kernel's own input ``lin`` -> ``got``, or
        their MXFP8 (vLLM's rule) ``quant``."""

        def check(kappa):
            lo, hi = _norm_span(lin, w, eps, kappa)
            if got is not None:
                _within(tag, got, lo, hi)
            if quant is not None:
                _mx_within(tag, quant[0], quant[1], lo, hi, R.mx_code_vllm)

        stage("rstd", check)

    def ffn_io(tag, M, res, post, comb, pre, outs, hard=False):
        """The FFN seam and the MoE of the last launch, from the partials its
        attention pushed to this rank."""
        sc, lay = (
            runner.scratch(M),
            scratch_layout(M, tp, experts, topk_k, runner.ug_groups),
        )
        ep = int(runner.epoch[0]) - 1
        t2 = 2 * ep + 2
        attn, ffn = _check_reduces(tag, runner, M, ep, outs, res, post, comb)

        def pairs(name, n):
            o, _ = lay[name]
            return R.pairs(sc[o:], n, t2)

        out, res_out, post_out, comb_out, pre_out = outs
        # K2.5: the attention all-reduce (above), the seam
        mixes(f"{tag} FFN seam", "ffn", res_out, post_out, comb_out, pre_out)
        lin = _collapse(res_out, pre)
        assert torch.equal(pairs("lin", M * HIDDEN // 2).view(bf).view(M, HIDDEN), lin)
        o = lay["normed"][0]
        normed = sc[o : o + M * HIDDEN * 2].view(bf).view(M, HIDDEN)
        norm(f"{tag} FFN norm", lin, norm_w["ffn"], got=normed)
        # K2.6: the MoE input quantized for the routed and the shared experts
        xr, xr_s = mx_rows(sc, lay["x8r"], lay["x8rs"], M, HIDDEN)
        xs, xs_s = mx_rows(sc, lay["x8"], lay["x8s"], M, HIDDEN)
        for (q, c), rule in (((xr, xr_s), _code_aiter), ((xs, xs_s), R.mx_code_vllm)):
            want_q, want_c = R.mx_quantize(normed, rule)
            assert torch.equal(c.long(), want_c), tag
            assert torch.equal(q.view(torch.float8_e4m3fn).double(), want_q), tag
        # K2.7: router logits, the four K parts summed in order
        parts = pairs("logit", M * experts * 4).view(torch.float32).view(M, experts, 4)
        logits = parts[..., 0]
        for p in range(1, 4):
            logits = logits + parts[..., p]
        gate_w, gate_b = router[hard]
        nd, gd = normed.double(), gate_w.double()
        err = R.err_over_cond(logits, nd @ gd.T, nd, gd)
        stage("logit", lambda k: _le(f"{tag} logits", err, k))
        # K2.8: routing on the kernel's logits
        route = pairs("route", 8 * M * 2 * topk_k).view(8, M, 2 * topk_k)
        assert (route == route[:1]).all(), tag
        ids, w = route[0, :, :topk_k].long(), route[0, :, topk_k:].view(torch.float32)
        assert (ids.sort(1).values.diff(dim=1) > 0).all() and (ids < experts).all()
        ref_ids, _, margin = R.route(logits, gate_b, topk_k, "sqrt_softplus", 1.5)
        _, ref_w, _ = R.route(logits, gate_b, topk_k, "sqrt_softplus", 1.5, ids=ids)

        # the kernel's softplus rounds 1 + e^x to f32: within 2^-23, so each
        # score within d of the reference's, which the weights carry
        sp = torch.nn.functional.softplus(logits.double())
        d = sp.sqrt() - (sp - 2.0**-23).clamp(min=0).sqrt()
        di, si = d.gather(1, ids), sp.gather(1, ids).sqrt()
        allow = (1.5 * di + ref_w * di.sum(-1, keepdim=True)) / si.sum(-1, keepdim=True)
        err = ((w.double() - ref_w).abs() - allow).clamp(min=0) / ref_w
        stage("route", lambda k: _le(f"{tag} routing weights", err, k))
        sel = sp.sqrt() + gate_b.double()
        edge = sel.sort(-1, descending=True).indices[:, topk_k - 1 : topk_k + 1]
        bound = 2 * (KAPPA["route"] * sel.abs().amax(-1) + d.gather(1, edge).amax(-1))
        if hard:
            assert (margin == 0).all(), tag
            assert (logits[:, TIE[0]] == logits[:, TIE[1]]).all(), tag
            margin = torch.full_like(margin, torch.inf)
        checked = R.assert_topk(ids, ref_ids, margin, bound)
        assert checked >= M * 3 // 4, f"{tag}: {checked} of {M} rows past ties"
        # K2.9: the shared expert, then its down projection's input quant
        X8 = R.dequant_fp8(xs, xs_s, (1, 32))
        smid = pairs("smid", M * shi).view(torch.float32).view(M, shi)
        gate_up = [(X8 @ W.T, R.blockmax_cond(X8, W)) for W in (SGU[:shi], SGU[shi:])]

        def shared(kappa):
            (gr, gc), (ur, uc) = gate_up
            glo, ghi = _span(gr, KAPPA["fp8"] * gc)
            ulo, uhi = _span(ur, KAPPA["fp8"] * uc)
            a_lo, a_hi = _silu(glo.clamp(max=10), ghi.clamp(max=10), kappa)
            plo, phi = _mul(
                R.bf16_rne(a_lo),
                R.bf16_rne(a_hi),
                ulo.clamp(-10, 10),
                uhi.clamp(-10, 10),
            )
            _within(
                f"{tag} shared act", smid, *map(R.bf16_rne, _widen(plo, phi, F32_2))
            )

        stage("act", shared)
        q, c = R.mx_quantize(smid, R.mx_code_vllm)
        smq = pairs("smq", M * shi // 4).view(torch.uint8).view(M, shi)
        smc = pairs("smc", M * shi // 32).view(M, shi // 32)
        assert torch.equal(smc.long(), c), tag
        assert torch.equal(smq.view(torch.float8_e4m3fn).double(), q), tag
        # K2.10: the routed up / gate GEMVs and activation, a pick a MID row in
        # (expert, token) order
        P = M * topk_k
        tok = torch.arange(M, device=device).repeat_interleave(topk_k)
        order = torch.argsort(ids.flatten() * M + tok)
        row_of = torch.empty_like(order)
        row_of[order] = torch.arange(P, device=device)
        o = lay["mid"][0]
        mid = sc[o : o + P * inter].view(P, inter)
        o = lay["mids"][0]
        steps = inter // 128
        perm = [(j % 4) * steps + j // 4 for j in range(inter // 32)]
        mids = sc[o : o + P * inter // 8].view(i32).view(P, inter // 32)[:, perm]
        XR = R.dequant_fp8(xr, xr_s, (1, 32))
        MID = R.dequant_fp8(mid, mids, (1, 32))
        gu, down = [], torch.zeros(2, P, HIDDEN, dtype=f64, device=device)
        for ex in ids.unique().tolist():
            rows = order[ids.flatten()[order] == ex]
            W13 = R.dequant_mxfp4(raw[0][ex], raw[1][ex])
            W2 = R.dequant_mxfp4(raw[2][ex], raw[3][ex])
            x = XR[tok[rows]]
            # the real intermediate: MID past it (the loader's padding) unwritten
            gate, up = W13[:real], W13[inter : inter + real]
            gu.append(
                (
                    row_of[rows],
                    x @ gate.T,
                    R.blockmax_cond(x, gate),
                    x @ up.T,
                    R.blockmax_cond(x, up),
                )
            )
            m = MID[row_of[rows]]
            down[0, rows], down[1, rows] = m @ W2.T, R.blockmax_cond(m, W2)

        # K2.11: down, weighted, summed in bf16 in top-k order, plus the shared
        # expert's down -> this rank's FFN partial
        SMQ = R.dequant_fp8(smq, smc, (1, 32))
        sh_ref, sh_cond = SMQ @ SW2.T, R.blockmax_cond(SMQ, SW2)

        def ffn_partial(kappa):
            lo = hi = torch.zeros(M, HIDDEN, dtype=f64, device=device)
            for k in range(topk_k):
                at = torch.arange(M, device=device) * topk_k + k
                v, cnd = down[0, at], down[1, at]
                wk = w[:, k : k + 1].double()
                vlo = R.bf16_rne(R.f32_rne((v - kappa * cnd) * wk))
                vhi = R.bf16_rne(R.f32_rne((v + kappa * cnd) * wk))
                lo = R.bf16_rne(R.f32_rne(lo + vlo))
                hi = R.bf16_rne(R.f32_rne(hi + vhi))
            slo, shi_ = _span(sh_ref, KAPPA["fp8"] * sh_cond)
            lo, hi = R.bf16_rne(R.f32_rne(lo + slo)), R.bf16_rne(R.f32_rne(hi + shi_))
            _within(f"{tag} FFN partial", ffn[rank], lo, hi)

        stage("fp4_down", ffn_partial)

        def routed_act(kappa):
            for rows, gr, gc, ur, uc in gu:
                glo, ghi = (
                    (gr - kappa * gc).clamp(max=10),
                    (gr + kappa * gc).clamp(max=10),
                )
                ulo, uhi = (
                    (ur - kappa * uc).clamp(-10, 10),
                    (ur + kappa * uc).clamp(-10, 10),
                )
                a_lo, a_hi = _silu(glo, ghi, KAPPA["act"])
                lo, hi = _mul(a_lo, a_hi, ulo, uhi)
                lo, hi = _widen(lo, hi, F32_2)
                _mx_within(
                    f"{tag} MID", mid[rows, :real], mids[rows, : real // 32],
                    R.bf16_rne(lo), R.bf16_rne(hi), _code_aiter,
                )  # fmt: skip

        stage("fp4_ug", routed_act)
        if hard:
            clamped = sum(int((g[1] > 10).sum()) for g in gu) / (P * real)
            assert 0.01 <= clamped <= 0.2, f"{tag}: {clamped:.3f} of gates past 10"
            assert (gate_up[0][0] > 10).any() and (gate_up[1][0].abs() > 10).any()
            assert (logits > 20).any(), tag
            assert (xr_s == _code_aiter(torch.zeros(1, device=device))).any(), tag

    def layer_io(tag, case, s, cos_sin, ins, outs, k1, empty):
        """K1 and K2 of the last whole-layer launch, stage by stage."""
        x0, res0, post0, comb0, pre0 = ins
        M, n = s["M"], case.R * case.T
        sc, lay = (
            runner.scratch(M),
            scratch_layout(M, tp, experts, topk_k, runner.ug_groups),
        )
        ep = int(runner.epoch[0]) - 1
        t1 = 2 * ep + 1
        res_mid, post_a, comb_a, pre_a = (
            t[:M] for t in (runner.res_mid, runner.post_a, runner.comb_a, runner.pre_a)
        )
        pos = s["positions"][:n]
        cos, sin = (t.double() for t in cos_sin[pos].split(ROPE // 2, -1))
        # K1.1: the attention seam
        fold = case.entry == "fold"
        assert torch.equal(res_mid, _fold(x0, res0, post0, comb0) if fold else res0)
        mixes(f"{tag} attention seam", "attn", res_mid, post_a, comb_a, pre_a)
        lin = _collapse(res_mid, pre0)
        o = lay["lin"][0]
        got = R.pairs(k1[o:], M * HIDDEN // 2, t1).view(bf).view(M, HIDDEN)
        assert torch.equal(got, lin), tag
        xa = mx_rows(k1, lay["x8"], lay["x8s"], M, HIDDEN)
        normed = runner.normed[:M] if case.index else None
        norm(f"{tag} attention norm", lin, norm_w["attn"], normed, xa)
        zero = xa[0].view(torch.float8_e4m3fn).view(M, -1, 32).double() == 0
        assert not case.hard or zero.all(-1).any(), tag
        # K1.3: wqkv, two K halves
        o = lay["qkvp"][0]
        qkvp = R.pairs(k1[o:], 2 * M * 1792, t1).view(torch.float32).view(2, M, 1792)
        X8 = R.dequant_fp8(xa[0], xa[1], (1, 32))[:n]
        for p in range(2):
            ks = slice(p * HIDDEN // 2, (p + 1) * HIDDEN // 2)
            gemv(f"{tag} wqkv {p}", qkvp[p, :n], X8[:, ks], WQKV[:, ks], "fp8")
        # K1.4: q / kv norms, the q quant, the KV record
        qkv = (qkvp[0] + qkvp[1]).to(bf)[:n]
        qa = mx_rows(k1, lay["qx8"], lay["qx8s"], M, Q_RANK)
        qr = runner.qr[:n] if case.index else None
        norm(f"{tag} q_norm", qkv[:, :Q_RANK], q_norm, qr, [t[:n] for t in qa])
        flat_ = _flat(s["swa_cache"])
        d, scb = _record_bytes(s["swa_cache"], SWA_BLOCK, s["slot_mapping"][:n])
        rec, codes = flat_[d], flat_[scb]
        assert (codes[:, 7] == 0).all(), tag
        written = torch.zeros_like(flat_, dtype=torch.bool)
        written[d.flatten()] = written[scb.flatten()] = True
        assert torch.equal(flat_[~written], empty[~written]), tag
        assert not torch.equal(flat_[written], empty[written]), tag

        def kv_record(kappa):
            lo, hi = _norm_span(qkv[:, Q_RANK:], kv_norm, eps, kappa)
            _mx_within(
                f"{tag} KV record", rec[:, :NOPE], codes[:, :7], lo[:, :NOPE],
                hi[:, :NOPE], _code_record, block=64,
            )  # fmt: skip
            rope = rec[:, NOPE:].contiguous().view(bf)
            _within(f"{tag} KV rope", rope, *_rot(lo[:, NOPE:], hi[:, NOPE:], cos, sin))

        stage("rstd", kv_record)
        # K1.5: wq_b, RoPE
        QX = R.dequant_fp8(qa[0][:n], qa[1][:n], (1, 32))
        qref, qcond = QX @ WQB.T, R.blockmax_cond(QX, WQB)

        def q_rope(kappa):
            lo, hi = (t.view(n, H, HEAD_DIM) for t in _span(qref, kappa * qcond))
            rlo, rhi = _rot(lo[..., NOPE:], hi[..., NOPE:], cos[:, None], sin[:, None])
            lo, hi = (
                torch.cat([lo[..., :NOPE], rlo], -1),
                torch.cat([hi[..., :NOPE], rhi], -1),
            )
            _within(f"{tag} q", runner.q[:n], lo, hi)

        stage("fp8", q_rope)
        # K2.1-2.2: sparse attention with the sink over K1's key table, from
        # the records K1 wrote, inverse RoPE, the MXFP8 of each 32
        kt = runner.kt[: M * KEYS].view(M, KEYS)[:n].long()
        valid = (torch.arange(KEYS, device=device) < runner.klen[:n, None]) & (kt != -1)
        comp = valid & (kt < 0)
        keys = torch.zeros(n, KEYS, HEAD_DIM, dtype=f64, device=device)
        keys[valid & ~comp] = R.read_records(
            s["swa_cache"], SWA_BLOCK, kt[valid & ~comp]
        )
        if comp.any():
            keys[comp] = R.read_records(
                s["comp_cache"], s["comp_entries"], kt[comp] & 0x7FFFFFFF
            )
        args = (runner.q[:n], keys, HEAD_DIM**-0.5, sink, valid)
        aref = R.sparse_attention_sink(*args)
        acond = R.sparse_attention_sink(*args, values=keys.abs())
        ones = torch.ones_like(keys[..., :1])
        sink_share = 1 - R.sparse_attention_sink(*args, values=ones)[..., 0]
        xo = [t[:n] for t in mx_rows(sc, lay["xo"], lay["xos"], M, H * HEAD_DIM)]

        def attention(kappa):
            lo, hi = _span(aref, kappa * acond)
            rlo, rhi = _rot(
                lo[..., NOPE:], hi[..., NOPE:], cos[:, None], sin[:, None], True,
                lambda x: x,
            )  # fmt: skip
            lo, hi = (
                torch.cat([lo[..., :NOPE], rlo], -1),
                torch.cat([hi[..., :NOPE], rhi], -1),
            )
            lo, hi = lo.flatten(1), hi.flatten(1)
            _mx_within(f"{tag} XO", xo[0], xo[1], lo, hi, R.mx_code_vllm)

        stage("attn", attention)
        if case.short:
            assert sink_share.max() > 4 * KAPPA["attn"], (
                f"{tag}: the sink never matters"
            )
        # K2.3-2.4: wo_a by group, wo_b -> this rank's attention partial
        XO = R.dequant_fp8(xo[0], xo[1], (1, 32))
        xb = [t[:n] for t in mx_rows(sc, lay["x8b"], lay["x8bs"], M, G * O_RANK)]
        for gi in range(G):
            ks, rs = (
                slice(gi * 4096, (gi + 1) * 4096),
                slice(gi * O_RANK, (gi + 1) * O_RANK),
            )
            q, c = xb[0][:, rs], xb[1][:, gi * 32 : (gi + 1) * 32]
            gemv(f"{tag} wo_a {gi}", None, XO[:, ks], WOA[rs], "fp8", (q, c))
        attn, _ = _peer(runner, M, ep)
        X8B = R.dequant_fp8(xb[0], xb[1], (1, 32))
        gemv(f"{tag} wo_b", attn[rank, :n], X8B, WOB, "fp8")
        ffn_io(tag, M, res_mid, post_a, comb_a, pre_a, outs, case.hard)

    def check(case: Case):
        ratio = case.ratio
        with set_current_vllm_config(VllmConfig()):
            rope = build_deepseek_v4_rope(
                cfg, head_dim=HEAD_DIM, rope_head_dim=ROPE,
                max_position_embeddings=cfg.max_position_embeddings,
                compress_ratio=ratio,
            )  # fmt: skip
        cos_sin = rope.cos_sin_cache.to(device=device, dtype=torch.float32)
        s = _decode_step(
            ratio, case.R, case.T, kv_norm, cos_sin, seed=7 + ratio + sd,
            short=case.short, pad=case.pad, causal=not draft,
        )  # fmt: skip
        M, n = s["M"], case.R * case.T
        norm_w.update({sub: sm["norm"] for sub, sm in seams.items()})
        if case.hard:
            norm_w["ffn"] = (norm_w["ffn"].float() * 6).to(bf)
            for sub in norm_w:
                norm_w[sub] = norm_w[sub].clone()
                norm_w[sub][64:96] = 0
        gate_w, gate_b = router[case.hard]
        mw = MonoLayerWeights(
            attn=AttnWeights(
                layer_id=layer_id, wqkv=wqkv, wqkv_scale=wqkv_s, q_norm=q_norm,
                kv_norm=kv_norm, wq_b=wq_b, wq_b_scale=wq_b_s, wo_a=wo_a,
                wo_a_scale=wo_a_s, wo_b=wo_b, wo_b_scale=wo_b_s, attn_sink=sink,
                cos_sin=cos_sin, ratio=ratio,
            ),
            hc_attn_fn=seams["attn"]["fn"], hc_attn_scale=seams["attn"]["scale"],
            hc_attn_base=seams["attn"]["base"], attn_norm=norm_w["attn"],
            hc_ffn_fn=seams["ffn"]["fn"], hc_ffn_scale=seams["ffn"]["scale"],
            hc_ffn_base=seams["ffn"]["base"], ffn_norm=norm_w["ffn"],
            gate_w=gate_w, bias=gate_b,
            w13=e.w13_weight, w13_s=e.w13_weight_scale, w2=e.w2_weight,
            w2_s=e.w2_weight_scale, sgu=sh.gate_up_proj.weight,
            sgu_s=u8(sh.gate_up_proj.weight_scale), sw2=sh.down_proj.weight,
            sw2_s=u8(sh.down_proj.weight_scale),
        )  # fmt: skip
        # the layouts vLLM's loading leaves, as the model checks them
        mw.attn.check(Dims(tp))
        mw.check(tp, experts)

        # the layer's inputs at the attention seam (the same on every rank)
        gi = torch.Generator().manual_seed(11 + sd)
        res0 = torch.randn(M, HC, HIDDEN, generator=gi)
        res0 = (
            (res0 * torch.rsqrt(res0.pow(2).mean(-1, keepdim=True))).to(bf).to(device)
        )
        x0 = (torch.randn(M, HIDDEN, generator=gi) * 0.5).to(bf).to(device)
        comb0 = torch.rand(M, HC, HC, generator=gi) + 0.1
        for _ in range(20):
            comb0 = comb0 / comb0.sum(-1, keepdim=True)
            comb0 = comb0 / comb0.sum(-2, keepdim=True)
        comb0 = comb0.to(device)
        post0 = (torch.sigmoid(torch.randn(M, HC, 1, generator=gi)) * 2).to(device)
        pre0 = (torch.sigmoid(torch.randn(M, HC, generator=gi)) + 1e-6).to(device)
        if case.entry == "embedding":
            # the embedding broadcast to the streams; vLLM's seam takes stream
            # 0 without a pre-mix, the kernels with runner.first_pre_mix
            res0 = x0.unsqueeze(1).expand(-1, HC, -1).contiguous()
            pre0 = runner.first_pre_mix(M)
        cache = _flat(s["swa_cache"])
        empty = cache.clone()
        # this rank's unreduced wo_b output, for the FFN launch
        part = (torch.randn(M, HIDDEN, generator=gs) * 0.1).to(bf).to(device)

        if case.vllm:
            # vLLM's decode chain
            if case.entry == "fold":
                res, post, comb, xa, attn_pre = seam(
                    "attn", res0, x0, post0, comb0, pre0
                )
            else:
                res, post, comb, xa, attn_pre = seam(
                    "attn",
                    res0,
                    None,
                    None,
                    None,
                    pre0 if case.entry == "settled" else None,
                )
            swa_ind, swa_ptr = build_ragged_indices_from_dense(
                s["swa_indices"].reshape(M, -1), s["swa_lens"]
            )
            top_ind = top_ptr = None
            if ratio:
                top_ind, top_ptr, _ = compute_global_topk_ragged_indices_and_indptr(
                    s["topk"], s["token_to_req"], s["comp_block_table"],
                    s["comp_entries"], s["slot_mapping"] >= 0,
                )  # fmt: skip
            x8, xs = mxfp8_e4m3_quantize(xa)
            qr_kv = rocm_mxfp8_block32_gemm(x8, xs, wqkv, wqkv_s, bf)
            qr, kv = fused_q_kv_rmsnorm(
                qr_kv[:, :Q_RANK], qr_kv[:, Q_RANK:], q_norm, kv_norm, cfg.rms_norm_eps
            )
            q8, qs = mxfp8_e4m3_quantize(qr)
            q = rocm_mxfp8_block32_gemm(q8, qs, wq_b, wq_b_s, bf).view(M, H, HEAD_DIM)
            swa2d = s["swa_cache"].view(s["swa_cache"].shape[0], -1)
            q_roped = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q, kv, swa2d, s["slot_mapping"], s["positions"], cos_sin, H,
                cfg.rms_norm_eps, SWA_BLOCK, False, False, True, False,
            )  # fmt: skip
            attn = torch.empty(M, H, HEAD_DIM, dtype=bf, device=device)
            rocm_aiter_ops.triton_sparse_mla_fwd(
                q_roped, s["swa_cache"], attn, HEAD_DIM**-0.5, swa_ptr, swa_ind,
                attn_sink=sink, extra_kv_buffer=s["comp_cache"],
                extra_kv_indptr=top_ptr, extra_kv_indices=top_ind,
            )  # fmt: skip
            a8 = torch.empty(M, H * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
            a_s = torch.empty(M, H * HEAD_DIM // 32, dtype=torch.uint8, device=device)
            rocm_inverse_rope_mxfp8_rows(attn, s["positions"], cos_sin, ROPE, a8, a_s)
            z = torch.ops.vllm.rocm_dsv41_mxfp8_wo_a_bmm(
                a8, a_s, wo_a, wo_a_s, G, O_RANK
            )
            z8, zs = mxfp8_e4m3_quantize(z)
            part = rocm_mxfp8_block32_gemm(z8, zs, wo_b, wo_b_s, bf).contiguous()
            a = tensor_model_parallel_all_reduce(part.clone())
            res2, post2, comb2, xf, ffn_pre = seam("ffn", res, a, post, comb, attn_pre)
            out = moe_out(xf)
            cache.copy_(empty)

        # ---- the whole layer: K1 + K2
        meta = dict(
            topk_indices=s["topk"], comp_cache=s["comp_cache"],
            comp_block_table=s["comp_block_table"], fold=case.entry == "fold",
        )  # fmt: skip
        if case.index:
            topk = torch.full_like(s["topk"], -1)
            meta.update(topk_indices=topk, select=lambda *_: topk.copy_(s["topk"]))
        ins = (x0, res0, post0, comb0, pre0)
        args = (
            mw, *ins, s["positions"], s["slot_mapping"], s["swa_cache"],
            s["swa_indices"], s["swa_lens"], s["token_to_req"],
        )  # fmt: skip
        taps: dict = {}
        with _k1_taps(runner, taps):
            o = runner.forward(*args, **meta)
        torch.accelerator.synchronize()
        tag = f"{'draft ' if draft else ''}{case}"
        # the keys K2 attended over, as the step's metadata lists them
        keys, lens = _key_table(s, ratio)
        assert torch.equal(runner.kt[: M * KEYS].view(M, KEYS).cpu(), keys), tag
        assert torch.equal(runner.klen[: 2 * M].view(2, M).cpu(), lens), tag
        layer_io(tag, case, s, cos_sin, ins, o, taps["k1"], empty)
        res_a, post_a, comb_a, pre_a = (
            t[:M].clone()
            for t in (runner.res_mid, runner.post_a, runner.comb_a, runner.pre_a)
        )

        off = scratch_layout(M, tp, experts, topk_k, runner.ug_groups)["normed"][0]
        normed = runner.scratch(M)[off : off + M * HIDDEN * 2].view(bf).view(M, HIDDEN)
        o_n = [t[:n] for t in o]
        if case.vllm:
            res_n, res2_n, xf_n, normed_n = res[:n], res2[:n], xf[:n], normed[:n]
            assert _cos(runner.res_mid[:n], res_n) > 0.99999, tag
            for got, want in (
                (runner.post_a, post),
                (runner.comb_a, comb),
                (runner.pre_a, attn_pre),
            ):
                torch.testing.assert_close(
                    got[:n].view_as(want[:n]), want[:n], rtol=1e-4, atol=1e-5
                )
            same_in = moe_out(normed_n)
            # relative errors about 3x what the kernels show (0.3-0.5%)
            assert _cos(o_n[1], res2_n) > 0.9999 and _rel(o_n[1], res2_n) < 0.01, tag
            assert _cos(normed_n, xf_n) > 0.999 and _rel(normed_n, xf_n) < 0.012, tag
            assert _cos(o_n[0], same_in) > 0.9999, tag
            assert _rel(o_n[0], same_in) < 0.015, tag
            # per token, where both FFN inputs pick the same experts
            same = (top(xf_n) == top(normed_n)).all(1)
            assert same.float().mean() >= 0.5, tag
            tok = torch.nn.functional.cosine_similarity(
                o_n[0].float(), out[:n].float(), dim=1
            )
            assert tok[same].min() > 0.99, tag

        if case.pad:
            # the pad rows leave the step's rows as they are without them
            cut = _rows(s, 0, n)
            alone = runner.forward(
                mw, *(t[:n].contiguous() for t in ins),
                *(cut[k] for k in (
                    "positions", "slot_mapping", "swa_cache", "swa_indices",
                    "swa_lens", "token_to_req",
                )),
                **meta,
            )  # fmt: skip
            for got, want in zip(o_n, alone):
                assert torch.equal(got, want), tag

        ffn = None
        if not draft:
            # ---- the FFN launch, from an unreduced wo_b output
            fin = (
                (res, post, comb, attn_pre)
                if case.vllm
                else (res_a, post_a, comb_a, pre_a)
            )
            f = runner.ffn(mw, part, *fin)
            torch.accelerator.synchronize()
            attn_parts, _ = _peer(runner, M, int(runner.epoch[0]) - 1)
            assert torch.equal(attn_parts[rank], part), tag
            ffn_io(f"{tag} FFN launch", M, *fin, f, case.hard)
            if case.vllm:
                torch.testing.assert_close(f[1][:n], res2_n, rtol=2**-8, atol=1e-6)
                torch.testing.assert_close(normed[:n], xf_n, rtol=2**-7, atol=1e-6)
                assert _cos(f[0][:n], out[:n]) > 0.9999, tag
                assert _rel(f[0][:n], out[:n]) < 0.015, tag
            ffn = runner._outs(M, fin[0])
        torch.accelerator.synchronize()

        whole = runner._outs(M, res0)

        def step():
            runner.forward(*args, **meta, outs=whole)
            if ffn is None:
                return [*whole]
            runner.ffn(mw, part, *fin, outs=ffn)
            return [*whole, *ffn]

        return step

    return runner, check


def _replay(steps: list, launches: dict) -> None:
    """The hand-off protocol under back-to-back launches: the steps alternating
    in one graph, every replay bit for bit the first's. ``launches``: each
    runner's launches in a run of the steps; every one moves its runner's
    epoch, which the ranks agree on."""
    from vllm.distributed import tensor_model_parallel_all_gather

    first = {r: int(r.epoch[0]) for r in launches}
    outs = [t for step in steps for t in step()]
    torch.accelerator.synchronize()
    ref = [t.clone() for t in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in steps:
            step()
    for i in range(REPLAYS):
        graph.replay()
        if i % 50 == 49 or i == REPLAYS - 1:
            torch.accelerator.synchronize()
            bad = [
                j for j, (t, r) in enumerate(zip(outs, ref)) if not torch.equal(t, r)
            ]
            assert not bad, f"replay {i}: outputs {bad} differ from the first run"
    for runner, n in launches.items():
        epoch = runner.epoch[:1].clone()
        assert int(epoch) == first[runner] + n * (REPLAYS + 1), (int(epoch), first)
        ranks = tensor_model_parallel_all_gather(epoch, 0)
        assert torch.equal(ranks, epoch.expand_as(ranks)), ranks.tolist()


def _check_layers(rank: int, tp: int, device) -> None:
    runner, check = _bench(rank, tp, device)
    steps = [check(Case(1, 1, 6, vllm=True)), check(Case(2, 8, 6, vllm=True))]
    _replay(steps, {runner: 2 * len(steps)})


def _check_variants(rank: int, tp: int, device) -> None:
    """Steps of 1 to 33 rows (a GEMV pass takes 16), the first layer's and an
    Engram layer's settled residual, a pad row alone in a pass, contexts under
    the window and the top-k, and inputs that hit the clamps and zero groups."""
    runner, check = _bench(rank, tp, device)
    steps = [
        check(Case(0, 1, 1, entry="embedding")),
        check(Case(0, 16, 1, pad=1, short=True)),
        check(Case(2, 17, 1, short=True)),
        check(Case(1, 33, 1, entry="settled", short=True)),
        check(Case(1, 2, 6, hard=True)),
    ]
    _replay(steps, {runner: 2 * len(steps)})


def _check_draft(rank: int, tp: int, device) -> None:
    """The DSpark draft's runner on its first layer (the broadcast embedding)
    and a later one, beside the target's runner in one graph: each has its own
    epoch and peer memory."""
    draft, check_draft = _bench(rank, tp, device, draft=True)
    target, check_target = _bench(rank, tp, device)
    drafts = [
        check_draft(Case(0, 8, DRAFTS, entry="embedding", short=True)),
        check_draft(Case(0, 1, DRAFTS)),
    ]
    steps = [check_target(Case(1, 1, 6)), *drafts]
    _replay(steps, {target: 2, draft: len(drafts)})


def _check_index_source(rank: int, tp: int, device) -> None:
    """An index source's K2 lists its keys itself, after vLLM's indexer: it
    must not see the key counts the previous launch (a window-only layer's)
    left."""
    _, check = _bench(rank, tp, device)
    check(Case(0, 17, 1, short=True))
    check(Case(2, 17, 1, short=True, index=True))


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
@pytest.mark.parametrize("tp", [2, 4])
def test_mono_decode_layer_matches_vllm(monkeypatch: pytest.MonkeyPatch, tp: int):
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    multi_process_parallel(monkeypatch, tp, 1, _mono_numerics)


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
@pytest.mark.parametrize("tp", [2, 4])
def test_mono_decode_variants_match_vllm(monkeypatch: pytest.MonkeyPatch, tp: int):
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    multi_process_parallel(monkeypatch, tp, 1, _mono_variants)


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
@pytest.mark.parametrize("tp", [2, 4])
def test_mono_draft_layer_matches_vllm(monkeypatch: pytest.MonkeyPatch, tp: int):
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    multi_process_parallel(monkeypatch, tp, 1, _mono_draft)


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
def test_mono_index_source_matches_vllm(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    multi_process_parallel(monkeypatch, 2, 1, _mono_index)


# ---------------------------------------------------------------- gfx942


@ray.remote(num_gpus=1, max_calls=1)
def _mono_replay_942(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.workspace import init_workspace_manager

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        rocm_aiter_ops.refresh_env_variables()
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        _replay_942(rank, tp_size, device)


def _replay_942(rank: int, tp: int, device) -> None:
    """The composed gfx942 runner at TP4 on the copies of the weights that the
    model makes at load. The steps have 6 rows (one GEMV pass), 8 rows (two
    passes, the second one padded) and 42 rows (seven passes, the most that
    the gfx942 kernels take). The padded pass must leave the real rows alone,
    and the steps alternating in one CUDA graph must give the first run's
    outputs bit for bit at every replay."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v41.amd import mono_decode
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS, Dims
    from vllm.models.deepseek_v41.amd.mono.runner import (
        AttnWeights,
        DSV41MonoLayer,
        MonoLayerWeights,
    )
    from vllm.models.deepseek_v41.amd.mono.weights942 import linear_copy
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.utils.torch_utils import set_default_torch_dtype

    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tp,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=65536,
        language_model_only=True,
    ).create_engine_config()
    cfg = vc.model_config.hf_config
    cfg = getattr(cfg, "text_config", cfg)
    # replicated tensors alike on every rank; each rank's own TP shards
    g = torch.Generator().manual_seed(1000)
    gs = torch.Generator().manual_seed(1001 + rank)
    H, G = N_HEADS // tp, O_GROUPS // tp
    bf = torch.bfloat16

    with set_current_vllm_config(vc), set_default_torch_dtype(bf), device:
        moe = DeepseekV4MoE(vc, prefix="model.layers.21.ffn")
    e, sh = moe.experts.routed_experts, moe.shared_experts
    # The hooks that the model sets at load (mono_decode._copy_942_at_load).
    # The routed experts are copied from the plain tensors before vLLM's MoE
    # backend converts them, and the shared expert from the bf16 weights that
    # vLLM dequantizes at load.
    mono_decode._capture_plain_experts(e)
    for lin in (sh.gate_up_proj, sh.down_proj):
        mono_decode._copy_linear_at_load(lin)
    _random_moe(
        moe,
        torch.Generator(device=device).manual_seed(2001 + rank),
        torch.Generator(device=device).manual_seed(2000),
        inter=mono_decode.INTER_TP4,
    )
    process_weights_after_loading(moe, vc.model_config, device)

    def f32(*shape, s=1.0, gen=g):
        return (torch.randn(*shape, generator=gen) * s).to(device)

    def attn_copy(n: int, k: int, gen):
        """A random MXFP8 attention linear as vLLM keeps it on gfx942, bf16
        values with each 32 x 32 block's code repeated over the block's 32
        rows, and its copy for the kernels, as the load hook makes it."""
        w8, s = _mxfp8(n, k, gen, device)
        mul = torch.exp2(s.float() - 127.0)
        mul = mul.repeat_interleave(32, 0).repeat_interleave(32, 1)
        return linear_copy((w8.float() * mul).to(bf), s.repeat_interleave(32, 0))

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
    wqkv = attn_copy(1792, HIDDEN, g)
    wq_b = attn_copy(H * HEAD_DIM, Q_RANK, gs)
    wo_a = attn_copy(G * O_RANK, H // G * HEAD_DIM, gs)
    wo_b = attn_copy(HIDDEN, G * O_RANK, gs)
    q_norm, kv_norm = (1 + f32(Q_RANK, s=0.1)).to(bf), (1 + f32(HEAD_DIM, s=0.1)).to(bf)
    sink = f32(H, gen=gs)
    w13, w13_s, w2, w2_s = e.mono942
    (sgu, sgu_s), (sw2, sw2_s) = sh.gate_up_proj.mono942, sh.down_proj.mono942
    runner = DSV41MonoLayer(tp, rank, get_tp_group().cpu_group, device)

    def inputs(M: int):
        """The layer's inputs at the attention seam, the same on every rank:
        x, residual, post mix, res mix, pre mix."""
        gi = torch.Generator().manual_seed(11)
        res = torch.randn(M, HC, HIDDEN, generator=gi)
        res = (res * torch.rsqrt(res.pow(2).mean(-1, keepdim=True))).to(bf).to(device)
        x = (torch.randn(M, HIDDEN, generator=gi) * 0.5).to(bf).to(device)
        comb = torch.rand(M, HC, HC, generator=gi) + 0.1
        for _ in range(20):
            comb = comb / comb.sum(-1, keepdim=True)
            comb = comb / comb.sum(-2, keepdim=True)
        post = (torch.sigmoid(torch.randn(M, HC, 1, generator=gi)) * 2).to(device)
        pre = (torch.sigmoid(torch.randn(M, HC, generator=gi)) + 1e-6).to(device)
        return x, res, post, comb.to(device), pre

    def launch(mw, s, ins, part, whole=None, ffn=None):
        """The whole layer (K1 + K2) and the FFN launch of one step. The FFN
        launch takes the step's inputs as its attention seam's outputs."""
        x, res, post, comb, pre = ins
        whole = runner.forward(
            mw, x, res, post, comb, pre, s["positions"], s["slot_mapping"],
            s["swa_cache"], s["swa_indices"], s["swa_lens"], s["token_to_req"],
            topk_indices=s["topk"], comp_cache=s["comp_cache"],
            comp_block_table=s["comp_block_table"], outs=whole,
        )  # fmt: skip
        ffn = runner.ffn(mw, part, res, post, comb, pre, outs=ffn)
        return [*whole, *ffn]

    steps = []  # each case's launches, on fixed buffers, for the replay stress
    for ratio, reqs, T in ((1, 1, 6), (0, 8, 1), (2, 7, 6)):
        with set_current_vllm_config(VllmConfig()):
            rope = build_deepseek_v4_rope(
                cfg, head_dim=HEAD_DIM, rope_head_dim=ROPE,
                max_position_embeddings=cfg.max_position_embeddings,
                compress_ratio=ratio,
            )  # fmt: skip
        cos_sin = rope.cos_sin_cache.to(device=device, dtype=torch.float32)
        s = _decode_step(ratio, reqs, T, kv_norm, cos_sin, seed=7 + ratio, fnuz=True)
        M = s["M"]
        attn = AttnWeights(
            layer_id=21, wqkv=wqkv[0], wqkv_scale=wqkv[1], q_norm=q_norm,
            kv_norm=kv_norm, wq_b=wq_b[0], wq_b_scale=wq_b[1], wo_a=wo_a[0],
            wo_a_scale=wo_a[1], wo_b=wo_b[0], wo_b_scale=wo_b[1], attn_sink=sink,
            cos_sin=cos_sin, ratio=ratio,
        )  # fmt: skip
        attn.check(Dims(tp))
        mw = MonoLayerWeights(
            attn=attn,
            hc_attn_fn=seams["attn"]["fn"], hc_attn_scale=seams["attn"]["scale"],
            hc_attn_base=seams["attn"]["base"], attn_norm=seams["attn"]["norm"],
            hc_ffn_fn=seams["ffn"]["fn"], hc_ffn_scale=seams["ffn"]["scale"],
            hc_ffn_base=seams["ffn"]["base"], ffn_norm=seams["ffn"]["norm"],
            gate_w=moe.gate.weight, bias=moe.gate.e_score_correction_bias,
            w13=w13, w13_s=w13_s, w2=w2, w2_s=w2_s, sgu=sgu, sgu_s=sgu_s,
            sw2=sw2, sw2_s=sw2_s,
        )  # fmt: skip
        mw.check_942(moe.n_routed_experts)
        ins = inputs(M)
        # this rank's unreduced wo_b output, for the FFN launch
        part = (torch.randn(M, HIDDEN, generator=gs) * 0.1).to(bf).to(device)
        whole, ffn = runner._outs(M, ins[1]), runner._outs(M, ins[1])
        outs = launch(mw, s, ins, part, whole, ffn)
        tag = f"ratio {ratio}, {reqs} x {T}"
        # (output, rows) of every output with a value that is not finite
        nonfinite = [
            (j, (~t.isfinite()).reshape(M, -1).any(1).nonzero().flatten().tolist())
            for j, t in enumerate(outs)
            if not bool(t.isfinite().all())
        ]
        assert not nonfinite, f"{tag}: rows that are not finite, by output: {nonfinite}"
        # the arch-independent stages exactly: the key table, the attention
        # seam's residual, the all-reduces of the whole layer and the FFN launch
        keys, lens = _key_table(s, ratio)
        assert torch.equal(runner.kt[: M * KEYS].view(M, KEYS).cpu(), keys), tag
        assert torch.equal(runner.klen[: 2 * M].view(2, M).cpu(), lens), tag
        x, res, post, comb, pre = ins
        assert torch.equal(runner.res_mid[:M], _fold(x, res, post, comb)), tag
        ep = int(runner.epoch[0]) - 1
        _check_reduces(tag, runner, M, ep, outs[5:], res, post, comb)
        mid = (runner.res_mid[:M], runner.post_a[:M], runner.comb_a[:M])
        _check_reduces(tag, runner, M, ep - 1, outs[:5], *mid)

        if M == 8:
            # The second GEMV pass holds rows 6 and 7 and four padding rows.
            # Each row must come out as it does in a step without the other
            # rows: rows 0 .. 5 as a step of 6 rows, rows 6 and 7 as a step of
            # 2 rows.
            for a, b in ((0, 6), (6, 8)):
                cut = [t[a:b].contiguous() for t in (*ins, part)]
                alone = launch(mw, _rows(s, a, b), cut[:5], cut[5])
                for i, (t, want) in enumerate(zip(outs, alone)):
                    assert torch.equal(t[a:b], want), (
                        f"{tag}: output {i} of rows {a} .. {b - 1} differs from "
                        f"a step of those rows alone, by up to "
                        f"{(t[a:b].float() - want.float()).abs().max().item()}"
                    )

        steps.append(
            lambda mw=mw, s=s, ins=ins, part=part, whole=whole, ffn=ffn: launch(
                mw, s, ins, part, whole, ffn
            )
        )

    # the hand-off protocol under back-to-back launches: the three step widths
    # alternating in one graph, every replay bit for bit the first's
    outs = [t for step in steps for t in step()]
    torch.accelerator.synchronize()
    ref = [t.clone() for t in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in steps:
            step()
    for i in range(REPLAYS):
        graph.replay()
        if i % 50 == 49 or i == REPLAYS - 1:
            torch.accelerator.synchronize()
            bad = [
                j for j, (t, r) in enumerate(zip(outs, ref)) if not torch.equal(t, r)
            ]
            assert not bad, f"replay {i}: outputs {bad} differ from the first run"


@pytest.mark.skipif(not _on_gfx942(), reason="the gfx942 mono kernels")
def test_mono_decode_942_replays(monkeypatch: pytest.MonkeyPatch):
    # The gfx942 kernels build for TP4 only.
    if torch.accelerator.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    multi_process_parallel(monkeypatch, 4, 1, _mono_replay_942)
