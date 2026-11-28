# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An index source layer's compressor tail in one launch
(Triton: launch-bound glue, ~nothing to win from asm).

vLLM runs four kernels after the compressor GEMM: on the aux stream
`_fused_save_compress_norm` (ring save, ratio-2 softmax pool, RMSNorm →
bf16 latent) and `_rope_quant_insert` (the compressed-cache record); after
the streams join, on the critical path, the indexer's `wk` (hipBLASLt) and
aiter's `_k_norm_rope_mxfp4_cache`. `comp_tail` is vLLM's first kernel
(same grid, programs and ring handling; ROCm's unjoined row loads, no PCP)
whose latent epilogue also does the other three for that token, so the whole
chain stays on the aux stream:

1. latent = bf16(RMSNorm(pooled)) (`_store_latent`), also kept in `lat`
   (the K GEMV reads it back in chunks: one global round trip, same program);
2. the compressed insert, fp8_ds_mla 584 B record (`_rope_quant_insert_kernel`
   with gfx950's NaN sanitising) at the compressed cache's slot;
3. k = bf16(latent · wkᵀ) (fp32, 4 chunks of 128: not hipBLASLt's order, so
   k can be 1 bf16 ulp off), k_norm → bf16, GPT-J RoPE on the last 64 dims at
   the group start → bf16, MXFP4 (aiter's ceil scales and pack instruction)
   in aiter's page shuffle at the indexer K cache's slot
   (`_k_norm_rope_mxfp4_cache_kernel`).

Slot mappings keep vLLM's meaning (−1 skips); records are written only at
group ends, as vLLM's kernels do.

rstd (the index layers' front, k1_runner.seam_front): the GEMM ran on y' = bf16(x1 ·
attn_norm.w) without the norm's rstd; each token's kv_score row is scaled by
rstd[t] = rsq(Σx1² / 5120 + eps) as it is loaded, from the attention seam's
partials (`ffn_seam.source_v2`: `[token][160 slices][32]` f32, Σx1² at 25).
The ring keeps the scaled rows.
"""

import torch
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

from vllm.triton_utils import tl, triton

D, KD = 512, 128  # latent dim, indexer head dim


@triton.jit
def _finish(
    pooled,
    t,
    positions,
    norm_w,
    eps,
    lat,
    ccache,
    cslots,
    ccs,
    ccs_stride,
    cpage_stride,
    wk,
    knorm_w,
    keps,
    kcs,
    kcs_stride,
    kcache,
    kslots,
    kpage_stride,
    kpage,
    CBLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    NPT: tl.constexpr,
    DPT: tl.constexpr,
    SLANES: tl.constexpr,
):
    d = tl.arange(0, 512)
    position = tl.load(positions + t)
    gpos = position // RATIO * RATIO
    # 1. latent (_store_latent)
    w = tl.load(norm_w + d).to(tl.float32)
    variance = tl.sum(pooled * pooled, 0) / 512
    latb = (pooled * tl.rsqrt(variance + eps) * w).to(tl.bfloat16)
    tl.store(lat + t.to(tl.int64) * 512 + d, latb)
    normed = latb.to(tl.float32)

    # 2. compressed insert (_rope_quant_insert_kernel, 584 B records,
    # SANITIZE_CACHE_NANS)
    slot = tl.load(cslots + t)
    if slot >= 0:
        page = ccache + (slot // CBLOCK).to(tl.int64) * cpage_stride
        values = page + (slot % CBLOCK) * 576
        scales = page + CBLOCK * 576 + (slot % CBLOCK) * 8
        quant = tl.reshape(normed, (8, 64))
        amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
        exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
        scaled = quant * tl.reshape(tl.exp2(-exponent), (8, 1))
        fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
        packed = tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))
        tl.store(values + d, packed, d < 448)
        s8 = tl.arange(0, 8)
        encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 254.0)
        tl.store(scales + s8, encoded.to(tl.uint8), s8 < 7)
        tl.store(scales + 7, tl.full((), 0, tl.uint8))
        even, odd = tl.split(tl.reshape(normed, (256, 2)))
        pair = tl.arange(0, 256) - 224
        cs = ccs + gpos * ccs_stride
        c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
        s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
        rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
        rotated = tl.where(rotated == rotated, rotated, 0.0)
        rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
        tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)

    # 3. indexer K store
    kslot = tl.load(kslots + t)
    if kslot >= 0:
        tl.debug_barrier()  # every lane's latent store is in before the chunk reads
        n = tl.arange(0, 128)
        acc = tl.zeros((128,), tl.float32)
        for c4 in tl.static_range(4):
            xc = tl.load(lat + t.to(tl.int64) * 512 + c4 * 128 + n).to(tl.float32)
            wc = tl.load(wk + n[:, None] * 512 + c4 * 128 + n[None, :]).to(tl.float32)
            acc += tl.sum(wc * xc[None, :], 1)
        k = acc.to(tl.bfloat16).to(tl.float32)
        kw = tl.load(knorm_w + n).to(tl.float32)
        kvar = tl.sum(k * k, axis=0) / 128
        k = (k * tl.rsqrt(kvar + keps) * kw).to(tl.bfloat16).to(tl.float32)
        ke, ko = tl.split(tl.reshape(k, (64, 2)))
        rp = tl.arange(0, 64) - 32
        is_rope = rp >= 0
        cidx = tl.maximum(rp, 0)
        kb = kcs + gpos * kcs_stride
        cos = tl.load(kb + cidx, mask=is_rope, other=1.0)
        sin = tl.load(kb + 32 + cidx, mask=is_rope, other=0.0)
        ne = (ke * cos - ko * sin).to(tl.bfloat16).to(tl.float32)
        no = (ko * cos + ke * sin).to(tl.bfloat16).to(tl.float32)
        k_rot = tl.reshape(tl.join(ne, no), (1, 128))
        kq, e8 = _mxfp4_quant_op(k_rot, 128, 1, 32, SCALING_MODE=1, USE_ASM=True)
        kq = tl.reshape(kq, (64,))
        e8 = tl.reshape(e8, (4,))
        pg = kcache + (kslot // kpage).to(tl.int64) * kpage_stride
        p = kslot % kpage
        byte = tl.arange(0, 64)
        si = tl.arange(0, 4)
        S_HI: tl.constexpr = 4 // SLANES
        grp = p // NPT
        lane = p % NPT
        voff = grp * (NPT * 64) + byte // DPT * (NPT * DPT) + lane * DPT + byte % DPT
        soff = grp * (NPT * 4) + si % SLANES * (NPT * S_HI) + lane * S_HI + si // SLANES
        tl.store(pg + voff, kq)
        tl.store(pg + kpage * 64 + soff, e8)


@triton.jit
def _ld(p, off, SPLIT: tl.constexpr, SPLIT_STRIDE: tl.constexpr):
    """raw[off], or the sum of comp_gemm's SPLIT partial planes in plane order."""
    v = tl.load(p + off)
    for s in tl.static_range(1, SPLIT):
        v += tl.load(p + s * SPLIT_STRIDE + off)
    return v


@triton.jit
def _rstd(rs_part, t, rs_eps, RSTD: tl.constexpr):
    """Token t's attention-norm rstd from the seam partials (RSTD:
    `[token][160 slices][32]` f32, Σx1² at 25), else 1."""
    if RSTD:
        cs = tl.arange(0, 256)
        sq = tl.load(
            rs_part + t.to(tl.int64) * (160 * 32) + cs * 32 + 25, cs < 160, other=0.0
        )
        return tl.rsqrt(tl.sum(sq, 0) / 5120 + rs_eps)
    else:
        return 1.0


@triton.jit(do_not_specialize=["num_tokens", "num_reqs"])
def _comp_tail_kernel(
    raw,
    positions,
    state,
    state_slots,
    query_start_loc,
    req_ids,
    norm_w,
    eps,
    lat,
    ccache,
    cslots,
    ccs,
    ccs_stride,
    cpage_stride,
    wk,
    knorm_w,
    keps,
    kcs,
    kcs_stride,
    kcache,
    kslots,
    kpage_stride,
    kpage,
    num_tokens,
    num_reqs,
    rs_part,
    rs_eps,
    RAW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    CBLOCK: tl.constexpr,
    NPT: tl.constexpr,
    DPT: tl.constexpr,
    SLANES: tl.constexpr,
    SPLIT: tl.constexpr,
    SPLIT_STRIDE: tl.constexpr,
    RSTD: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, 512)
    if RATIO == 2:  # noqa: SIM102
        if pid < num_reqs:
            # request program: the group the chunk's first token closes with its ring
            # row, then the ring tail
            start = tl.load(query_start_loc + pid)
            end = tl.load(query_start_loc + pid + 1)
            if start >= end:
                return
            state_slot = tl.load(state_slots + start)
            if state_slot < 0:
                return
            position = tl.load(positions + start)
            if (position + 1) % 2 == 0:
                ring = state + (state_slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                prev = ring + ((position - 1) % STATE_BLOCK) * STATE_ROW_STRIDE
                current = raw + start.to(tl.int64) * RAW_STRIDE
                kv_prev = tl.load(prev + d)
                score_prev = tl.load(prev + 512 + d)
                rs = _rstd(rs_part, start, rs_eps, RSTD)
                kv_current = _ld(current, d, SPLIT, SPLIT_STRIDE) * rs
                score_current = _ld(current, 512 + d, SPLIT, SPLIT_STRIDE) * rs
                tl.debug_barrier()  # every lane has read the ring before the tail store
                peak = tl.maximum(score_prev, score_current)
                weight_prev = tl.exp(score_prev - peak)
                weight_current = tl.exp(score_current - peak)
                pooled = (kv_prev * weight_prev + kv_current * weight_current) / (
                    weight_prev + weight_current
                )
                _finish(
                    pooled,
                    start,
                    positions,
                    norm_w,
                    eps,
                    lat,
                    ccache,
                    cslots,
                    ccs,
                    ccs_stride,
                    cpage_stride,
                    wk,
                    knorm_w,
                    keps,
                    kcs,
                    kcs_stride,
                    kcache,
                    kslots,
                    kpage_stride,
                    kpage,
                    CBLOCK,
                    RATIO,
                    NPT,
                    DPT,
                    SLANES,
                )
            num_rows = tl.minimum(end - start, STATE_BLOCK)
            for k in tl.range(0, num_rows):
                token = end - num_rows + k
                slot = tl.load(state_slots + token)
                if slot >= 0:
                    row = (
                        state
                        + (slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                        + (slot % STATE_BLOCK) * STATE_ROW_STRIDE
                    )
                    src = raw + token.to(tl.int64) * RAW_STRIDE
                    rs = _rstd(rs_part, token, rs_eps, RSTD)
                    tl.store(row + d, _ld(src, d, SPLIT, SPLIT_STRIDE) * rs)
                    tl.store(row + 512 + d, _ld(src, 512 + d, SPLIT, SPLIT_STRIDE) * rs)
            return

    # group program: one token (ratio 1) or the token of the pair that ends a group
    # (ratio 2)
    t = (pid - num_reqs) * RATIO
    if RATIO == 2:
        t += (tl.load(positions + t) + 1) % 2
    state_slot = tl.load(state_slots + t, t < num_tokens, other=-1)
    if state_slot < 0:
        return
    if RATIO == 1:
        pooled = _ld(raw + t.to(tl.int64) * RAW_STRIDE, d, SPLIT, SPLIT_STRIDE) * _rstd(
            rs_part, t, rs_eps, RSTD
        )
    else:
        if t == tl.load(query_start_loc + tl.load(req_ids + t)):
            return
        rows = raw + (t.to(tl.int64) - 1 + tl.arange(0, 2)[:, None]) * RAW_STRIDE
        if RSTD:
            rs = tl.where(
                tl.arange(0, 2) == 0,
                _rstd(rs_part, t - 1, rs_eps, RSTD),
                _rstd(rs_part, t, rs_eps, RSTD),
            )
        else:
            rs = tl.full((2,), 1.0, tl.float32)
        kv = _ld(rows, d[None, :], SPLIT, SPLIT_STRIDE) * rs[:, None]
        score = _ld(rows, 512 + d[None, :], SPLIT, SPLIT_STRIDE) * rs[:, None]
        pooled = tl.sum(kv * tl.softmax(score, 0), 0)
    _finish(
        pooled,
        t,
        positions,
        norm_w,
        eps,
        lat,
        ccache,
        cslots,
        ccs,
        ccs_stride,
        cpage_stride,
        wk,
        knorm_w,
        keps,
        kcs,
        kcs_stride,
        kcache,
        kslots,
        kpage_stride,
        kpage,
        CBLOCK,
        RATIO,
        NPT,
        DPT,
        SLANES,
    )


def comp_tail(
    kv_score,
    positions,
    ratio,
    state_cache,
    state_slots,
    query_start_loc,
    req_ids,
    norm_w,
    eps,
    lat,
    ccache,
    cslots,
    ccs,
    wk,
    knorm_w,
    keps,
    kcs,
    kcache,
    kslots,
    shuffle,
    split: int = 1,
    rs_part=None,
    rs_eps: float = 0.0,
    num_warps: int = 4,
):
    """kv_score fp32 [T, 512·ratio] → ring save (ratio 2), compressed insert
    (ccache uint8 [blocks, CBLOCK, 584] pages, slots cslots, cos/sin ccs) and
    the indexer K store (kcache uint8 pages of kpage entries × 68 B, slots
    kslots, cos/sin kcs, shuffle = (n_per_tile, d_per_tile, scale_lanes)).
    state_slots: the ring slots (ratio 2) or, as vLLM, the compressed cache's
    slots (ratio 1; they only mark valid tokens). lat: bf16 scratch ≥ [T, 512].
    split > 1: kv_score is comp_gemm's partials [split, ≥ T, 512·ratio] (summed
    here in plane order; the ring gets the sums). rs_part (the attention seam's
    partials [≥ T, 160, 32] f32) + rs_eps (attn_norm's): rows × rstd[t]."""
    assert (
        ratio in (1, 2)
        and kv_score.dtype == torch.float32
        and kv_score.shape[-1] == 512 * ratio
    )
    if split > 1:
        assert (
            kv_score.dim() == 3
            and kv_score.shape[0] == split
            and kv_score.is_contiguous()
        )
        sstride = kv_score.stride(0)
        kv_score = kv_score[0]
    else:
        sstride = 0
    assert kv_score.stride(1) == 1 and kv_score.stride(0) % 16 == 0
    assert ccache.dtype == torch.uint8 and ccache.shape[-1] == 584, (
        "fp8_ds_mla records only"
    )
    assert wk.shape == (KD, D) and wk.is_contiguous() and wk.dtype == torch.bfloat16
    num_tokens = state_slots.numel()
    assert (
        cslots.numel() >= num_tokens
        and kslots.numel() >= num_tokens
        and lat.shape[0] >= num_tokens
    )
    if num_tokens == 0:
        return
    kpages = kcache.view(kcache.shape[0], -1)
    kpage = kpages.shape[1] // (KD // 2 + KD // 32)
    cpages = ccache.view(ccache.shape[0], -1)
    if ratio == 2:
        assert state_cache.shape[2] == 1024 and state_cache.stride(2) == 1
        num_reqs = query_start_loc.numel() - 1
        st = (state_cache.stride(0), state_cache.stride(1), state_cache.shape[1])
    else:
        state_cache, query_start_loc, req_ids = kv_score, state_slots, state_slots
        num_reqs, st = 0, (1, 1, 1)
    npt, dpt, sl = (int(v) for v in shuffle)
    _comp_tail_kernel[(num_reqs + triton.cdiv(num_tokens, ratio),)](
        kv_score,
        positions,
        state_cache,
        state_slots,
        query_start_loc,
        req_ids,
        norm_w,
        eps,
        lat,
        ccache,
        cslots,
        ccs,
        ccs.stride(0),
        cpages.stride(0),
        wk,
        knorm_w,
        keps,
        kcs,
        kcs.stride(0),
        kcache,
        kslots,
        kpages.stride(0),
        kpage,
        num_tokens,
        num_reqs,
        rs_part if rs_part is not None else kv_score,
        float(rs_eps),
        RAW_STRIDE=kv_score.stride(0),
        STATE_STRIDE=st[0],
        STATE_ROW_STRIDE=st[1],
        STATE_BLOCK=st[2],
        RATIO=ratio,
        CBLOCK=ccache.shape[1],
        NPT=npt,
        DPT=dpt,
        SLANES=sl,
        SPLIT=split,
        SPLIT_STRIDE=sstride,
        RSTD=rs_part is not None,
        num_warps=num_warps,
    )


@triton.jit
def _comp_gemm_kernel(
    x,
    w,
    part,
    M,
    stride_x,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    PSTRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """part`[s] [M, N]` fp32 = x[:, K slice s] · w[:, K slice s]ᵀ (bf16)."""
    pid = tl.program_id(0)
    NT: tl.constexpr = N // BN
    mt = tl.cdiv(M, BM)
    pn = pid % NT
    pm = (pid // NT) % mt
    s = pid // NT // mt
    KS: tl.constexpr = K // SPLIT
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    mm = om[:, None] < M
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(s * KS, s * KS + KS, BK):
        a = tl.load(
            x + om[:, None].to(tl.int64) * stride_x + k + ok[None, :],
            mask=mm,
            other=0.0,
        )
        b = tl.load(w + on[:, None] * K + k + ok[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(part + s * PSTRIDE + om[:, None] * N + on[None, :], acc, mask=mm)


def gemm_cfg(m: int, n: int = 1024) -> dict:
    """comp_gemm's tiles by M and N (tuned with cold weights; µs vs hipBLASLt): N 1024:
    5.4 / 6.3 / 8.8 vs 9.6 / 12.1 / 14.4 at M 64 / 128 / 256; N 512: 3.8 / 4.3 / 5.3 vs
    8.0 / 8.3 / 10.9."""
    if n == 512:
        c = (
            (64, 32, 64, 16)
            if m <= 64
            else (64, 64, 64, 16)
            if m <= 128
            else (64, 64, 128, 8)
        )
    else:
        c = (
            (64, 32, 128, 8)
            if m <= 64
            else (64, 64, 128, 8)
            if m <= 128
            else (64, 64, 128, 4)
        )
    return dict(BM=c[0], BN=c[1], BK=c[2], split=c[3], num_warps=4, num_stages=3)


def comp_gemm(x, w, part, cfg=None):
    """X bf16 [M, K] (unit column stride), w bf16 [N, K] → part fp32
    [split, ≥ M, N] (`comp_tail(.., split=)`'s kv_score); returns it."""
    M, K = x.shape
    N = w.shape[0]
    cfg = cfg or gemm_cfg(M, N)
    sp, bm, bn, bk = cfg["split"], cfg["BM"], cfg["BN"], cfg["BK"]
    assert x.stride(1) == 1 and w.is_contiguous() and K % (sp * bk) == 0 and N % bn == 0
    assert (
        part.dim() == 3
        and part.shape[0] >= sp
        and part.shape[1] >= M
        and part.shape[2] == N
        and part.is_contiguous()
    )
    _comp_gemm_kernel[(sp * triton.cdiv(M, bm) * (N // bn),)](
        x,
        w,
        part,
        M,
        x.stride(0),
        N=N,
        K=K,
        SPLIT=sp,
        PSTRIDE=part.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=cfg["num_warps"],
        num_stages=cfg.get("num_stages", 2),
    )
    return part[:sp]
