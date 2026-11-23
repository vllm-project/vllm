# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels of the Hopper DSA layer (`dsa.HopperDsaLayer`).

Every kernel is launched with programmatic dependent launch. Before
`gdc_wait` it touches only weights and constant tables; it reads a
predecessor's output only after the wait, and releases its dependents
(`gdc_launch_dependents`) only after the wait as well, at a point chosen per
kernel. A dependent's wait then covers every earlier grid's writes.
"""

from vllm.triton_utils import tl, triton

INV_SQRT_128 = tl.constexpr(0.08838834764831845)  # 1 / sqrt(128), exact in FP32
FP8_MAX = tl.constexpr(448.0)
_BITS = 12  # first radix digit of the long-row select
TOP_BITS = tl.constexpr(_BITS)
TOP_BINS = tl.constexpr(1 << _BITS)


@triton.jit(do_not_specialize=["M", "N0", "N1"])
def gemm_kernel(
    x_ptr,
    w0_ptr,
    w1_ptr,
    w2_ptr,
    o_ptr,
    ws_ptr,
    ctr_ptr,
    M,
    sxm,
    sxb,
    swb,
    s_om,
    sob,
    N0,
    N1,
    N: tl.constexpr,
    K: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    NSTAGE: tl.constexpr,
    EARLY: tl.constexpr,
):
    """`o[m, n] = sum_k x[m, k] w[n, k]`, swap-AB (weight rows are the MMA M).

    The weight rows come from up to three tensors: rows `[0, N0)` of w0, then
    `[N0, N1)` of w1, then `[N1, N)` of w2; the boundaries are multiples of
    BN, so a tile reads one tensor. Program axis 2 is a batch of independent
    GEMMs (heads) offset by the `*b` strides. With SPLIT > 1 every split writes
    an FP32 tile, and the last split of a tile to finish (atomic counter) adds
    them in split order and stores: deterministic, no output atomics.

    With EARLY the dependents are released right after the PDL wait, else
    after the main loop.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)
    n0 = pid_n * BN
    if n0 < N0:
        w_ptr = w0_ptr
        first = 0
    elif n0 < N1:
        w_ptr = w1_ptr
        first = N0
    else:
        w_ptr = w2_ptr
        first = N1
    rn = n0 + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    k0 = pid_k * (K // SPLIT)
    wp = (
        w_ptr
        + pid_b.to(tl.int64) * swb
        + (rn - first)[:, None] * SWN
        + (k0 + rk)[None, :] * SWK
    )
    xp = x_ptr + pid_b * sxb + rm[None, :] * sxm + (k0 + rk)[:, None]
    nm = rn[:, None] < N
    mm = rm[None, :] < M
    tl.extra.cuda.gdc_wait()
    if EARLY:
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k in tl.range(0, K // SPLIT, BK, num_stages=NSTAGE):
        w = tl.load(wp + k * SWK, mask=nm, other=0.0)
        xt = tl.load(xp + k, mask=mm, other=0.0)
        acc = tl.dot(w, xt, acc)
    if not EARLY:
        tl.extra.cuda.gdc_launch_dependents()
    last = True
    if SPLIT > 1:
        tile = pid_b * tl.num_programs(0) + pid_n
        tb = tl.arange(0, BN)[:, None] * BM + tl.arange(0, BM)[None, :]
        tl.store(ws_ptr + (tile * SPLIT + pid_k) * (BN * BM) + tb, acc)
        tl.debug_barrier()
        old = tl.atomic_add(ctr_ptr + tile, 1, sem="acq_rel")
        last = old == SPLIT - 1
        if last:
            acc = tl.load(
                ws_ptr + (tile * SPLIT) * (BN * BM) + tb, cache_modifier=".cg"
            )
            for i in tl.static_range(1, SPLIT):
                acc += tl.load(
                    ws_ptr + (tile * SPLIT + i) * (BN * BM) + tb,
                    cache_modifier=".cg",
                )
            tl.store(ctr_ptr + tile, 0)
    if last:
        tl.store(
            o_ptr + pid_b * sob + rm[None, :] * s_om + rn[:, None],
            acc.to(o_ptr.dtype.element_ty),
            mask=nm & mm,
        )


@triton.jit
def _ld_bf16(y_ptr, off, mask):
    """A G1 output element as the BF16 value a BF16 GEMM would have stored."""
    return tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _layer_norm(x, w, b, eps, D: tl.constexpr):
    mean = tl.sum(x, 0) / D
    d = x - mean
    var = tl.sum(d * d, 0) / D
    return (d * tl.rsqrt(var + eps) * w + b).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (GROUPS * 2 * STRIDE,))


@triton.jit
def _hadamard(x, R: tl.constexpr):
    """The Hadamard-128 transform of each of the R 128-wide rows of the flat x,
    in the butterfly order and FP32 arithmetic of `fwht128_quant_fp8`."""
    x = _hadamard128_stage(x, R * 64, 1)
    x = _hadamard128_stage(x, R * 32, 2)
    x = _hadamard128_stage(x, R * 16, 4)
    x = _hadamard128_stage(x, R * 8, 8)
    x = _hadamard128_stage(x, R * 4, 16)
    x = _hadamard128_stage(x, R * 2, 32)
    x = _hadamard128_stage(x, R, 64)
    return x * INV_SQRT_128


@triton.jit(do_not_specialize=["M", "eps", "keps"])
def post_g1_kernel(
    y_ptr,
    qaw_ptr,
    kvw_ptr,
    eps,
    qc_ptr,
    kv_ptr,
    kv_sb,
    kslot_ptr,
    kn_w_ptr,
    kn_b_ptr,
    keps,
    ape_ptr,
    pos_ptr,
    islot_ptr,
    tslot_ptr,
    tail_ptr,
    idx8_ptr,
    idx32_ptr,
    M,
    SY: tl.constexpr,
    QA: tl.constexpr,
    KVR: tl.constexpr,
    GATE: tl.constexpr,
    KEY: tl.constexpr,
    BLOCK: tl.constexpr,
    KV_BS: tl.constexpr,
    NEXT_N: tl.constexpr,
    HD: tl.constexpr,
    KPOOL: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLK: tl.constexpr,
    TAIL_HALF: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    KPOOL_PROGRAMS: tl.constexpr,
):
    """Everything between the a-projection (G1, FP32 `[M, SY]`) and q_b.

    Programs `[0, 2M)`: `fused_q_kv_rmsnorm` of token `p // 2`, the q part
    (p even, into qc) or the kv part (p odd, into its MLA KV cache slot unless
    the slot is -1), with the stock kernel's block, warps and arithmetic, so a
    BF16 input gives its bits. Programs `[2M, 3M)` (`KPOOL_PROGRAMS`): the
    indexer key LayerNorm and `_kpool_decode_update_batched_kernel` for one
    token. A token completing a pool reads the earlier tokens of the pool from
    the tail ring when they are from earlier steps, else recomputes them from
    G1 (same values as the stock kernel's in-order stash), so the tokens of a
    request need no ordering. Rows are grouped per request, NEXT_N each.
    """
    pid = tl.program_id(0)
    tl.extra.cuda.gdc_wait()
    if pid < 2 * M:
        m = pid // 2
        block = tl.arange(0, BLOCK)
        if pid % 2 == 0:
            mask = block < QA
            w = tl.load(qaw_ptr + block, mask=mask, other=0.0).to(tl.float32)
            x = _ld_bf16(y_ptr, m * SY + block, mask)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / QA + eps)
            tl.store(qc_ptr + m * QA + block, (x * rrms * w).to(tl.bfloat16), mask=mask)
        else:
            mask = block < KVR
            w = tl.load(kvw_ptr + block, mask=mask, other=0.0).to(tl.float32)
            x = _ld_bf16(y_ptr, m * SY + QA + block, mask)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / KVR + eps)
            slot = tl.load(kslot_ptr + m)
            if slot >= 0:
                dst = kv_ptr + (slot // KV_BS) * kv_sb + (slot % KV_BS) * KVR
                tl.store(dst + block, (x * rrms * w).to(tl.bfloat16), mask=mask)
    elif KPOOL_PROGRAMS:
        idx = pid - 2 * M
        t = idx % NEXT_N
        offs = tl.arange(0, HD)
        kn_w = tl.load(kn_w_ptr + offs)
        kn_b = tl.load(kn_b_ptr + offs)
        key = _layer_norm(_ld_bf16(y_ptr, idx * SY + KEY + offs, offs < HD), kn_w, kn_b,
                          keps, HD)  # fmt: skip
        score = _ld_bf16(y_ptr, idx * SY + GATE + offs, offs < HD)
        cache_loc = tl.load(islot_ptr + idx)
        pos = tl.load(pos_ptr + idx).to(tl.int32)
        safe_pos = tl.maximum(pos, 0)
        slot = safe_pos % KPOOL
        tail_slot = tl.load(tslot_ptr + idx)
        base = tail_ptr + tl.maximum(tail_slot, 0).to(tl.int64) // RING * TAIL_BLK
        if (cache_loc >= 0) & (pos >= 0) & (slot == KPOOL - 1):
            start = safe_pos - slot
            peak = tl.full((HD,), -float("inf"), tl.float32)
            for p in tl.static_range(KPOOL):
                back = slot - p  # tokens before the current one
                ring = (back > t) & (offs < HD)
                row = base + TAIL_HALF + ((start + p) % RING) * HD + offs
                s = tl.load(row, mask=ring, other=0.0).to(tl.float32)
                s_now = _ld_bf16(y_ptr, (idx - back) * SY + GATE + offs, ~ring)
                s = tl.where(ring, s, s_now) + tl.load(ape_ptr + p * HD + offs)
                peak = tl.maximum(peak, s)
            acc = tl.full((HD,), 0.0, tl.float32)
            denom = tl.full((HD,), 0.0, tl.float32)
            for p in tl.static_range(KPOOL):
                back = slot - p
                ring = (back > t) & (offs < HD)
                row = base + ((start + p) % RING) * HD + offs
                s = tl.load(row + TAIL_HALF, mask=ring, other=0.0).to(tl.float32)
                s_now = _ld_bf16(y_ptr, (idx - back) * SY + GATE + offs, ~ring)
                s = tl.where(ring, s, s_now) + tl.load(ape_ptr + p * HD + offs)
                prob = tl.exp(s - peak)
                denom += prob
                k = tl.load(row, mask=ring, other=0.0).to(tl.float32)
                if back == 0:
                    k_now = key
                else:
                    k_now = _layer_norm(
                        _ld_bf16(y_ptr, (idx - back) * SY + KEY + offs, ~ring),
                        kn_w, kn_b, keps, HD,
                    )  # fmt: skip
                acc += tl.where(ring, k, k_now) * prob
            x = (acc / denom).to(tl.bfloat16).to(tl.float32)
            x = _hadamard(x, 1).to(tl.bfloat16).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
            scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / FP8_MAX))))
            quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
            loc = cache_loc.to(tl.int64)
            page = loc // PAGE
            in_page = loc % PAGE
            tl.store(idx8_ptr + page * PAGE_BYTES + in_page * HD + offs, quantized)
            tl.store(
                idx32_ptr + page * (PAGE_BYTES // 4) + (PAGE * HD) // 4 + in_page, scale
            )
        stash = (pos >= 0) & (tail_slot >= 0) & (offs < HD)
        row = base + (safe_pos % RING) * HD + offs
        tl.store(row, key, mask=stash)
        tl.store(row + TAIL_HALF, score, mask=stash)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _radix_keys(lg):
    """FP32 logits as int32 keys of the same order (signed compare)."""
    k = lg.to(tl.int32, bitcast=True)
    return tl.where(k < 0, k ^ 0x7FFFFFFF, k)


@triton.jit
def _digit(k, i: tl.constexpr):
    """Byte i (0 = most significant) of the key in unsigned order."""
    if i == 0:
        return ((k >> 24) + 128) & 255
    return (k >> (24 - 8 * i)) & 255


@triton.jit
def _top_digit(k):
    """The top TOP_BITS bits of the key in unsigned order."""
    return ((k >> (32 - TOP_BITS)) + (1 << (TOP_BITS - 1))) & ((1 << TOP_BITS) - 1)


@triton.jit
def _pick_bin(h, krem, NB: tl.constexpr = 256):
    """(bin b, count above b) of the histogram's largest bin whose suffix count
    reaches krem."""
    cs = tl.cumsum(h, 0)
    tot = tl.sum(h, 0)
    suffix = tot - cs + h
    bins = tl.arange(0, NB)
    b = tl.sum((suffix >= krem).to(tl.int32), 0) - 1
    above = tl.sum(tl.where(bins == b, suffix - h, 0), 0)
    return b, above


@triton.jit(do_not_specialize=["sibt", "smbt", "max_blocks", "stopk", "topw", "C"])
def select_kernel(
    iq_ptr,
    siq,
    wr_ptr,
    swr,
    pos_ptr,
    req_ptr,
    kslot_ptr,
    idx8_ptr,
    idx32_ptr,
    ibt_ptr,
    sibt,
    lg_ptr,
    slg,
    ctr_ptr,
    hist_ptr,
    mbt_ptr,
    smbt,
    max_blocks,
    sl_ptr,
    cnt_ptr,
    topk_ptr,
    stopk,
    topw,
    C,
    NH: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    MBS: tl.constexpr,
    MSTRIDE: tl.constexpr,
    KSEL: tl.constexpr,
    KPOOL: tl.constexpr,
    BP: tl.constexpr,
    NCH: tl.constexpr,
    NSTAGE: tl.constexpr,
    BLOCK_SEL: tl.constexpr,
    SLW: tl.constexpr,
    WRITE_LOGICAL: tl.constexpr,
    TOPW_BLOCK: tl.constexpr,
    had_ptr,
):
    """The decode indexer after its projections, one row per program row.

    Program (r, c) quantizes row r's indexer query (Hadamard-128 and FP8 with
    a power-of-two scale, as `fwht128_quant_fp8`), scales the head weights as
    `_fused_indexer_weight_scale` and writes the pool logits
    `sum_h w_h relu(q_h . k) * k_scale` of pool tiles c, c + NCH, ... into
    `lg`. For a row of at most BLOCK_SEL pools the last program of the row
    (atomic counter) picks the top KSEL pools in registers (radix select on the
    FP32 bits, ties to the lower pool index), expands them to tokens, appends
    the incomplete pool and writes the MLA KV rows of the selection to `sl`
    (count to `cnt`), and with WRITE_LOGICAL the token ids to the logical
    top-k buffer. Rows with at most KSEL pools select every token without
    logits. Padded rows (MLA slot -1) select nothing. For a longer row every
    program adds its logits' top TOP_BITS key bits to the row's histogram
    `hist`, and `select_long_kernel` finishes the row.
    """
    r = tl.program_id(0)
    c = tl.program_id(1)
    tl.extra.cuda.gdc_wait()
    pos = tl.load(pos_ptr + r).to(tl.int32)
    live = tl.load(kslot_ptr + r) >= 0
    ctx = tl.where(live, pos + 1, 0)
    npool = ctx // KPOOL
    req = tl.load(req_ptr + r)
    mine = c == 0
    if npool > KSEL:
        oh = tl.arange(0, NH)
        od = tl.arange(0, 128)
        x = tl.load(iq_ptr + r * siq + oh[:, None] * 128 + od[None, :])
        # Hadamard-128 as an MMA with the +-1 matrix `had_ptr`: the products are
        # exact and the sums of 128 BF16 values fit FP32, so this gives the
        # butterflies' values (up to exponent spreads > 2^9).
        hm = tl.load(had_ptr + od[:, None] * 128 + od[None, :])
        x = tl.dot(x, hm) * INV_SQRT_128
        x = x.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        qs = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / FP8_MAX))))
        qf = tl.minimum(tl.maximum(x / qs[:, None], -FP8_MAX), FP8_MAX)
        qb = qf.to(tl.float8e4nv).to(tl.bfloat16)
        weff = tl.load(wr_ptr + r * swr + oh) * qs * C
        for p0 in tl.range(c * BP, npool, NCH * BP, num_stages=NSTAGE):
            pp = p0 + tl.arange(0, BP)
            pm = pp < npool
            page = tl.load(ibt_ptr + req * sibt + pp // PAGE, mask=pm, other=0)
            base = page.to(tl.int64) * PAGE_BYTES
            sl_ = pp % PAGE
            kk = tl.load(idx8_ptr + base[:, None] + sl_[:, None] * 128 + od[None, :],
                         mask=pm[:, None], other=0.0)  # fmt: skip
            ks = tl.load(idx32_ptr + base // 4 + (PAGE * 128) // 4 + sl_, mask=pm,
                         other=0.0)  # fmt: skip
            s = tl.dot(qb, tl.trans(kk.to(tl.bfloat16)))
            s = tl.maximum(s, 0.0) * ks[None, :]
            lg = tl.sum(s * weff[:, None], axis=0)
            tl.store(lg_ptr + r * slg + pp, lg, mask=pm)
            tl.atomic_add(hist_ptr + r * TOP_BINS + _top_digit(_radix_keys(lg)), 1,
                          mask=pm & (npool > BLOCK_SEL), sem="relaxed")  # fmt: skip
        mine = c < 0
        if npool <= BLOCK_SEL:
            tl.debug_barrier()
            done = tl.atomic_add(ctr_ptr + r, 1, sem="acq_rel")
            mine = done == NCH - 1
    tl.extra.cuda.gdc_launch_dependents()
    if mine:
        # In registers: select (or keep every pool) and expand in one pass.
        nsel = tl.minimum(npool, KSEL)
        j = tl.arange(0, KPOOL)
        out = sl_ptr + r * SLW
        ap = tl.arange(0, BLOCK_SEL)
        valid = ap < npool
        if npool > KSEL:
            tl.store(ctr_ptr + r, 0)
            k = _radix_keys(tl.load(lg_ptr + r * slg + ap, mask=valid, other=0.0,
                                    cache_modifier=".cg"))  # fmt: skip
            krem = KSEL
            cand = valid
            sel = ap < 0
            bins = tl.arange(0, 256)
            for i in tl.static_range(4):
                d = _digit(k, i)
                h = tl.histogram(tl.where(cand, d, 0), 256)
                h -= tl.where(bins == 0, BLOCK_SEL - tl.sum(cand.to(tl.int32), 0), 0)
                b, above = _pick_bin(h, krem)
                sel = sel | (cand & (d > b))
                cand = cand & (d == b)
                krem -= above
            sel = sel | (cand & (tl.cumsum(cand.to(tl.int32), 0) <= krem))
        else:
            sel = valid
        opos = tl.cumsum(sel.to(tl.int32), 0) - 1
        tok = ap * KPOOL
        blk = tl.load(mbt_ptr + req * smbt + tok // MBS,
                      mask=sel & (tok // MBS < max_blocks), other=0)  # fmt: skip
        first = blk * MSTRIDE + tok % MBS
        col = opos[:, None] * KPOOL + j[None, :]
        tl.store(out + col, first[:, None] + j[None, :], mask=sel[:, None])
        if WRITE_LOGICAL:
            tl.store(topk_ptr + r * stopk + col, tok[:, None] + j[None, :],
                     mask=sel[:, None])  # fmt: skip
        _select_tail(ctx, npool, nsel, req, out, mbt_ptr, smbt, max_blocks, cnt_ptr + r,
                     topk_ptr + r * stopk, topw, KPOOL, MBS, MSTRIDE, WRITE_LOGICAL,
                     TOPW_BLOCK)  # fmt: skip


@triton.jit
def _select_tail(ctx, npool, nsel, req, out, mbt_ptr, smbt, max_blocks, cnt_ptr,
                 topk_ptr, topw, KPOOL: tl.constexpr, MBS: tl.constexpr,
                 MSTRIDE: tl.constexpr, WRITE_LOGICAL: tl.constexpr,
                 TOPW_BLOCK: tl.constexpr):  # fmt: skip
    """After the selected pools: the incomplete pool's tokens, the row's count
    and, with WRITE_LOGICAL, the -1 fill of the logical top-k row."""
    j = tl.arange(0, KPOOL)
    ntail = ctx - npool * KPOOL
    tok = npool * KPOOL + j
    tm = (j < ntail) & (tok // MBS < max_blocks)
    blk = tl.load(mbt_ptr + req * smbt + tok // MBS, mask=tm, other=0)
    tl.store(out + nsel * KPOOL + j, blk * MSTRIDE + tok % MBS, mask=tm)
    count = nsel * KPOOL + tl.sum(tm.to(tl.int32), 0)
    tl.store(cnt_ptr, count)
    if WRITE_LOGICAL:
        tl.store(topk_ptr + nsel * KPOOL + j, tok, mask=tm)
        for c0 in range(count, topw, TOPW_BLOCK):
            cw = c0 + tl.arange(0, TOPW_BLOCK)
            tl.store(topk_ptr + cw, -1, mask=cw < topw)


@triton.jit(do_not_specialize=["smbt", "max_blocks", "stopk", "topw"])
def select_long_kernel(
    pos_ptr,
    req_ptr,
    kslot_ptr,
    lg_ptr,
    slg,
    hist_ptr,
    cand_ptr,
    psel_ptr,
    mbt_ptr,
    smbt,
    max_blocks,
    sl_ptr,
    cnt_ptr,
    topk_ptr,
    stopk,
    topw,
    KSEL: tl.constexpr,
    KPOOL: tl.constexpr,
    MBS: tl.constexpr,
    MSTRIDE: tl.constexpr,
    BLOCK_SEL: tl.constexpr,
    CAP: tl.constexpr,
    FB: tl.constexpr,
    SLW: tl.constexpr,
    EXP: tl.constexpr,
    WRITE_LOGICAL: tl.constexpr,
    TOPW_BLOCK: tl.constexpr,
):
    """The top KSEL pools of each row over BLOCK_SEL pools, from the logits and
    the top-digit histogram `select_kernel` left; other rows do nothing. Its own
    launch, so its tiles and warps do not set the logits programs' registers.

    The histogram (zeroed here after the read) gives the threshold bin without
    a pass over `lg`; one pass keeps the pools above it at the front of `psel`
    and gathers the bin's pools and keys to `cand`, each in pool order, and a
    radix select on the low bits picks the bin's share (ties to the lower pool
    index) after them. A bin of more than CAP pools (near constant logits)
    takes byte-wise passes over `lg` instead. Then the expansion to tokens and
    MLA KV rows as in `select_kernel`.
    """
    r = tl.program_id(0)
    tl.extra.cuda.gdc_wait()
    # The dependents' own wait still covers this grid's writes.
    tl.extra.cuda.gdc_launch_dependents()
    pos = tl.load(pos_ptr + r).to(tl.int32)
    live = tl.load(kslot_ptr + r) >= 0
    ctx = tl.where(live, pos + 1, 0)
    npool = ctx // KPOOL
    if npool > BLOCK_SEL:
        req = tl.load(req_ptr + r)
        row = lg_ptr + r * slg
        psel = psel_ptr + r * KSEL
        hb = tl.arange(0, TOP_BINS)
        hrow = hist_ptr + r * TOP_BINS + hb
        hist = tl.load(hrow, cache_modifier=".cg")
        tl.store(hrow, tl.zeros((TOP_BINS,), tl.int32))
        top, above = _pick_bin(hist, KSEL, TOP_BINS)
        ncand = tl.sum(tl.where(hb == top, hist, 0), 0)
        bins = tl.arange(0, 256)
        if ncand <= CAP:
            cid = cand_ptr + r * (2 * CAP)
            ckey = cid + CAP
            taken = 0
            nc = 0
            for b0 in range(0, npool, FB):
                ap = b0 + tl.arange(0, FB)
                valid = ap < npool
                k = _radix_keys(tl.load(row + ap, mask=valid, other=0.0,
                                        cache_modifier=".cg"))  # fmt: skip
                d = _top_digit(k)
                hi = (valid & (d > top)).to(tl.int32)
                eq = (valid & (d == top)).to(tl.int32)
                tl.store(psel + taken + tl.cumsum(hi, 0) - 1, ap, mask=hi != 0)
                at = nc + tl.cumsum(eq, 0) - 1
                tl.store(cid + at, ap, mask=eq != 0)
                tl.store(ckey + at, k, mask=eq != 0)
                taken += tl.sum(hi, 0)
                nc += tl.sum(eq, 0)
            tl.debug_barrier()
            # The low 32 - TOP_BITS bits of the bin's keys: 8, 8 and 4 bits.
            ci = tl.arange(0, CAP)
            cand = ci < ncand
            ap = tl.load(cid + ci, mask=cand, other=0, cache_modifier=".cg")
            k = tl.load(ckey + ci, mask=cand, other=0, cache_modifier=".cg")
            krem = KSEL - above
            sel = ci < 0
            for i in tl.static_range(3):
                if i == 0:
                    d = (k >> 12) & 255
                elif i == 1:
                    d = (k >> 4) & 255
                else:
                    d = k & 15
                h = tl.histogram(tl.where(cand, d, 0), 256)
                h -= tl.where(bins == 0, CAP - tl.sum(cand.to(tl.int32), 0), 0)
                b, ahead = _pick_bin(h, krem)
                sel = sel | (cand & (d > b))
                cand = cand & (d == b)
                krem -= ahead
            sel = sel | (cand & (tl.cumsum(cand.to(tl.int32), 0) <= krem))
            seli = sel.to(tl.int32)
            tl.store(psel + above + tl.cumsum(seli, 0) - 1, ap, mask=sel)
        else:
            # The threshold key byte by byte, then the selection in pool order.
            krem = KSEL
            thr = 0
            for i in tl.static_range(4):
                h = tl.zeros((256,), tl.int32)
                for b0 in range(0, npool, FB):
                    ap = b0 + tl.arange(0, FB)
                    valid = ap < npool
                    k = _radix_keys(tl.load(row + ap, mask=valid, other=0.0,
                                            cache_modifier=".cg"))  # fmt: skip
                    cand = valid
                    if i > 0:
                        shift = 32 - 8 * i
                        cand = cand & ((k >> shift) == (thr >> shift))
                    n_out = FB - tl.sum(cand.to(tl.int32), 0)
                    h += tl.histogram(tl.where(cand, _digit(k, i), 0), 256)
                    h -= tl.where(bins == 0, n_out, 0)
                b, above = _pick_bin(h, krem)
                krem -= above
                if i == 0:  # noqa: SIM108 (Triton JIT: a constexpr branch per pass)
                    thr = (b - 128) << 24
                else:
                    thr = thr | (b << (24 - 8 * i))
            taken = 0
            ties = 0
            for b0 in range(0, npool, FB):
                ap = b0 + tl.arange(0, FB)
                valid = ap < npool
                k = _radix_keys(tl.load(row + ap, mask=valid, other=0.0,
                                        cache_modifier=".cg"))  # fmt: skip
                eq = (valid & (k == thr)).to(tl.int32)
                sel = (valid & (k > thr)) | (
                    (eq != 0) & (tl.cumsum(eq, 0) + ties <= krem)
                )
                seli = sel.to(tl.int32)
                tl.store(psel + taken + tl.cumsum(seli, 0) - 1, ap, mask=sel)
                taken += tl.sum(seli, 0)
                ties += tl.sum(eq, 0)
        tl.debug_barrier()
        j = tl.arange(0, KPOOL)
        out = sl_ptr + r * SLW
        for e0 in range(0, KSEL, EXP):
            rank = e0 + tl.arange(0, EXP)
            pool = tl.load(psel + rank, cache_modifier=".cg")
            tok = pool * KPOOL
            blk = tl.load(mbt_ptr + req * smbt + tok // MBS,
                          mask=tok // MBS < max_blocks, other=0)  # fmt: skip
            first = blk * MSTRIDE + tok % MBS
            col = rank[:, None] * KPOOL + j[None, :]
            tl.store(out + col, first[:, None] + j[None, :])
            if WRITE_LOGICAL:
                tl.store(topk_ptr + r * stopk + col, tok[:, None] + j[None, :])
        _select_tail(ctx, npool, KSEL, req, out, mbt_ptr, smbt, max_blocks, cnt_ptr + r,
                     topk_ptr + r * stopk, topw, KPOOL, MBS, MSTRIDE, WRITE_LOGICAL,
                     TOPW_BLOCK)  # fmt: skip


@triton.jit(do_not_specialize=["stopk", "topw", "smbt", "max_blocks"])
def remap_kernel(
    topk_ptr,
    stopk,
    topw,
    req_ptr,
    kslot_ptr,
    mbt_ptr,
    smbt,
    max_blocks,
    sl_ptr,
    cnt_ptr,
    MBS: tl.constexpr,
    MSTRIDE: tl.constexpr,
    SLW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The MLA KV rows of a reused logical top-k row (token ids, -1 unused),
    compacted to the front: `triton_convert_req_index_to_global_index` with
    valid counts, for draft steps that skip the indexer."""
    r = tl.program_id(0)
    tl.extra.cuda.gdc_wait()
    req = tl.load(req_ptr + r)
    live = tl.load(kslot_ptr + r) >= 0
    taken = 0
    if live:
        for c0 in range(0, topw, BLOCK):
            cw = c0 + tl.arange(0, BLOCK)
            tok = tl.load(topk_ptr + r * stopk + cw, mask=cw < topw, other=-1)
            ok = (tok >= 0) & (tok // MBS < max_blocks)
            blk = tl.load(mbt_ptr + req * smbt + tok // MBS, mask=ok, other=0)
            oki = ok.to(tl.int32)
            dst = taken + tl.cumsum(oki, 0) - 1
            tl.store(sl_ptr + r * SLW + dst, blk * MSTRIDE + tok % MBS,
                     mask=ok & (dst < SLW))  # fmt: skip
            taken += tl.sum(oki, 0)
    tl.store(cnt_ptr + r, tl.minimum(taken, SLW))
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def attention_kernel(
    q_ptr,
    sl_ptr,
    cnt_ptr,
    kv_ptr,
    op_ptr,
    lse_ptr,
    scale_log2,
    S: tl.constexpr,
    CH: tl.constexpr,
    BT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SLW: tl.constexpr,
    NSTAGE: tl.constexpr,
):
    """Split `s` of row r's sparse MLA decode: the latent rows `sl[r]` in
    `[s CH, (s + 1) CH)` of the row's count, all H heads at once (V = K)."""
    r = tl.program_id(0)
    s = tl.program_id(1)
    oh = tl.arange(0, H)
    od = tl.arange(0, D)
    tl.extra.cuda.gdc_wait()
    cnt = tl.load(cnt_ptr + r)
    start = s * CH
    end = tl.minimum(start + CH, cnt)
    q = tl.load(q_ptr + r * (H * D) + oh[:, None] * D + od[None, :])
    m_i = tl.full((H,), -float("inf"), tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, D), tl.float32)
    for t0 in tl.range(start, end, BT, num_stages=NSTAGE):
        tt = t0 + tl.arange(0, BT)
        tm = tt < end
        slot = tl.load(sl_ptr + r * SLW + tt, mask=tm, other=0).to(tl.int64)
        k = tl.load(
            kv_ptr + slot[:, None] * D + od[None, :], mask=tm[:, None], other=0.0
        )
        sc = tl.dot(q, tl.trans(k)) * scale_log2
        sc = tl.where(tm[None, :], sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k)
        m_i = m_new
    tl.extra.cuda.gdc_launch_dependents()
    ok = l_i > 0
    o = tl.where(ok[:, None], acc / tl.where(ok, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(ok, m_i + tl.log2(tl.where(ok, l_i, 1.0)), -float("inf"))
    tl.store(op_ptr + ((r * S + s) * H + oh[:, None]) * D + od[None, :], o)
    tl.store(lse_ptr + (r * S + s) * H + oh, lse)


@triton.jit
def combine_kernel(
    op_ptr,
    lse_ptr,
    ob_ptr,
    S: tl.constexpr,
    SP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    """The S split partials of (row r, head h) merged into BF16 `ob[r, h]`."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    os_ = tl.arange(0, SP)
    sm = os_ < S
    od = tl.arange(0, D)
    tl.extra.cuda.gdc_wait()
    lse = tl.load(lse_ptr + (r * S + os_) * H + h, mask=sm, other=-float("inf"))
    mx = tl.max(lse, 0)
    mx = tl.where(mx == -float("inf"), 0.0, mx)
    w = tl.exp2(lse - mx)
    den = tl.sum(w, 0)
    o = tl.load(op_ptr + ((r * S + os_[:, None]) * H + h) * D + od[None, :],
                mask=sm[:, None], other=0.0)  # fmt: skip
    a = tl.sum(o * w[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.extra.cuda.gdc_launch_dependents()
    tl.store(ob_ptr + (r * H + h) * D + od, a.to(tl.bfloat16))
