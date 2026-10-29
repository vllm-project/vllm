# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-sharded speculative verification over the TP group.

Stock verification all-gathers the target logits, so every TP rank holds
``[num_logits, vocab]`` and then runs temperature, top-p and the rejection
sampler over the whole vocabulary: the same work on every rank.

Here TP rank ``r`` owns whole 8192-column blocks of the vocabulary (the
rejection sampler's statistics blocks, also aligned to its 1024-column
resampling blocks), split as evenly as possible, and processes only those:

* Target logits: the rank's own lm_head shard, plus the few columns of its
  owned range that sit in the right neighbour's shard, which that neighbour
  pushes. Nothing is all-gathered.
* Top-p: an exact select of the smallest prefix (by logit, then token id)
  whose probability mass reaches ``p``, run jointly on the owned columns of all
  ranks. Masses are summed in 64-bit fixed point, so the sums are exact and
  every rank (and every run) agrees on the kept set.
* Rejection sampling: the rejection sampler's per-block kernels run on the
  owned blocks only; the per-block partials are exchanged and the per-request
  kernels combine all blocks in the stock order. The drafted tokens' logits
  come from their owners. Given the same processed logits, the result is
  bitwise the stock one.

Exchanges push 8-byte words ``[tag | 32-bit payload]`` into every peer's
symmetric memory buffer, and readers poll for the tag (NCCL's LL protocol), so
no fence or flag is needed. Every verification step uses its own tag epoch and
alternates between two buffer halves, so a rank never overwrites words a slower
peer has yet to read.
"""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# The rejection sampler's statistics and resampling block sizes.
VOCAB_BLOCK_SIZE = 8192
RESAMPLE_BLOCK_SIZE = 1024
# Exchanges per verification step get distinct tags below this.
_TAGS_PER_STEP = 32
# Bounded spin (about a second) so a missing peer cannot hang the GPU; a
# timeout is counted in `errors`.
_SPIN_LIMIT = 1 << 24
_BLK = 256

# Top-p: token masses exp(x - max) in fixed point with this many fraction bits.
# Tokens more than _TOPP_RANGE below the row max have mass 0 and are skipped.
_TOPP_FRAC_BITS = 40
_TOPP_RANGE = 28.0
_TOPP_BINS = 256
# Tokens of the first round's boundary bin kept per row for the later rounds
# (more: those rounds scan the whole row).
_TOPP_CAP = 2048


def vocab_shard_range(vocab_size: int, world: int, rank: int) -> tuple[int, int]:
    """Columns [start, end) owned by ``rank``: whole VOCAB_BLOCK_SIZE blocks,
    spread as evenly as possible (lower ranks take the extra blocks)."""
    num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    base, extra = divmod(num_blocks, world)
    b0 = rank * base + min(rank, extra)
    b1 = b0 + base + (1 if rank < extra else 0)
    return min(vocab_size, b0 * VOCAB_BLOCK_SIZE), min(
        vocab_size, b1 * VOCAB_BLOCK_SIZE
    )


def vocab_shard_unsupported_reason(
    vocab_size: int, local_vocab_size: int, world: int
) -> str | None:
    """Why the owned ranges cannot be assembled from the TP lm_head shards."""
    if world == 1:
        return "tensor parallelism is 1"
    if triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE) < world:
        return "the vocabulary has fewer 8192-column blocks than TP ranks"
    if local_vocab_size * world < vocab_size:
        return "the lm_head shards do not cover the vocabulary"
    for r in range(world):
        start, end = vocab_shard_range(vocab_size, world, r)
        # Every owned range starts in its own shard and ends no later than the
        # right neighbour's shard.
        if start < r * local_vocab_size or end > (r + 2) * local_vocab_size:
            return "the owned ranges do not fit the lm_head shards"
    return None


# ----------------------------------------------------------------------------
# LL exchange kernels. A word is [31-bit tag | 32-bit payload]; the tag of
# exchange k in step `epoch` is (epoch * _TAGS_PER_STEP + k) & 0x7FFFFFFF.


@triton.jit
def _peer(ptrs, d):
    return tl.load(ptrs + d).to(tl.pointer_type(tl.int64))


@triton.jit
def _tag(epoch_ptr, k):
    return (tl.load(epoch_ptr).to(tl.int64) * 32 + k) & 0x7FFFFFFF


@triton.jit
def _half(epoch_ptr, half_words):
    return (tl.load(epoch_ptr).to(tl.int64) & 1) * half_words


@triton.jit
def _poll(p, m, tag, err_ptr, SPIN_LIMIT: tl.constexpr):
    w = tl.load(p, mask=m, other=0, volatile=True)
    bad = tl.sum((m & ((w >> 32) != tag)).to(tl.int32), axis=0)
    it = 0
    while (bad > 0) & (it < SPIN_LIMIT):
        w = tl.load(p, mask=m, other=0, volatile=True)
        bad = tl.sum((m & ((w >> 32) != tag)).to(tl.int32), axis=0)
        it += 1
    if bad > 0:
        tl.atomic_add(err_ptr, 1)
    return (w & 0xFFFFFFFF).to(tl.int32)


@triton.jit
def _push_kernel(
    src_ptr,  # int32 [rows, src_stride]
    src_stride,
    col_a,  # pushes columns [col_a, col_b) of every row
    col_b,
    buf_ptrs,  # int64 [WORLD]: the peers' buffers
    half_words,
    region,  # word offset of the region in a buffer half
    region_cols,  # words per region row
    slot_words,  # slot of this rank in the region (0: shared layout)
    epoch_ptr,
    k,
    DST: tl.constexpr,  # -1: every rank, else this rank only
    WORLD: tl.constexpr,
    BLK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    cols = col_a + tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = cols < col_b
    v = tl.load(src_ptr + r * src_stride + cols, mask=m, other=0).to(tl.int64)
    w = (_tag(epoch_ptr, k) << 32) | (v & 0xFFFFFFFF)
    off = _half(epoch_ptr, half_words) + region + slot_words + r * region_cols + cols
    if DST < 0:
        for d in tl.static_range(WORLD):
            tl.store(_peer(buf_ptrs, d) + off, w, mask=m)
    else:
        tl.store(_peer(buf_ptrs, DST) + off, w, mask=m)


@triton.jit
def _pull_kernel(
    buf_ptr,  # int64: this rank's buffer
    half_words,
    region,
    region_cols,
    dst_ptr,  # int32 [SLOTS, rows, ncols] (slot stride slot_words)
    dst_stride,
    dst_slot_stride,
    ncols,
    slot_words,
    epoch_ptr,
    k,
    err_ptr,
    SPIN_LIMIT: tl.constexpr,
    BLK: tl.constexpr,
):
    s = tl.program_id(2).to(tl.int64)
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = cols < ncols
    p = (
        buf_ptr
        + _half(epoch_ptr, half_words)
        + region
        + s * slot_words
        + r * region_cols
        + cols
    )
    v = _poll(p, m, _tag(epoch_ptr, k), err_ptr, SPIN_LIMIT)
    tl.store(dst_ptr + s * dst_slot_stride + r * dst_stride + cols, v, mask=m)


@triton.jit
def _owned_logits_kernel(
    out_ptr,  # fp32 [rows, num_cols]: the owned columns
    out_stride,
    local_ptr,  # [rows, local_cols]: this rank's lm_head shard
    local_stride,
    buf_ptr,  # this rank's buffer: the right neighbour's pushed columns
    half_words,
    region,
    region_cols,
    num_cols,
    num_local,  # owned columns taken from the local shard
    local_offset,  # local column of the first owned column
    epoch_ptr,
    k,
    err_ptr,
    PACK: tl.constexpr,  # columns per pushed word
    SPIN_LIMIT: tl.constexpr,
    BLK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = cols < num_cols
    x = tl.load(
        local_ptr + r * local_stride + local_offset + cols,
        mask=m & (cols < num_local),
        other=0.0,
    ).to(tl.float32)
    if (tl.program_id(1) + 1) * BLK > num_local:
        rcols = cols - num_local
        rm = m & (rcols >= 0)
        word = rcols // PACK
        p = buf_ptr + _half(epoch_ptr, half_words) + region + r * region_cols + word
        v = _poll(p, rm, _tag(epoch_ptr, k), err_ptr, SPIN_LIMIT)
        if PACK == 2:
            bits = ((v >> ((rcols % 2) * 16)) & 0xFFFF).to(tl.int16)
            y = bits.to(local_ptr.dtype.element_ty, bitcast=True).to(tl.float32)
        else:
            y = v.to(tl.float32, bitcast=True)
        x = tl.where(rm, y, x)
    tl.store(out_ptr + r * out_stride + cols, x.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _token_logits_kernel(
    out_ptr,  # fp32 [num_logits, 2]
    target_ptr,  # [num_logits, num_cols]: owned processed target logits
    target_stride,
    draft_ptr,  # [., n_spec, V] draft logits cache or None
    draft_stride_0,
    draft_stride_1,
    draft_sampled_ptr,  # [num_logits]
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    num_logits,
    num_steps,
    col_start,
    col_end,
    BLK: tl.constexpr,
):
    # The drafted token of each logit row (draft_sampled[i + 1]) if this rank
    # owns it, else -inf (the exchange takes the maximum over ranks).
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < num_logits
    tok = tl.load(draft_sampled_ptr + i + 1, mask=m & (i + 1 < num_logits), other=-1)
    tok = tok.to(tl.int64)
    own = m & (tok >= col_start) & (tok < col_end)
    t = tl.load(
        target_ptr + i.to(tl.int64) * target_stride + (tok - col_start),
        mask=own,
        other=float("-inf"),
    ).to(tl.float32)
    tl.store(out_ptr + i * 2, t, mask=m)
    d = tl.full((BLK,), float("-inf"), tl.float32)
    if draft_ptr is not None:
        req = tl.load(expanded_idx_mapping_ptr + i, mask=m, other=0).to(tl.int64)
        step = tl.load(expanded_local_pos_ptr + i, mask=m, other=0).to(tl.int64)
        d = tl.load(
            draft_ptr + req * draft_stride_0 + step * draft_stride_1 + tok,
            mask=own & (req >= 0) & (step < num_steps),
            other=float("-inf"),
        ).to(tl.float32)
    tl.store(out_ptr + i * 2 + 1, d, mask=m)


@triton.jit
def _max_slots_kernel(
    out_ptr,  # fp32 [n]
    src_ptr,  # fp32 [WORLD, n]
    n,
    WORLD: tl.constexpr,
    BLK: tl.constexpr,
):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < n
    acc = tl.full((BLK,), float("-inf"), tl.float32)
    for s in tl.static_range(WORLD):
        acc = tl.maximum(acc, tl.load(src_ptr + s * n + i, mask=m, other=0.0))
    tl.store(out_ptr + i, acc, mask=m)


# ----------------------------------------------------------------------------
# Exact top-p over the owned columns of all ranks.
#
# Per row: M = global max; token mass q(x) = floor(exp(x - M) * 2^40) (int64);
# Z = sum q; target T = ceil(p * Z). The kept set is every token whose
# (logit, -token id) is at least the boundary token's, where the boundary is
# the first token in descending order at which the cumulative mass reaches T.
# Round 0 buckets the tokens by value (256 bins over [M - 28, M]); rounds 1-4
# radix-select the 32-bit order key byte by byte inside the boundary bin. Each
# round's per-bucket masses are exchanged and summed (int64: exact, order
# free). Ties at the boundary value are resolved by token id: the j-th tied
# token in id order is found by its owner.
#
# Row state ST (int64 [rows, 8]): 0 target T, 1 mass strictly above the current
# prefix, 2 flag (0 untouched, 1 active, 2 keep all), 3 round-0 bin,
# 4 key prefix, 5 boundary token id.


@triton.jit
def _okey(x):
    # 32-bit key, monotone in x as unsigned (-0.0 maps to +0.0's key).
    b = tl.where(x == 0.0, 0, x.to(tl.int32, bitcast=True))
    return tl.where(b >= 0, b ^ (-2147483648), ~b).to(tl.uint32, bitcast=True)


@triton.jit
def _mass(x, mx):
    e = tl.exp2((x - mx) * 1.4426950408889634)
    return (e * 1099511627776.0).to(tl.int64)  # 2^40, truncated


@triton.jit
def _value_bin(x, mx, NB: tl.constexpr, RANGE: tl.constexpr):
    d = tl.minimum(tl.floor((mx - x) * (NB / RANGE)), NB - 1)
    return (NB - 1) - d.to(tl.int32)


@triton.jit
def _row_max(MM, row, n, WORLD: tl.constexpr):
    mx = -float("inf")
    for s in tl.static_range(WORLD):
        mx = tl.maximum(mx, tl.load(MM + s * n + row))
    return mx


@triton.jit
def _topp_max_kernel(L, LD, NCOLS, P, OUT, BLOCK: tl.constexpr):
    # grid (rows, splits); OUT fp32 [rows] pre-filled with -inf
    row = tl.program_id(0)
    if tl.load(P + row) >= 1.0:
        return
    ns = tl.num_programs(1)
    per = tl.cdiv(NCOLS, ns)
    lo = tl.program_id(1) * per
    hi = tl.minimum(lo + per, NCOLS)
    R = L + row.to(tl.int64) * LD
    offs = tl.arange(0, BLOCK)
    mx = tl.full((BLOCK,), -float("inf"), tl.float32)
    for i in range(lo, hi, BLOCK):
        mx = tl.maximum(
            mx, tl.load(R + i + offs, mask=i + offs < hi, other=-float("inf"))
        )
    tl.atomic_max(OUT + row, tl.max(mx), sem="relaxed")


@triton.jit
def _topp_hist_kernel(
    L,
    LD,
    NCOLS,
    P,
    MM,  # fp32 [WORLD, rows]: exchanged row maxima
    ST,
    HIST,  # int64 [rows, NB + 1] (pre-zeroed); slot NB: Z (round 0)
    CX,  # fp32 [rows, CAP]: the round-0 boundary bin's tokens
    CCNT,  # int32 [rows]: their count
    CAP,
    ROUND: tl.constexpr,
    WORLD: tl.constexpr,
    NB: tl.constexpr,
    RANGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.num_programs(0)
    mx = _row_max(MM, row, n, WORLD)
    S = ST + row * 8
    if ROUND == 0:
        if (tl.load(P + row) >= 1.0) | (mx == -float("inf")):
            return
    elif tl.load(S + 2) != 1:
        return
    H = HIST + row.to(tl.int64) * (NB + 1)
    split = tl.program_id(1)
    nsplit = tl.num_programs(1)
    offs = tl.arange(0, BLOCK)
    SHIFT: tl.constexpr = 24 - 8 * (ROUND - 1) if ROUND > 0 else 0
    if ROUND > 0:
        b0 = tl.load(S + 3).to(tl.int32)
        prefix = tl.load(S + 4).to(tl.uint32)
        cc = tl.load(CCNT + row)
        if cc <= CAP:
            per = tl.cdiv(cc, nsplit)
            lo = split * per
            hi = tl.minimum(lo + per, cc)
            C = CX + row.to(tl.int64) * CAP
            for i in range(lo, hi, BLOCK):
                m = i + offs < hi
                x = tl.load(C + i + offs, mask=m, other=-float("inf"))
                k = _okey(x)
                sel = m
                if ROUND > 1:
                    sel = sel & ((k >> (SHIFT + 8)) == prefix)
                q = _mass(x, mx)
                b = ((k >> SHIFT) & 255).to(tl.int32)
                tl.atomic_add(H + b, q, mask=sel & (q > 0), sem="relaxed")
            return
    per = tl.cdiv(NCOLS, nsplit)
    lo = split * per
    hi = tl.minimum(lo + per, NCOLS)
    R = L + row.to(tl.int64) * LD
    z = tl.zeros((BLOCK,), tl.int64)
    for i in range(lo, hi, BLOCK):
        m = i + offs < hi
        x = tl.load(R + i + offs, mask=m, other=-float("inf"))
        q = _mass(x, mx)
        live = m & (q > 0)
        vb = _value_bin(x, mx, NB, RANGE)
        if ROUND == 0:
            z += tl.where(live, q, 0)
            tl.atomic_add(H + vb, q, mask=live, sem="relaxed")
        else:
            k = _okey(x)
            sel = live & (vb == b0)
            if ROUND > 1:
                sel = sel & ((k >> (SHIFT + 8)) == prefix)
            b = ((k >> SHIFT) & 255).to(tl.int32)
            tl.atomic_add(H + b, q, mask=sel, sem="relaxed")
    if ROUND == 0:
        tl.atomic_add(H + NB, tl.sum(z), sem="relaxed")


@triton.jit
def _topp_select_kernel(
    HALL,  # int64 [WORLD, rows, NB + 1]: exchanged histograms
    P,
    MM,
    ST,
    ROUND: tl.constexpr,
    WORLD: tl.constexpr,
    NB: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.num_programs(0)
    S = ST + row * 8
    if ROUND == 0:
        p = tl.load(P + row)
        active = (p < 1.0) & (_row_max(MM, row, n, WORLD) > -float("inf"))
        tl.store(S + 2, active.to(tl.int64))
        tl.store(S + 4, 0)
        if active == 0:
            return
    elif tl.load(S + 2) != 1:
        return
    bk = tl.arange(0, NB)
    h = tl.zeros((NB,), tl.int64)
    z = tl.zeros((), tl.int64)
    for s in tl.static_range(WORLD):
        src = HALL + (s * n + row).to(tl.int64) * (NB + 1)
        h += tl.load(src + bk)
        if ROUND == 0:
            z += tl.load(src + NB)
    if ROUND == 0:
        target = tl.ceil(tl.load(P + row).to(tl.float64) * z.to(tl.float64))
        target = tl.maximum(target.to(tl.int64), 1)
        tl.store(S + 0, target)
        above = tl.zeros((), tl.int64)
    else:
        target = tl.load(S + 0)
        above = tl.load(S + 1)
    # suffix[b] = mass of buckets >= b; the crossing bucket is the highest b
    # with above + suffix[b] >= target (it always has positive mass).
    suffix = tl.sum(h, axis=0) - tl.cumsum(h, 0) + h
    bstar = tl.max(tl.where(above + suffix >= target, bk, -1), axis=0)
    if bstar < 0:
        tl.store(S + 2, 2)  # unreachable with exact sums; keep the whole row
        return
    tl.store(S + 1, above + tl.sum(tl.where(bk > bstar, h, 0), axis=0))
    if ROUND == 0:
        tl.store(S + 3, bstar.to(tl.int64))
    else:
        tl.store(S + 4, (tl.load(S + 4) << 8) | bstar.to(tl.int64))


@triton.jit
def _topp_compact_kernel(
    L,
    LD,
    NCOLS,
    MM,
    ST,
    CX,
    CCNT,
    CAP,
    WORLD: tl.constexpr,
    NB: tl.constexpr,
    RANGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # grid (rows, splits): append the round-0 boundary bin's tokens to CX;
    # CCNT (pre-zeroed) ends as their count (> CAP: later rounds scan the row).
    row = tl.program_id(0)
    n = tl.num_programs(0)
    S = ST + row * 8
    if tl.load(S + 2) != 1:
        return
    mx = _row_max(MM, row, n, WORLD)
    b0 = tl.load(S + 3).to(tl.int32)
    ns = tl.num_programs(1)
    per = tl.cdiv(NCOLS, ns)
    lo = tl.program_id(1) * per
    hi = tl.minimum(lo + per, NCOLS)
    R = L + row.to(tl.int64) * LD
    C = CX + row.to(tl.int64) * CAP
    offs = tl.arange(0, BLOCK)
    for i in range(lo, hi, BLOCK):
        m = i + offs < hi
        x = tl.load(R + i + offs, mask=m, other=-float("inf"))
        sel = m & (_mass(x, mx) > 0) & (_value_bin(x, mx, NB, RANGE) == b0)
        cnt = tl.sum(sel.to(tl.int32), axis=0)
        if cnt > 0:
            start = tl.atomic_add(CCNT + row, cnt, sem="relaxed")
            pos = start + tl.cumsum(sel.to(tl.int32), 0) - 1
            tl.store(C + pos, x, mask=sel & (pos < CAP))


@triton.jit
def _topp_tie_count_kernel(
    L, LD, NCOLS, ST, CNTS, NSP: tl.constexpr, BLOCK: tl.constexpr
):
    # grid (rows, NSP): tokens equal to the boundary value, per column split.
    row = tl.program_id(0)
    sp = tl.program_id(1)
    S = ST + row * 8
    c = 0
    if tl.load(S + 2) == 1:
        bkey = tl.load(S + 4).to(tl.uint32)
        per = tl.cdiv(NCOLS, NSP)
        lo = sp * per
        hi = tl.minimum(lo + per, NCOLS)
        R = L + row.to(tl.int64) * LD
        offs = tl.arange(0, BLOCK)
        for i in range(lo, hi, BLOCK):
            m = i + offs < hi
            x = tl.load(R + i + offs, mask=m, other=-float("inf"))
            c += tl.sum((m & (_okey(x) == bkey)).to(tl.int32), axis=0)
    tl.store(CNTS + row * NSP + sp, c)


@triton.jit
def _topp_tie_index_kernel(
    L,
    LD,
    NCOLS,
    COL0,
    MM,
    ST,
    CNTS,
    CALL,
    BIDX,
    RANK,
    WORLD: tl.constexpr,
    NSP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # CALL int32 [WORLD, rows, NSP]: every rank's per-split tie counts. The
    # j-th tied token (id order = rank order, then column order) is the
    # boundary; its owner writes its id, every other rank -1.
    row = tl.program_id(0)
    n = tl.num_programs(0)
    S = ST + row * 8
    res = tl.full((), -1, tl.int64)
    if tl.load(S + 2) == 1:
        mx = _row_max(MM, row, n, WORLD)
        bkey = tl.load(S + 4).to(tl.uint32)
        kb = bkey.to(tl.int32, bitcast=True)
        bx = tl.where(kb < 0, kb ^ (-2147483648), ~kb).to(tl.float32, bitcast=True)
        qb = _mass(bx, mx)
        need = tl.load(S + 0) - tl.load(S + 1)
        j = tl.maximum((need + qb - 1) // tl.maximum(qb, 1), 1)
        sidx = tl.arange(0, NSP)
        before = tl.zeros((), tl.int64)
        total = tl.zeros((), tl.int64)
        for s in tl.static_range(WORLD):
            cs = tl.sum(tl.load(CALL + (s * n + row) * NSP + sidx), axis=0)
            total += cs
            if s < RANK:
                before += cs
        j = tl.minimum(j, total)
        mine = tl.load(CNTS + row * NSP + sidx).to(tl.int64)
        jl = j - before
        if (jl >= 1) & (jl <= tl.sum(mine, axis=0)):
            cum = tl.cumsum(mine, 0)
            sp = tl.min(tl.where(cum >= jl, sidx, NSP), axis=0)
            seen = tl.sum(tl.where(sidx < sp, mine, 0), axis=0)
            per = tl.cdiv(NCOLS, NSP)
            lo = sp * per
            hi = tl.minimum(lo + per, NCOLS)
            R = L + row.to(tl.int64) * LD
            offs = tl.arange(0, BLOCK)
            for i in range(lo, hi, BLOCK):
                m = i + offs < hi
                x = tl.load(R + i + offs, mask=m, other=-float("inf"))
                t = (m & (_okey(x) == bkey)).to(tl.int64)
                c = seen + tl.cumsum(t, 0)
                hit = (t > 0) & (c == jl)
                res = tl.maximum(
                    res, tl.max(tl.where(hit, COL0 + i + offs, -1), axis=0)
                )
                seen += tl.sum(t, axis=0)
    tl.store(BIDX + row, res.to(tl.int32))


@triton.jit
def _topp_mask_kernel(
    L, LD, NCOLS, COL0, MM, ST, BALL, WORLD: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    n = tl.num_programs(0)
    S = ST + row * 8
    if tl.load(S + 2) != 1:
        return  # untouched (p >= 1 or no finite logit) or keep-all
    mx = _row_max(MM, row, n, WORLD)
    bkey = tl.load(S + 4).to(tl.uint32)
    bidx = -1
    for s in tl.static_range(WORLD):
        bidx = tl.maximum(bidx, tl.load(BALL + s * n + row))
    ns = tl.num_programs(1)
    per = tl.cdiv(NCOLS, ns)
    lo = tl.program_id(1) * per
    hi = tl.minimum(lo + per, NCOLS)
    R = L + row.to(tl.int64) * LD
    offs = tl.arange(0, BLOCK)
    for i in range(lo, hi, BLOCK):
        m = i + offs < hi
        x = tl.load(R + i + offs, mask=m, other=-float("inf"))
        k = _okey(x)
        keep = (k > bkey) | ((k == bkey) & (COL0 + i + offs <= bidx)) | (x == mx)
        tl.store(R + i + offs, tl.where(keep, x, float("-inf")), mask=m)


# ----------------------------------------------------------------------------


class VocabShard:
    """This rank's owned vocabulary range, and the exchanges over the TP group.

    All exchanges of one verification step must be issued between
    `begin_step` and `end_step`, in the same order on every rank.
    """

    def __init__(
        self,
        vocab_size: int,
        local_vocab_size: int,
        max_num_logits: int,
        max_num_reqs: int,
        device: torch.device,
        verify: bool = True,
    ):
        """``verify=False``: only `owned_logits` is used (smaller buffer)."""
        import torch.distributed._symmetric_memory as symm_mem

        group = get_tp_group()
        self.group = group
        self.world = group.world_size
        self.rank = group.rank_in_group
        reason = vocab_shard_unsupported_reason(
            vocab_size, local_vocab_size, self.world
        )
        assert reason is None, reason
        self.vocab_size = vocab_size
        self.local_vocab_size = local_vocab_size
        self.col_start, self.col_end = vocab_shard_range(
            vocab_size, self.world, self.rank
        )
        self.num_cols = self.col_end - self.col_start
        self.block_start = self.col_start // VOCAB_BLOCK_SIZE
        self.block_end = triton.cdiv(self.col_end, VOCAB_BLOCK_SIZE)
        self.num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        self.tp_start = self.rank * local_vocab_size
        self.tp_end = min(vocab_size, self.tp_start + local_vocab_size)
        # Owned columns from the right neighbour's shard, and the columns this
        # rank sends to its left neighbour.
        self.num_recv = max(0, self.col_end - self.tp_end)
        left_end = vocab_shard_range(vocab_size, self.world, self.rank - 1)[1]
        self.num_send = max(0, left_end - self.tp_start) if self.rank > 0 else 0
        self.max_num_logits = max_num_logits
        self.max_num_reqs = max_num_reqs

        # Words per buffer half: the largest step (bounded per region below).
        n, w = max_num_logits, self.world
        sliver = max(
            max(
                0,
                vocab_shard_range(vocab_size, w, r)[1]
                - min(vocab_size, (r + 1) * local_vocab_size),
            )
            for r in range(w)
        )
        nsp = _tie_splits(n)
        words = n * sliver  # owned columns from the right neighbour
        words += (
            0
            if not verify
            else (
                w * n  # row maxima
                + 2 * w * n * 2 * (_TOPP_BINS + 1)  # two alternating histograms
                + w * n * nsp  # tie counts
                + w * n  # boundary ids
                + n * self.num_blocks * 6  # rejection statistics
                + w * n * 2  # drafted tokens' logits
                + n * self.num_blocks  # residual mass
                + max_num_reqs * triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE) * 4
            )
        )
        self.half_words = words + 64
        self.buf = symm_mem.empty(2 * self.half_words, dtype=torch.int64, device=device)
        self.buf.zero_()
        self._handle = symm_mem.rendezvous(self.buf, group.device_group.group_name)
        self.buf_ptrs = torch.tensor(
            list(self._handle.buffer_ptrs), dtype=torch.int64, device=device
        )
        # Tags never match a zeroed word: the first epoch is 1.
        self.epoch = torch.ones(1, dtype=torch.int32, device=device)
        # Exchange timeouts (expected to stay 0).
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self._offset = 0
        self._k = 0
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=group.cpu_group)

    # -- step bookkeeping ----------------------------------------------------

    def begin_step(self) -> None:
        self._offset = 0
        self._k = 0

    def end_step(self) -> None:
        self.epoch.add_(1)

    def _region(self, words: int, reuse: int | None = None) -> tuple[int, int]:
        """A region of this step (fresh, or ``reuse``) and a fresh tag.

        Reusing a region within a step is safe once every rank has read it:
        a peer only pushes the next exchange after pulling this one from all
        ranks, i.e. after all of them pushed it, i.e. after they were done with
        the exchange before it."""
        k = self._k
        assert k < _TAGS_PER_STEP
        self._k += 1
        if reuse is not None:
            return reuse, k
        off = self._offset
        assert off + words <= self.half_words, "vocab shard exchange buffer overflow"
        self._offset += words
        return off, k

    def reserve(self, words: int) -> int:
        """A region of this step for `allgather_slots(reuse=...)`."""
        off = self._offset
        assert off + words <= self.half_words, "vocab shard exchange buffer overflow"
        self._offset += words
        return off

    # -- primitives ----------------------------------------------------------

    def _push(
        self,
        src32: torch.Tensor,
        col_a: int,
        col_b: int,
        region: int,
        region_cols: int,
        slot_words: int,
        k: int,
        dst: int = -1,
    ) -> None:
        rows = src32.shape[0]
        if rows == 0 or col_b <= col_a:
            return
        _push_kernel[(rows, triton.cdiv(col_b - col_a, _BLK))](
            src32,
            src32.stride(0),
            col_a,
            col_b,
            self.buf_ptrs,
            self.half_words,
            region,
            region_cols,
            slot_words,
            self.epoch,
            k,
            DST=dst,
            WORLD=self.world,
            BLK=_BLK,
        )

    def _pull(
        self,
        dst32: torch.Tensor,
        region: int,
        region_cols: int,
        slot_words: int,
        k: int,
    ) -> None:
        # dst32: int32 [slots, rows, ncols]
        slots, rows, ncols = dst32.shape
        if rows == 0 or ncols == 0:
            return
        _pull_kernel[(rows, triton.cdiv(ncols, _BLK), slots)](
            self.buf,
            self.half_words,
            region,
            region_cols,
            dst32,
            dst32.stride(1),
            dst32.stride(0),
            ncols,
            slot_words,
            self.epoch,
            k,
            self.errors,
            SPIN_LIMIT=_SPIN_LIMIT,
            BLK=_BLK,
        )

    def allgather_cols(
        self, tensors: list[torch.Tensor], col_a: int, col_b: int
    ) -> None:
        """In place: fill every column of each ``[rows, cols]`` tensor from its
        owner; this rank's are [col_a, col_b). All pushes go out before the
        first pull, so the exchanges overlap."""
        pending = []
        for t in tensors:
            assert t.is_contiguous()
            t32 = t.view(torch.int32)
            f = t32.shape[1] // t.shape[1]
            region, k = self._region(t32.numel())
            self._push(t32, col_a * f, col_b * f, region, t32.shape[1], 0, k)
            pending.append((t32, region, k))
        for t32, region, k in pending:
            self._pull(t32.unsqueeze(0), region, t32.shape[1], 0, k)

    def allgather_slots(
        self, t: torch.Tensor, reuse: int | None = None
    ) -> torch.Tensor:
        """``[rows, w]`` per rank -> ``[world, rows, w]`` (slot = rank)."""
        t = t.contiguous()
        t32 = t.view(torch.int32)
        rows, w = t32.shape
        slot = rows * w
        region, k = self._region(self.world * slot, reuse)
        self._push(t32, 0, w, region, w, self.rank * slot, k)
        out = torch.empty((self.world, *t.shape), dtype=t.dtype, device=t.device)
        self._pull(out.view(torch.int32).view(self.world, rows, w), region, w, slot, k)
        return out

    def owned_logits(
        self, local_logits: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """``[rows, num_cols]``: this rank's owned columns of the logits whose
        TP shards are ``local_logits`` ([rows, local_vocab_size]), as ``dtype``
        (the values are exact in fp32 or the logits' own dtype)."""
        assert local_logits.shape[1] == self.local_vocab_size
        rows = local_logits.shape[0]
        esize = local_logits.element_size()
        assert esize in (2, 4) and local_logits.stride(1) == 1
        pack = 4 // esize
        # Every rank sizes the region the same way: the widest exchange.
        region_cols = (
            max(
                triton.cdiv(
                    max(
                        0,
                        vocab_shard_range(self.vocab_size, self.world, r)[1]
                        - min(self.vocab_size, (r + 1) * self.local_vocab_size),
                    ),
                    pack,
                )
                for r in range(self.world)
            )
            or 1
        )
        region, k = self._region(rows * region_cols)
        if self.num_send > 0:
            src = local_logits
            if not src.is_contiguous():
                src = src.contiguous()
            if pack == 2 and src.shape[1] % 2:
                src = torch.nn.functional.pad(src, (0, 1))
            src32 = src.view(torch.int32)
            self._push(
                src32,
                0,
                triton.cdiv(self.num_send, pack),
                region,
                region_cols,
                0,
                k,
                dst=self.rank - 1,
            )
        assert dtype in (torch.float32, local_logits.dtype)
        out = torch.empty(rows, self.num_cols, dtype=dtype, device=local_logits.device)
        if rows > 0:
            num_local = self.num_cols - self.num_recv
            _owned_logits_kernel[(rows, triton.cdiv(self.num_cols, 1024))](
                out,
                out.stride(0),
                local_logits,
                local_logits.stride(0),
                self.buf,
                self.half_words,
                region,
                region_cols,
                self.num_cols,
                num_local,
                self.col_start - self.tp_start,
                self.epoch,
                k,
                self.errors,
                PACK=pack,
                SPIN_LIMIT=_SPIN_LIMIT,
                BLK=1024,
            )
        return out

    def token_logits(
        self,
        target_logits: torch.Tensor,  # [num_logits, num_cols] owned, processed
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        """fp32 ``[num_logits, 2]``: the target and draft logits of each logit
        row's drafted token (``draft_sampled[i + 1]``), from its owner."""
        n = target_logits.shape[0]
        mine = torch.empty(n, 2, dtype=torch.float32, device=target_logits.device)
        if n > 0:
            _token_logits_kernel[(triton.cdiv(n, _BLK),)](
                mine,
                target_logits,
                target_logits.stride(0),
                draft_logits,
                draft_logits.stride(0) if draft_logits is not None else 0,
                draft_logits.stride(1) if draft_logits is not None else 0,
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                n,
                draft_logits.shape[1] if draft_logits is not None else 0,
                self.col_start,
                self.col_end,
                BLK=_BLK,
            )
        every = self.allgather_slots(mine)
        out = torch.empty_like(mine)
        if n > 0:
            _max_slots_kernel[(triton.cdiv(2 * n, _BLK),)](
                out, every, 2 * n, WORLD=self.world, BLK=_BLK
            )
        return out

    # -- top-p -----------------------------------------------------------------

    def top_p(self, logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """In place: mask the owned columns ``logits`` (fp32 [rows, num_cols])
        to each row's exact top-p set over the whole vocabulary."""
        rows, ncols = logits.shape
        assert ncols == self.num_cols and logits.dtype == torch.float32
        if rows == 0:
            return logits
        dev = logits.device
        p = top_p.to(torch.float32).contiguous()
        nsp = _tie_splits(rows)
        nsplit = max(1, min(16, 1024 // rows))
        mx = torch.full((rows,), float("-inf"), dtype=torch.float32, device=dev)
        _topp_max_kernel[(rows, nsp)](
            logits, logits.stride(0), ncols, p, mx, BLOCK=1024, num_warps=4
        )
        mm = self.allgather_slots(mx.unsqueeze(1)).view(self.world, rows)
        st = torch.empty(rows, 8, dtype=torch.int64, device=dev)
        cx = torch.empty(rows, _TOPP_CAP, dtype=torch.float32, device=dev)
        ccnt = torch.zeros(rows, dtype=torch.int32, device=dev)
        # Two histogram regions, alternating between rounds.
        hist_words = self.world * rows * 2 * (_TOPP_BINS + 1)
        hist_regions = (self.reserve(hist_words), self.reserve(hist_words))
        for rnd in range(5):
            if rnd == 1:
                _topp_compact_kernel[(rows, nsp)](
                    logits,
                    logits.stride(0),
                    ncols,
                    mm,
                    st,
                    cx,
                    ccnt,
                    _TOPP_CAP,
                    WORLD=self.world,
                    NB=_TOPP_BINS,
                    RANGE=_TOPP_RANGE,
                    BLOCK=1024,
                    num_warps=4,
                )
            hist = torch.zeros(rows, _TOPP_BINS + 1, dtype=torch.int64, device=dev)
            _topp_hist_kernel[(rows, nsplit)](
                logits,
                logits.stride(0),
                ncols,
                p,
                mm,
                st,
                hist,
                cx,
                ccnt,
                _TOPP_CAP,
                ROUND=rnd,
                WORLD=self.world,
                NB=_TOPP_BINS,
                RANGE=_TOPP_RANGE,
                BLOCK=1024 if rnd == 0 else 256,
                num_warps=4,
            )
            hall = self.allgather_slots(hist, reuse=hist_regions[rnd % 2])
            _topp_select_kernel[(rows,)](
                hall,
                p,
                mm,
                st,
                ROUND=rnd,
                WORLD=self.world,
                NB=_TOPP_BINS,
                num_warps=4,
            )
        cnts = torch.empty(rows, nsp, dtype=torch.int32, device=dev)
        _topp_tie_count_kernel[(rows, nsp)](
            logits,
            logits.stride(0),
            ncols,
            st,
            cnts,
            NSP=nsp,
            BLOCK=1024,
            num_warps=4,
        )
        call = self.allgather_slots(cnts)
        bidx = torch.empty(rows, 1, dtype=torch.int32, device=dev)
        _topp_tie_index_kernel[(rows,)](
            logits,
            logits.stride(0),
            ncols,
            self.col_start,
            mm,
            st,
            cnts,
            call,
            bidx,
            self.rank,
            WORLD=self.world,
            NSP=nsp,
            BLOCK=1024,
            num_warps=4,
        )
        ball = self.allgather_slots(bidx).view(self.world, rows)
        _topp_mask_kernel[(rows, nsp)](
            logits,
            logits.stride(0),
            ncols,
            self.col_start,
            mm,
            st,
            ball,
            WORLD=self.world,
            BLOCK=1024,
            num_warps=4,
        )
        return logits


def _tie_splits(rows: int) -> int:
    return 16 if rows <= 64 else (8 if rows <= 128 else 4)
