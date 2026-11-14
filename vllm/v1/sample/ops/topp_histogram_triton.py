# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Histogram-bracket top-p masking for large batches.

The monolithic Triton top-k/top-p kernel runs one program per row and searches
the top-p pivot with repeated full-vocab sweeps, so at large batches (e.g. 256
rows of a 200k vocab) it costs about a millisecond. Such batches reach
``apply_top_k_top_p`` from the sampler when the fused FlashInfer sampler is not
used (a greedy or seeded row in the batch, processed logprobs, or FlashInfer
unavailable), and from the rejection samplers, which mask the processed fp32
logits of speculative-decoding verification before verifying. This pipeline
finds the exact top-p boundary in four streaming passes over the row, each
split across several programs per row:

1. min/max per slice.
2. sum of exp(x - max) and a histogram of (max - x) over ``NUM_BINS`` bins.
   A per-row program then brackets the boundary between the first bin whose
   upper mass bound reaches p and the first whose lower bound does (usually one
   or two adjacent bins).
3. the exact mass above the bracket, and a gather of the bracket's tokens into
   per-slice scratch (no atomics). A per-row program builds a fine histogram
   over the gathered tokens, gathers the few tokens of the fine bracket and
   sorts them by (logit desc, index asc) to find the exact boundary token.
4. the mask.

Kept set: the smallest prefix of tokens ordered by (logit desc, index asc)
whose probability mass reaches p, plus every token equal to the row max. It
matches a float64 reference except where the reference's cumulative mass at
the boundary is within fp32 rounding of p (the masses are fp32 with
compensated summation); there the boundary can move by one token.

When a bracket holds more tokens than the scratch that sorts it, typically
many exact ties (bf16 logits upcast to fp32 share few distinct values), the
boundary is found by walking the bracket's distinct values downwards, adding
each value's mass and resolving the boundary value's ties by index: over the
gathered tokens, or over the row if even the gather overflowed (slow; seen
only on synthetic tie-heavy rows). Only a bracket with more than
``_MAX_DISTINCT`` distinct values beyond the scratch keeps its whole coarse
bracket, a superset whose mass still reaches p.

Rows with p >= 1 are left untouched, as are rows whose finite logits are all
equal or that hold a single finite logit.
"""

from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import (
    WarmupChoices,
    WarmupIntRange,
    _when,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import num_compute_units

# Smallest batch routed here; smaller batches keep the split-row pipeline.
MIN_BATCH_SIZE = 32
# Coarse bins over [max - R, max], R = min(max - min, _MAX_RANGE) in logit units.
NUM_BINS = 4096
# Fine bins over the gathered coarse bracket.
NUM_FINE_BINS = 1024
# Tokens of the fine bracket sorted exactly per row.
FINE_CAPACITY = 256
# Expected coarse-bracket tokens per row; each slice gets twice its share.
ROW_CAPACITY = 8192
# Tokens more than this many nats below the max share the last bin.
_MAX_RANGE = 40.0
# Programs per row: aim for this many programs per SM, slices >= _MIN_SLICE.
_PROGRAMS_PER_SM = 4
_MAX_SLICES = 32
_MIN_SLICE = 4096
# Bracket bounds are relaxed by this relative margin against rounding.
_BOUND_MARGIN = 1e-5
# Distinct values a too-crowded fine bracket is walked through before a row
# falls back to its coarse bracket.
_MAX_DISTINCT = 64

_LOG2E = tl.constexpr(1.4426950408889634)
# Per-row state floats: flag, max, sum_exp, inv_bin_width, first_bin, last_bin,
# boundary_logit, boundary_index (int32 bits).
_RS = tl.constexpr(8)

# One grow-only scratch workspace per device.
_WORKSPACE: dict[torch.device, dict[str, torch.Tensor]] = {}


def _num_slices(batch_size: int, vocab_size: int, num_sm: int) -> int:
    slices = 1
    while (
        slices < _MAX_SLICES
        and slices * batch_size < _PROGRAMS_PER_SM * num_sm
        and vocab_size // (2 * slices) >= _MIN_SLICE
    ):
        slices *= 2
    return slices


def _slice_capacity(slices: int) -> int:
    return next_power_of_2(min(ROW_CAPACITY, max(512, 2 * ROW_CAPACITY // slices)))


@triton.jit
def _kahan_add(total, comp, value):
    """Compensated (Kahan) summation step: returns (total, compensation)."""
    y = value - comp
    t = total + y
    return t, (t - total) - y


@triton.jit
def _row_range(STATS, row, S: tl.constexpr, NB: tl.constexpr, MAX_RANGE: tl.constexpr):
    s = tl.arange(0, S)
    row_max = tl.max(tl.load(STATS + (row * S + s) * 2 + 0))
    row_min = tl.min(tl.load(STATS + (row * S + s) * 2 + 1))
    rng = tl.minimum(row_max - row_min, MAX_RANGE)
    inv_w = NB / tl.maximum(rng, 1e-6)
    return row_max, row_min, rng, inv_w


@triton.jit
def _coarse_bin(x, row_max, inv_w, NB: tl.constexpr):
    b = ((row_max - x) * inv_w).to(tl.int32)
    return tl.minimum(tl.maximum(b, 0), NB - 1)


@triton.jit
def _fine_bin(x, row_max, d_hi, inv_w2, NBF: tl.constexpr):
    # Distances below row_max, so that no large terms cancel.
    fb = (((row_max - x) - d_hi) * inv_w2).to(tl.int32)
    return tl.minimum(tl.maximum(fb, 0), NBF - 1)


@triton.jit
def _max_count(v, cnt, x, valid):
    """Running (largest valid value, its count), updated with one chunk."""
    m = tl.max(tl.where(valid, x, -float("inf")))
    c = tl.sum((valid & (x == m)).to(tl.int32))
    cnt = tl.where(m > v, c, tl.where(m == v, cnt + c, cnt))
    return tl.maximum(v, m), cnt


@triton.jit
def _topp_hist_minmax_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    P,
    STATS,
    VOCAB_SIZE: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // S
    if tl.load(P + row) >= 1.0:
        return
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = (pid % S) * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    mx = tl.full((BLOCK,), -float("inf"), tl.float32)
    mn = tl.full((BLOCK,), float("inf"), tl.float32)
    for i in range(start, end, BLOCK):
        x = tl.load(ROW + i + offs, mask=i + offs < end, other=-float("inf"))
        mx = tl.maximum(mx, x)
        mn = tl.minimum(mn, tl.where(x > -float("inf"), x, float("inf")))
    tl.store(STATS + pid * 2 + 0, tl.max(mx))
    tl.store(STATS + pid * 2 + 1, tl.min(mn))


@triton.jit
def _topp_hist_coarse_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    P,
    STATS,
    HIST,
    ZPART,
    VOCAB_SIZE: tl.constexpr,
    S: tl.constexpr,
    NB: tl.constexpr,
    MAX_RANGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // S
    if tl.load(P + row) >= 1.0:
        return
    row_max, row_min, rng, inv_w = _row_range(STATS, row, S, NB, MAX_RANGE)
    if not (rng > 0.0):
        return
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = (pid % S) * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    z = tl.zeros((BLOCK,), tl.float32)
    zc = tl.zeros((BLOCK,), tl.float32)  # Kahan compensation
    hist = tl.zeros((NB,), tl.int32)
    for i in range(start, end, BLOCK):
        x = tl.load(ROW + i + offs, mask=i + offs < end, other=-float("inf"))
        finite = x > -float("inf")
        z, zc = _kahan_add(
            z, zc, tl.where(finite, tl.exp2((x - row_max) * _LOG2E), 0.0)
        )
        hist += tl.histogram(_coarse_bin(x, row_max, inv_w, NB), NB, mask=finite)
    tl.store(ZPART + pid, tl.sum(z - zc))
    tl.store(HIST + pid.to(tl.int64) * NB + tl.arange(0, NB), hist)


@triton.jit
def _topp_hist_bracket_kernel(
    P,
    STATS,
    HIST,
    ZPART,
    ROWSTATE,
    S: tl.constexpr,
    NB: tl.constexpr,
    MAX_RANGE: tl.constexpr,
    MARGIN: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.load(P + row)
    flag = 0.0
    if p < 1.0:
        row_max, row_min, rng, inv_w = _row_range(STATS, row, S, NB, MAX_RANGE)
        if rng > 0.0:
            z = tl.sum(tl.load(ZPART + row * S + tl.arange(0, S)))
            target = p * z
            b = tl.arange(0, NB)
            cnt = tl.zeros((NB,), tl.int32)
            for j in range(S):
                cnt += tl.load(HIST + (row * S + j).to(tl.int64) * NB + b)
            c = cnt.to(tl.float32)
            w = 1.0 / inv_w
            bf = b.to(tl.float32)
            upper = c * tl.exp2(-bf * w * _LOG2E) * (1.0 + MARGIN)
            # The last bin also holds everything below max - R when R is capped.
            lower_f = tl.where(
                (b == NB - 1) & (row_max - row_min > rng),
                0.0,
                tl.exp2(-(bf + 1.0) * w * _LOG2E) * (1.0 - MARGIN),
            )
            lower = c * lower_f
            # Per-bin bounds are loose by about w of the mass they sum, too
            # loose near p = 1. Also bound the mass up to each bin by the
            # exact total minus the bounds of the bins below it.
            below_u = tl.cumsum(upper, 0, reverse=True) - upper
            below_l = tl.cumsum(lower, 0, reverse=True) - lower
            cu = tl.minimum(tl.cumsum(upper, 0), z * (1.0 + MARGIN) - below_l)
            cl = tl.maximum(tl.cumsum(lower, 0), z * (1.0 - MARGIN) - below_u)
            first_bin = tl.min(tl.where(cu >= target, b, NB - 1))
            last_bin = tl.min(tl.where(cl >= target, b, NB - 1))
            flag = 1.0
            base = ROWSTATE + row * _RS
            tl.store(base + 1, row_max)
            tl.store(base + 2, z)
            tl.store(base + 3, inv_w)
            tl.store(base + 4, first_bin.to(tl.float32))
            tl.store(base + 5, last_bin.to(tl.float32))
    tl.store(ROWSTATE + row * _RS, flag)


@triton.jit
def _topp_hist_gather_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    ROWSTATE,
    ABOVE,
    COUNTS,
    CAND_X,
    CAND_I,
    VOCAB_SIZE: tl.constexpr,
    S: tl.constexpr,
    NB: tl.constexpr,
    SLICE_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // S
    base = ROWSTATE + row * _RS
    if tl.load(base) == 0.0:
        return
    row_max = tl.load(base + 1)
    inv_w = tl.load(base + 3)
    first_bin = tl.load(base + 4).to(tl.int32)
    last_bin = tl.load(base + 5).to(tl.int32)
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = (pid % S) * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    above = tl.zeros((BLOCK,), tl.float32)
    above_c = tl.zeros((BLOCK,), tl.float32)  # Kahan compensation
    n = 0
    out = pid.to(tl.int64) * SLICE_CAP
    for i in range(start, end, BLOCK):
        x = tl.load(ROW + i + offs, mask=i + offs < end, other=-float("inf"))
        finite = x > -float("inf")
        b = _coarse_bin(x, row_max, inv_w, NB)
        e = tl.exp2((x - row_max) * _LOG2E)
        above, above_c = _kahan_add(
            above, above_c, tl.where(finite & (b < first_bin), e, 0.0)
        )
        cand = finite & (b >= first_bin) & (b <= last_bin)
        pos = n + tl.cumsum(cand.to(tl.int32), 0) - 1
        ok = cand & (pos < SLICE_CAP)
        tl.store(CAND_X + out + pos, x, mask=ok)
        tl.store(CAND_I + out + pos, i + offs, mask=ok)
        n += tl.sum(cand.to(tl.int32))
    tl.store(ABOVE + pid, tl.sum(above - above_c))
    tl.store(COUNTS + pid, n)


@triton.jit
def _topp_hist_boundary_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    P,
    ROWSTATE,
    ABOVE,
    COUNTS,
    CAND_X,
    CAND_I,
    FINE,
    S: tl.constexpr,
    NB: tl.constexpr,
    NBF: tl.constexpr,
    SLICE_CAP: tl.constexpr,
    FINE_CAP: tl.constexpr,
    CHUNK: tl.constexpr,
    ROW_CHUNK: tl.constexpr,
    MARGIN: tl.constexpr,
    MAX_DISTINCT: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = ROWSTATE + row * _RS
    if tl.load(base) == 0.0:
        return
    target = tl.load(P + row) * tl.load(base + 2)
    row_max = tl.load(base + 1)
    w = 1.0 / tl.load(base + 3)
    first_bin = tl.load(base + 4)
    last_bin = tl.load(base + 5)
    s = tl.arange(0, S)
    counts = tl.load(COUNTS + row * S + s)
    above = tl.sum(tl.load(ABOVE + row * S + s))
    mode = 1.0
    boundary_x = -float("inf")
    boundary_i = tl.full((), 2147483647, tl.int32)
    if tl.max(counts) > SLICE_CAP:
        # The coarse bracket overflowed the gather scratch (extreme ties or a
        # very flat tail): walk the bracket's distinct values over the row.
        ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
        inv_w = tl.load(base + 3)
        lo_bin = first_bin.to(tl.int32)
        hi_bin = last_bin.to(tl.int32)
        r = tl.arange(0, ROW_CHUNK)
        prev = float("inf")
        done = 0
        it = 0
        while (done == 0) & (it < MAX_DISTINCT):
            it += 1
            v = -float("inf")
            cnt = 0
            for i in range(0, VOCAB_SIZE, ROW_CHUNK):
                x = tl.load(ROW + i + r, mask=i + r < VOCAB_SIZE, other=-float("inf"))
                b = _coarse_bin(x, row_max, inv_w, NB)
                inb = (x > -float("inf")) & (b >= lo_bin) & (b <= hi_bin) & (x < prev)
                v, cnt = _max_count(v, cnt, x, inb)
            e = tl.exp2((v - row_max) * _LOG2E)
            if (v == -float("inf")) | (above + cnt * e >= target):
                need = tl.maximum(target - above, 0.0)
                kk = tl.minimum(tl.maximum(tl.ceil(need / e).to(tl.int32), 1), cnt)
                seen = 0
                for i in range(0, VOCAB_SIZE, ROW_CHUNK):
                    x = tl.load(
                        ROW + i + r, mask=i + r < VOCAB_SIZE, other=-float("inf")
                    )
                    eq = x == v
                    pos = seen + tl.cumsum(eq.to(tl.int32), 0)
                    hit = eq & (pos == kk)
                    if tl.sum(hit.to(tl.int32)) > 0:
                        boundary_i = tl.sum(tl.where(hit, i + r, 0))
                    seen += tl.sum(eq.to(tl.int32))
                boundary_x = v
                done = 1
            else:
                above += cnt * e
                prev = v
        if done == 0:
            mode = 2.0  # keep the whole coarse bracket
    else:
        # Fine bins over the bracket: distances [d_hi, d_hi + (last-first+1) w)
        # below row_max.
        d_hi = first_bin * w
        w2 = (last_bin - first_bin + 1.0) * w / NBF
        inv_w2 = 1.0 / w2
        k = tl.arange(0, CHUNK)
        hist = tl.zeros((NBF,), tl.int32)
        for j in range(S):
            nj = tl.load(COUNTS + row * S + j)
            src = (row * S + j).to(tl.int64) * SLICE_CAP
            for c0 in range(0, SLICE_CAP, CHUNK):
                if c0 < nj:
                    m = c0 + k < nj
                    x = tl.load(CAND_X + src + c0 + k, mask=m, other=-float("inf"))
                    fb = _fine_bin(x, row_max, d_hi, inv_w2, NBF)
                    hist += tl.histogram(fb, NBF, mask=m)
        fbin = tl.arange(0, NBF)
        c = hist.to(tl.float32)
        ff = fbin.to(tl.float32)
        upper = c * tl.exp2(-(d_hi + ff * w2) * _LOG2E) * (1.0 + MARGIN)
        lower = c * tl.exp2(-(d_hi + (ff + 1.0) * w2) * _LOG2E) * (1.0 - MARGIN)
        cu = above + tl.cumsum(upper, 0)
        cl = above + tl.cumsum(lower, 0)
        f_first = tl.min(tl.where(cu >= target, fbin, NBF - 1))
        f_last = tl.min(tl.where(cl >= target, fbin, NBF - 1))
        # Exact mass above the fine bracket; compact the fine bracket's tokens.
        above_fine = tl.zeros((CHUNK,), tl.float32)
        above_fine_c = tl.zeros((CHUNK,), tl.float32)
        nf = 0
        fine_row = FINE + row.to(tl.int64) * FINE_CAP * 2
        for j in range(S):
            nj = tl.load(COUNTS + row * S + j)
            src = (row * S + j).to(tl.int64) * SLICE_CAP
            for c0 in range(0, SLICE_CAP, CHUNK):
                if c0 < nj:
                    m = c0 + k < nj
                    x = tl.load(CAND_X + src + c0 + k, mask=m, other=-float("inf"))
                    ix = tl.load(CAND_I + src + c0 + k, mask=m, other=0)
                    fb = _fine_bin(x, row_max, d_hi, inv_w2, NBF)
                    above_fine, above_fine_c = _kahan_add(
                        above_fine,
                        above_fine_c,
                        tl.where(
                            m & (fb < f_first), tl.exp2((x - row_max) * _LOG2E), 0.0
                        ),
                    )
                    cand = m & (fb >= f_first) & (fb <= f_last)
                    pos = nf + tl.cumsum(cand.to(tl.int32), 0) - 1
                    ok = cand & (pos < FINE_CAP)
                    # +0.0 for -0.0: the sort key must order like the mask's ==.
                    xk = tl.where(x == 0.0, 0.0, x).to(tl.int32, bitcast=True)
                    tl.store(fine_row + pos * 2, xk, mask=ok)
                    tl.store(fine_row + pos * 2 + 1, ix, mask=ok)
                    nf += tl.sum(cand.to(tl.int32))
        above += tl.sum(above_fine - above_fine_c)
        if nf > FINE_CAP:
            # Too many tokens in the fine bracket to sort, typically ties (bf16
            # logits upcast to fp32 share few distinct values). Walk the
            # bracket's distinct values downwards, adding each value's mass,
            # and resolve the boundary value's ties by index.
            prev = float("inf")
            done = 0
            it = 0
            while (done == 0) & (it < MAX_DISTINCT):
                it += 1
                v = -float("inf")
                cnt = 0
                for j in range(S):
                    nj = tl.load(COUNTS + row * S + j)
                    src = (row * S + j).to(tl.int64) * SLICE_CAP
                    for c0 in range(0, SLICE_CAP, CHUNK):
                        if c0 < nj:
                            m = c0 + k < nj
                            x = tl.load(
                                CAND_X + src + c0 + k, mask=m, other=-float("inf")
                            )
                            fb = _fine_bin(x, row_max, d_hi, inv_w2, NBF)
                            inb = m & (fb >= f_first) & (fb <= f_last) & (x < prev)
                            v, cnt = _max_count(v, cnt, x, inb)
                e = tl.exp2((v - row_max) * _LOG2E)
                if (v == -float("inf")) | (above + cnt * e >= target):
                    # Tokens of value v to keep, in index order: the first kk.
                    need = tl.maximum(target - above, 0.0)
                    kk = tl.minimum(tl.maximum(tl.ceil(need / e).to(tl.int32), 1), cnt)
                    seen = 0
                    for j in range(S):
                        nj = tl.load(COUNTS + row * S + j)
                        src = (row * S + j).to(tl.int64) * SLICE_CAP
                        for c0 in range(0, SLICE_CAP, CHUNK):
                            if c0 < nj:
                                m = c0 + k < nj
                                x = tl.load(
                                    CAND_X + src + c0 + k, mask=m, other=-float("inf")
                                )
                                ix = tl.load(CAND_I + src + c0 + k, mask=m, other=0)
                                eq = m & (x == v)
                                pos = seen + tl.cumsum(eq.to(tl.int32), 0)
                                hit = eq & (pos == kk)
                                if tl.sum(hit.to(tl.int32)) > 0:
                                    boundary_i = tl.sum(tl.where(hit, ix, 0))
                                seen += tl.sum(eq.to(tl.int32))
                    boundary_x = v
                    done = 1
                else:
                    above += cnt * e
                    prev = v
            if done == 0:
                mode = 2.0  # more distinct values than the walk allows
        else:
            q = tl.arange(0, FINE_CAP)
            valid = q < nf
            xb = tl.load(fine_row + q * 2, mask=valid, other=0)
            ix = tl.load(fine_row + q * 2 + 1, mask=valid, other=2147483647)
            # Sort by (logit desc, index asc): ascending on
            # (~orderable(logit)) << 32 | index, orderable() being monotone.
            orderable = tl.where(xb >= 0, xb ^ (-2147483648), ~xb)
            key = (~orderable).to(tl.uint32).to(tl.uint64) << 32
            key = key | ix.to(tl.uint32).to(tl.uint64)
            key = tl.where(valid, key, 0xFFFFFFFFFFFFFFFF)
            key = tl.sort(key, 0)
            sorted_i = (key & 0xFFFFFFFF).to(tl.uint32).to(tl.int32)
            sorted_o = (~((key >> 32).to(tl.uint32))).to(tl.int32)
            sorted_b = tl.where(sorted_o < 0, sorted_o ^ (-2147483648), ~sorted_o)
            sorted_x = sorted_b.to(tl.float32, bitcast=True)
            mass = tl.where(valid, tl.exp2((sorted_x - row_max) * _LOG2E), 0.0)
            cum = above + tl.cumsum(mass, 0)
            first = tl.min(tl.where(valid & (cum >= target), q, FINE_CAP))
            if first == FINE_CAP:
                # Rounding left the target unreached: keep the whole bracket.
                first = tl.maximum(nf - 1, 0)
            sel = q == first
            boundary_x = tl.sum(tl.where(sel, sorted_x, 0.0))
            boundary_i = tl.sum(tl.where(sel, sorted_i, 0))
    tl.store(base + 6, boundary_x)
    tl.store(base + 7, boundary_i.to(tl.float32, bitcast=True))
    tl.store(base, mode)


@triton.jit
def _topp_hist_mask_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    ROWSTATE,
    MASK_VALUE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    S: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // S
    base = ROWSTATE + row * _RS
    mode = tl.load(base)
    if mode == 0.0:
        return
    row_max = tl.load(base + 1)
    inv_w = tl.load(base + 3)
    last_bin = tl.load(base + 5).to(tl.int32)
    boundary_x = tl.load(base + 6)
    boundary_i = tl.load(base + 7).to(tl.int32, bitcast=True)
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = (pid % S) * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    for i in range(start, end, BLOCK):
        m = i + offs < end
        x = tl.load(ROW + i + offs, mask=m, other=-float("inf"))
        if mode == 2.0:
            keep = _coarse_bin(x, row_max, inv_w, NB) <= last_bin
        else:
            keep = (x > boundary_x) | ((x == boundary_x) & (i + offs <= boundary_i))
        keep = keep | (x == row_max)
        tl.store(ROW + i + offs, tl.where(keep, x, MASK_VALUE), mask=m)


# --- dispatch + warmup ------------------------------------------------------


def max_sampler_batch_size(vllm_config: Any) -> int:
    """Largest number of logits rows the sampler masks in one call."""
    return vllm_config.scheduler_config.max_num_seqs * (
        vllm_config.num_speculative_tokens + 1
    )


_SCRATCH_DTYPES = dict(
    p=torch.float32,
    stats=torch.float32,
    hist=torch.int32,
    zpart=torch.float32,
    rowstate=torch.float32,
    above=torch.float32,
    counts=torch.int32,
    cand_x=torch.float32,
    cand_i=torch.int32,
    fine=torch.int32,
)


def _warmup_case(
    vllm_config: Any, batch_size: int, logits_stride: int, names: tuple[str, ...]
) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    case: dict[str, Any] = dict(
        logits=TritonWarmupTensor(
            torch.float32, shape=(batch_size, vocab_size), strides=(logits_stride, 1)
        ),
        num_sm=num_compute_units(),
    )
    for name in names:
        case[name] = (
            float("-inf")
            if name == "mask_value"
            else TritonWarmupTensor(_SCRATCH_DTYPES[name])
        )
    return case


def _warmup_inputs(*names: str) -> Any:
    """Warmup cases of a launcher taking logits, ``names`` and num_sm."""

    def warmup_inputs(
        vllm_config: Any, names: tuple[str, ...] = names
    ) -> dict[str, Any]:
        _when(current_platform.is_cuda())
        batch_size: Any = WarmupIntRange(
            MIN_BATCH_SIZE, max_sampler_batch_size(vllm_config) + 1
        )
        logits_stride: Any = WarmupChoices(16, 2)
        return _warmup_case(vllm_config, batch_size, logits_stride, names)

    return warmup_inputs


def _layout(logits: torch.Tensor, num_sm: int) -> tuple[int, int, int]:
    batch_size, vocab_size = logits.shape
    slices = _num_slices(batch_size, vocab_size, num_sm)
    return batch_size, vocab_size, slices


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_hist_minmax_kernel,
    warmup_inputs=_warmup_inputs("p", "stats"),
)
def _launch_minmax(
    logits: torch.Tensor, p: torch.Tensor, stats: torch.Tensor, num_sm: int
) -> DispatchSpec:
    batch_size, vocab_size, slices = _layout(logits, num_sm)
    return (batch_size * slices,), dict(
        VOCAB_SIZE=vocab_size, S=slices, BLOCK=2048, num_warps=8
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_hist_coarse_kernel,
    warmup_inputs=_warmup_inputs("p", "stats", "hist", "zpart"),
)
def _launch_coarse(
    logits: torch.Tensor,
    p: torch.Tensor,
    stats: torch.Tensor,
    hist: torch.Tensor,
    zpart: torch.Tensor,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size, slices = _layout(logits, num_sm)
    return (batch_size * slices,), dict(
        VOCAB_SIZE=vocab_size,
        S=slices,
        NB=NUM_BINS,
        MAX_RANGE=_MAX_RANGE,
        BLOCK=2048,
        num_warps=8,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_hist_bracket_kernel,
    warmup_inputs=_warmup_inputs("p", "stats", "hist", "zpart", "rowstate"),
)
def _launch_bracket(
    logits: torch.Tensor,
    p: torch.Tensor,
    stats: torch.Tensor,
    hist: torch.Tensor,
    zpart: torch.Tensor,
    rowstate: torch.Tensor,
    num_sm: int,
) -> DispatchSpec:
    batch_size, _, slices = _layout(logits, num_sm)
    return (batch_size,), dict(
        S=slices, NB=NUM_BINS, MAX_RANGE=_MAX_RANGE, MARGIN=_BOUND_MARGIN, num_warps=8
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_hist_gather_kernel,
    warmup_inputs=_warmup_inputs("rowstate", "above", "counts", "cand_x", "cand_i"),
)
def _launch_gather(
    logits: torch.Tensor,
    rowstate: torch.Tensor,
    above: torch.Tensor,
    counts: torch.Tensor,
    cand_x: torch.Tensor,
    cand_i: torch.Tensor,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size, slices = _layout(logits, num_sm)
    return (batch_size * slices,), dict(
        VOCAB_SIZE=vocab_size,
        S=slices,
        NB=NUM_BINS,
        SLICE_CAP=_slice_capacity(slices),
        BLOCK=512,
        num_warps=4,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_hist_boundary_kernel,
    warmup_inputs=_warmup_inputs(
        "p", "rowstate", "above", "counts", "cand_x", "cand_i", "fine"
    ),
)
def _launch_boundary(
    logits: torch.Tensor,
    p: torch.Tensor,
    rowstate: torch.Tensor,
    above: torch.Tensor,
    counts: torch.Tensor,
    cand_x: torch.Tensor,
    cand_i: torch.Tensor,
    fine: torch.Tensor,
    num_sm: int,
) -> DispatchSpec:
    batch_size, _, slices = _layout(logits, num_sm)
    slice_cap = _slice_capacity(slices)
    return (batch_size,), dict(
        S=slices,
        NB=NUM_BINS,
        NBF=NUM_FINE_BINS,
        SLICE_CAP=slice_cap,
        FINE_CAP=FINE_CAPACITY,
        CHUNK=min(slice_cap, 1024),
        ROW_CHUNK=4096,
        MARGIN=_BOUND_MARGIN,
        MAX_DISTINCT=_MAX_DISTINCT,
        VOCAB_SIZE=logits.shape[1],
        num_warps=8,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_hist_mask_kernel,
    warmup_inputs=_warmup_inputs("rowstate", "mask_value"),
)
def _launch_mask(
    logits: torch.Tensor, rowstate: torch.Tensor, mask_value: float, num_sm: int
) -> DispatchSpec:
    batch_size, vocab_size, slices = _layout(logits, num_sm)
    return (batch_size * slices,), dict(
        VOCAB_SIZE=vocab_size, S=slices, NB=NUM_BINS, BLOCK=2048, num_warps=8
    )


def apply_top_p_histogram(
    logits: torch.Tensor,
    p: torch.Tensor,
    mask_value: float = float("-inf"),
    num_sm: int | None = None,
) -> torch.Tensor:
    """Mask ``logits`` ([batch, vocab] fp32, vocab contiguous) to top-p in place.

    ``p`` is a [batch] fp32 tensor; rows with p >= 1 are left untouched.
    No host synchronization, so it can be captured in a CUDA graph. The
    scratch is grow-only per device: a capture stays valid until a larger batch
    reallocates it.
    """
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert logits.stride(1) == 1
    batch_size, vocab_size = logits.shape
    if num_sm is None:
        num_sm = num_compute_units(logits.device.index)
    programs = batch_size * _num_slices(batch_size, vocab_size, num_sm)
    cands = programs * _slice_capacity(programs // batch_size)
    numels = dict(
        stats=programs * 2,  # (max, min) per program
        hist=programs * NUM_BINS,
        zpart=programs,
        rowstate=batch_size * _RS.value,
        above=programs,
        counts=programs,
        cand_x=cands,
        cand_i=cands,
        fine=batch_size * FINE_CAPACITY * 2,  # (logit bits, index) pairs
    )
    ws = _WORKSPACE.setdefault(logits.device, {})
    for name, numel in numels.items():
        if name not in ws or ws[name].numel() < numel:
            ws[name] = torch.empty(
                next_power_of_2(numel),
                dtype=_SCRATCH_DTYPES[name],
                device=logits.device,
            )
    p = p.to(torch.float32)
    _launch_minmax(logits, p, ws["stats"], num_sm)
    _launch_coarse(logits, p, ws["stats"], ws["hist"], ws["zpart"], num_sm)
    _launch_bracket(
        logits, p, ws["stats"], ws["hist"], ws["zpart"], ws["rowstate"], num_sm
    )
    _launch_gather(
        logits,
        ws["rowstate"],
        ws["above"],
        ws["counts"],
        ws["cand_x"],
        ws["cand_i"],
        num_sm,
    )
    _launch_boundary(
        logits,
        p,
        ws["rowstate"],
        ws["above"],
        ws["counts"],
        ws["cand_x"],
        ws["cand_i"],
        ws["fine"],
        num_sm,
    )
    _launch_mask(logits, ws["rowstate"], mask_value, num_sm)
    return logits


def reset_workspace() -> None:
    _WORKSPACE.clear()


def register_warmups() -> None:
    for launch in (
        _launch_minmax,
        _launch_coarse,
        _launch_bracket,
        _launch_gather,
        _launch_boundary,
        _launch_mask,
    ):
        launch.register_warmup()
