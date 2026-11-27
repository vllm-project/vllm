# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata and schedules for the exact DoSA prefill operator."""

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as ld


@triton.jit(do_not_specialize=["N"])
def rank_query_tiles(
    W, RANK, N, ROWS: tl.constexpr, H: tl.constexpr, HIGH: tl.constexpr
):
    PLAN, COUNTS = RANK
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    valid = row < ROWS
    src = tl.minimum(row, N - 1)
    w = tl.load(W + src[:, None] * H + tl.arange(0, H)[None, :], valid[:, None], 0)
    npos = tl.sum((w > 0).to(tl.int32), 1)
    tier = (
        (npos > 14).to(tl.int32) + (npos > 16).to(tl.int32) + (npos > HIGH).to(tl.int32)
    )
    # At most128 rows contribute: no byte carries into the adjacent class.
    one = tl.where(valid, tl.full((), 1, tl.uint32) << (8 * tier), 0)
    prefix = tl.cumsum(one, 0).to(tl.uint32)
    local = ((prefix >> (8 * tier)) & 255).to(tl.int32) - 1
    tl.store(PLAN + row, (tier << 7) | local, valid)
    packed_count = tl.sum(one, 0).to(tl.uint32)
    tl.store(COUNTS + tl.program_id(0), packed_count.to(tl.int32, bitcast=True))


@triton.jit(do_not_specialize=["N", "OFFSET"])
def stage_queries(
    INPUTS,
    OUTPUTS,
    RANK,
    N,
    OFFSET,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    HIGH: tl.constexpr,
    B: tl.constexpr,
):
    PLAN, COUNTS, ORDER = RANK
    q, w, start, end = INPUTS
    out_q, out_w, out_start, out_end, BUDGET8, BUDGET4 = OUTPUTS
    row = tl.program_id(0)
    code = tl.load(PLAN + row)
    tier, local = code >> 7, code & 127
    tiles = tl.arange(0, B)
    totals = tl.load(COUNTS + tiles, 128 * tiles < ROWS, 0).to(tl.uint32, bitcast=True)
    before = tl.where(tiles < row // 128, (totals >> (8 * tier)) & 255, 0)
    for k in tl.static_range(3):
        before += tl.where(tier > k, (totals >> (8 * k)) & 255, 0)
    dest = tl.sum(before.to(tl.int32), 0) + local
    tl.store(ORDER + dest, row)
    src = tl.minimum(row, N - 1)
    d, h = tl.arange(0, H * 128), tl.arange(0, H)
    tl.store(out_q + dest * H * 128 + d, tl.load(q + src * H * 128 + d))
    tl.store(out_w + dest * H + h, tl.load(w + src * H + h))
    tl.store(out_start + dest, tl.load(start + src) - OFFSET)
    tl.store(out_end + dest, tl.load(end + src) - OFFSET)
    if dest % 8 == 7:
        paid = tl.where(tier == 0, 14, tl.where(tier == 1, 16, HIGH))
        tl.store(BUDGET8 + dest // 8, paid)
        tl.store(BUDGET4 + 2 * (dest // 8) + tl.arange(0, 2), paid)


@triton.jit
def add_up(x, y):
    return ld.add_ru(x, y)


@triton.jit(do_not_specialize=["N"])
def key_finish(PS, PB, S, BAD, N, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(S, tl.max(tl.load(PS + i, i < N, 0.0), 0))
    tl.store(BAD, tl.max(tl.load(PB + i, i < N, 0), 0))


@triton.jit
def query_check(
    W,
    H14,
    H16,
    H20,
    BUDGET,
    S,
    KBAD,
    SB,
    TAU,
    ERR,
    CAPTURE_FLAG,
    SAFE,
    CAPTURE_STRIDE: tl.constexpr,
    GROUP: tl.constexpr,
    DOT: tl.constexpr,
    G: tl.constexpr,
    NATIVE_FACTOR: tl.constexpr,
    HEADS: tl.constexpr,
    HIGH: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.load(BUDGET + row // GROUP)
    valid_family = (b == 14) | (b == 16) | (b == HIGH)
    hp = tl.where(b == 14, H14, tl.where(b == 16, H16, H20))
    hint = tl.load(hp + row)
    incoming = tl.load(ERR + row)
    if incoming != 0:
        # The original selected planner starts invalid on a seed/extent error.
        tl.store(CAPTURE_FLAG + row * CAPTURE_STRIDE, incoming)
        if valid_family:
            hint = 1
            tl.store(hp + row, hint)
    h = tl.arange(0, HEADS)
    w = tl.load(W + row * HEADS + h)
    mag = w.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    domain = (mag == 0) | ((mag >= 0x15800000) & (mag <= 0x6E800000))
    sw = tl.reduce(tl.abs(w).to(tl.float64), 0, add_up)
    one = tl.full((), 1.0, tl.float64)
    d = ld.mul_ru(
        ld.add_ru(one, tl.full((), DOT, tl.float64)),
        tl.full((), 25690112.0, tl.float64),
    )
    pre = ld.mul_ru(ld.mul_ru(d, sw), ld.add_ru(one, tl.full((), G, tl.float64)))
    double_pre = ld.mul_ru(tl.full((), 2.0, tl.float64), pre)
    native_pre = ld.mul_ru(ld.mul_ru(d, sw), tl.full((), NATIVE_FACTOR, tl.float64))
    smax = ld.add_ru(
        ld.mul_ru(tl.load(S).to(tl.float64), native_pre),
        tl.full((), 4 * 2.0**-126, tl.float64),
    )
    limit = tl.full((), 3.3895313892515355e38, tl.float64)
    tau = tl.load(TAU + row)
    bad = (
        ~valid_family
        | (hint != 2)
        | (tl.load(KBAD) != 0)
        | (tl.load(SB + row) != 0)
        | (tl.sum((~domain).to(tl.int32)) != 0)
        | (sw > tl.full((), 2.0**100, tl.float64))
        | ~(double_pre < tl.full((), 3.4028234663852886e38, tl.float64))
        | ~(smax < limit)
        | ~(tl.abs(tau) < float("inf"))
    )
    if bad:
        tl.atomic_or(ERR + row, 1024)
    tl.store(SAFE + row, tl.where(bad | (incoming != 0), -float("inf"), tau))


@triton.jit
def begin_slab(
    STATE,
    SNAPSHOT,
    LIVE,
    ROWS: tl.constexpr,
    CAP: tl.constexpr,
    SEED: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """Initialize the flags and queues before the seed and plan branches."""
    error, pool, solo, common = STATE
    row = tl.program_id(0) * 256 + tl.arange(0, 256)
    nk = tl.load(LIVE)
    bad = ((nk < SEED) | (nk > CAP)).to(tl.int32)
    tl.store(error + row, bad, row < ROWS)
    tl.store(SNAPSHOT + row * STRIDE, bad, row < ROWS)
    if tl.program_id(0) == 0:
        i = tl.arange(0, 2)
        tl.store(pool, 0)
        tl.store(solo + i, 0)
        tl.store(common + i, 0)


@triton.jit
def key_parts_live(
    KS,
    RHO,
    PS,
    PB,
    LIVE,
    CAP: tl.constexpr,
    SEED: tl.constexpr,
    WORKERS: tl.constexpr,
    BLOCK: tl.constexpr,
    MIXED_MX: tl.constexpr = False,
):
    worker = tl.program_id(0)
    nk = tl.load(LIVE)
    valid = (nk >= SEED) & (nk <= CAP)
    maximum = tl.full((BLOCK,), 0.0, tl.float32)
    bad = tl.full((BLOCK,), 0, tl.int32)
    if valid:
        parts = (nk + BLOCK - 1) // BLOCK
        for part in range(worker, parts, WORKERS):
            i = part * BLOCK + tl.arange(0, BLOCK)
            m = i < nk
            scale = tl.load(KS + i, m, 1.0)
            bits = scale.to(tl.int32, bitcast=True)
            rho = tl.load(RHO + i, m, 0)
            # Nonuniform native MX keys are never pruned by a scalar certificate.
            valid_rho = (rho >= 0) & (rho < 127)
            if MIXED_MX:
                valid_rho |= rho == 255
            # Repeated UE8M0 scale: raw/MX commutation requires e in [-108,100].
            ok = (
                (bits >= 0x09800000)
                & (bits <= 0x71800000)
                & ((bits & 0x007FFFFF) == 0)
                & valid_rho
            )
            maximum = tl.maximum(maximum, tl.where(m & ok, scale, 0.0))
            bad = bad | (m & ~ok).to(tl.int32)
    else:
        bad = tl.full((BLOCK,), 1, tl.int32)
    tl.store(PS + worker, tl.max(maximum, 0))
    tl.store(PB + worker, tl.max(bad, 0))


@triton.jit(do_not_specialize=["NV"])
def _add_votes(IDS, VOTES, N: tl.constexpr, NV, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(IDS + i, i < N, -1)
    tl.atomic_add(VOTES + key, 1, (i < N) & (key >= 0) & (key < NV), sem="relaxed")


@triton.jit(do_not_specialize=["NQ"])
def _spread_queries(Q, W, OQ, OW, NQ, HEADS: tl.constexpr):
    anchor = tl.program_id(0)
    row = (anchor * (NQ - 1) // (tl.num_programs(0) - 1)).to(tl.int64)
    i = tl.arange(0, HEADS * 128)
    tl.store(OQ + anchor * HEADS * 128 + i, tl.load(Q + row * HEADS * 128 + i))
    weight = tl.load(W + row * HEADS + i, i < HEADS, 0.0)
    tl.store(OW + anchor * HEADS + i, weight, i < HEADS)


@triton.jit
def _decoded_ue8m0(exponent):
    bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["KEYS"])
def recovery_pack_scales(SCALE, PACKED, KEYS, BLOCK: tl.constexpr):
    key = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(SCALE + key, key < KEYS, 0)
    bits = scale.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) if SCALE.dtype.element_ty == tl.float32 else bits) & 255
    # Code0 decodes to F32 bits0x00400000; unused zero padding also has exponent0.
    # Valid intervals exclude the owner's zero K/scale padding.
    packed = exponent * tl.full((), 0x01010101, tl.uint32)
    tl.store(PACKED + key, packed, key < KEYS)


@triton.jit
def finish_slab(ERROR, COUNTS, ORDER, ORIGINAL, ROWS: tl.constexpr):
    """Publish counts and recovery flags in original query order."""
    counts_out, error_out = ORIGINAL
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    active = row < ROWS
    dest = tl.load(ORDER + row, active, 0)
    valid = (dest >= 0) & (dest < ROWS)
    _trap_if(tl.sum((active & ~valid).to(tl.int32), 0) != 0)
    tl.store(error_out + dest, tl.load(ERROR + row, active, 0), active & valid)
    tl.store(counts_out + dest, tl.load(COUNTS + row, active, 0), active & valid)


@triton.jit
def _trap_if(condition):
    tl.inline_asm_elementwise(
        "{ .reg .pred bad; setp.ne.u32 bad, $1, 0; @bad trap; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[condition.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["N", "MAX"])
def _carry_histogram(
    V,
    P,
    N,
    MAX,
    STRIDE: tl.constexpr,
    BINS: tl.constexpr,
):
    """Publish I16 cumulative counts for each original 8192-ID block."""
    ids = tl.program_id(0) * 8192 + tl.arange(0, 8192)
    live = ids < N
    votes = tl.minimum(tl.maximum(tl.load(V + ids, ids < N, 0), 0), MAX)
    hist = tl.histogram(votes, BINS, mask=live & (votes != 0))
    bins = tl.arange(0, BINS)
    hist = tl.where(bins == 0, tl.sum((live & (votes == 0)).to(tl.int32)), hist)
    cumulative = tl.cumsum(hist, reverse=True)
    tl.store(P + tl.program_id(0) * STRIDE + bins, cumulative, bins <= MAX)


@triton.jit(do_not_specialize=["K", "MAX", "BLOCKS"])
def _carry_cutoff(
    P,
    S,
    K,
    MAX,
    BLOCKS,
    STRIDE: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
):
    """Find exact cutoff, small-ID tie quota and ascending block offsets."""
    blocks = tl.arange(0, BLOCK_PAD)
    live = blocks < BLOCKS
    lo, hi = tl.full((), 0, tl.int32), MAX + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        total = tl.sum(tl.load(P + blocks * STRIDE + mid, live, 0).to(tl.int32))
        enough = total >= K
        lo, hi = tl.where(enough, mid, lo), tl.where(enough, hi, mid)
    ge = tl.load(P + blocks * STRIDE + lo, live, 0).to(tl.int32)
    gt = tl.load(P + blocks * STRIDE + lo + 1, live & (lo < MAX), 0).to(tl.int32)
    eq = ge - gt
    before = tl.cumsum(eq) - eq
    needed = K - tl.sum(gt)
    quota = tl.minimum(tl.maximum(needed - before, 0), eq)
    sizes = gt + quota
    offsets = tl.cumsum(sizes) - sizes
    tl.store(S + blocks, offsets, live)
    tl.store(S + BLOCK_PAD + blocks, quota, live)
    tl.store(S + 2 * BLOCK_PAD, lo)


@triton.jit(do_not_specialize=["N", "MAX"])
def _carry_emit(V, HOT, S, N, MAX, BLOCK_PAD: tl.constexpr):
    """Emit ascending HOT IDs and reset votes."""
    block = tl.program_id(0)
    ids = block * 8192 + tl.arange(0, 8192)
    live = ids < N
    votes = tl.minimum(tl.maximum(tl.load(V + ids, live, 0), 0), MAX)
    threshold = tl.load(S + 2 * BLOCK_PAD)
    match = live & (votes == threshold)
    quota = tl.load(S + BLOCK_PAD + block)
    selected = live & (
        (votes > threshold) | (match & (tl.cumsum(match.to(tl.int32)) <= quota))
    )
    rank = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(HOT + tl.load(S + block) + rank, ids.to(tl.int64), selected)
    tl.store(V + ids, 0, live)


@triton.jit
def _magnitude_unit(code):
    exponent, mantissa = code >> 3, code & 7
    return tl.where(
        exponent == 0, mantissa, (8 + mantissa) << tl.maximum(exponent - 1, 0)
    )


@triton.jit
def _exact_rho(square, kappa, live):
    """Floating point proposes only; exact current/previous inequalities certify."""
    estimate = tl.sqrt(square.to(tl.float32) * (1.0 / 128.0)) * (1.0 / 512.0)
    proposal = estimate.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)
    unit = _magnitude_unit(proposal).to(tl.int64)
    proposal += (128 * unit * unit < square).to(tl.int32)
    unit = _magnitude_unit(proposal).to(tl.int64)
    previous = _magnitude_unit(tl.maximum(proposal - 1, 0)).to(tl.int64)
    certified = (
        (proposal <= kappa)
        & (128 * unit * unit >= square)
        & ((proposal == 0) | (128 * previous * previous < square))
    )
    result = proposal
    if tl.sum((live & ~certified).to(tl.int32)) != 0:
        lo, hi = tl.full(square.shape, 0, tl.int32), kappa
        for _ in range(7):
            mid = (lo + hi) // 2
            unit = _magnitude_unit(mid).to(tl.int64)
            enough = 128 * unit * unit >= square
            lo = tl.where((lo < hi) & ~enough, mid + 1, lo)
            hi = tl.where((lo < hi) & enough, mid, hi)
        result = tl.where(certified, proposal, lo)
    return result


@triton.jit
def _gather_key_rows(
    K,
    SCALE,
    DK,
    DS,
    RHO,
    PACKED_KS,
    source,
    rows,
    update,
    gather_live,
    TILE_ROWS: tl.constexpr,
):
    dims = tl.arange(0, 128)
    raw = tl.load(K + source[:, None] * 128 + dims[None, :], gather_live[:, None], 0)
    tl.store(DK + rows[:, None] * 128 + dims[None, :], raw, update[:, None])
    scale = tl.load(SCALE + source, gather_live, 0)
    packed_bad = tl.full((TILE_ROWS,), False, tl.int1)
    mixed = tl.full((TILE_ROWS,), False, tl.int1)
    if SCALE.dtype.element_ty == tl.int32:
        packed = scale.to(tl.uint32)
        exponent = packed & 255
        if PACKED_KS is not None:
            tl.store(PACKED_KS + rows, scale, update)
            mixed = packed != exponent * tl.full((), 0x01010101, tl.uint32)
            second = (packed >> 8) & 255
            third = (packed >> 16) & 255
            fourth = packed >> 24
            # Use the largest component for the native score overflow bound.
            # Out-of-envelope packed values retain the original recovery.
            minimum = tl.minimum(
                tl.minimum(exponent, second), tl.minimum(third, fourth)
            )
            packed_bad = mixed & (minimum < 19)
            exponent = tl.maximum(
                tl.maximum(exponent, second), tl.maximum(third, fourth)
            )
        scale = _decoded_ue8m0(exponent)
    tl.store(DS + rows, scale, update)
    bits = scale.to(tl.uint32, bitcast=True)
    codes = (raw & 127).to(tl.int32)
    units = _magnitude_unit(codes).to(tl.int64)
    units = tl.where(codes == 127, 0, units)
    square, kappa = tl.sum(units * units, 1), tl.max(codes, 1)
    bad = ((bits - 1).to(tl.uint32) >= 0x5F800000) | (kappa == 127) | packed_bad
    rho = _exact_rho(square, kappa, update & ~bad & ~mixed)
    # E4M3 NaN makes all three screens keep the key; native MX reads its
    # original packed scales during seed scoring, refinement and recovery.
    tl.store(RHO + rows, tl.where(bad, 127, tl.where(mixed, 255, rho)), update)


@triton.jit(do_not_specialize=["TOTAL", "ROWS", "MAP_SIZE", "VOTE_SIZE", "RECENT"])
def _map_votes(
    OUT,
    MAP,
    STATUS,
    VOTES,
    TOTAL,
    ROWS,
    MAP_SIZE,
    TOPK: tl.constexpr,
    VOTE_SIZE,
    RECENT,
):
    """Check metadata before reads, map winners in place and vote recent rows."""
    lanes = tl.arange(0, 256)
    for base in range(tl.program_id(0) * 256, TOTAL, tl.num_programs(0) * 256):
        ids = base + lanes
        status = tl.load(STATUS + base // TOPK)
        _trap_if(status != 0)
        physical = tl.load(OUT + ids, ids < TOTAL, 0)
        valid = (physical >= 0) & (physical < MAP_SIZE)
        _trap_if(tl.sum(((ids < TOTAL) & ~valid).to(tl.int32)) != 0)
        logical = tl.load(MAP + physical, (ids < TOTAL) & valid, 0)
        valid = (logical >= 0) & (logical < MAP_SIZE) & (logical <= 0xFFFFF)
        _trap_if(tl.sum(((ids < TOTAL) & ~valid).to(tl.int32)) != 0)
        tl.store(OUT + ids, logical, ids < TOTAL)
        if VOTE_SIZE > 0:
            vote_ids = tl.minimum(tl.maximum(logical, 0), VOTE_SIZE - 1)
            tl.atomic_add(
                VOTES + vote_ids,
                1,
                (ids < TOTAL) & (ids >= (ROWS - RECENT) * TOPK),
                sem="relaxed",
            )


# Copyright (c) 2025 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


@triton.jit
def compact_groups(BUDGET, IDS, INV, N, PAID: tl.constexpr, B: tl.constexpr):
    g = tl.arange(0, B)
    ok = g < N
    b0 = tl.load(BUDGET + 2 * g, ok, 0)
    b1 = tl.load(BUDGET + 2 * g + 1, ok, 0)
    active = ok & ((b0 == PAID) | (b1 == PAID))
    pos = tl.cumsum(active.to(tl.int32), 0) - 1
    tl.store(IDS + pos, g, active)
    tl.store(INV + g, tl.where(active, pos, -1), ok)
    return tl.sum(active.to(tl.int32), 0)


@triton.jit
def choose_parts(count, tiles, sms: tl.constexpr, grid, EXACT: tl.constexpr):
    """Minimize full waves times tiles per part, then break ties by part count."""
    groups = tl.maximum(count, 1)
    t = tl.maximum(tiles, 1)
    cap = tl.maximum(1, tl.minimum(t, grid // groups))
    if EXACT:
        # ceil(t/p) changes at these breakpoints for t <= 8192.
        j = tl.arange(0, 256)
        d = j % 128 + 1
        p = tl.where(j < 128, d, (t - 1) // d + 1)
    else:
        wave = tl.cdiv(groups, sms) + tl.arange(0, 8)
        p = tl.maximum(1, tl.minimum(cap, (wave * sms) // groups))
    legal = p <= cap
    step = tl.minimum(p, cap)
    cost = tl.cdiv(groups * step, sms) * tl.cdiv(t, step)
    best = tl.min(tl.where(legal, cost, 2147483647), 0)
    winner = tl.min(tl.where(legal & (cost == best), p, 2147483647), 0)
    if EXACT:
        return tl.where((count > 0) & (tiles > 0), winner, 1)
    return winner


@triton.jit(do_not_specialize=["N", "GRID"])
def prepare_live(
    BUDGET,
    LIVE_NK,
    IDS,
    INV,
    CONFIG,
    N,
    GRID,
    SMS: tl.constexpr,
    MAX_K: tl.constexpr,
    SEED: tl.constexpr,
    TILE: tl.constexpr,
    PAID: tl.constexpr,
    B: tl.constexpr,
    WHOLE_WAVES: tl.constexpr = False,
    OFFSET=None,
):
    tl.static_assert(SMS > 0)
    tl.static_assert((PAID == 14 and TILE == 256) or (PAID == 16 and TILE == 128))
    tl.static_assert(SEED >= 0)
    tl.static_assert(SEED <= MAX_K)
    tl.static_assert(MAX_K <= 1048576)
    tl.static_assert(SEED % 256 == 0)
    tl.static_assert((MAX_K - SEED + TILE - 1) // TILE <= 8192)
    count = compact_groups(BUDGET, IDS, INV, N, PAID, B)
    nk = tl.load(LIVE_NK)
    valid = (nk >= SEED) & (nk <= MAX_K)
    # Clip before subtraction to avoid integer overflow. Invalid input still
    # publishes count=-1; clipping does not admit an invalid logical extent.
    safe_nk = tl.minimum(tl.maximum(nk, SEED), MAX_K)
    tiles = (safe_nk - SEED + TILE - 1) // TILE
    chosen = tl.full((), 1, tl.int32)
    if valid:
        # Whole resident waves need only one K span per query group. Divide
        # just the remaining groups using the same tile/wave objective.
        bulk = count // SMS * SMS if WHOLE_WAVES else 0
        chosen = choose_parts(count - bulk, tiles, SMS, GRID - bulk, PAID == 14)
    tl.store(CONFIG, tl.where(valid, count, -1))
    tl.store(CONFIG + 1, chosen)
    if OFFSET is not None:
        tl.store(CONFIG + 2, tl.load(OFFSET) * (256 // TILE))
