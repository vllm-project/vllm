# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata and schedules for the exact DoSA prefill operator."""

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as ld


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
    SAFE,
    GROUP: tl.constexpr,
    DOT: tl.constexpr,
    G: tl.constexpr,
    BF16_FACTOR: tl.constexpr,
    HEADS: tl.constexpr,
    HIGH: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.load(BUDGET + row // GROUP)
    valid_family = (b == 14) | (b == 16) | (b == HIGH)
    hp = tl.where(b == 14, H14, tl.where(b == 16, H16, H20))
    hint = tl.load(hp + row)
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
    native_pre = ld.mul_ru(ld.mul_ru(d, sw), tl.full((), BF16_FACTOR, tl.float64))
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
    tl.store(SAFE + row, tl.where(bad | (tl.load(ERR + row) != 0), -float("inf"), tau))


@triton.jit
def check_live_extent(
    ERROR,
    LIVE,
    Q: tl.constexpr,
    CAP: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    nk = tl.load(LIVE)
    tl.atomic_or(ERROR + row, 1, (row < Q) & ((nk < SEED) | (nk > CAP)))


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
            # Repeated UE8M0 scale: raw/MX commutation requires e in [-108,100].
            ok = (
                (bits >= 0x09800000)
                & (bits <= 0x71800000)
                & ((bits & 0x007FFFFF) == 0)
                & (rho >= 0)
                & (rho < 127)
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
    row = (anchor * (NQ - 1) // 3).to(tl.int64)
    i = tl.arange(0, HEADS * 128)
    tl.store(OQ + anchor * HEADS * 128 + i, tl.load(Q + row * HEADS * 128 + i))
    weight = tl.load(W + row * HEADS + i, i < HEADS, 0.0)
    tl.store(OW + anchor * HEADS + i, weight, i < HEADS)


@triton.jit(do_not_specialize=["N"])
def decode_ue8m0(SOURCE, OUTPUT, N, BLOCK: tl.constexpr):
    """Decode the repeated UE8M0 byte of each FP8 index-K scale losslessly."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    exponent = tl.load(SOURCE + i * 4, i < N, 127).to(tl.int32)
    bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    tl.store(OUTPUT + i, bits.to(tl.float32, bitcast=True), i < N)


@triton.jit(do_not_specialize=["KEYS"])
def recovery_pack_scales(SCALE, PACKED, KEYS, BLOCK: tl.constexpr):
    key = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(SCALE + key, key < KEYS, 0.0).to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    # Code0 decodes to F32 bits0x00400000; unused zero padding also has exponent0.
    # Valid intervals exclude the owner's zero K/scale padding.
    packed = exponent * tl.full((), 0x01010101, tl.uint32)
    tl.store(PACKED + key, packed, key < KEYS)


@triton.jit(do_not_specialize=["ROWS", "CAPACITY"])
def recovery_intervals(
    START, END, ERROR, OUT_START, OUT_END, LENGTH, LIVE, ROWS, CAPACITY
):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    active = (row < ROWS) & (tl.load(ERROR + row, row < ROWS, 0) != 0)
    begin = tl.load(START + row, active, 0)
    end = tl.load(END + row, active, 0)
    logical = tl.load(LIVE)
    valid = (logical >= 0) & (logical <= CAPACITY) & (begin >= 0)
    valid &= (begin <= end) & (end <= logical)
    tl.atomic_or(ERROR + row, 2, active & ~valid)
    begin = tl.where(active & valid, begin, 0)
    end = tl.where(active & valid, end, 0)
    tl.store(OUT_START + row, begin, row < ROWS)
    tl.store(OUT_END + row, end, row < ROWS)
    tl.store(LENGTH + row, end - begin, row < ROWS)


@triton.jit
def recovery_publish(
    SELECTED, ERROR, ORDER, ORIGIN, OUTPUT, TOPK: tl.constexpr, STRIDE: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.program_id(1) * 256 + tl.arange(0, 256)
    failed = tl.load(ERROR + row) != 0
    destination = tl.load(ORDER + row)
    selected = tl.load(SELECTED + row * TOPK + column, failed, -1)
    begin = tl.load(ORIGIN + row, failed, 0)
    selected = tl.where(selected >= 0, selected + begin, -1)
    tl.store(OUTPUT + destination * STRIDE + column, selected, failed)


@triton.jit(do_not_specialize=["ROWS", "CAPACITY"])
def recovery_classify_errors(SAFE, ERROR, START, END, LIVE, FAST, SLOW, ROWS, CAPACITY):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    active = row < ROWS
    error = tl.load(ERROR + row, active, 0)
    safe = tl.load(SAFE + row, active, -float("inf"))
    begin = tl.load(START + row, active, 0)
    end = tl.load(END + row, active, 0)
    logical = tl.load(LIVE)
    valid = (logical >= 0) & (logical <= CAPACITY) & (begin >= 0)
    valid &= (begin <= end) & (end <= logical) & (end - begin >= 2048)
    finite = (safe.to(tl.uint32, bitcast=True) & 0x7F800000) != 0x7F800000
    fast = finite & ((error & (16 | 2048)) != 0) & ((error & ~(17 | 2048)) == 0) & valid
    tl.store(FAST + row, tl.where(fast, error, 0), active)
    tl.store(SLOW + row, tl.where(fast, 0, error), active)


@triton.jit(do_not_specialize=["ROWS"])
def recovery_merge_errors(ERROR, FAST, SLOW, ROWS):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    active = row < ROWS
    error = tl.load(ERROR + row, active, 0)
    merged = error | tl.load(FAST + row, active, 0) | tl.load(SLOW + row, active, 0)
    tl.store(ERROR + row, merged, active)


@triton.jit(do_not_specialize=["ROWS"])
def recovery_row_map(ERROR, END, MAP, STATE, LENGTH, ROWS, BLOCK: tl.constexpr):
    """Compact current classifier-admitted rows; lengths retain original row order."""
    row = tl.arange(0, BLOCK)
    failed = (row < ROWS) & (tl.load(ERROR + row, row < ROWS, 0) != 0)
    slot = tl.cumsum(failed.to(tl.int32)) - 1
    tl.store(MAP + slot, row, failed)
    tl.store(LENGTH + row, tl.load(END + row, failed, 0), row < ROWS)
    count = tl.sum(failed.to(tl.int32))
    tl.store(STATE, count)
    tl.store(STATE + 1, (count > 0).to(tl.int32))


@triton.jit
def recovery_gather_rows(Q, W, MAP, STATE, PACKED_Q, PACKED_W, HEADS: tl.constexpr):
    """Write every packed slot, including Q4 padding, before native raw scoring."""
    slot = tl.program_id(0)
    active = slot < tl.load(STATE)
    row = tl.load(MAP + slot, active, 0)
    column = tl.arange(0, HEADS * 128)
    q = tl.load(Q + row.to(tl.int64) * HEADS * 128 + column, active, 0)
    tl.store(PACKED_Q + slot.to(tl.int64) * HEADS * 128 + column, q)
    head = tl.arange(0, HEADS)
    w = tl.load(W + row * HEADS + head, active, 0.0)
    tl.store(PACKED_W + slot * HEADS + head, w)


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


@triton.jit(do_not_specialize=["N", "FIRST", "MAX"])
def _carry_histogram(
    V,
    P,
    N,
    FIRST,
    MAX,
    STRIDE: tl.constexpr,
    BINS: tl.constexpr,
):
    """Publish I16 cumulative counts for each original 8192-ID block."""
    ids = tl.program_id(0) * 8192 + tl.arange(0, 8192)
    live = (ids < N) & (ids >= FIRST)
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
    boundary = live & (needed > before) & (needed <= before + eq)
    tie_block = tl.min(tl.where(boundary, blocks, BLOCKS - 1))
    tie_take = tl.sum(tl.where(boundary, needed - before, 0))
    tl.store(S + 6 + blocks, offsets, live)
    tl.store(S + 1, lo)
    tl.store(S + 2, tie_block)
    tl.store(S + 3, tie_take)


@triton.jit(do_not_specialize=["N", "FIRST", "MAX"])
def _carry_emit(V, HOT, S, N, FIRST, MAX):
    """Emit globally ascending HOT IDs; reset all N votes, including excluded IDs."""
    block = tl.program_id(0)
    ids = block * 8192 + tl.arange(0, 8192)
    live = ids < N
    votes = tl.minimum(tl.maximum(tl.load(V + ids, live, 0), 0), MAX)
    threshold, tie_block, tie_take = tl.load(S + 1), tl.load(S + 2), tl.load(S + 3)
    eq = live & (ids >= FIRST) & (votes == threshold)
    eq_rank = tl.cumsum(eq.to(tl.int32)) - 1
    quota = tl.where(block < tie_block, 8192, tl.where(block == tie_block, tie_take, 0))
    selected = live & (ids >= FIRST) & ((votes > threshold) | (eq & (eq_rank < quota)))
    rank = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(HOT + tl.load(S + 6 + block) + rank, ids.to(tl.int64), selected)
    tl.store(V + ids, 0, live)


@triton.jit(do_not_specialize=["END", "epoch"])
def _mark_hot(HOT, STATE, SIZE: tl.constexpr, END, epoch, TILE: tl.constexpr):
    """Mark HOT uniquely and preserve sticky metadata errors."""
    EPOCH, _, _, _, _, C = STATE
    ids = tl.program_id(0) * TILE + tl.arange(0, TILE)
    hot = tl.load(HOT + ids, ids < SIZE, 0)
    valid = (ids < SIZE) & (hot >= 0) & (hot < END)
    previous = tl.atomic_xchg(EPOCH + hot, epoch, valid, sem="relaxed")
    duplicate = tl.where(valid, previous == epoch, False)
    error = tl.where(duplicate, 1, 0)
    error |= tl.where((ids < SIZE) & ~valid, 2, 0)
    tl.atomic_or(C + 2, tl.max(error), sem="relaxed")
    if tl.program_id(0) == 0:
        tl.store(C, 0)
        tl.store(C + 1, 0)


@triton.jit(do_not_specialize=["END", "epoch"])
def _compact_pairs(HOT, STATE, SIZE: tl.constexpr, END, epoch, TILE: tl.constexpr):
    """Reserve endpoint ranks in scratch disjoint from the final permutation."""
    EPOCH, RANK, _, A, B, C = STATE
    ids = tl.program_id(0) * TILE + tl.arange(0, TILE)
    hot = tl.load(HOT + ids, ids < SIZE, 0)
    take_a = (ids < SIZE) & (hot >= SIZE) & (hot < END)
    take_b = (ids < SIZE) & (tl.load(EPOCH + ids, ids < SIZE, 0) != epoch)
    rank_a, rank_b = tl.cumsum(take_a.to(tl.int32)), tl.cumsum(take_b.to(tl.int32))
    base_a = tl.atomic_add(C, tl.sum(take_a.to(tl.int32)), sem="relaxed")
    base_b = tl.atomic_add(C + 1, tl.sum(take_b.to(tl.int32)), sem="relaxed")
    rank_a, rank_b = base_a + rank_a - 1, base_b + rank_b - 1
    tl.store(A + rank_a, hot, take_a)
    tl.store(B + rank_b, ids, take_b)
    tl.store(RANK + hot, rank_a, take_a)
    tl.store(RANK + ids, rank_b, take_b)


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


@triton.jit(do_not_specialize=["N", "epoch"])
def _publish_gather(
    K, SCALE, DK, DS, RHO, STATE, N, SIZE: tl.constexpr, epoch, TILE_ROWS: tl.constexpr
):
    """Publish pair swaps while gathering raw K/scales and integer-certified rho."""
    EPOCH, RANK, PERM, A, B, C = STATE
    count_a, count_b, status = tl.load(C), tl.load(C + 1), tl.load(C + 2)
    if tl.program_id(0) == 0:
        tl.atomic_or(C + 2, tl.where(count_a != count_b, 8, 0), sem="relaxed")
    _trap_if((status != 0) | (count_a != count_b))
    rows = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    live = rows < N
    marked = tl.load(EPOCH + rows, live, 0) == epoch
    take_a = live & (rows >= SIZE) & marked
    take_b = live & (rows < SIZE) & ~marked
    endpoint = take_a | take_b
    stored = tl.load(RANK + rows, endpoint, 0)
    rank_ok = (stored >= 0) & (stored < count_a)
    _trap_if(tl.sum((endpoint & ~rank_ok).to(tl.int32)) != 0)
    opposite = tl.load(tl.where(take_a, B, A) + stored, endpoint & rank_ok, 0)
    source = tl.where(endpoint, opposite, rows)
    source_ok = (source >= 0) & (source < N)
    _trap_if(tl.sum((live & ~source_ok).to(tl.int32)) != 0)
    tl.store(PERM + rows, source, live)
    dims = tl.arange(0, 128)
    gather_live = live & source_ok
    raw = tl.load(K + source[:, None] * 128 + dims[None, :], gather_live[:, None], 0)
    tl.store(DK + rows[:, None] * 128 + dims[None, :], raw, live[:, None])
    scale = tl.load(SCALE + source, gather_live, 0)
    tl.store(DS + rows, scale, live)
    bits = scale.to(tl.uint32, bitcast=True)
    codes = (raw & 127).to(tl.int32)
    units = _magnitude_unit(codes).to(tl.int64)
    units = tl.where(codes == 127, 0, units)
    square, kappa = tl.sum(units * units, 1), tl.max(codes, 1)
    bad = ((bits - 1).to(tl.uint32) >= 0x5F800000) | (kappa == 127)
    rho = _exact_rho(square, kappa, live & ~bad)
    tl.store(RHO + rows, tl.where(bad, 127, rho), live)


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
def exact_parts(count, tiles, sms: tl.constexpr, grid):
    # The callers bound tiles <= 8192 before using the exact256 breakpoints.
    groups = tl.maximum(count, 1)
    t = tl.maximum(tiles, 1)
    cap = tl.maximum(1, tl.minimum(t, grid // groups))
    chosen = tl.full((), 1, tl.int32)
    if (count > 0) & (tiles > 0):
        j = tl.arange(0, 256)
        d = j % 128 + 1
        p = tl.where(j < 128, d, (t - 1) // d + 1)
        valid = p <= cap
        pe = tl.minimum(p, cap)
        cost = ((groups * pe + sms - 1) // sms) * ((t + pe - 1) // pe)
        best = tl.min(tl.where(valid, cost, 2147483647), 0)
        chosen = tl.min(tl.where(valid & (cost == best), p, 2147483647), 0)
    return chosen


@triton.jit
def eight_wave_parts(count, tiles, sms: tl.constexpr, grid):
    groups = tl.maximum(count, 1)
    minwaves = (groups + sms - 1) // sms
    maxparts = tl.maximum(1, tl.minimum(tl.maximum(tiles, 1), grid // groups))
    chosen = tl.full((), 1, tl.int32)
    best = tl.full((), 2147483647, tl.int32)
    for delta in tl.static_range(8):
        wave = minwaves + delta
        p = tl.maximum(1, tl.minimum(maxparts, (wave * sms) // groups))
        cost = ((groups * p + sms - 1) // sms) * ((tiles + p - 1) // p)
        better = (cost < best) | ((cost == best) & (p < chosen))
        chosen = tl.where(better, p, chosen)
        best = tl.minimum(best, cost)
    return chosen


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
        if PAID == 14:
            chosen = exact_parts(count, tiles, SMS, GRID)
        else:
            chosen = eight_wave_parts(count, tiles, SMS, GRID)
    tl.store(CONFIG, tl.where(valid, count, -1))
    tl.store(CONFIG + 1, chosen)
