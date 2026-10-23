# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata and schedules for the exact DoSA prefill operator."""

import torch

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
    SEED_DIAG: tl.constexpr,
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
        | (tl.load(SB + row * 4 + 3) == 3 if SEED_DIAG else tl.load(SB + row) != 0)
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
def clear_live_table(
    TABLE,
    ERROR,
    STATUS,
    LIVE,
    Q: tl.constexpr,
    STRIDE: tl.constexpr,
    CAP: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    nk = tl.load(LIVE)
    valid = (nk >= SEED) & (nk <= CAP)
    if valid:
        columns = (nk - SEED + 127) // 128
        for begin in range(0, columns, BLOCK):
            c = begin + tl.arange(0, BLOCK)
            tl.store(TABLE + row * STRIDE + c, -1, c < columns)
    else:
        if row < Q:
            tl.atomic_or(ERROR + row, 1)
        if row == 0:
            tl.atomic_or(STATUS, 2)


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


@triton.jit
def _upcast_logits_kernel(src_ptr, dst_ptr, row_ends_ptr, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    end = tl.load(row_ends_ptr + row)
    src_ptr += row.to(tl.int64) * stride
    dst_ptr += row.to(tl.int64) * stride
    for start in range(tl.program_id(1) * BLOCK, end, tl.num_programs(1) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < end
        vals = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, vals.to(tl.float32), mask=mask)


def as_fp32_logits(logits: torch.Tensor, row_ends: torch.Tensor) -> torch.Tensor:
    """Upcast SM100's BF16 MQA logits for the FP32 top-k kernels.

    Only columns below ``row_ends`` (int32, one per row) are converted, the
    span the consumers read; DeepGEMM's aligned row stride is kept.
    """
    if logits.dtype == torch.float32:
        return logits
    out = torch.empty_strided(
        logits.shape, logits.stride(), dtype=torch.float32, device=logits.device
    )
    num_rows, width = logits.shape
    if num_rows == 0 or width == 0:
        return out
    assert row_ends.numel() == num_rows and row_ends.dtype == torch.int32
    block = 1024
    splits = min(triton.cdiv(width, block), max(1, 1024 // num_rows))
    _upcast_logits_kernel[(num_rows, splits)](
        logits, out, row_ends.contiguous(), logits.stride(0), BLOCK=block
    )
    return out


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
    SELECTED, ERROR, ORDER, START, OUTPUT, TOPK: tl.constexpr, STRIDE: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.program_id(1) * 256 + tl.arange(0, 256)
    failed = tl.load(ERROR + row) != 0
    destination = tl.load(ORDER + row)
    selected = tl.load(SELECTED + row * TOPK + column, failed, -1)
    begin = tl.load(START + row, failed, 0)
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
    fast = finite & ((error & 16) != 0) & ((error & ~17) == 0) & valid
    tl.store(FAST + row, tl.where(fast, error, 0), active)
    tl.store(SLOW + row, tl.where(fast, 0, error), active)


@triton.jit(do_not_specialize=["ROWS"])
def recovery_merge_errors(ERROR, FAST, SLOW, ROWS):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    active = row < ROWS
    error = tl.load(ERROR + row, active, 0)
    merged = error | tl.load(FAST + row, active, 0) | tl.load(SLOW + row, active, 0)
    tl.store(ERROR + row, merged, active)
