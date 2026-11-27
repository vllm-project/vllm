# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Maintain the request key bank, seed and exact sideband partitions."""

import torch

from vllm.model_executor.layers.dosa import SEED
from vllm.model_executor.layers.dosa_ops import _gather_key_rows
from vllm.triton_utils import tl, triton


@triton.jit
def _sample_pairs(
    SCALE,
    RHO,
    COMMON,
    S0,
    R0,
    S1,
    R1,
    SAMPLES: tl.constexpr,
    SEED: tl.constexpr = 12288,
):
    common = tl.load(COMMON)
    n = common - SEED
    i = tl.arange(0, SAMPLES)
    key = SEED + (2 * i + 1) * n // (2 * SAMPLES)
    scale = tl.load(SCALE + key, n > 0, 1.0)
    rho = tl.load(RHO + key, n > 0, -1)
    bits = scale.to(tl.uint32, bitcast=True)
    valid = (n > 0) & (scale > 0) & (scale < float("inf")) & (rho >= 0) & (rho < 127)
    match = (
        (bits[:, None] == bits[None, :])
        & (rho[:, None] == rho[None, :])
        & valid[:, None]
        & valid[None, :]
    )
    counts = tl.sum(match.to(tl.int32), 1)
    for band in tl.static_range(2):
        winner = tl.argmax(counts, 0, tie_break_left=True)
        position = SEED + (2 * winner + 1) * n // (2 * SAMPLES)
        picked_scale = tl.load(SCALE + position, n > 0, 1.0)
        picked_rho = tl.load(RHO + position, n > 0, -1)
        exists = tl.max(counts, 0) > 0
        tl.store(S0 if band == 0 else S1, tl.where(exists, picked_scale, 1.0))
        tl.store(R0 if band == 0 else R1, tl.where(exists, picked_rho, -1))
        same = (bits == picked_scale.to(tl.uint32, bitcast=True)) & (rho == picked_rho)
        counts = tl.where(same, 0, counts)


def _key_bank(owner):
    packed = owner.fp32.packed_scales if owner.fp32 is not None else None
    return owner.pk.view(torch.uint8), owner.ps, owner.rho, owner.permutation, packed


def partition(owner):
    """Choose the two hot bands and place their keys with the repair pass."""
    _sample_pairs[(1,)](
        owner.ps,
        owner.rho,
        owner._common_end,
        owner._pair_scale,
        owner._pair_rho,
        owner._second_scale,
        owner._second_rho,
        128,
        num_warps=4,
    )
    repair_bands(owner)


@triton.jit
def swap_pairs(
    K,
    SCALE,
    RHO,
    PERM,
    PACKED,
    INVERSE,
    A,
    B,
    COUNT,
    MAX_PAIRS: tl.constexpr,
    K_STRIDE: tl.constexpr = 128,
    PAIRS: tl.constexpr = 4,
):
    """Swap disjoint physical endpoints, preserving bytes, metadata and both maps.

    A/B are I32[MAX_PAIRS], COUNT is I32[1]. Every endpoint in the live prefix
    must be valid, and all2*COUNT endpoints must be distinct. Other than this
    disjointness, endpoints may belong to any bank regions.
    """
    count = tl.load(COUNT)
    first = tl.program_id(0) * PAIRS
    while first < count:
        pair = first + tl.arange(0, PAIRS)
        active = (pair < count) & (pair < MAX_PAIRS)
        a = tl.load(A + pair, active, 0)
        b = tl.load(B + pair, active, 0)
        d = tl.arange(0, 128)
        ka = tl.load(K + a[:, None] * K_STRIDE + d[None, :], active[:, None], 0)
        kb = tl.load(K + b[:, None] * K_STRIDE + d[None, :], active[:, None], 0)
        sa = tl.load(SCALE + a, active, 0)
        sb = tl.load(SCALE + b, active, 0)
        ra = tl.load(RHO + a, active, 0)
        rb = tl.load(RHO + b, active, 0)
        la = tl.load(PERM + a, active, 0)
        lb = tl.load(PERM + b, active, 0)
        # All old payloads are loaded before either endpoint is overwritten.
        tl.store(K + a[:, None] * K_STRIDE + d[None, :], kb, active[:, None])
        tl.store(K + b[:, None] * K_STRIDE + d[None, :], ka, active[:, None])
        tl.store(SCALE + a, sb, active)
        tl.store(SCALE + b, sa, active)
        tl.store(RHO + a, rb, active)
        tl.store(RHO + b, ra, active)
        tl.store(PERM + a, lb, active)
        tl.store(PERM + b, la, active)
        tl.store(INVERSE + la, b, active)
        tl.store(INVERSE + lb, a, active)
        if PACKED is not None:
            pa = tl.load(PACKED + a, active, 0)
            pb = tl.load(PACKED + b, active, 0)
            tl.store(PACKED + a, pb, active)
            tl.store(PACKED + b, pa, active)
        first += tl.num_programs(0) * PAIRS


def _swap_bank_pairs(owner, left, right, count, limit):
    swap_pairs[(2 * owner.sms,)](
        *_key_bank(owner),
        owner.pair_rank,
        left,
        right,
        count,
        MAX_PAIRS=limit,
        num_warps=4,
    )


@triton.jit
def _append(INPUT, OUTPUT, INV, BEGIN, END, TILE: tl.constexpr):
    keys, ks = INPUT
    dst, scale_dst, rho_dst, perm, packed = OUTPUT
    row = BEGIN + tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = row < END
    _gather_key_rows(
        keys, ks, dst, scale_dst, rho_dst, packed, row, row, valid, valid, TILE
    )
    tl.store(perm + row, row, valid)
    tl.store(INV + row, row, valid)


@triton.jit(do_not_specialize=["epoch"])
def _mark_seed(HOT, EPOCH, epoch, B: tl.constexpr):
    ids = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(EPOCH + tl.load(HOT + ids), epoch)


def append_and_seed(owner, keys, scales, count, previous):
    _append[(triton.cdiv(count - previous, 32),)](
        (keys.view(torch.uint8), scales),
        _key_bank(owner),
        owner.pair_rank,
        previous,
        count,
        32,
        num_warps=8,
    )
    _mark_seed[(SEED // 256,)](owner.hot, owner.hot_epoch, owner.plan_epoch, 256)
    seed = owner.hot, owner.hot_epoch, owner.permutation, owner.pair_rank
    _exchange(owner, seed, SEED, 256, SEED, epoch=owner.plan_epoch)


@triton.jit
def _match_band(
    SCALE,
    RHO,
    PICKED_SCALE,
    PICKED_RHO,
    COMMON,
    FIRST_END,
    FLAGS,
    COUNTS,
    SECOND: tl.constexpr,
    N: tl.constexpr,
    SEED: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0)
    i = block * B + tl.arange(0, B)
    row = SEED + i
    common = tl.load(COMMON)
    first = tl.load(FIRST_END) if SECOND else SEED
    valid = (i < N) & (row < common) & (row >= first)
    scale = tl.load(SCALE + row, valid, 0.0)
    rho = tl.load(RHO + row, valid, 0)
    code = tl.load(PICKED_RHO)
    match = (
        valid
        & (code >= 0)
        & (rho == code)
        & (
            scale.to(tl.int32, bitcast=True)
            == tl.load(PICKED_SCALE).to(tl.int32, bitcast=True)
        )
    )
    tl.store(FLAGS + i, match, i < N)
    tl.store(COUNTS + block, tl.sum(match.to(tl.int64), 0))


@triton.jit
def _seed_endpoints(DOMAIN, i, epoch, N: tl.constexpr):
    hot, epochs, permutation, inverse = DOMAIN
    wanted = tl.load(hot + i, i < N, 0)
    source = tl.load(inverse + wanted, i < N, 0)
    old = tl.load(permutation + i, i < N, 0)
    incoming = (i < N) & (source >= N)
    vacated = (i < N) & (tl.load(epochs + old, i < N, epoch) != epoch)
    return source, i, incoming, vacated


@triton.jit
def _band_endpoints(
    DOMAIN, SETUP, i, N: tl.constexpr, Blocks: tl.constexpr, Seed: tl.constexpr
):
    flags, common, first_end, end_ptr = DOMAIN
    first = Seed if first_end is None else tl.load(first_end)
    if SETUP is None:
        end = tl.load(end_ptr)
    else:
        match_prefix, live, band_live, band_config, cold_live, cold_config = SETUP
        count = tl.load(match_prefix + Blocks - 1) // 256 * 256
        end = first + count
        if tl.program_id(0) == 0:
            tl.store(end_ptr, end)
            tl.store(cold_live, tl.load(live) + Seed - end)
            tl.store(cold_config + 2, (end - Seed) // 256)
            if first_end is not None:
                tl.store(band_live, Seed + count)
                tl.store(band_config + 2, (first - Seed) // 256)
    row = Seed + i
    valid = (i < N) & (row < tl.load(common))
    match = tl.load(flags + i, i < N, False)
    fill = valid & (row >= first) & (row < end) & ~match
    donor = valid & (row >= end) & match
    return row, row, fill, donor


@triton.jit(do_not_specialize=["epoch"])
def _plan_swaps(
    DOMAIN,
    SETUP,
    epoch,
    COUNTS,
    PREFIX,
    PAIRS,
    N: tl.constexpr,
    B: tl.constexpr,
    BLOCKS: tl.constexpr,
    SEED: tl.constexpr,
    IS_SEED: tl.constexpr,
):
    block = tl.program_id(0)
    left_out, right_out, count = PAIRS
    if PREFIX is not None:
        if block == 0:
            tl.store(count, (tl.load(PREFIX + BLOCKS - 1) & 0xFFFFFFFF).to(tl.int32))
        before = tl.load(PREFIX + block - 1, block > 0, 0)
        if tl.load(PREFIX + block) == before:
            return
    i = block * B + tl.arange(0, B)
    if IS_SEED:
        left, right, take_left, take_right = _seed_endpoints(DOMAIN, i, epoch, N)
    else:
        left, right, take_left, take_right = _band_endpoints(
            DOMAIN, SETUP, i, N, BLOCKS, SEED
        )
    packed = take_left.to(tl.int64) | (take_right.to(tl.int64) << 32)
    if PREFIX is None:
        tl.store(COUNTS + block, tl.sum(packed, 0))
    else:
        rank = before + tl.cumsum(packed, 0)
        tl.store(left_out + (rank & 0xFFFFFFFF).to(tl.int32) - 1, left, take_left)
        tl.store(right_out + (rank >> 32).to(tl.int32) - 1, right, take_right)


def _exchange(owner, domain, n, tile, limit, *, setup=None, epoch=None):
    """Pair endpoints in each side's enumeration order using shared scratch."""
    blocks = triton.cdiv(n, tile)
    if epoch is None:
        counts, prefix = owner._pair_bits[:blocks], owner._pair_scan[:blocks]
        pairs = owner._fill_slots, owner._donors, owner._num_repairs
    else:
        counts, prefix = owner._seed_scan
        pairs = owner._seed_swaps
    args = pairs, n, tile, blocks, SEED, epoch is not None
    _plan_swaps[(blocks,)](domain, setup, epoch, counts, None, *args, num_warps=4)
    torch.cumsum(counts, 0, out=prefix)
    _plan_swaps[(blocks,)](domain, None, epoch, counts, prefix, *args, num_warps=4)
    _swap_bank_pairs(owner, *pairs, limit)


def repair_bands(owner):
    """Restore the two hot bands in order, reusing the dead candidate arena."""
    seed = SEED
    n = owner.max_k - seed
    blocks = triton.cdiv(n, 512)
    counts = owner._pair_bits[:blocks]
    prefix = owner._pair_scan[:blocks]
    for second in (False, True):
        scale = owner._second_scale if second else owner._pair_scale
        rho = owner._second_rho if second else owner._pair_rho
        end = owner._second_end if second else owner._hot_end
        _match_band[(blocks,)](
            owner.ps,
            owner.rho,
            scale,
            rho,
            owner._common_end,
            owner._hot_end,
            owner._flags,
            counts,
            second,
            n,
            seed,
            512,
            num_warps=4,
        )
        torch.cumsum(counts, 0, out=prefix)
        first = owner._hot_end if second else None
        domain = owner._flags, owner._common_end, first, end
        setup = (
            prefix,
            owner.live_nk,
            owner._second_live,
            owner._second_config,
            owner._cold_live,
            owner._cold_config,
        )
        _exchange(owner, domain, n, 512, owner.max_k, setup=setup)


def allocate_layout(owner):
    k = owner.max_k
    arena = owner.id_pool.view(torch.uint8).reshape(-1)
    blocks = triton.cdiv(k - SEED, 512)
    # Byte flags, two i64 scan planes and two i32 endpoint lists.
    if arena.numel() < 9 * k + 16 * blocks:
        return False
    owner._flags = arena[:k].view(torch.bool)
    offset = k

    def scratch(elements, dtype):
        nonlocal offset
        width = dtype.itemsize
        offset = triton.cdiv(offset, width) * width
        end = offset + width * elements
        result = arena[offset:end].view(dtype)
        offset = end
        return result

    owner._pair_bits = scratch(blocks, torch.int64)
    owner._pair_scan = scratch(blocks, torch.int64)
    owner._fill_slots = scratch(k, torch.int32)
    owner._donors = scratch(k, torch.int32)
    owner._num_repairs = owner._empty((1,), torch.int32)
    owner._pair_scale = owner._empty((1,)).fill_(1)
    owner._second_scale = owner._empty((1,)).fill_(1)
    owner._pair_rho = owner._empty((1,), torch.int32).fill_(-1)
    owner._second_rho = owner._empty((1,), torch.int32).fill_(-1)
    owner._hot_end = owner._empty((1,), torch.int32).fill_(SEED)
    owner._second_end = owner._empty((1,), torch.int32).fill_(SEED)
    owner._cold_live = owner._empty((1,), torch.int32).fill_(SEED)
    owner._second_live = owner._empty((1,), torch.int32).fill_(SEED)
    owner._common_end = owner._empty((1,), torch.int32).fill_(SEED)
    owner._second_config = owner._empty((3,), torch.int32).zero_()
    owner._cold_config = owner._empty((3,), torch.int32).zero_()
    if owner.heads == 32 and owner.fp32 is not None:
        owner._second_h18_config = owner._empty((3,), torch.int32).zero_()
        owner._cold_h18_config = owner._empty((3,), torch.int32).zero_()
        owner._h18_choice = owner._empty((1,), torch.int32).zero_()
    if owner.heads == 32:
        owner._sampled_ids = owner._empty((1,), torch.int32)
    owner._first_slab = owner._empty((1,), torch.int32).zero_()
    owner._layout_active = owner._empty((1,), torch.int32).zero_()
    owner._first_reorder = owner._empty((1,), torch.int32).zero_()
    if owner.heads == 32:
        owner._cutoff = owner._empty((owner.rows,))
        owner._second_cutoff = owner._empty((owner.rows,))
    owner._high_cutoff = owner._empty((owner.rows,))
    owner._high_cutoff_second = owner._empty((owner.rows,))
    return True


@triton.jit
def reset_ranges(
    COMMON,
    FIRST,
    LAYOUT,
    HOT_END,
    SECOND_END,
    SECOND_LIVE,
    COLD_LIVE,
    SECOND_CONFIG,
    COLD_CONFIG,
    END,
    LIVE,
    FIRST_CALL: tl.constexpr,
    SEED: tl.constexpr,
):
    tl.store(COMMON, END)
    tl.store(FIRST, 1 if FIRST_CALL else 0)
    if FIRST_CALL:
        tl.store(LAYOUT, 0)
    if FIRST_CALL or tl.load(LAYOUT) == 0:
        tl.store(HOT_END, SEED)
        tl.store(SECOND_END, SEED)
        tl.store(SECOND_LIVE, SEED)
        tl.store(COLD_LIVE, LIVE)
        tl.store(SECOND_CONFIG + 2, 0)
        tl.store(COLD_CONFIG + 2, 0)


@triton.jit
def decide_layout(
    IDS, LIVE, COMMON, LAYOUT, CAPACITY: tl.constexpr, SEED: tl.constexpr
):
    sample = tl.load(IDS).to(tl.int64)
    keys = tl.load(LIVE).to(tl.int64) - SEED
    fits = (sample * 8 * keys < CAPACITY * 1024 * 128) & (tl.load(COMMON) >= 524288)
    tl.store(LAYOUT, fits.to(tl.int32))


@triton.jit
def first_reorder(LAYOUT, FIRST, ACTIVATE):
    enabled = (tl.load(LAYOUT) != 0) & (tl.load(FIRST) != 0)
    tl.store(ACTIVATE, enabled.to(tl.int32))


@triton.jit
def choose_h18(BUDGET, LAYOUT, OUT, GROUPS, SMS: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(LAYOUT) == 0:
        tl.store(OUT, 0)
        return
    g = tl.arange(0, BLOCK)
    a = tl.load(BUDGET + 2 * g, g < GROUPS, 0)
    b = tl.load(BUDGET + 2 * g + 1, g < GROUPS, 0)
    count = tl.sum(((a == 16) | (b == 16)).to(tl.int32), 0)
    tl.store(OUT, (count >= SMS).to(tl.int32))


@triton.jit
def choose_high_layout(
    BUDGET, COMMON, OUT, GROUPS, SMS: tl.constexpr, BLOCK: tl.constexpr
):
    g = tl.arange(0, BLOCK)
    tags = tl.load(BUDGET + g, g < GROUPS, 0)
    count = tl.sum((tags == 28).to(tl.int32), 0)
    tl.store(OUT, ((tl.load(COMMON) >= 524288) & (count >= SMS)).to(tl.int32))


@triton.jit
def disable_on_pool_overflow(LAYOUT, HEAD, PAGES: tl.constexpr):
    if tl.load(HEAD) > PAGES:
        tl.store(LAYOUT, 0)


@triton.jit
def disable_on_pressure(LAYOUT, PRESSURE):
    if tl.load(PRESSURE) != 0:
        tl.store(LAYOUT, 0)


@triton.jit
def _ordinary(code):
    return tl.where(code < 0x80000000, ~code, code ^ 0x80000000).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _discard(core, rho, guard, raw, out, scale, tau, HIGH: tl.constexpr):
    if HIGH:
        value = tl.inline_asm_elementwise(
            """{
            .reg .f32 x,rs,tail,u;
            add.rn.f32 x,$1,$2;
            mul.rn.f32 rs,$3,$4;
            add.rn.f32 tail,x,rs;
            mul.rn.f32 u,$5,tail;
            add.rn.f32 $0,u,$6;
            }""",
            constraints="=f,f,f,f,f,f,f",
            args=[core, raw, rho, guard, scale, out],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        value = tl.inline_asm_elementwise(
            """{
            .reg .f32 first, second;
            add.rp.f32 first, $1, $2;
            fma.rp.f32 second, $3, $4, first;
            fma.rp.f32 $0, $5, second, $6;
            }""",
            constraints="=f,f,f,f,f,f,f",
            args=[core, raw, rho, guard, scale, out],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return value < tau


@triton.jit
def _prepare_cutoffs(
    META,
    RAW,
    OUT,
    TAU,
    HINT,
    BUDGET,
    SCALE0,
    RHO0,
    SCALE1,
    RHO1,
    FIRST_END,
    SECOND_END,
    CUT0,
    CUT1,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    PAID: tl.constexpr,
    B: tl.constexpr,
):
    elem = tl.program_id(0) * B + tl.arange(0, B)
    row = elem // 2
    pair = elem % 2
    valid = row < ROWS
    high: tl.constexpr = PAID >= 20
    if high:
        lower: tl.constexpr = 0x287FFFFF
        upper: tl.constexpr = 0xD7800000
        sentinel: tl.constexpr = -float(2**48)
        guard_slot: tl.constexpr = PAID + 2
    else:
        lower: tl.constexpr = 0x00800000
        upper: tl.constexpr = 0xFF7FFFFF
        sentinel: tl.constexpr = -float("inf")
        guard_slot: tl.constexpr = PAID + 2
    first = tl.load(FIRST_END)
    second = tl.load(SECOND_END)
    nonempty_band = tl.where(pair == 0, first > 12288, second > first)
    scale = tl.load(tl.where(pair == 0, SCALE0, SCALE1))
    code = tl.load(tl.where(pair == 0, RHO0, RHO1))
    pair_ok = (code >= 0) & (code < 127) & (scale > 0) & (scale < float("inf"))
    if high:
        pair_ok &= scale <= 2.0**64
    # Bitcast E4M3 code, then exact promotion; no host decode or temporary byte.
    rho = (
        tl.where(pair_ok, code, 0)
        .to(tl.uint8)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    budget = tl.load(BUDGET + row // 4, valid, 0)
    hint = tl.load(HINT + row, valid, 0)
    active = valid & nonempty_band & pair_ok & (budget == PAID) & (hint == 2)
    destination = tl.where(pair == 0, CUT0, CUT1) + row
    if tl.sum(active.to(tl.int32), 0) == 0:
        tl.store(destination, sentinel, valid)
        return
    guard = tl.load(META + row * STRIDE + guard_slot, active, 0.0)
    raw = tl.load(RAW + row, active, 0.0)
    out = tl.load(OUT + row, active, 0.0)
    tau = tl.load(TAU + row, active, -float("inf"))
    lo = tl.full((B,), lower, tl.uint32)
    hi = tl.full((B,), upper, tl.uint32)
    # The prepared F1 screen also discards a negative overflow when it is
    # below finite SAFE. Including it makes the lower search bound monotone.
    nonempty = _discard(_ordinary(lo), rho, guard, raw, out, scale, tau, high)
    for _ in range(32):
        mid = lo + (hi - lo + 1) // 2
        rejected = (mid == lower) | _discard(
            _ordinary(mid), rho, guard, raw, out, scale, tau, high
        )
        lo = tl.where(rejected, mid, lo)
        hi = tl.where(rejected, hi, mid - 1)
    cut = tl.where(nonempty, _ordinary(lo), sentinel)
    tl.store(destination, tl.where(active, cut, sentinel), valid)


def prepare_cutoffs(owner, paid, cut0, cut1, active_budget=None):
    """Invert the existing RU (LOW) or RN (HIGH) screen for each band."""
    _, _, meta, raw, out, hint = owner.plans[paid]
    budget = owner.budget4 if active_budget is None else active_budget
    _prepare_cutoffs[(triton.cdiv(owner.rows * 2, 256),)](
        meta,
        raw,
        out,
        owner.safe_tau,
        hint,
        budget,
        owner._pair_scale,
        owner._pair_rho,
        owner._second_scale,
        owner._second_rho,
        owner._hot_end,
        owner._second_end,
        cut0,
        cut1,
        owner.rows,
        meta.stride(0),
        paid,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
