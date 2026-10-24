# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Call-local HOT voting, logical ID mapping and lossless FP8 key gathering."""

import torch

from vllm.triton_utils import tl, triton


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
    tl.store(S + 6 + BLOCKS, tl.sum(sizes))
    tl.store(S, 0)
    tl.store(S + 1, lo)
    tl.store(S + 2, tie_block)
    tl.store(S + 3, tie_take)
    tl.store(S + 4, K)
    tl.store(S + 5, BLOCKS)


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


def carry_votes_topk_reset_(votes, hot, partial, state, k, max_vote, min_index):
    """I32[N], I64[K], I16[ceil(N/8192),max_vote+1], I32[6+blocks+1].

    N<=2^20, 0<K<=N-min_index and max_vote is the unchanged recent-row bound.
    The owner serializes all buffers; only HOT[:K] and active scratch are written.
    """
    count, stride = votes.numel(), partial.stride(0)
    blocks = triton.cdiv(count, 8192)
    take = min(k, count - min_index)
    with torch.accelerator.device_index(votes.device.index):
        _carry_histogram[(blocks,)](
            votes,
            partial,
            count,
            min_index,
            max_vote,
            stride,
            8 if max_vote <= 4 else triton.next_power_of_2(stride),
            num_warps=8,
        )
        _carry_cutoff[(1,)](
            partial,
            state,
            take,
            max_vote,
            blocks,
            stride,
            triton.next_power_of_2(partial.shape[0]),
            num_warps=8,
        )
        _carry_emit[(blocks,)](
            votes, hot, state, count, min_index, max_vote, num_warps=8
        )


@triton.jit(do_not_specialize=["START", "END", "epoch"])
def _undo_mark(
    HOT,
    EPOCH,
    PERM,
    A,
    B,
    C,
    SIZE: tl.constexpr,
    START,
    END,
    epoch,
    TILE: tl.constexpr,
):
    """Undo previous pairs, mark HOT uniquely, preserve sticky metadata errors."""
    ids = tl.program_id(0) * TILE + tl.arange(0, TILE)
    old_count = tl.minimum(tl.maximum(tl.load(C), 0), SIZE)
    a, b = tl.load(A + ids, ids < old_count, 0), tl.load(B + ids, ids < old_count, 0)
    tl.store(PERM + a, a, ids < old_count)
    tl.store(PERM + b, b, ids < old_count)
    hot = tl.load(HOT + ids, ids < SIZE, START)
    valid = (ids < SIZE) & (hot >= START) & (hot < END)
    previous = tl.atomic_xchg(EPOCH + hot, epoch, valid, sem="relaxed")
    duplicate = tl.where(valid, previous == epoch, False)
    error = tl.where(duplicate, 1, 0)
    error |= tl.where((ids < SIZE) & ~valid, 2, 0)
    tl.atomic_or(C + 3, tl.max(error), sem="relaxed")
    if tl.program_id(0) == 0:
        tl.store(C + 1, 0)
        tl.store(C + 2, 0)


@triton.jit(do_not_specialize=["START", "END", "epoch"])
def _compact_pairs(
    HOT,
    EPOCH,
    RANK,
    A,
    B,
    C,
    SIZE: tl.constexpr,
    START,
    END,
    epoch,
    TILE: tl.constexpr,
):
    """Reserve A/B pairs and write only endpoint-local ranks to disjoint scratch."""
    ids = tl.program_id(0) * TILE + tl.arange(0, TILE)
    hot = tl.load(HOT + ids, ids < SIZE, START)
    take_a = (ids < SIZE) & (hot >= START + SIZE) & (hot < END)
    window = START + ids
    take_b = (ids < SIZE) & (tl.load(EPOCH + window, ids < SIZE, 0) != epoch)
    rank_a, rank_b = tl.cumsum(take_a.to(tl.int32)), tl.cumsum(take_b.to(tl.int32))
    base_a = tl.atomic_add(C + 1, tl.sum(take_a.to(tl.int32)), sem="relaxed")
    base_b = tl.atomic_add(C + 2, tl.sum(take_b.to(tl.int32)), sem="relaxed")
    rank_a, rank_b = base_a + rank_a - 1, base_b + rank_b - 1
    tl.store(A + rank_a, hot, take_a)
    tl.store(B + rank_b, window, take_b)
    tl.store(RANK + hot, rank_a, take_a)
    tl.store(RANK + window, rank_b, take_b)


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


@triton.jit(do_not_specialize=["N", "START", "epoch"])
def _publish_gather(
    K,
    SCALE,
    DK,
    DS,
    RHO,
    EPOCH,
    RANK,
    PERM,
    A,
    B,
    C,
    N,
    SIZE: tl.constexpr,
    START,
    epoch,
    TILE_ROWS: tl.constexpr,
):
    """Publish pair swaps while gathering raw K/scales and integer-certified rho."""
    count_a, count_b, status = tl.load(C + 1), tl.load(C + 2), tl.load(C + 3)
    if tl.program_id(0) == 0:
        tl.atomic_or(C + 3, tl.where(count_a != count_b, 8, 0), sem="relaxed")
    _trap_if((status != 0) | (count_a != count_b))
    rows = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    live = rows < N
    marked = tl.load(EPOCH + rows, live, 0) == epoch
    take_a = live & (rows >= START + SIZE) & marked
    take_b = live & (rows >= START) & (rows < START + SIZE) & ~marked
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
    if tl.program_id(0) == 0:
        tl.store(C, count_a)


def plan_and_permuted_gather_rho_out(
    hot,
    hot_epoch,
    permutation,
    pair_rank,
    swap_a,
    swap_b,
    counts,
    window_start,
    common_end,
    epoch,
    keys,
    scales,
    dst_keys,
    dst_scales,
    rho,
):
    """I64[seed] HOT; I32[N] epoch/PERM/rank/rho; I32[seed] A/B; I32[4] counts.

    Keys are contiguous FP8[N,128], scales positive F32[N]; raw values are copied.
    The window and unique HOT are inside [window_start,common_end)<=N. The owner
    initializes identity permutation/zero epochs, handles epoch rollover, and
    serializes this scratch. Rank is disjoint from PERM, K/scales and outputs.
    Compact writes endpoint ranks; gather reads only those ranks and writes
    PERM without reading it. Rank's nonendpoints and tail are untouched.
    K/scale/rho output padding after N is untouched.
    Undo may restore old permutation endpoints beyond the current N, inside
    the owner's full persistent allocation.
    """
    size, count = hot.numel(), dst_keys.shape[0]
    grid = (triton.cdiv(size, 256),)
    with torch.accelerator.device_index(hot.device.index):
        _undo_mark[grid](
            hot,
            hot_epoch,
            permutation,
            swap_a,
            swap_b,
            counts,
            size,
            window_start,
            common_end,
            epoch,
            256,
            num_warps=8,
        )
        _compact_pairs[grid](
            hot,
            hot_epoch,
            pair_rank,
            swap_a,
            swap_b,
            counts,
            size,
            window_start,
            common_end,
            epoch,
            256,
            num_warps=8,
        )
        _publish_gather[(triton.cdiv(count, 32),)](
            keys.view(-1).view(torch.uint8),
            scales,
            dst_keys.view(-1).view(torch.uint8),
            dst_scales,
            rho,
            hot_epoch,
            pair_rank,
            permutation,
            swap_a,
            swap_b,
            counts,
            count,
            size,
            window_start,
            epoch,
            32,
            num_warps=8,
        )


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


def map_topk_vote_stats_(out, index_map, status, votes, recent):
    """I32[rows,2048] output, I32[N] map/votes, I32[rows] status; recent<=rows.

    A zero-length votes tensor disables voting. Keep all buffers live and serial
    on the current stream; malformed status/physical/logical IDs always trap.
    """
    with torch.accelerator.device_index(out.device.index):
        sms = torch.cuda.get_device_properties(out.device).multi_processor_count
        blocks = min(triton.cdiv(out.numel(), 256), sms * 8)
        _map_votes[(blocks,)](
            out,
            index_map,
            status,
            votes,
            out.numel(),
            out.shape[0],
            index_map.numel(),
            out.shape[1],
            votes.numel(),
            recent,
            num_warps=8,
        )
