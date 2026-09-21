# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    n_live,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)

    # [B1] LENGTH GATE, DEVICE-SIDE.  `nblocks` is cdiv(width, BLOCK_SIZE) and
    # `width` on the ratio-1 index layers is the *model* limit (1048576 -> 131072
    # blocks), not this row's context.  Only cdiv(end - start, BLOCK_SIZE) blocks
    # can hold a finite score; the rest were being filled with -inf purely so the
    # downstream top-k could discard them again.  Publish the live count for the
    # selector and skip every tile beyond it.  The bound is recomputed from
    # `ends` on each launch, so it stays correct under FULL cudagraph replay
    # (graphs are not keyed by context length).
    live_blocks = (end - start + BLOCK_SIZE - 1) // BLOCK_SIZE
    live_blocks = min(max(live_blocks, 0), nblocks)
    if tl.program_id(1) == 0:
        tl.store(n_live + row, live_blocks.to(tl.int32))
    if tl.program_id(1) * TILE >= live_blocks:
        # Tail tiles are left UNINITIALISED on purpose: `select_candidate_blocks`
        # bounds its top-k by `n_live`, so no consumer ever reads them.
        return
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < live_blocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value > -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    # [B1] Length-gate the clear. `nblocks` is derived from the *model* limit
    # (cdiv(1048576, 8) = 131072 on the ratio-1 index layers), but
    # `_mask_candidates_kernel` only ever loads a flag for a column below this
    # row's `end`, so slots past that are written and never read. Clearing the
    # live prefix instead of the whole row turns a fixed 129-iteration serial
    # loop into one proportional to the context actually in flight.
    live_blocks = (end - start + BLOCK_SIZE - 1) // BLOCK_SIZE
    live_blocks = min(max(live_blocks, 0), nblocks)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(live_blocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= live_blocks)
    # The clamped-candidate edge slot is read unconditionally, so it must stay
    # cleared even when it sits far outside the live prefix.
    tl.store(flags + row * (nblocks + 1) + nblocks, 0)
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile_base = tl.program_id(1) * TILE
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    # [B1] Length gate. Every column at or past `end` is outside this row's
    # causal bound, so the only thing this kernel does there is store -inf
    # into a region no consumer reads: both `top_k_per_row_decode` and
    # `top_k_per_row_prefill` scan `[start, end)`. The compress_ratio-2 index
    # layers (2/8/14) already run with no mask at all and leave that tail
    # holding whatever the shared workspace last had in it, which is the
    # standing proof that the tail is unread.
    #
    # The gate is evaluated per program from a tensor the kernel already
    # loads, NOT from a host-side max over `ends`: the decode indexer replays
    # from a FULL cuda graph that is not keyed by context length, so a
    # host-derived bound would be frozen at capture and silently under-mask on
    # a replay with longer contexts. The launch geometry is unchanged.
    if tile_base >= end:
        return
    cols = tile_base + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    # The score buffer is an internal temporary, so pin it to fp32 rather than
    # inheriting the logits dtype: widening is exact and order-preserving (it
    # cannot change which blocks win, or which scores tie), and the per-row
    # bounded top-k below is an fp32 kernel.
    scores = torch.empty((rows, nblocks), device=logits.device, dtype=torch.float32)
    n_live = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        n_live,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
    )
    # [B1] `scores.topk(...)` is a full-width kernel: it sorts all `nblocks`
    # columns for every row, so on the ratio-1 index layers it paid for 131072
    # blocks whether the request held 8k of context or 1M.  That made
    # select_candidate_blocks cost the same at every context length.  This build
    # already ships a per-row top-k that takes a DEVICE-SIDE length bound and is
    # what the dense decode path uses for its own top-k over the logits; the
    # block-score selection simply was not wired to it.  Bounding it by `n_live`
    # reads only the live prefix -- which is also what makes it safe for
    # `_block_scores_kernel` to leave the tail unwritten.
    #
    # It matches `scores.topk(...)` + `_store_candidates_kernel` slot for slot:
    # valid ids fill slots [0, min(topk_blocks, n_live)) and -1 pads the suffix,
    # which is the ordering `candidate_slot_bound`/`_mask_candidates_kernel`
    # rely on downstream.
    assert out.is_contiguous() and out.dtype == torch.int32
    assert out.shape[1] == topk_blocks
    torch.ops._C.top_k_per_row_decode(
        scores,
        1,
        n_live,
        out,
        rows,
        scores.stride(0),
        scores.stride(1),
        topk_blocks,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        row_ke,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )


# ---------------------------------------------------------------------------
# [B1] Candidate-only scoring for indexer CONSUMER layers.
#
# On a consumer layer the shipped prefill sequence is
#
#     logits = rocm_fp8_mqa_logits(q, k, w, ks, ke)   # scores ALL `width` cols
#     apply_candidate_mask(logits, ks, ke, cand, bs)  # -inf on all but the
#                                                     # candidate columns
#     top_k_per_row_prefill(logits, ks, ke, ...)      # re-scans all `width`
#
# The candidate set published by the source layer is already in hand BEFORE
# the scoring GEMM runs, and it spans at most ``topk_blocks * block_size``
# columns (16384 for V4.1-Flash).  Everything the GEMM computes outside that
# set is written to HBM, overwritten with -inf, re-read by the top-k and
# discarded.  The scored region is sized to the KV buffer rather than to the
# columns the request can actually consume -- three O(width) passes to keep
# O(cap) values.
#
# ``candidate_mqa_logits`` scores only the candidate columns into a compact
# ``[rows, cap + block_size]`` buffer.  The mask pass disappears (a
# non-candidate column is never produced) and the top-k runs over the compact
# buffer, so all three stages become O(cap).
#
# Restricting the INPUT is what makes this work, and it is only possible
# because K is read transposed: one column is HEAD_SIZE contiguous bytes
# whether or not the columns are adjacent, so an arbitrary column gather
# costs the same bandwidth as a dense sweep.  Gathering the *logits* after
# the fact -- which is the obvious alternative -- moves 8*4 = 32-byte runs
# and gives most of the bus back.
#
# EXACT EQUIVALENCE, including the edge fold.  ``apply_candidate_mask`` folds
# a candidate whose first column lands at or beyond ``width`` into a flag
# that keeps column ``width - 1``.  That branch is unreachable while producer
# and consumer share a compress ratio, but narrowing the contract to "the
# cases I expect" is how a silent divergence gets shipped, so the kernel
# reproduces it: such a slot is scored at column ``width - 1`` and lands in
# the reserved compact column ``cap``.  Several out-of-width slots write the
# same value to that same column, so the race is benign and the reference's
# de-duplication is preserved for free.
# ---------------------------------------------------------------------------


@triton.jit
def _candidate_mqa_logits_kernel(
    Q_ptr,            # fp8 [M, H, D]
    KV_ptr,           # fp8 [N, D]
    kv_scales_ptr,    # fp32 [N]
    weights_ptr,      # fp32 [M, H]
    row_ks_ptr,       # int32 [M]
    row_ke_ptr,       # int32 [M]
    cand_ptr,         # int32 [M, n_cand], request-local block ids, -1 padded
    out_ptr,          # fp32 [M, cap + BLOCK_SIZE]
    seq_len_kv,       # N == the width the dense path would have scored
    n_cand,
    cap,              # n_cand * BLOCK_SIZE; also the edge column index
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_ke: tl.constexpr,
    stride_cand_r: tl.int64,
    stride_cand_c: tl.int64,
    stride_out_r: tl.int64,
    stride_out_c: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # candidate block width, in columns
    BLOCK_KV: tl.constexpr,     # columns per tile; multiple of BLOCK_SIZE
    EDGE_FOLD: tl.constexpr,
):
    row_id = tl.program_id(0)
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_out_r > 0)

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_block = tl.load(
        weights_ptr + row_id * stride_w_s + h_inds * stride_w_h, cache_modifier=".cg"
    ).to(tl.float32)

    start_ind = tl.load(row_ks_ptr + row_id * stride_ks)
    end_ind = tl.minimum(tl.load(row_ke_ptr + row_id * stride_ke), seq_len_kv)

    out_row_ptr = out_ptr + row_id.to(tl.int64) * stride_out_r
    cand_row_ptr = cand_ptr + row_id.to(tl.int64) * stride_cand_r

    if EDGE_FOLD:
        # Written before the loop so an untriggered fold reads as -inf.
        tl.store(out_row_ptr + cap * stride_out_c, -float("inf"))

    SLOTS: tl.constexpr = BLOCK_KV // BLOCK_SIZE
    lane = tl.arange(0, BLOCK_KV)
    slot_of_lane = lane // BLOCK_SIZE
    off_in_block = lane % BLOCK_SIZE

    for tile in tl.range(0, n_cand, SLOTS):
        slot = tile + slot_of_lane
        slot_ok = slot < n_cand
        block = tl.load(
            cand_row_ptr + slot * stride_cand_c, mask=slot_ok, other=-1
        ).to(tl.int32)

        base = start_ind + block * BLOCK_SIZE
        nat_col = base + off_in_block
        if EDGE_FOLD:
            oow = (block >= 0) & (base >= seq_len_kv)
            col = tl.where(oow, seq_len_kv - 1, nat_col)
        else:
            oow = block < -1  # statically false
            col = nat_col
        live = slot_ok & (block >= 0) & (col >= start_ind) & (col < end_ind)
        # Keep every dereferenced address in range, padded slots included.
        col = tl.where(live, col, 0)

        kv_ptrs = KV_ptr + col[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
        kv_block = tl.load(kv_ptrs, mask=live[None, :], other=0.0)
        kv_scales = tl.load(kv_scales_ptr + col, mask=live, other=0.0)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] @ [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)

        # An out-of-width slot contributes only the edge column, so its own
        # compact columns must read -inf.
        nat_live = live & ~oow
        tl.store(
            out_row_ptr + (tile * BLOCK_SIZE + lane) * stride_out_c,
            tl.where(nat_live, scores, -float("inf")),
            mask=slot_ok,
        )
        if EDGE_FOLD:
            # Every out-of-width slot resolves to the same column, so the
            # several writers store the same value: the race is benign and
            # the reference's de-duplication comes for free.
            tl.store(
                out_row_ptr + (cap + 0 * lane) * stride_out_c,
                scores,
                mask=live & oow & (off_in_block == 0),
            )


@triton.jit(do_not_specialize=["topk", "n_cand", "cap", "seq_len_kv"])
def _remap_compact_indices_kernel(
    compact_ptr,   # int32 [M, topk], columns of the compact buffer
    cand_ptr,      # int32 [M, n_cand]
    row_ks_ptr,    # int32 [M]
    out_ptr,       # int32 [M, topk], columns relative to row_ks
    stride_c_r: tl.int64,
    stride_c_c: tl.int64,
    stride_cand_r: tl.int64,
    stride_cand_c: tl.int64,
    stride_ks: tl.constexpr,
    stride_o_r: tl.int64,
    stride_o_c: tl.int64,
    topk,
    n_cand,
    cap,
    seq_len_kv,
    BLOCK_SIZE: tl.constexpr,
    EDGE_FOLD: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    in_range = cols < topk
    idx = tl.load(compact_ptr + row * stride_c_r + cols * stride_c_c, in_range, other=-1)
    start = tl.load(row_ks_ptr + row * stride_ks)

    slot = idx // BLOCK_SIZE
    off = idx % BLOCK_SIZE
    good = in_range & (idx >= 0) & (slot < n_cand)
    block = tl.load(
        cand_ptr + row * stride_cand_r + slot * stride_cand_c, good, other=-1
    )
    remapped = tl.where(good & (block >= 0), block * BLOCK_SIZE + off, -1)
    if EDGE_FOLD:
        edge_rel = seq_len_kv - 1 - start
        remapped = tl.where(
            in_range & (idx == cap) & (edge_rel >= 0), edge_rel, remapped
        )
    tl.store(out_ptr + row * stride_o_r + cols * stride_o_c, remapped, in_range)


# Row stride of the compact buffer is padded to this many fp32 columns. The
# top-k that follows reads the buffer row-major; an unpadded stride of
# ``cap + block_size`` puts every row start 32 bytes off a 128-byte boundary,
# so each row's scan is misaligned against the cache line for its whole
# length. Alignment, not size, is what this padding buys.
_COMPACT_ALIGN_COLS = 128

# Tile shape / pipeline depth for the candidate-scoring kernel, measured on
# gfx950 across row counts 64..4096 and widths 64k..128k; see
# eval/eval_b1_candidate_scoring.py.
_COMPACT_BLOCK_KV = 256
_COMPACT_NUM_WARPS = 4
_COMPACT_MFMA_NONKDIM = 16
_COMPACT_NUM_STAGES = 3


def candidate_scoring_width(n_cand: int, block_size: int) -> int:
    """Allocation width of the compact score buffer.

    ``cap`` columns of candidate scores, then the reserved edge column, then
    padding to a cache-line-aligned row stride.
    """
    total = n_cand * block_size + block_size
    return (
        (total + _COMPACT_ALIGN_COLS - 1) // _COMPACT_ALIGN_COLS
    ) * _COMPACT_ALIGN_COLS


def candidate_scoring_admits(
    row_context: int,
    n_cand: int,
    block_size: int,
    crossover_ratio: float,
) -> bool:
    """Length gate for candidate-only scoring.

    The compact path costs O(``n_cand * block_size``) per row whatever the
    context; the dense path costs O(``row_context``) per row.  Below the
    capacity the candidate set already covers the whole row and the compact
    path is pure overhead, so the gate is a RATIO -- and ``crossover_ratio``
    has to come from a measurement on the target architecture, not from
    another box's table.

    ``row_context`` must be a PER-ROW span, not the packed width of a
    multi-request chunk's K buffer.  The two differ by the packing factor,
    and gating on the packed width admits chunks of many short requests
    where every row is already fully covered by its candidate set: measured
    on gfx950 at span/capacity 0.5, the compact path runs 0.58x-0.69x
    however wide the packed buffer is.
    """
    cap = n_cand * block_size
    if cap <= 0 or n_cand <= 0:
        return False
    return row_context > crossover_ratio * cap


_COMPACT_BOUNDS_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def compact_topk_bounds(
    rows: int, n_cand: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row bounds for a top-k over the compact buffer: ``[0, cap + 1)``.

    Constant for a given capacity, so they are built once and sliced rather
    than reallocated per chunk.
    """
    cap = n_cand * block_size
    key = (cap, str(device))
    entry = _COMPACT_BOUNDS_CACHE.get(key)
    if entry is None or entry[0].numel() < rows:
        size = max(rows, 1024)
        zeros = torch.zeros(size, dtype=torch.int32, device=device)
        ends = torch.full((size,), cap + 1, dtype=torch.int32, device=device)
        entry = (zeros, ends)
        _COMPACT_BOUNDS_CACHE[key] = entry
    return entry[0][:rows], entry[1][:rows]


def candidate_mqa_logits(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    row_ks: torch.Tensor,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    out: torch.Tensor | None = None,
    edge_fold: bool = True,
    block_kv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Score only the candidate columns, into a compact buffer.

    ``candidate_blocks`` holds request-local block ids relative to ``row_ks``
    -- the space ``select_candidate_blocks`` writes and
    ``apply_candidate_mask`` reads -- with ``-1`` padding.  Compact column
    ``slot * block_size + off`` carries the score of packed column
    ``row_ks[row] + candidate_blocks[row, slot] * block_size + off``; compact
    column ``cap`` carries the edge-fold column; padded slots and columns
    outside ``[ks, ke)`` read ``-inf``.
    """
    assert q.is_cuda and k_fp8.is_cuda
    m, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    n_cand = candidate_blocks.shape[1]
    cap = n_cand * block_size
    assert candidate_blocks.shape[0] >= m
    assert weights.shape[0] >= m
    assert num_heads & (num_heads - 1) == 0, "NUM_HEADS must be a power of two"
    assert head_size & (head_size - 1) == 0, "HEAD_SIZE must be a power of two"
    assert block_size & (block_size - 1) == 0, "block_size must be a power of two"

    if out is None:
        out = torch.empty(
            (m, candidate_scoring_width(n_cand, block_size)),
            dtype=torch.float32,
            device=q.device,
        )
    if m == 0 or cap == 0:
        return out

    # A tile must cover a whole number of candidate blocks. Unlike the dense
    # kernel, whose KV pointer advances by a constant and can therefore be
    # prefetched arbitrarily far ahead, this kernel's KV address depends on a
    # load from the candidate list. The tile shape and stage count are the
    # levers that hide that dependency; the defaults below were measured on
    # gfx950, not inherited.
    if block_kv is None:
        block_kv = max(_COMPACT_BLOCK_KV, block_size)
    assert block_kv % block_size == 0
    if num_warps is None:
        num_warps = _COMPACT_NUM_WARPS
    if num_stages is None:
        num_stages = _COMPACT_NUM_STAGES

    _candidate_mqa_logits_kernel[(m,)](
        q,
        k_fp8,
        kv_scales.reshape(-1),
        weights,
        row_ks,
        row_ke,
        candidate_blocks,
        out,
        seq_len_kv,
        n_cand,
        cap,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        row_ks.stride(0),
        row_ke.stride(0),
        candidate_blocks.stride(0),
        candidate_blocks.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_KV=block_kv,
        EDGE_FOLD=edge_fold,
        num_warps=num_warps,
        num_stages=num_stages,
        # The dot is [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV], so its M
        # is NUM_HEADS and does NOT grow with the row count. The dense
        # kernel's `16 if seq_len <= 1024 else 32` heuristic keys on
        # something this dot's shape does not depend on; carrying it over
        # would only force a recompile at an arbitrary row count.
        matrix_instr_nonkdim=_COMPACT_MFMA_NONKDIM,
        waves_per_eu=2,
    )
    return out


def remap_compact_indices(
    compact_indices: torch.Tensor,
    candidate_blocks: torch.Tensor,
    row_ks: torch.Tensor,
    block_size: int,
    seq_len_kv: int,
    out: torch.Tensor | None = None,
    edge_fold: bool = True,
) -> torch.Tensor:
    """Map compact-buffer columns to packed columns relative to ``row_ks``."""
    rows, topk = compact_indices.shape
    n_cand = candidate_blocks.shape[1]
    if out is None:
        out = torch.empty_like(compact_indices)
    if rows == 0 or topk == 0:
        return out
    tile = 256
    _remap_compact_indices_kernel[(rows, triton.cdiv(topk, tile))](
        compact_indices,
        candidate_blocks,
        row_ks,
        out,
        *compact_indices.stride(),
        *candidate_blocks.stride(),
        row_ks.stride(0),
        *out.stride(),
        topk,
        n_cand,
        n_cand * block_size,
        seq_len_kv,
        block_size,
        edge_fold,
        tile,
    )
    return out


# ---------------------------------------------------------------------------
# [B1] Candidate-only scoring for the DECODE path.
#
# The prefill kernel above reads a contiguous, already-gathered ``[N, D]`` K
# buffer.  Decode has no such buffer: it reads the paged indexer cache through
# a block table, in the SHUFFLE layout ``indexer_k_quant_and_cache_triton``
# writes and the AITER ``Preshuffle=True`` kernel reads.  So the compact path
# cannot be reached by reusing the prefill kernel -- the gather has to move
# INTO the scorer.
#
# Why decode is where this matters most on ROCm.  ``_supports_native_decode``
# returns ``next_n in (1, 2)`` off CUDA, so with DSpark (next_n=6)
# ``_use_flattening`` is True and every verify row becomes its own
# single-token request with its own block-table row and its own causal
# length.  The dense scorer therefore re-reads the whole KV span once per
# verify row -- there is no next_n amortisation to give back.  Reading only
# the candidate columns divides that by ``ctx / cap`` per row.
#
# LENGTH GATE, DEVICE-SIDE.  ``select_candidate_blocks`` builds the candidate
# list with a per-row top-k bounded by the row's live block count, which emits
# ids for slots ``[0, min(topk_blocks, n_live))`` and -1 for the rest, so every
# live block id sits in a LEADING slot and the -1 padding is a suffix.  A row whose context is ``end`` can have at
# most ``cdiv(end, block_size)`` live blocks, so
#
#     live_slots = min(n_cand, cdiv(end, block_size))
#
# is an exact upper bound on the slots worth touching.  Gating the slot loop
# on it makes the compact path cost ``O(min(ctx, cap))`` per row instead of
# ``O(cap)``: at contexts below the capacity it degenerates to the dense
# column count rather than over-running it.  The bound is recomputed on
# device from ``ends_ptr`` on every launch, so it is correct under FULL
# cuda-graph replay, which is not keyed by context length.
#
# The same bound is the top-k's row end: compact columns at or past
# ``live_slots * block_size`` are never written by this kernel, and the
# workspace arena they live in is shared, so the top-k must not scan them.
# The kernel publishes ``live_slots * block_size`` into ``bounds_ptr`` for
# exactly that use.
#
# NO EDGE FOLD.  ``_candidate_flags_kernel``'s out-of-width clamp needs
# ``start + block * block_size >= width``.  At decode ``start`` is 0, block
# ids come from a top-k over ``cdiv(width, block_size)`` blocks, and width is
# ``max_model_len``; the branch is unreachable.  It is asserted away rather
# than reproduced.
# ---------------------------------------------------------------------------


@triton.jit
def _paged_candidate_mqa_logits_kernel(
    Q_ptr,          # fp8 [R, H, D]
    KV_ptr,         # fp8 [num_blocks, page_size * D] (strided rows)
    KVS_ptr,        # fp32 [num_blocks, page_size]    (strided rows)
    W_ptr,          # fp32 [R, H]
    ends_ptr,       # int32 [R]   per-row causal length
    BT_ptr,         # int32 [R, bt_width]
    cand_ptr,       # int32 [R, n_cand]  block ids, -1 padded suffix
    out_ptr,        # fp32 [R, >= cap]
    bounds_ptr,     # int32 [R]   written: live compact columns
    kv_stride: tl.int64,
    kvs_stride: tl.int64,
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_end: tl.constexpr,
    stride_bt_r: tl.int64,
    stride_bt_c: tl.constexpr,
    stride_cand_r: tl.int64,
    stride_cand_c: tl.int64,
    stride_out_r: tl.int64,
    stride_out_c: tl.int64,
    n_cand,
    num_blocks,
    bt_width,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # candidate block width, in columns
    BLOCK_KV: tl.constexpr,     # columns per tile; multiple of BLOCK_SIZE
    PAGE_SIZE: tl.constexpr,    # KV cache block size, in tokens
    LAYOUT: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tl.assume(row >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_out_r > 0)
    tl.assume(kv_stride > 0)
    tl.assume(kvs_stride > 0)

    SLOTS: tl.constexpr = BLOCK_KV // BLOCK_SIZE
    slot_base = tl.program_id(1) * (SLOTS * TILES)

    end = tl.load(ends_ptr + row * stride_end)
    live_slots = (end + BLOCK_SIZE - 1) // BLOCK_SIZE
    live_slots = tl.maximum(tl.minimum(live_slots, n_cand), 0)
    if tl.program_id(1) == 0:
        tl.store(bounds_ptr + row, live_slots * BLOCK_SIZE)
    if slot_base >= live_slots:
        return

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)
    q_block = tl.load(
        Q_ptr
        + row.to(tl.int64) * stride_q_s
        + h_inds * stride_q_h
        + d_inds[None, :] * stride_q_d,
        cache_modifier=".cg",
    )
    w_block = tl.load(
        W_ptr + row.to(tl.int64) * stride_w_s + h_inds * stride_w_h,
        cache_modifier=".cg",
    ).to(tl.float32)

    if LAYOUT == "SHUFFLE":
        d_off = (d_inds // HEAD_TILE) * (HEAD_TILE * BLOCK_TILE) + d_inds % HEAD_TILE
    else:
        d_off = d_inds
    d_off = d_off.to(tl.int64)[:, None]

    cand_row = cand_ptr + row.to(tl.int64) * stride_cand_r
    bt_row = BT_ptr + row.to(tl.int64) * stride_bt_r
    out_row = out_ptr + row.to(tl.int64) * stride_out_r

    lane = tl.arange(0, BLOCK_KV)
    slot_of_lane = lane // BLOCK_SIZE
    off_in_block = lane % BLOCK_SIZE

    for t in tl.static_range(TILES):
        tile_slot0 = slot_base + t * SLOTS
        slot = tile_slot0 + slot_of_lane
        slot_ok = slot < live_slots
        block = tl.load(
            cand_row + slot * stride_cand_c, mask=slot_ok, other=-1
        ).to(tl.int32)

        col = block * BLOCK_SIZE + off_in_block
        live = slot_ok & (block >= 0) & (col < end)
        page = tl.where(live, col // PAGE_SIZE, 0)
        live = live & (page < bt_width)
        phys = tl.load(
            bt_row + tl.where(live, page, 0) * stride_bt_c, mask=live, other=-1
        )
        live = live & (phys >= 0) & (phys < num_blocks)
        safe_phys = tl.where(live, phys, 0).to(tl.int64)
        page_off = tl.where(live, col % PAGE_SIZE, 0)

        if LAYOUT == "SHUFFLE":
            base = (
                safe_phys * kv_stride
                + (page_off // BLOCK_TILE).to(tl.int64) * (HEAD_SIZE * BLOCK_TILE)
                + (page_off % BLOCK_TILE).to(tl.int64) * HEAD_TILE
            )
        else:
            base = safe_phys * kv_stride + page_off.to(tl.int64) * HEAD_SIZE

        kv_block = tl.load(KV_ptr + base[None, :] + d_off, mask=live[None, :], other=0.0)
        kv_scales = tl.load(
            KVS_ptr + safe_phys * kvs_stride + page_off, mask=live, other=0.0
        )

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] @ [HEAD_SIZE, BLOCK_KV]
        # Reduction order follows the DECODE reference
        # (`fp8_paged_mqa_logits_torch`): relu, weight, sum over heads, then
        # one dequant multiply per column.  The prefill kernel scales before
        # the relu, which is the same value (`kv_scale > 0` commutes with both
        # relu and the sum) at H times the multiplies.
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        scores = scores * kv_scales

        tl.store(
            out_row + (tile_slot0 * BLOCK_SIZE + lane) * stride_out_c,
            tl.where(live, scores, -float("inf")),
            mask=slot_ok,
        )


@triton.jit(do_not_specialize=["topk", "n_cand"])
def _remap_decode_compact_kernel(
    compact_ptr,   # int32 [R, topk], columns of the compact buffer
    cand_ptr,      # int32 [R, n_cand]
    ends_ptr,      # int32 [R]
    out_ptr,       # int32 [R, topk], absolute columns
    stride_c_r: tl.int64,
    stride_c_c: tl.int64,
    stride_cand_r: tl.int64,
    stride_cand_c: tl.int64,
    stride_end: tl.constexpr,
    stride_o_r: tl.int64,
    stride_o_c: tl.int64,
    topk,
    n_cand,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    in_range = cols < topk
    idx = tl.load(compact_ptr + row * stride_c_r + cols * stride_c_c, in_range, other=-1)
    end = tl.load(ends_ptr + row * stride_end)

    slot = idx // BLOCK_SIZE
    off = idx % BLOCK_SIZE
    good = in_range & (idx >= 0) & (slot < n_cand)
    block = tl.load(
        cand_ptr + row * stride_cand_r + slot * stride_cand_c, good, other=-1
    )
    col = block * BLOCK_SIZE + off
    # The compact row bound is the CANDIDATE capacity, not the causal length,
    # so a row with fewer live candidate columns than `topk` can hand back a
    # column at or past `end`.  The dense path scans `[0, end)` and can never
    # do that; emit -1 (the buffer's own "no token" value) instead of an index
    # outside the causal range.
    remapped = tl.where(good & (block >= 0) & (col < end), col, -1)
    tl.store(out_ptr + row * stride_o_r + cols * stride_o_c, remapped, in_range)


# Tile shape / pipeline depth for the paged candidate scorer, measured on
# gfx950; see eval/eval_b1_decode_candidate_scoring.py.  Unlike the prefill
# kernel this one runs at 64-192 rows, far short of the 256 CUs, so the slot
# range is split across a second grid dimension rather than walked serially
# by one CTA per row.
_PAGED_BLOCK_KV = 256
_PAGED_TILES = 1
_PAGED_NUM_WARPS = 4
_PAGED_NUM_STAGES = 2
_PAGED_MFMA_NONKDIM = 16


def paged_candidate_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    head_dim: int,
    out: torch.Tensor,
    out_bounds: torch.Tensor,
    block_kv: int | None = None,
    tiles: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score only the candidate columns of a paged decode row, compactly.

    ``q`` is ``[rows, heads, head_dim]`` fp8 -- one row per FLATTENED verify
    slot, which is what ROCm decode produces.  ``kv_cache`` is the packed
    indexer cache ``[num_blocks, page_size, 1, head_dim + 4]`` (uint8; fp8
    values then fp32 scales), read through ``block_table`` exactly as
    ``indexer_k_quant_and_cache_triton`` wrote it.

    Compact column ``slot * block_size + off`` carries the score of absolute
    column ``candidate_blocks[row, slot] * block_size + off``.  Columns whose
    slot is padded or whose absolute column is at or past ``context_lens[row]``
    read ``-inf``.  ``out_bounds[row]`` receives the number of compact columns
    this launch actually wrote; everything at or past it is untouched
    workspace and must not be scanned.
    """
    from vllm.platforms import current_platform

    assert q.is_cuda and kv_cache.is_cuda
    rows, num_heads, head_size = q.shape
    assert head_size == head_dim
    num_blocks = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    n_cand = candidate_blocks.shape[1]
    cap = n_cand * block_size
    assert candidate_blocks.shape[0] >= rows
    assert weights.shape[0] >= rows
    assert block_table.shape[0] >= rows
    assert context_lens.numel() >= rows
    assert out.shape[0] >= rows and out.shape[1] >= cap
    assert out_bounds.numel() >= rows
    assert num_heads & (num_heads - 1) == 0, "NUM_HEADS must be a power of two"
    assert head_size & (head_size - 1) == 0, "HEAD_SIZE must be a power of two"
    assert block_size & (block_size - 1) == 0, "block_size must be a power of two"
    assert page_size % block_size == 0, (
        "a candidate block must not straddle two KV pages"
    )

    if rows == 0 or cap == 0:
        return out, out_bounds

    flat = kv_cache.view(num_blocks, -1)
    assert flat.shape[1] == page_size * (head_dim + 4)
    fp8_dtype = current_platform.fp8_dtype()
    kv_value = flat[:, : page_size * head_dim].view(fp8_dtype)
    kv_scale = flat[:, page_size * head_dim :].view(torch.float32)
    layout = "NORMAL" if page_size == 1 else "SHUFFLE"
    head_tile = head_tile_size // flat.element_size()

    if block_kv is None:
        block_kv = max(_PAGED_BLOCK_KV, block_size)
    assert block_kv % block_size == 0
    if tiles is None:
        tiles = _PAGED_TILES
    if num_warps is None:
        num_warps = _PAGED_NUM_WARPS
    if num_stages is None:
        num_stages = _PAGED_NUM_STAGES

    slots_per_program = (block_kv // block_size) * tiles
    grid = (rows, triton.cdiv(n_cand, slots_per_program))
    _paged_candidate_mqa_logits_kernel[grid](
        q,
        kv_value,
        kv_scale,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        out,
        out_bounds,
        kv_value.stride(0),
        kv_scale.stride(0),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        context_lens.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        candidate_blocks.stride(0),
        candidate_blocks.stride(1),
        out.stride(0),
        out.stride(1),
        n_cand,
        num_blocks,
        block_table.shape[1],
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_KV=block_kv,
        PAGE_SIZE=page_size,
        LAYOUT=layout,
        BLOCK_TILE=block_tile_size,
        HEAD_TILE=head_tile,
        TILES=tiles,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=_PAGED_MFMA_NONKDIM,
        waves_per_eu=2,
    )
    return out, out_bounds


def remap_decode_compact_indices(
    compact_indices: torch.Tensor,
    candidate_blocks: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map compact-buffer columns back to absolute KV columns."""
    rows, topk = compact_indices.shape
    n_cand = candidate_blocks.shape[1]
    if out is None:
        out = torch.empty_like(compact_indices)
    if rows == 0 or topk == 0:
        return out
    tile = 256
    _remap_decode_compact_kernel[(rows, triton.cdiv(topk, tile))](
        compact_indices,
        candidate_blocks,
        context_lens,
        out,
        *compact_indices.stride(),
        *candidate_blocks.stride(),
        context_lens.stride(0),
        *out.stride(),
        topk,
        n_cand,
        block_size,
        tile,
    )
    return out
