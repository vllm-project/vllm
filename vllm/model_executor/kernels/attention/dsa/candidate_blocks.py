# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

# Programs per row for the kernels that walk a row's own [start, end) range.
# The grid is static (rows x this), so the kernels replay unchanged inside CUDA
# graphs while the work stays proportional to each row's context length. The
# logits buffer can be as wide as max_model_len (1M for DeepSeek-V4), so a grid
# over the buffer width would sweep a million columns per row every step.
_ROW_PROGRAMS = 8


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
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
    ROW_PROGRAMS: tl.constexpr,
):
    # Per-block max of the row's logits over [start, end); blocks past the
    # row's range are not written here: `scores` is pre-filled with -inf.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end_raw = tl.load(ends + row // ROW_REPEAT * stride_end)
    end = tl.minimum(end_raw, width)
    row_blocks = tl.cdiv(tl.maximum(end - start, 0), BLOCK_SIZE)
    row_blocks = tl.minimum(row_blocks, nblocks)
    # The row's newest block is pinned to +inf. It is computed from the raw
    # end, as before, so a row whose end lies past the buffer width still pins
    # the block the packed-column clamp later maps to the last column.
    pin = (end_raw - start - 1) // BLOCK_SIZE
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    for tile in range(tl.program_id(1), tl.cdiv(row_blocks, TILE), ROW_PROGRAMS):
        blocks = tile * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (blocks[:, None] < row_blocks)
            & (offsets[None, :] < BLOCK_SIZE)
            & (cols < end),
            other=-float("inf"),
        )
        reduced = tl.reduce(values, 1, _max_with_nan)
        reduced = tl.where((end_raw > start) & (blocks == pin), float("inf"), reduced)
        tl.store(scores + row * nblocks + blocks, reduced, blocks < row_blocks)
    if tl.program_id(1) == 0:
        if (end_raw > start) & (pin >= row_blocks) & (pin < nblocks):
            tl.store(scores + row * nblocks + pin, float("inf"))


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
    # Only the flags of the row's own blocks (plus the sentinel slot at
    # `nblocks`) are ever read by the mask kernel, so only those are cleared.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    end = tl.minimum(end, width)
    row_blocks = tl.cdiv(tl.maximum(end - start, 0), BLOCK_SIZE)
    row_blocks = tl.minimum(row_blocks, nblocks)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(row_blocks, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots < row_blocks)
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
    ROW_PROGRAMS: tl.constexpr,
):
    # Walk only the row's own [start, end) columns. Columns outside it are
    # never read by the row top-k (it takes the same bounds), so they are left
    # alone instead of being filled with -inf across the whole buffer width.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    end = tl.minimum(end, width)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    offsets = tl.arange(0, TILE)
    for tile in range(tl.program_id(1), tl.cdiv(tl.maximum(end - start, 0), TILE), ROW_PROGRAMS):
        cols = start + tile * TILE + offsets
        valid = cols < end
        block = (cols - start) // BLOCK_SIZE
        keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
        keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            valid & ~keep,
        )


@triton.jit
def _ordered_key(x):
    # fp32 -> uint32 whose unsigned order matches the float order.
    b = x.to(tl.int32, bitcast=True)
    flip = tl.where(b < 0, -1, -2147483648)  # negatives: flip all bits; else: flip sign
    return (b ^ flip).to(tl.uint32, bitcast=True)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _topk_candidates_kernel(
    scores,
    starts,
    ends,
    output,
    out_stride_row,
    out_stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    # Exact top-K block ids of one row over its own block scores, with a static
    # launch shape: a row with at most K blocks keeps them all; a longer row
    # goes through a 4-pass radix select on the fp32 keys. Output order is
    # unspecified (the consumer only tests membership); -1 pads the rest.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end_raw = tl.load(ends + row // ROW_REPEAT * stride_end)
    end = tl.minimum(end_raw, width)
    n = tl.cdiv(tl.maximum(end - start, 0), BLOCK_SIZE)
    n = tl.minimum(n, nblocks)
    # A pin past the row's range (end beyond the buffer width) was written by
    # the score kernel at index pin < nblocks with +inf; it is a candidate too.
    pin = (end_raw - start - 1) // BLOCK_SIZE
    has_far_pin = (end_raw > start) & (pin >= n) & (pin < nblocks)
    n_eff = tl.where(has_far_pin, n + 1, n)
    out_cols = tl.arange(0, K)
    if n_eff <= K:
        ids = tl.where(out_cols < n, out_cols, -1)
        ids = tl.where((out_cols == n) & has_far_pin, pin, ids)
        tl.store(output + row * out_stride_row + out_cols * out_stride_col, ids)
    else:
        offs = tl.arange(0, TILE)
        n_scan = tl.where(has_far_pin, pin + 1, n)
        d_bins = tl.arange(0, 256)
        remaining = tl.full([1], K, tl.int32)
        prefix = tl.full([1], 0, tl.uint32)
        for shift in tl.static_range(24, -8, -8):
            hist = tl.zeros([256], tl.int32)
            for t in range(0, tl.cdiv(n_scan, TILE)):
                idx = t * TILE + offs
                valid = (idx < n) | ((idx == pin) & has_far_pin)
                v = tl.load(scores + row * nblocks + idx, valid, other=-float("inf"))
                key = _ordered_key(v)
                if shift == 24:
                    match = valid
                else:
                    match = valid & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
                digit = tl.where(match, ((key >> shift) & 255).to(tl.int32), 0)
                hist += tl.histogram(digit, 256)
                hist -= tl.where(d_bins == 0, tl.sum((~match).to(tl.int32), 0), 0)
            total = tl.sum(hist, 0)
            above = total - tl.cumsum(hist, 0)   # keys whose digit is strictly greater
            # threshold digit: the largest d with at least `remaining` keys at digit >= d
            cand = tl.where(above + hist >= remaining, d_bins, -1)
            d = tl.max(cand, 0)
            above_d = tl.sum(tl.where(d_bins == d, above, 0), 0)
            remaining = remaining - above_d
            prefix = prefix | (d.to(tl.uint32) << shift)
        # prefix is the K-th largest key: take every key above it and
        # `remaining` of the keys equal to it, in index order.
        n_written = tl.zeros([1], tl.int32)
        n_ties = tl.zeros([1], tl.int32)
        for t in range(0, tl.cdiv(n_scan, TILE)):
            idx = t * TILE + offs
            valid = (idx < n) | ((idx == pin) & has_far_pin)
            v = tl.load(scores + row * nblocks + idx, valid, other=-float("inf"))
            key = _ordered_key(v)
            gt = valid & (key > prefix)
            eq = valid & (key == prefix)
            eq_i = eq.to(tl.int32)
            eq_rank = tl.cumsum(eq_i, 0) - eq_i + n_ties
            take = gt | (eq & (eq_rank < remaining))
            take_i = take.to(tl.int32)
            pos = tl.cumsum(take_i, 0) - take_i + n_written
            tl.store(output + row * out_stride_row + pos * out_stride_col, idx, take & (pos < K))
            n_written += tl.sum(take_i, 0)
            n_ties += tl.sum(eq_i, 0)
        tl.store(output + row * out_stride_row + out_cols * out_stride_col, -1, out_cols >= n_written)


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
    scores = logits.new_empty((rows, nblocks))
    _block_scores_kernel[(rows, _ROW_PROGRAMS)](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
        _ROW_PROGRAMS,
    )
    # Exact per-row top-k over the row's own block scores with a static grid
    # (the row length is read on the device), instead of torch.topk over the
    # full `nblocks` width of the buffer for every row.
    _topk_candidates_kernel[(rows,)](
        scores,
        row_ks,
        row_ke,
        out,
        *out.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        topk_blocks,
        row_ks is not None,
        row_repeat,
        1024,
    )

def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask each row's in-range logits down to its request-local candidates."""
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
    _mask_candidates_kernel[(rows, _ROW_PROGRAMS)](
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
        _ROW_PROGRAMS,
    )
