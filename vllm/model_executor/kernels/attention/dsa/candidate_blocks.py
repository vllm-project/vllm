# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

# Programs along the column axis of the kernels that walk a row's live range.
# The grid is (rows, this) and both terms come from tensor shapes only, so a
# full cudagraph replays the same launch; only the loop trip count inside the
# kernel follows each row's context length. The decode logits buffer is
# max_model_len wide (1M for DeepSeek-V4.1), so a grid sized by the buffer
# width launches 1024 programs per row of which the first few do all the work.
_ROW_PROGRAMS = 8  # floor, so a wide batch keeps a few programs per row
_TOTAL_PROGRAMS = 8192  # ceiling on the total, so a narrow batch still fills


def _row_programs(rows: int, tiles: int) -> int:
    """Column programs per row: enough to fill the device, at most `tiles`."""
    return max(1, min(tiles, max(_ROW_PROGRAMS, triton.cdiv(_TOTAL_PROGRAMS, rows))))


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
):
    # Score only the blocks the row's own [start, end) covers, striding over
    # them with a fixed number of programs. `scores` arrives pre-filled with
    # -inf, which is what the old full-width grid wrote for the rest.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    row_blocks = tl.minimum(
        tl.cdiv(tl.maximum(tl.minimum(end, width) - start, 0), BLOCK_SIZE), nblocks
    )
    # The row's newest block is pinned to +inf, off the unclamped end as before.
    pin = (end - start - 1) // BLOCK_SIZE
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    tile = tl.program_id(1)
    num_tiles = tl.cdiv(row_blocks, TILE)
    while tile < num_tiles:
        blocks = tile * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (blocks[:, None] < row_blocks)
            & (offsets[None, :] < BLOCK_SIZE)
            & (cols < end)
            & (cols < width),
            other=-float("inf"),
        )
        reduced = tl.reduce(values, 1, _max_with_nan)
        reduced = tl.where((end > start) & (blocks == pin), float("inf"), reduced)
        tl.store(scores + row * nblocks + blocks, reduced, blocks < row_blocks)
        tile += tl.num_programs(1)
    # A row whose end runs past the buffer width pins a block outside the range
    # walked above; the full-width grid used to reach it.
    if (tl.program_id(1) == 0) & (end > start) & (pin >= row_blocks) & (pin < nblocks):
        tl.store(scores + row * nblocks + pin, float("inf"))


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
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
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
    # Stride over the row's live [0, end) columns instead of the whole buffer
    # width. Columns at or past `end` are not read afterwards -- the row top-k
    # bounds its scan by the same ends -- so the buffer tail is left alone.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    col0 = tl.program_id(1) * TILE
    step = tl.num_programs(1) * TILE
    while col0 < end:
        cols = col0 + tl.arange(0, TILE)
        live = (cols < end) & (cols < width)
        valid = live & (cols >= start)
        block = (cols - start) // BLOCK_SIZE
        keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
        keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            live & ~(valid & keep),
        )
        col0 += step


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
    scores = logits.new_full((rows, nblocks), -float("inf"))
    _block_scores_kernel[(rows, _row_programs(rows, triton.cdiv(nblocks, 128)))](
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
    )
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask each row's live logits down to its request-local candidates."""
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
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, _row_programs(rows, triton.cdiv(width, 1024)))](
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
