# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import has_deep_select
from vllm.model_executor.layers.indexer_topk import (
    deep_select_topk,
    get_deep_select_stride_requirement,
    is_deep_select_supported,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    valid_blocks,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    stride_scores,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    FULL_WIDTH: tl.constexpr,
):
    # Writes each row's valid block count; FULL_WIDTH also scores the -inf tail.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    col_end = tl.minimum(end, width)
    num_valid = tl.cdiv(tl.maximum(col_end - start, 0), BLOCK_SIZE)
    if tl.program_id(1) == 0:
        tl.store(valid_blocks + row, num_valid)
    num_blocks = nblocks if FULL_WIDTH else num_valid
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    num_tiles = tl.cdiv(num_blocks, TILE)
    for tile_id in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        blocks = tile_id * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (offsets[None, :] < BLOCK_SIZE) & (cols < col_end),
            other=-float("inf"),
        ).to(tl.float32)
        reduced = tl.reduce(values, 1, _max_with_nan)
        # Pin the newest block. DeepSelect leaves NaN rows partially written,
        # so also pin NaN blocks (torch.topk ranks NaN first too).
        pinned = (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE)
        if not FULL_WIDTH:
            pinned |= reduced != reduced
        reduced = tl.where(pinned, float("inf"), reduced)
        tl.store(scores + row * stride_scores + blocks, reduced, blocks < num_blocks)


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
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    # Dead-store elimination: every consumer of the masked logits reads only
    # [start, end) per row (top_k_per_row_prefill/decode and _block_scores are
    # extent-bounded; the DCP merge is a no-op at dcp_world_size == 1), so
    # poisoning columns outside that range is wasted bandwidth. Skip tiles
    # beyond the row extent entirely.
    if tl.program_id(1) * TILE >= end:
        return
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        valid & ~keep,
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
    # Logits are max_model_len wide (CUDA-graph static) but rows are shorter:
    # DeepSelect stops at each row's valid block count. Order does not matter.
    use_deep_select = has_deep_select() and nblocks >= topk_blocks
    if use_deep_select:
        in_align, out_align = get_deep_select_stride_requirement()
        padded = round_up(nblocks, in_align // logits.element_size())
        scores = logits.new_empty((rows, padded))[:, :nblocks]
        use_deep_select = (
            out.stride(1) == 1
            and out.stride(0) * out.element_size() % out_align == 0
            and is_deep_select_supported(scores, topk_blocks)
        )
    else:
        scores = logits.new_empty((rows, nblocks))
    valid_blocks = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _block_scores_kernel[(rows, 32)](
        logits,
        row_ks,
        row_ke,
        scores,
        valid_blocks,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        scores.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        TILE=128,
        FULL_WIDTH=not use_deep_select,
    )
    if use_deep_select:
        deep_select_topk(scores, topk_blocks, end=valid_blocks, output_idx=out)
        return
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
