# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import has_deep_select
from vllm.model_executor.layers.indexer_topk import (
    AUTO_COOPERATIVE_MAX_ROWS,
    RADIX_TOPK_WORKSPACE_SIZE,
    deep_select_topk,
    get_deep_select_stride_requirement,
    is_deep_select_supported,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.worker.workspace import current_workspace_manager


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
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
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


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_prefix_kernel(
    logits,
    ends,
    scores,
    lengths,
    out,
    stride_row,
    stride_col,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    end = tl.load(ends + row)
    # Blocks from cdiv(end, BLOCK_SIZE) on score -inf and are not pinned, so
    # the top-k reads only this prefix of the row, the only part stored.
    limit = tl.minimum((tl.maximum(end, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE, nblocks)
    if tl.program_id(1) == 0:
        tl.store(lengths + row, limit.to(tl.int32))
        # cooperative_topk may leave slots unwritten when its boundary bin
        # overflows; they must read -1, never a stale block id.
        for tile in range(0, K, TILE):
            slots = tile + tl.arange(0, TILE)
            tl.store(out + row * K + slots, -1, slots < K)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
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
        (end > 0) & (blocks == (end - 1) // BLOCK_SIZE), float("inf"), reduced
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < limit)


@triton.jit(do_not_specialize=["nblocks"])
def _finalize_candidates_kernel(
    out,
    scores,
    lengths,
    nblocks,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    index = tl.load(out + row * K + cols, cols < K, other=-1)
    valid = (index >= 0) & (index < tl.load(lengths + row))
    value = tl.load(
        scores + row * nblocks + index, (cols < K) & valid, other=-float("inf")
    )
    # Only -inf blocks become padding, as in select_candidate_blocks.
    tl.store(
        out + row * K + cols,
        tl.where(valid & (value != -float("inf")), index, -1),
        cols < K,
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


def supports_cooperative_candidates(num_rows: int, num_blocks: int, k: int) -> bool:
    """Whether `select_candidate_blocks_cooperative` takes ``num_rows`` rows of
    ``num_blocks`` blocks and ``k`` candidates."""
    return (
        0 < num_rows <= AUTO_COOPERATIVE_MAX_ROWS
        and k in (512, 1024, 2048)
        # cooperative_topk reads the scores with TMA: 16-byte aligned rows.
        and num_blocks % 4 == 0
    )


def select_candidate_blocks_cooperative(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
) -> None:
    """`select_candidate_blocks` for decode rows, on ``cooperative_topk``.

    The score kernel also writes each row's valid block count, so
    ``cooperative_topk`` scans only the valid prefix of the padded row. The
    selected set of non-negative ids matches `select_candidate_blocks` except
    for which blocks whose fp32 scores tie exactly at the selection boundary
    are kept. ``cooperative_topk`` keeps at most ``k`` (rows of up to 16384
    blocks) or 1024 per cluster CTA (longer rows) candidates of its fp16
    boundary bin, so a more crowded boundary bin is selected approximately.
    The pinned newest block is kept unless at least ``k`` blocks of the row
    score NaN or overflow fp16. Rows are not sorted and -1 padding may sit
    anywhere; every consumer of the candidate buffer treats a row as a set.

    Args:
        logits: [rows, width] fp32 logits; row ``i`` is valid below
            ``row_ends[i]``.
        row_ends: Contiguous [rows] int32 row lengths in logits columns.
        block_size: Columns per candidate block.
        out: Contiguous [rows, k] int32 candidate block ids, -1 padded; see
            `supports_cooperative_candidates` for the supported shapes.

    """
    rows, width = logits.shape
    k = out.shape[1]
    nblocks = triton.cdiv(width, block_size)
    assert supports_cooperative_candidates(rows, nblocks, k)
    assert out.dtype == torch.int32 and out.shape[0] == rows and out.is_contiguous()
    scores = logits.new_empty((rows, nblocks))
    lengths = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _block_scores_prefix_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ends,
        scores,
        lengths,
        out,
        *logits.stride(),
        width,
        nblocks,
        block_size,
        k,
        128,
    )
    (workspace,) = current_workspace_manager().get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    # Rows of at most k blocks take all of them and pad with -1; the last
    # argument (max_seq_len) is unused by the kernel.
    torch.ops._C.cooperative_topk(scores, lengths, out, workspace, k, nblocks)
    _finalize_candidates_kernel[(rows, triton.cdiv(k, 1024))](
        out, scores, lengths, nblocks, k, 1024
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
