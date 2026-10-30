# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-block selection with ``cooperative_topk`` for decode rows.

`select_candidate_blocks` scores every block of the padded logits row and
runs ``torch.topk`` over all of them. At long context most of a decode row is
padding, so here the score kernel also writes each row's valid block count and
``cooperative_topk`` scans only that prefix.

The selected set of non-negative ids matches `select_candidate_blocks` except
for which blocks whose fp32 scores tie exactly at the selection boundary are
kept. ``cooperative_topk`` keeps at most ``k`` (rows of up to 16384 blocks) or
1024 per cluster CTA (longer rows) candidates of its fp16 boundary bin, so a
more crowded boundary bin is selected approximately. The pinned newest block is
kept unless at least ``k`` blocks of the row score NaN or overflow fp16. Rows
are not sorted and -1 padding may sit anywhere; every consumer of the candidate
buffer treats a row as a set.
"""

import torch

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import _max_with_nan
from vllm.model_executor.layers.indexer_topk import (
    AUTO_COOPERATIVE_MAX_ROWS,
    RADIX_TOPK_WORKSPACE_SIZE,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
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
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
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
