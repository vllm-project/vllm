# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_rows_kernel(
    source_ptr,
    destination_ptr,
    row_ids_ptr,
    num_rows,
    block_size: tl.constexpr,
    source_block_stride,
    destination_row_stride,
    row_words: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    words = tl.arange(0, BLOCK_WORDS)
    word_mask = words < row_words
    for i in range(ROWS_PER_PROGRAM):
        row = tl.program_id(0) * ROWS_PER_PROGRAM + i
        if row < num_rows:
            source_row = tl.load(row_ids_ptr + row).to(tl.int64)
            if source_row >= 0:
                block = source_row // block_size
                offset = source_row - block * block_size
                src = source_ptr + block * source_block_stride + offset * row_words
                values = tl.load(src + words, mask=word_mask)
                tl.store(
                    destination_ptr + row * destination_row_stride + words,
                    values,
                    mask=word_mask,
                )


def gather_rows(
    source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor
) -> None:
    """`destination[i] = source.rows[row_ids[i]]` for `row_ids[i] >= 0`.

    `source` is `[num_blocks, block_size, row]` with contiguous rows and any
    block stride; rows are copied as raw bytes, so the payload dtypes may differ.
    """
    num_rows = row_ids.numel()
    if num_rows == 0:
        return
    row_bytes = source.shape[-1] * source.element_size()
    destination_rows = destination.view(num_rows, -1)
    if (
        row_bytes % 4
        or destination_rows.shape[1] * destination.element_size() != row_bytes
        or source.stride(-1) != 1
        or source.stride(1) * source.element_size() != row_bytes
        or (source.stride(0) * source.element_size()) % 4
        or destination_rows.stride(1) != 1
        or (destination_rows.stride(0) * destination.element_size()) % 4
    ):
        raise ValueError("HiSparse row gather needs contiguous, 4-byte rows.")
    row_words = row_bytes // 4
    block_words = triton.next_power_of_2(row_words)
    rows_per_program = 4
    _gather_rows_kernel[(triton.cdiv(num_rows, rows_per_program),)](
        source.view(torch.int32),
        destination_rows.view(torch.int32),
        row_ids,
        num_rows,
        source.shape[1],
        source.stride(0) * source.element_size() // 4,
        destination_rows.stride(0) * destination.element_size() // 4,
        row_words,
        BLOCK_WORDS=block_words,
        ROWS_PER_PROGRAM=rows_per_program,
    )


@triton.jit
def _mark_referenced_blocks_kernel(
    flags_ptr,
    enabled_ptr,
    block_table_ptr,
    block_table_stride,
    req_ids_ptr,
    topk_ptr,
    topk_stride,
    top_k,
    block_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    if tl.load(enabled_ptr) == 0:
        return
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    positions = tl.load(
        topk_ptr + token * topk_stride + cols, mask=cols < top_k, other=-1
    )
    valid = positions >= 0
    req = tl.load(req_ids_ptr + token).to(tl.int64)
    blocks = tl.load(
        block_table_ptr + req * block_table_stride + positions // block_size,
        mask=valid,
    )
    tl.store(flags_ptr + blocks, 1, mask=valid)


def mark_referenced_blocks(
    block_table: torch.Tensor,
    req_ids: torch.Tensor,
    topk: torch.Tensor,
    block_size: int,
    num_blocks: int,
    enabled: torch.Tensor,
) -> torch.Tensor:
    """One bool per block of `block_table`: whether any `topk` entry reads it.

    Marking is skipped on device when the 0-dim `enabled` is false, leaving
    every block unmarked.
    """
    flags = torch.zeros(num_blocks, dtype=torch.uint8, device=topk.device)
    num_tokens, top_k = topk.shape
    if num_tokens:
        block_k = 1024
        _mark_referenced_blocks_kernel[(num_tokens, triton.cdiv(top_k, block_k))](
            flags,
            enabled,
            block_table,
            block_table.stride(0),
            req_ids,
            topk,
            topk.stride(0),
            top_k,
            block_size,
            BLOCK_K=block_k,
        )
    return flags.view(torch.bool)
