# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather the accepted target context without modifying its paged KV cache."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_context(
    cache,
    block_table,
    topk,
    source_rows,
    anchors,
    active,
    output,
    indices,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    RECORD: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TOPK_STRIDE: tl.constexpr,
    TOPK_ROWS: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    WINDOW: tl.constexpr,
):
    request = tl.program_id(0) // WIDTH
    column = tl.program_id(0) % WIDTH
    enabled = tl.load(active + request)
    anchor = tl.load(anchors + request)
    if WINDOW:
        position = anchor - WIDTH + column
    else:
        row = tl.load(source_rows + request)
        enabled &= (row >= 0) & (row < TOPK_ROWS)
        position = tl.load(
            topk + row.to(tl.int64) * TOPK_STRIDE + column,
            enabled,
            other=-1,
        )
    logical_block = position // CACHE_BLOCK
    valid = (
        enabled
        & (position >= 0)
        & (position < anchor)
        & (logical_block >= 0)
        & (logical_block < TABLE_COLS)
    )
    block = tl.load(
        block_table + request.to(tl.int64) * TABLE_STRIDE + logical_block,
        valid,
        other=-1,
    )
    valid &= (block >= 0) & (block < NUM_BLOCKS)
    page = cache + block.to(tl.int64) * CACHE_STRIDE
    slot = position % CACHE_BLOCK
    dim = tl.arange(0, 512)
    if RECORD == 528:
        data = page + slot * 528
        scale = tl.load(data + 512 + dim // 32, valid, other=127)
        value = tl.load(data + dim, valid, other=0).to(tl.float8e4nv, bitcast=True)
        result = value.to(tl.float32) * tl.exp2(scale.to(tl.float32) - 127)
    else:
        data = page + slot * 576
        scale = tl.load(
            page + CACHE_BLOCK * 576 + slot * 8 + dim // 64,
            valid & (dim < 448),
            other=127,
        )
        value = tl.load(data + dim, valid & (dim < 448), other=0).to(
            tl.float8e4nv, bitcast=True
        )
        tail = tl.load(
            (data + 448).to(tl.pointer_type(tl.bfloat16)) + dim - 448,
            valid & (dim >= 448),
            other=0,
        )
        result = tl.where(
            dim < 448,
            value.to(tl.float32) * tl.exp2(scale.to(tl.float32) - 127),
            tail.to(tl.float32),
        )
    output_row = request.to(tl.int64) * OUTPUT_STRIDE + OFFSET + column
    tl.store(output + output_row * 512 + dim, result)
    tl.store(
        indices + request.to(tl.int64) * INDEX_STRIDE + OFFSET + column,
        tl.where(valid, output_row, -1),
    )


def gather_dfly_context(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    topk: torch.Tensor,
    source_rows: torch.Tensor,
    anchors: torch.Tensor,
    active: torch.Tensor,
    output: torch.Tensor,
    indices: torch.Tensor,
    *,
    offset: int,
    width: int,
    window: bool,
) -> None:
    """Gather top-k or the preceding SWA window into a request-local BF16 buffer.

    The caller supplies only the current batch's block tables. Rejected target
    tokens, inactive requests and invalid sparse indices produce zero KV and
    an index of -1, including when the buffers held another request previously.
    """
    if cache.dtype != torch.uint8 or cache.shape[-1] not in (528, 584):
        raise ValueError("DFly2 requires the FP8 DeepSeek-V4/V4.1 KV record")
    assert output.shape[-1] == 512 and output.stride(-1) == 1
    assert output.stride(1) == 512 and indices.stride(1) == 1
    assert topk.stride(1) == 1 and block_table.stride(1) == 1
    assert offset + width <= output.shape[1]
    assert offset + width <= indices.shape[1]
    batch = output.shape[0]
    assert batch <= block_table.shape[0]
    _gather_context[(batch * width,)](
        cache,
        block_table,
        topk,
        source_rows,
        anchors,
        active,
        output,
        indices,
        CACHE_STRIDE=cache.stride(0),
        CACHE_BLOCK=block_size,
        NUM_BLOCKS=cache.shape[0],
        RECORD=cache.shape[-1],
        TABLE_STRIDE=block_table.stride(0),
        TABLE_COLS=block_table.shape[1],
        TOPK_STRIDE=topk.stride(0),
        TOPK_ROWS=topk.shape[0],
        OUTPUT_STRIDE=output.stride(0) // 512,
        INDEX_STRIDE=indices.stride(0),
        WIDTH=width,
        OFFSET=offset,
        WINDOW=window,
        num_warps=4,
    )
