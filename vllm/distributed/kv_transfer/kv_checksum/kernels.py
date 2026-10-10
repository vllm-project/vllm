# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for one layer's part of KV block checksums: the sums of
``_LayerView.accumulate``'s PyTorch path."""

import torch

from vllm.triton_utils import tl, triton

# Pieces one program sums.
_CHUNK = 16 * 1024


# One variant per layer shape; the output column may be any strided view.
@triton.jit(do_not_specialize=["first_slice", "out_ptr", "out_stride"])
def _layer_checksum_kernel(
    pieces_ptr,  # int64 [num_blocks, kernel_blocks, heads, tokens, pieces]
    stride_block,
    stride_kernel_block,
    stride_head,
    stride_token,
    stride_piece,
    block_ids_ptr,  # int64 [num_rows]
    first_tokens_ptr,  # int64 [num_rows]
    valid_tokens_ptr,  # int64 [num_rows]
    out_ptr,  # int64 [num_rows], accumulated into
    out_stride,
    first_slice,
    kernel_blocks: tl.constexpr,
    heads: tl.constexpr,
    tokens: tl.constexpr,
    num_pieces: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    block_id = tl.load(block_ids_ptr + row)
    first_token = tl.load(first_tokens_ptr + row).to(tl.uint64)
    valid_tokens = tl.load(valid_tokens_ptr + row)
    block_ptr = pieces_ptr + block_id * stride_block
    slice_size = tokens * num_pieces
    num_pieces_in_block = kernel_blocks * heads * slice_size
    # uint64 arithmetic wraps around: mod 2**64.
    first_slice = first_slice.to(tl.uint64)
    acc = tl.zeros((BLOCK,), dtype=tl.uint64)
    chunk_start = chunk * CHUNK
    chunk_end = tl.minimum(chunk_start + CHUNK, num_pieces_in_block)
    for start in range(chunk_start, chunk_end, BLOCK):
        # One flat index over [kernel_block, head, token, piece].
        offsets = start + tl.arange(0, BLOCK)
        piece = offsets % num_pieces
        token = offsets // num_pieces % tokens
        head = offsets // slice_size % heads
        kernel_block = offsets // (heads * slice_size)
        token_index = kernel_block * tokens + token
        mask = (offsets < chunk_end) & (token_index < valid_tokens)
        x = tl.load(
            block_ptr
            + kernel_block.to(tl.int64) * stride_kernel_block
            + head.to(tl.int64) * stride_head
            + token.to(tl.int64) * stride_token
            + piece.to(tl.int64) * stride_piece,
            mask=mask,
            other=0,
        )
        slice_weight = 2 * (first_slice + head.to(tl.uint64)) + 1
        position = (first_token + token_index.to(tl.uint64)) * num_pieces + piece.to(
            tl.uint64
        )
        acc += x.to(tl.uint64, bitcast=True) * (slice_weight * (2 * position + 1))
    # Chunks of a row add up in any order: integer addition is exact.
    tl.atomic_add(out_ptr + row * out_stride, tl.sum(acc).to(tl.int64, bitcast=True))


def layer_checksum(
    pieces: torch.Tensor,
    first_slice: int,
    block_ids: torch.Tensor,
    first_tokens: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Add one layer's weighted piece sum of each block row into ``out``.

    Args:
        pieces: The layer's cache as int64
            ``[num_blocks, kernel_blocks, heads, tokens, pieces]``.
        first_slice: ``layer * H + head`` of the first head in ``pieces``.
        block_ids: ``[n]`` int64 logical block ids.
        first_tokens: ``[n]`` int64 position in the request of each block's
            first token.
        valid_tokens: ``[n]`` int64 tokens to include from each block's start.
        out: ``[n]`` int64 accumulator.

    """
    _, kernel_blocks, heads, tokens, num_pieces = pieces.shape
    num_pieces_in_block = kernel_blocks * heads * tokens * num_pieces
    grid = (block_ids.shape[0], triton.cdiv(num_pieces_in_block, _CHUNK))
    _layer_checksum_kernel[grid](
        pieces,
        *pieces.stride(),
        block_ids,
        first_tokens,
        valid_tokens,
        out,
        out.stride(0),
        first_slice,
        kernel_blocks,
        heads,
        tokens,
        num_pieces,
        CHUNK=_CHUNK,
        BLOCK=min(triton.next_power_of_2(num_pieces_in_block), 1024),
    )
