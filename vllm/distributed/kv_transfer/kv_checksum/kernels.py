# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for the KV block checksums (see ``checksum.py``) of several
layers of one layout, in one launch."""

import torch

from vllm.triton_utils import tl, triton

# Pieces one program sums.
_CHUNK = 16 * 1024


# One variant per layout; the output may have any number of columns.
@triton.jit(do_not_specialize=["out_stride"])
def _layers_checksum_kernel(
    pieces_ptr,  # int [num_blocks, kernel_blocks, heads, tokens, pieces]
    stride_block,
    stride_kernel_block,
    stride_head,
    stride_token,
    stride_piece,
    layers_ptr,  # int64 [num_layers, 3]: base address, first slice, column
    block_ids_ptr,  # int64 [num_rows]
    first_tokens_ptr,  # int64 [num_rows]
    valid_tokens_ptr,  # int64 [num_rows]
    out_ptr,  # int64 [num_rows, num_columns], accumulated into
    out_stride,
    kernel_blocks: tl.constexpr,
    heads: tl.constexpr,
    tokens: tl.constexpr,
    num_pieces: tl.constexpr,
    PIECE_BITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_LAYER: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    layer = tl.program_id(2)
    if SINGLE_LAYER:
        # The cache's own pointer, whose alignment Triton knows: up to a third
        # faster than one loaded from the table.
        layer_ptr = pieces_ptr
    else:
        layer_ptr = tl.load(layers_ptr + 3 * layer).to(
            tl.pointer_type(pieces_ptr.dtype.element_ty)
        )
    first_slice = tl.load(layers_ptr + 3 * layer + 1).to(tl.uint64)
    column = tl.load(layers_ptr + 3 * layer + 2)
    block_id = tl.load(block_ids_ptr + row)
    first_token = tl.load(first_tokens_ptr + row).to(tl.uint64)
    valid_tokens = tl.load(valid_tokens_ptr + row)
    block_ptr = layer_ptr + block_id * stride_block
    slice_size = tokens * num_pieces
    num_pieces_in_block = kernel_blocks * heads * slice_size
    # uint64 arithmetic wraps around: mod 2**64.
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
        # Pieces narrower than 64 bits count as unsigned.
        x = x.to(tl.int64)
        if PIECE_BITS < 64:
            x = x & ((1 << PIECE_BITS) - 1)
        x = x.to(tl.uint64, bitcast=True)
        acc += x * (slice_weight * (2 * position + 1))
    # Chunks and layers add up in any order: integer addition is exact.
    tl.atomic_add(
        out_ptr + row * out_stride + column, tl.sum(acc).to(tl.int64, bitcast=True)
    )


def layers_checksum(
    pieces: torch.Tensor,
    layers: torch.Tensor,
    block_ids: torch.Tensor,
    first_tokens: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Add each layer's weighted piece sum of each block row into its column
    of ``out``.

    Args:
        pieces: The first layer's cache as integer pieces
            ``[num_blocks, kernel_blocks, heads, tokens, pieces]``, read as
            unsigned. Every layer's cache has its shape, strides and dtype.
        layers: ``[num_layers, 3]`` int64 on the device: each layer's cache
            address, ``layer * H + head`` of its first head, and its column
            of ``out``.
        block_ids: ``[n]`` int64 logical block ids.
        first_tokens: ``[n]`` int64 position in the request of each block's
            first token.
        valid_tokens: ``[n]`` int64 tokens to include from each block's start.
        out: ``[n, num_columns]`` int64 accumulator with unit column stride.

    """
    _, kernel_blocks, heads, tokens, num_pieces = pieces.shape
    num_pieces_in_block = kernel_blocks * heads * tokens * num_pieces
    grid = (block_ids.shape[0], triton.cdiv(num_pieces_in_block, _CHUNK), len(layers))
    _layers_checksum_kernel[grid](
        pieces,
        *pieces.stride(),
        layers,
        block_ids,
        first_tokens,
        valid_tokens,
        out,
        out.stride(0),
        kernel_blocks,
        heads,
        tokens,
        num_pieces,
        PIECE_BITS=8 * pieces.element_size(),
        CHUNK=_CHUNK,
        BLOCK=min(triton.next_power_of_2(num_pieces_in_block), 1024),
        SINGLE_LAYER=len(layers) == 1,
    )
