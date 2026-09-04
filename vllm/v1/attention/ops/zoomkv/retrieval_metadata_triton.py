# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused GPU metadata preparation for ZoomKV retrieval."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _actual_num_chunks_kernel(
    seq_lens_ptr,
    out_ptr,
    seq_lens_stride,
    sink_size,
    local_size,
    block_size,
    start_block,
    max_chunks,
    batch,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < batch
    seq_len = tl.load(
        seq_lens_ptr + offsets * seq_lens_stride, mask=mask, other=0
    ).to(tl.int32)
    # clamp(seq_len - sink, 0, local_size) followed by
    # max(seq_len - local_tokens, sink) simplifies to this expression.
    local_start = tl.maximum(sink_size, seq_len - local_size)
    actual = local_start // block_size - start_block
    actual = tl.maximum(0, tl.minimum(actual, max_chunks))
    tl.store(out_ptr + offsets, actual, mask=mask)


def build_actual_num_chunks(
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    sink_size: int,
    local_size: int,
    block_size: int,
    start_block: int,
    max_chunks: int,
) -> torch.Tensor:
    """Build per-request retrieval widths without generic elementwise ops."""
    if not seq_lens.is_cuda or not out.is_cuda:
        raise ValueError("seq_lens and out must be CUDA tensors")
    if seq_lens.dim() != 1 or out.dim() != 1:
        raise ValueError("seq_lens and out must be one-dimensional")
    if out.dtype != torch.int32:
        raise ValueError("out must have dtype torch.int32")
    if out.numel() < seq_lens.numel():
        raise ValueError("out is smaller than seq_lens")

    batch = seq_lens.numel()
    if batch == 0:
        return out[:0]
    block = 128
    _actual_num_chunks_kernel[(triton.cdiv(batch, block),)](
        seq_lens,
        out,
        seq_lens.stride(0),
        sink_size,
        local_size,
        block_size,
        start_block,
        max_chunks,
        batch,
        BLOCK=block,
    )
    return out[:batch]
