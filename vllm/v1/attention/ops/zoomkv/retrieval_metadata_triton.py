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


@triton.jit
def _stage_budget_kernel(
    actual_chunks_ptr,
    parent_lengths_ptr,
    large_ks_ptr,
    sub_lengths_ptr,
    small_ks_ptr,
    dense_ks_ptr,
    final_ks_ptr,
    batch,
    heads,
    factor,
    large_ratio,
    small_ratio,
    dense_ratio,
    max_large,
    max_small,
    dense_topk,
    sparse_topk,
    final_topk,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = batch * heads
    mask = offsets < rows
    batch_idx = offsets // heads
    actual = tl.load(actual_chunks_ptr + batch_idx, mask=mask, other=0)

    parent_len = (actual + factor - 1) // factor
    large_k = tl.ceil(parent_len.to(tl.float32) * large_ratio).to(tl.int32)
    large_k = tl.where(parent_len > 0, tl.maximum(1, large_k), 0)
    large_k = tl.minimum(large_k, max_large)

    sub_len = large_k * factor
    # A selected partial parent can create up to factor-1 internal holes. The
    # small-stage ratio is applied to the request's reachable child capacity;
    # physical kernels still mark any partial-parent holes as -inf.
    small_capacity = tl.minimum(actual, sub_len)
    small_k = tl.ceil(small_capacity.to(tl.float32) * small_ratio).to(tl.int32)
    small_k = tl.where(small_capacity > 0, tl.maximum(1, small_k), 0)
    small_k = tl.minimum(small_k, max_small)

    dense_k = (small_k.to(tl.float32) * dense_ratio).to(tl.int32)
    dense_k = tl.where(small_k > 0, tl.maximum(1, dense_k), 0)
    dense_k = tl.minimum(dense_k, small_k)
    final_candidates = (
        dense_k * dense_topk + (small_k - dense_k) * sparse_topk
    )
    final_k = tl.minimum(final_candidates, final_topk)

    tl.store(parent_lengths_ptr + offsets, parent_len, mask=mask)
    tl.store(large_ks_ptr + offsets, large_k, mask=mask)
    tl.store(sub_lengths_ptr + offsets, sub_len, mask=mask)
    tl.store(small_ks_ptr + offsets, small_k, mask=mask)
    tl.store(dense_ks_ptr + offsets, dense_k, mask=mask)
    tl.store(final_ks_ptr + offsets, final_k, mask=mask)


def build_stage_budgets(
    actual_num_chunks: torch.Tensor,
    parent_lengths: torch.Tensor,
    large_ks: torch.Tensor,
    sub_lengths: torch.Tensor,
    small_ks: torch.Tensor,
    dense_ks: torch.Tensor,
    final_ks: torch.Tensor,
    *,
    factor: int,
    large_ratio: float,
    small_ratio: float,
    dense_ratio: float,
    max_large: int,
    max_small: int,
    dense_topk: int,
    sparse_topk: int,
    final_topk: int,
) -> None:
    """Build fixed-shape, per-request/head retrieval budgets on the GPU."""
    outputs = (
        parent_lengths,
        large_ks,
        sub_lengths,
        small_ks,
        dense_ks,
        final_ks,
    )
    if not actual_num_chunks.is_cuda or any(not out.is_cuda for out in outputs):
        raise ValueError("stage budget tensors must be CUDA tensors")
    if actual_num_chunks.dtype != torch.int32 or any(
        out.dtype != torch.int32 for out in outputs
    ):
        raise ValueError("stage budget tensors must have dtype torch.int32")
    shape = parent_lengths.shape
    if len(shape) != 2 or any(out.shape != shape for out in outputs):
        raise ValueError("stage budget outputs must share shape [batch, heads]")
    if shape[0] != actual_num_chunks.numel():
        raise ValueError("stage budget batch does not match actual_num_chunks")
    rows = shape[0] * shape[1]
    if rows == 0:
        return
    block = 128
    _stage_budget_kernel[(triton.cdiv(rows, block),)](
        actual_num_chunks,
        *outputs,
        shape[0],
        shape[1],
        factor,
        large_ratio,
        small_ratio,
        dense_ratio,
        max_large,
        max_small,
        dense_topk,
        sparse_topk,
        final_topk,
        BLOCK=block,
    )
