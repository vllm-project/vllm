# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable (CUDA SM75) FP8 MQA logits for the sparse indexer.

The DeepGEMM ``fp8_fp4_mqa_logits`` kernel is unavailable below SM90 because
its ``normal_kernel_cuda`` path does not implement fp8 (e4m3) matmul on Turing.
This module vendors a copy of the portable ``_fp8_mqa_logits_kernel`` from
``vllm/v1/attention/ops/triton_fp8_mqa_logits.py`` (same body, renamed to the
fp16-typed ``_fp16_mqa_logits_kernel``) with a CUDA launcher.

Unlike the AMD ``fp8_mqa_logits_gfx942`` launcher, Triton on SM75 cannot even
*load* ``float8_e4m3fn`` tensors (``fp8e4nv`` is unsupported on this
architecture), so the fp8 values are converted losslessly to fp16 before the
dot (e4m3 has 3 mantissa bits, fp16 has 10, so every fp8 value is exact).
``tl.dot(..., input_precision="ieee")`` keeps a full-precision accumulation
path instead of the fp16 tensor-core path.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

__all__ = ["fp8_mqa_logits_triton"]


@triton.jit
def _fp16_mqa_logits_kernel(
    Q_ptr,  # fp16 [seq_len, H, D]
    KV_ptr,  # fp16 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )
    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets
    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    scores = scores * kv_scales[None, :]
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = tl.sum(scores, axis=0)
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits on CUDA via the portable Triton kernel.

    Drop-in replacement for the DeepGEMM ``fp8_fp4_mqa_logits`` contract on
    SM75 (Turing): FP8 Q/K values with fp32 K scales, per-head weights and
    per-row start/end windows.

    Args:
        q: Query tensor of shape ``[M, H, D]``, FP8 dtype.
        k_fp8: Key tensor of shape ``[N, D]``, FP8 dtype.
        kv_scales: K scales of shape ``[N]`` (or ``[N, 1]`` -- viewed as
            ``[N]``), float32.
        weights: Per-head weights of shape ``[M, H]``, float32.
        cu_starts: Start indices (inclusive) of shape ``[M]``, int32.
        cu_ends: End indices (exclusive) of shape ``[M]``, int32.

    Returns:
        Logits of shape ``[M, N]``, float32 -- positions outside
        ``[cu_starts[i], cu_ends[i])`` for row ``i`` are ``-inf``.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    # fp8e4nv is not loadable inside Triton on SM75; e4m3 -> fp16 is lossless.
    q_f16 = q.to(torch.float16).contiguous()
    k_f16 = k_fp8.to(torch.float16).contiguous()
    kv_scales_1d = kv_scales.reshape(-1)

    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # fp16 KV tiles are 2 bytes/elem, so a (BLOCK_KV=128, num_stages=2)
    # double-buffered tile (64 KiB) overshoots Turing's 64 KiB LDS budget.
    # (64, 2) keeps the double-buffer pipeline while using ~32 KiB.
    _fp16_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q_f16,
        KV_ptr=k_f16,
        kv_scales_ptr=kv_scales_1d,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=q_f16.stride(0),
        stride_q_h=q_f16.stride(1),
        stride_q_d=q_f16.stride(2),
        stride_kv_s=k_f16.stride(0),
        stride_kv_d=k_f16.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_KV=64,
        num_warps=4,
        num_stages=2,
    )

    return logits
