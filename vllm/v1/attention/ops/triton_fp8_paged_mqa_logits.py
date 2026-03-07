# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for FP8 paged MQA logits."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,  # [B, next_n, H, D], float16
    k_data_ptr,  # [num_blocks, block_size, D], uint8(fp8 bitcast)
    k_scale_ptr,  # [num_blocks, block_size], float32
    weights_ptr,  # [B * next_n, H], float32
    context_lens_ptr,  # [B], int32
    block_tables_ptr,  # [B, max_blocks], int32/int64
    logits_ptr,  # [B * next_n, max_model_len], float32
    next_n,
    num_heads,
    head_dim,
    block_size,
    max_model_len,
    # q strides
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    # k_data strides
    stride_kd_blk,
    stride_kd_pos,
    stride_kd_d,
    # k_scale strides
    stride_ks_blk,
    stride_ks_pos,
    # weights strides
    stride_w_row,
    stride_w_h,
    # context_lens stride
    stride_ctx,
    # block_tables strides
    stride_bt_b,
    stride_bt_blk,
    # logits strides
    stride_o_row,
    stride_o_col,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)

    b = pid_row // next_n
    n = pid_row % next_n

    context_len = tl.load(context_lens_ptr + b * stride_ctx)
    q_pos = context_len - next_n + n

    block_start = pid_block * block_size

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)

    kv_pos = block_start + offs_k
    in_block = offs_k < block_size
    in_ctx = kv_pos < context_len
    out_bounds = in_block & (kv_pos < max_model_len)

    block_active = block_start < context_len
    physical_block_id = tl.full((), 0, dtype=tl.int64)
    if block_active:
        physical_block_id = tl.load(
            block_tables_ptr + b * stride_bt_b + pid_block * stride_bt_blk
        ).to(tl.int64)

    token_valid = block_active & in_block & in_ctx

    # Load K tile [BLOCK_K, BLOCK_D] from packed FP8 bytes.
    k_ptrs = (
        k_data_ptr
        + physical_block_id * stride_kd_blk
        + offs_k[:, None] * stride_kd_pos
        + offs_d[None, :] * stride_kd_d
    )
    k_mask = token_valid[:, None] & (offs_d[None, :] < head_dim)
    k_u8 = tl.load(k_ptrs, mask=k_mask, other=0).to(tl.uint8)
    k_vals = k_u8.to(tl.float8e4nv, bitcast=True).to(tl.float16)

    # Load scales [BLOCK_K] and apply dequantization in fp16 for MMA throughput.
    k_scale = tl.load(
        k_scale_ptr + physical_block_id * stride_ks_blk + offs_k * stride_ks_pos,
        mask=token_valid,
        other=0.0,
    ).to(tl.float16)
    k_vals = k_vals * k_scale[:, None]

    # Load Q tile [BLOCK_H, BLOCK_D].
    q_ptrs = (
        q_ptr
        + b * stride_q_b
        + n * stride_q_n
        + offs_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q_mask = (offs_h[:, None] < num_heads) & (offs_d[None, :] < head_dim)
    q_vals = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)

    # Weights [BLOCK_H].
    w = tl.load(
        weights_ptr + pid_row * stride_w_row + offs_h * stride_w_h,
        mask=offs_h < num_heads,
        other=0.0,
    )

    # scores: [BLOCK_H, BLOCK_K] = Q @ K^T
    scores = tl.dot(q_vals, tl.trans(k_vals))
    scores = tl.maximum(scores, 0.0) * w[:, None]
    acc = tl.sum(scores, axis=0)

    causal = kv_pos <= q_pos
    write_valid = token_valid & causal
    out_vals = tl.where(write_valid, acc, float("-inf"))

    out_ptrs = logits_ptr + pid_row * stride_o_row + kv_pos * stride_o_col
    tl.store(out_ptrs, out_vals, mask=out_bounds)


def fp8_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 paged MQA logits using Triton."""
    if not q_fp8.is_cuda:
        raise ValueError("fp8_paged_mqa_logits_triton requires CUDA tensors")

    if q_fp8.ndim != 4:
        raise ValueError(f"q_fp8 must be 4D [B, next_n, H, D], got {q_fp8.shape}")
    if kv_cache_fp8.ndim != 4:
        raise ValueError(
            f"kv_cache_fp8 must be 4D [num_blocks, block_size, 1, D+4], "
            f"got {kv_cache_fp8.shape}"
        )

    batch_size, next_n, num_heads, head_dim = q_fp8.shape
    num_blocks, block_size, _, packed_dim = kv_cache_fp8.shape

    if packed_dim != head_dim + 4:
        raise ValueError(
            f"kv_cache_fp8 last dim must be head_dim + 4 ({head_dim + 4}), "
            f"got {packed_dim}"
        )

    # DeepGEMM-compatible packed layout expects contiguous memory.
    if not kv_cache_fp8.is_contiguous():
        kv_cache_fp8 = kv_cache_fp8.contiguous()

    # Convert Q once outside the kernel to avoid repeated per-block conversion.
    q = q_fp8.to(torch.float16)

    # Split fused KV cache as zero-copy views:
    #   [num_blocks, block_size * D] uint8 FP8 bytes
    #   [num_blocks, block_size] float32 scales
    kv_flat = kv_cache_fp8.view(num_blocks, -1)
    split = block_size * head_dim
    k_data = kv_flat[:, :split].view(num_blocks, block_size, head_dim)
    k_scale = kv_flat[:, split:].view(num_blocks, block_size, 4).view(torch.float32)

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    block_cols = block_tables.shape[1]
    grid = (batch_size * next_n, block_cols)

    block_k = triton.next_power_of_2(block_size)
    block_d = triton.next_power_of_2(head_dim)
    block_h = triton.next_power_of_2(max(1, num_heads))

    # Heuristics tuned for HxD(<=64x128)-by-KV(64) decode tiles.
    num_warps = 8 if (block_d >= 128 or block_h >= 64) else 4

    _fp8_paged_mqa_logits_kernel[grid](
        q,
        k_data,
        k_scale,
        weights,
        context_lens,
        block_tables,
        logits,
        next_n,
        num_heads,
        head_dim,
        block_size,
        max_model_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_data.stride(0),
        k_data.stride(1),
        k_data.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=2,
    )

    return logits
