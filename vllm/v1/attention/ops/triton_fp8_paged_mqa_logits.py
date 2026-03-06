# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for FP8 paged MQA logits.

This kernel mirrors ``_fp8_paged_mqa_logits_torch_impl`` in
``vllm/utils/deep_gemm.py`` while remaining CUDA Graph friendly:
- single Triton launch (no Python loops over batches/blocks),
- fixed output shape with masked writes,
- on-the-fly FP8 + per-token scale dequantization in device code.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,  # [B, next_n, H, D], float8
    kv_cache_ptr,  # [num_blocks, block_size, 1, D+4], uint8
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
    # kv_cache stride
    stride_kv_block,
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

    kv_pos = block_start + offs_k
    in_block = offs_k < block_size
    in_ctx = kv_pos < context_len
    out_bounds = in_block & (kv_pos < max_model_len)

    # Blocks at/after context_len are fully invalid; keep default -inf.
    block_active = block_start < context_len

    physical_block_id = tl.full((), 0, dtype=tl.int64)
    if block_active:
        physical_block_id = tl.load(
            block_tables_ptr + b * stride_bt_b + pid_block * stride_bt_blk
        ).to(tl.int64)

    token_valid = block_active & in_block & in_ctx

    # kv_cache layout is split in each block:
    # [block_size * head_dim bytes of FP8 K] + [block_size * 4 bytes of FP32 scale]
    # then viewed as [block_size, 1, head_dim + 4].
    block_base = kv_cache_ptr + physical_block_id * stride_kv_block

    # Load per-token FP8 K values from the K region.
    k_ptrs = block_base + offs_k[:, None] * head_dim + offs_d[None, :]
    k_mask = token_valid[:, None] & (offs_d[None, :] < head_dim)
    k_u8 = tl.load(k_ptrs, mask=k_mask, other=0).to(tl.uint8)
    k_vals = k_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    # Reconstruct float32 scale from 4 uint8 bytes at tail of each token.
    scale_base = block_base + block_size * head_dim + offs_k * 4
    b0 = tl.load(scale_base + 0, mask=token_valid, other=0).to(tl.uint32)
    b1 = tl.load(scale_base + 1, mask=token_valid, other=0).to(tl.uint32)
    b2 = tl.load(scale_base + 2, mask=token_valid, other=0).to(tl.uint32)
    b3 = tl.load(scale_base + 3, mask=token_valid, other=0).to(tl.uint32)
    scale_bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    k_scale = scale_bits.to(tl.float32, bitcast=True)
    k_scale = tl.where(token_valid, k_scale, 0.0)

    k_vals = k_vals * k_scale[:, None]

    # Accumulate logits over heads:
    # logits[pos] = sum_h relu(dot(q[h], k[pos])) * weight[h]
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    q_row_base = q_ptr + b * stride_q_b + n * stride_q_n
    w_row_base = weights_ptr + pid_row * stride_w_row

    for h in range(BLOCK_H):
        if h < num_heads:
            q_ptrs = q_row_base + h * stride_q_h + offs_d * stride_q_d
            q_vals = tl.load(q_ptrs, mask=offs_d < head_dim, other=0.0).to(tl.float32)
            w = tl.load(w_row_base + h * stride_w_h)

            scores = tl.sum(q_vals[None, :] * k_vals, axis=1)
            scores = tl.maximum(scores, 0.0) * w
            acc += scores

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
    """Compute FP8 paged MQA logits using Triton.

    Semantics match ``_fp8_paged_mqa_logits_torch_impl``.
    """
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
    _, block_size, _, packed_dim = kv_cache_fp8.shape

    if packed_dim != head_dim + 4:
        raise ValueError(
            f"kv_cache_fp8 last dim must be head_dim + 4 ({head_dim + 4}), "
            f"got {packed_dim}"
        )

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

    num_warps = 4
    if block_d >= 128:
        num_warps = 8

    _fp8_paged_mqa_logits_kernel[grid](
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        logits,
        next_n,
        num_heads,
        head_dim,
        block_size,
        max_model_len,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        kv_cache_fp8.stride(0),
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
        num_stages=1,
    )

    return logits
