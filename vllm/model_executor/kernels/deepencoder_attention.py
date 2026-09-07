# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _deepencoder_rel_pos_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    rel_h_ptr,
    rel_w_ptr,
    out_ptr,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_rhb,
    stride_rhh,
    stride_rhm,
    stride_rhk,
    stride_rwb,
    stride_rwh,
    stride_rwm,
    stride_rwk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    key_width: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < num_tokens

    q_offsets = (
        batch * stride_qb
        + head * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_offsets, mask=mask_m[:, None], other=0.0)

    row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    qk_scale = scale * 1.4426950408889634

    for start_n in range(0, num_tokens, BLOCK_N):
        key_indices = start_n + offs_n
        mask_n = key_indices < num_tokens
        k_offsets = (
            batch * stride_kb
            + head * stride_kh
            + key_indices[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptr + k_offsets, mask=mask_n[None, :], other=0.0)
        scores = tl.dot(q, k) * qk_scale

        key_h = key_indices // key_width
        key_w = key_indices % key_width
        rel_h_offsets = (
            batch * stride_rhb
            + head * stride_rhh
            + offs_m[:, None] * stride_rhm
            + key_h[None, :] * stride_rhk
        )
        rel_w_offsets = (
            batch * stride_rwb
            + head * stride_rwh
            + offs_m[:, None] * stride_rwm
            + key_w[None, :] * stride_rwk
        )
        score_mask = mask_m[:, None] & mask_n[None, :]
        rel_h = tl.load(rel_h_ptr + rel_h_offsets, mask=score_mask, other=0.0)
        rel_w = tl.load(rel_w_ptr + rel_w_offsets, mask=score_mask, other=0.0)
        scores += (rel_h + rel_w) * 1.4426950408889634
        scores = tl.where(score_mask, scores, -float("inf"))

        next_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        alpha = tl.math.exp2(row_max - next_max)
        next_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator *= alpha[:, None]

        v_offsets = (
            batch * stride_vb
            + head * stride_vh
            + key_indices[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        value = tl.load(v_ptr + v_offsets, mask=mask_n[:, None], other=0.0)
        accumulator = tl.dot(probabilities.to(value.dtype), value, accumulator)
        row_max = next_max
        row_sum = next_sum

    output = accumulator / row_sum[:, None]
    out_offsets = (
        batch * stride_ob
        + head * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptr + out_offsets, output, mask=mask_m[:, None])


def deepencoder_rel_pos_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
    height: int,
    width: int,
    scale: float,
) -> torch.Tensor:
    """Run DeepEncoder global attention with fused relative-position bias."""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype == rel_h.dtype == rel_w.dtype
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    batch, heads, num_tokens, head_dim = q.shape
    assert head_dim == 64
    assert num_tokens == height * width
    assert rel_h.shape == (batch, heads, num_tokens, height)
    assert rel_w.shape == (batch, heads, num_tokens, width)

    output = torch.empty_like(q)
    block_m = 64
    block_n = 64
    grid = (triton.cdiv(num_tokens, block_m), batch * heads)
    _deepencoder_rel_pos_attention_kernel[grid](
        q,
        k,
        v,
        rel_h,
        rel_w,
        output,
        scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *rel_h.stride(),
        *rel_w.stride(),
        *output.stride(),
        num_tokens=num_tokens,
        num_heads=heads,
        key_width=width,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=64,
        num_warps=4,
        num_stages=2,
    )
    return output
