# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton norm, RoPE, Per-Layer Embedding (PLE), and MoE custom ops
for Gemma 4."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _fused_qkv_norm_rope_kernel(
    qkv_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    wq_ptr,
    wk_ptr,
    out_q_ptr,
    out_k_ptr,
    out_v_ptr,
    stride_qkv_m,
    stride_cache_pos,
    stride_out_q_m,
    stride_out_k_m,
    stride_out_v_m,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    eps: tl.constexpr,
    is_kv_shared: tl.constexpr,
    is_k_eq_v: tl.constexpr = False,
):
    """Fuses QKV split + Q/K/V RMSNorm + Neox RoPE in a single launch.

    Grid: (num_q_heads + 2 * num_kv_heads, M) when not is_kv_shared and not is_k_eq_v,
          (num_q_heads + num_kv_heads, M) when is_k_eq_v,
          or (num_q_heads, M) when is_kv_shared.
    """
    head_idx = tl.program_id(0)
    m_idx = tl.program_id(1)

    offs_half = tl.arange(0, BLOCK_HALF)
    half_mask = offs_half < HALF_DIM

    pos = tl.load(positions_ptr + m_idx)
    cos_ptrs = cos_sin_cache_ptr + pos * stride_cache_pos + offs_half
    sin_ptrs = cos_sin_cache_ptr + pos * stride_cache_pos + (HALF_DIM + offs_half)
    cos = tl.load(cos_ptrs, mask=half_mask, other=1.0)
    sin = tl.load(sin_ptrs, mask=half_mask, other=0.0)

    q_size: tl.constexpr = num_q_heads * HEAD_DIM
    kv_size: tl.constexpr = num_kv_heads * HEAD_DIM

    if head_idx < num_q_heads:
        # Query head: RMSNorm with wq + Neox RoPE
        head_off = head_idx * HEAD_DIM
        x1 = tl.load(
            qkv_ptr + m_idx * stride_qkv_m + head_off + offs_half,
            mask=half_mask,
            other=0.0,
        )
        x2 = tl.load(
            qkv_ptr + m_idx * stride_qkv_m + head_off + HALF_DIM + offs_half,
            mask=half_mask,
            other=0.0,
        )

        x1_f32 = x1.to(tl.float32)
        x2_f32 = x2.to(tl.float32)
        sum_sq = tl.sum(x1_f32 * x1_f32 + x2_f32 * x2_f32, axis=0)
        var = sum_sq / HEAD_DIM
        r = tl.rsqrt(var + eps)

        # Exact BF16 RMSNorm parity rule: cast before weight multiplication
        x1_norm = (x1_f32 * r).to(tl.bfloat16)
        x2_norm = (x2_f32 * r).to(tl.bfloat16)

        w1 = tl.load(wq_ptr + offs_half, mask=half_mask, other=1.0)
        w2 = tl.load(wq_ptr + HALF_DIM + offs_half, mask=half_mask, other=1.0)
        x1_norm = x1_norm * w1
        x2_norm = x2_norm * w2

        # Neox-style RoPE rotation: o1 = x1*cos - x2*sin, o2 = x2*cos + x1*sin in FP32
        cos_f32 = cos.to(tl.float32)
        sin_f32 = sin.to(tl.float32)
        x1_norm_f32 = x1_norm.to(tl.float32)
        x2_norm_f32 = x2_norm.to(tl.float32)
        o1 = (x1_norm_f32 * cos_f32 - x2_norm_f32 * sin_f32).to(tl.bfloat16)
        o2 = (x2_norm_f32 * cos_f32 + x1_norm_f32 * sin_f32).to(tl.bfloat16)

        out_off = m_idx * stride_out_q_m + head_off
        tl.store(out_q_ptr + out_off + offs_half, o1, mask=half_mask)
        tl.store(out_q_ptr + out_off + HALF_DIM + offs_half, o2, mask=half_mask)

    elif not is_kv_shared and head_idx < num_q_heads + num_kv_heads:
        # Key head: RMSNorm with wk + Neox RoPE
        # If is_k_eq_v, also store RMSNorm(k, eps, has_weight=False) into out_v
        kv_idx = head_idx - num_q_heads
        head_off = q_size + kv_idx * HEAD_DIM
        x1 = tl.load(
            qkv_ptr + m_idx * stride_qkv_m + head_off + offs_half,
            mask=half_mask,
            other=0.0,
        )
        x2 = tl.load(
            qkv_ptr + m_idx * stride_qkv_m + head_off + HALF_DIM + offs_half,
            mask=half_mask,
            other=0.0,
        )

        x1_f32 = x1.to(tl.float32)
        x2_f32 = x2.to(tl.float32)
        sum_sq = tl.sum(x1_f32 * x1_f32 + x2_f32 * x2_f32, axis=0)
        var = sum_sq / HEAD_DIM
        r = tl.rsqrt(var + eps)

        x1_norm = (x1_f32 * r).to(tl.bfloat16)
        x2_norm = (x2_f32 * r).to(tl.bfloat16)

        if is_k_eq_v:
            out_v_off = m_idx * stride_out_v_m + kv_idx * HEAD_DIM
            tl.store(out_v_ptr + out_v_off + offs_half, x1_norm, mask=half_mask)
            tl.store(
                out_v_ptr + out_v_off + HALF_DIM + offs_half, x2_norm, mask=half_mask
            )

        w1 = tl.load(wk_ptr + offs_half, mask=half_mask, other=1.0)
        w2 = tl.load(wk_ptr + HALF_DIM + offs_half, mask=half_mask, other=1.0)
        x1_norm = x1_norm * w1
        x2_norm = x2_norm * w2

        cos_f32 = cos.to(tl.float32)
        sin_f32 = sin.to(tl.float32)
        x1_norm_f32 = x1_norm.to(tl.float32)
        x2_norm_f32 = x2_norm.to(tl.float32)
        o1 = (x1_norm_f32 * cos_f32 - x2_norm_f32 * sin_f32).to(tl.bfloat16)
        o2 = (x2_norm_f32 * cos_f32 + x1_norm_f32 * sin_f32).to(tl.bfloat16)

        out_off = m_idx * stride_out_k_m + kv_idx * HEAD_DIM
        tl.store(out_k_ptr + out_off + offs_half, o1, mask=half_mask)
        tl.store(out_k_ptr + out_off + HALF_DIM + offs_half, o2, mask=half_mask)

    elif not is_kv_shared and not is_k_eq_v:
        # Value head: RMSNorm (has_weight=False), NO RoPE
        v_idx = head_idx - (num_q_heads + num_kv_heads)
        head_off = q_size + kv_size + v_idx * HEAD_DIM
        x1 = tl.load(
            qkv_ptr + m_idx * stride_qkv_m + head_off + offs_half,
            mask=half_mask,
            other=0.0,
        )
        x2 = tl.load(
            qkv_ptr + m_idx * stride_qkv_m + head_off + HALF_DIM + offs_half,
            mask=half_mask,
            other=0.0,
        )

        x1_f32 = x1.to(tl.float32)
        x2_f32 = x2.to(tl.float32)
        sum_sq = tl.sum(x1_f32 * x1_f32 + x2_f32 * x2_f32, axis=0)
        var = sum_sq / HEAD_DIM
        r = tl.rsqrt(var + eps)

        x1_norm = (x1_f32 * r).to(tl.bfloat16)
        x2_norm = (x2_f32 * r).to(tl.bfloat16)

        out_off = m_idx * stride_out_v_m + v_idx * HEAD_DIM
        tl.store(out_v_ptr + out_off + offs_half, x1_norm, mask=half_mask)
        tl.store(out_v_ptr + out_off + HALF_DIM + offs_half, x2_norm, mask=half_mask)


@triton.jit
def _fused_post_attn_add_pre_ff_norm_kernel(
    attn_out_ptr,
    residual_ptr,
    post_w_ptr,
    pre_w_ptr,
    out_pre_ff_ptr,
    out_res_ptr,
    stride_attn_m,
    stride_res_m,
    stride_pre_m,
    stride_out_res_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_attn_norm + residual add + pre_ff_norm in 1 launch.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # 1. Load attn_out and post_w
    attn = tl.load(attn_out_ptr + m * stride_attn_m + offs, mask=mask, other=0.0)
    attn_f32 = attn.to(tl.float32)
    var1 = tl.sum(attn_f32 * attn_f32, axis=0) / H
    r1 = tl.rsqrt(var1 + eps)
    normed_attn = (attn_f32 * r1).to(tl.bfloat16)

    post_w = tl.load(post_w_ptr + offs, mask=mask, other=0.0)
    normed_attn = normed_attn * post_w

    # 2. Load residual, add in registers, store to out_res
    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    new_res = normed_attn + res
    tl.store(out_res_ptr + m * stride_out_res_m + offs, new_res, mask=mask)

    # 3. Pre-FF norm of new_res in registers
    new_res_f32 = new_res.to(tl.float32)
    var2 = tl.sum(new_res_f32 * new_res_f32, axis=0) / H
    r2 = tl.rsqrt(var2 + eps)
    normed_res = (new_res_f32 * r2).to(tl.bfloat16)

    pre_w = tl.load(pre_w_ptr + offs, mask=mask, other=0.0)
    pre_ff = normed_res * pre_w
    tl.store(out_pre_ff_ptr + m * stride_pre_m + offs, pre_ff, mask=mask)


@triton.jit
def _fused_post_ff_norm_add_scalar_kernel(
    mlp_out_ptr,
    residual_ptr,
    post_ff_w_ptr,
    layer_scalar_ptr,
    out_ptr,
    stride_mlp_m,
    stride_res_m,
    stride_out_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm + residual add + layer_scalar (dense without PLE).

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs, mask=mask, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_mlp = (mlp_f32 * r).to(tl.bfloat16)

    post_w = tl.load(post_ff_w_ptr + offs, mask=mask, other=0.0)
    normed_mlp = normed_mlp * post_w

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_mlp + res

    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    out = (h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_ptr + m * stride_out_m + offs, out, mask=mask)


@triton.jit
def _fused_post_ff_norm_add_kernel(
    mlp_out_ptr,
    residual_ptr,
    post_ff_w_ptr,
    out_h_ptr,
    stride_mlp_m,
    stride_res_m,
    stride_h_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm and residual add into h for dense PLE step 1.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs, mask=mask, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_mlp = (mlp_f32 * r).to(tl.bfloat16)

    post_w = tl.load(post_ff_w_ptr + offs, mask=mask, other=0.0)
    normed_mlp = normed_mlp * post_w

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_mlp + res
    tl.store(out_h_ptr + m * stride_h_m + offs, h, mask=mask)


@triton.jit
def _fused_gelu_tanh_mul_kernel(
    gate_raw_ptr,
    ple_in_ptr,
    out_ptr,
    stride_gate_m,
    stride_ple_m,
    stride_ple_d,
    stride_out_m,
    PLE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuses F.gelu(gate_raw, approximate='tanh') * per_layer_input directly.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < PLE_DIM

    x = tl.load(gate_raw_ptr + m * stride_gate_m + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    c: tl.constexpr = 0.7978845608028654
    arg = c * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
    th = tl.extra.cuda.libdevice.tanh(arg)
    gelu_val = 0.5 * x_f32 * (1.0 + th)

    ple = tl.load(
        ple_in_ptr + m * stride_ple_m + offs * stride_ple_d,
        mask=mask,
        other=0.0,
    )
    gated_ple = (gelu_val * ple.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + m * stride_out_m + offs, gated_ple, mask=mask)


@triton.jit
def _fused_post_ple_norm_add_scalar_kernel(
    ple_proj_ptr,
    h_ptr,
    post_ple_w_ptr,
    layer_scalar_ptr,
    out_ptr,
    stride_proj_m,
    stride_h_m,
    stride_out_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ple_norm(ple_proj) + h and * layer_scalar in registers.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    proj = tl.load(ple_proj_ptr + m * stride_proj_m + offs, mask=mask, other=0.0)
    proj_f32 = proj.to(tl.float32)
    var = tl.sum(proj_f32 * proj_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_proj = (proj_f32 * r).to(tl.bfloat16)

    w = tl.load(post_ple_w_ptr + offs, mask=mask, other=0.0)
    normed_proj = normed_proj * w

    h = tl.load(h_ptr + m * stride_h_m + offs, mask=mask, other=0.0)
    final_h = normed_proj + h

    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    out = (final_h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_ptr + m * stride_out_m + offs, out, mask=mask)


@triton.jit
def _fused_mlp_norm_and_moe_prenorm_kernel(
    mlp_out_ptr,
    residual_ptr,
    w_post1_ptr,
    w_pre2_ptr,
    out_h1_ptr,
    out_moe_in_ptr,
    stride_mlp_m,
    stride_res_m,
    stride_h1_m,
    stride_moe_in_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm_1(mlp_out) and pre_ff_norm_2(residual) simultaneously.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # 1. post_feedforward_layernorm_1 on mlp_out
    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs, mask=mask, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var1 = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r1 = tl.rsqrt(var1 + eps)
    normed_mlp = (mlp_f32 * r1).to(tl.bfloat16)

    w1 = tl.load(w_post1_ptr + offs, mask=mask, other=0.0)
    h1 = normed_mlp * w1
    tl.store(out_h1_ptr + m * stride_h1_m + offs, h1, mask=mask)

    # 2. pre_feedforward_layernorm_2 on residual
    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    res_f32 = res.to(tl.float32)
    var2 = tl.sum(res_f32 * res_f32, axis=0) / H
    r2 = tl.rsqrt(var2 + eps)
    normed_res = (res_f32 * r2).to(tl.bfloat16)

    w2 = tl.load(w_pre2_ptr + offs, mask=mask, other=0.0)
    moe_in = normed_res * w2
    tl.store(out_moe_in_ptr + m * stride_moe_in_m + offs, moe_in, mask=mask)


@triton.jit
def _fused_moe_combine_norm_add_scalar_kernel(
    h1_ptr,
    moe_out_ptr,
    residual_ptr,
    w_post2_ptr,
    w_post_ff_ptr,
    layer_scalar_ptr,
    out_ptr,
    stride_h1_m,
    stride_moe_m,
    stride_res_m,
    stride_out_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm_2, combination with h1, post_ff_norm, residual add,
    and layer scalar multiplication.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # 1. post_ff_norm_2 on moe_out
    moe = tl.load(moe_out_ptr + m * stride_moe_m + offs, mask=mask, other=0.0)
    moe_f32 = moe.to(tl.float32)
    var_m = tl.sum(moe_f32 * moe_f32, axis=0) / H
    r_m = tl.rsqrt(var_m + eps)
    normed_moe = (moe_f32 * r_m).to(tl.bfloat16)

    w_post2 = tl.load(w_post2_ptr + offs, mask=mask, other=0.0)
    normed_moe = normed_moe * w_post2

    # 2. Combine with h1
    h1 = tl.load(h1_ptr + m * stride_h1_m + offs, mask=mask, other=0.0)
    comb = h1 + normed_moe

    # 3. post_feedforward_layernorm on combined
    comb_f32 = comb.to(tl.float32)
    var_c = tl.sum(comb_f32 * comb_f32, axis=0) / H
    r_c = tl.rsqrt(var_c + eps)
    normed_c = (comb_f32 * r_c).to(tl.bfloat16)

    w_post_ff = tl.load(w_post_ff_ptr + offs, mask=mask, other=0.0)
    normed_c = normed_c * w_post_ff

    # 4. Residual add
    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_c + res

    # 5. Layer scalar
    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    out = (h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_ptr + m * stride_out_m + offs, out, mask=mask)


@triton.jit
def _fused_moe_combine_norm_add_kernel(
    h1_ptr,
    moe_out_ptr,
    residual_ptr,
    w_post2_ptr,
    w_post_ff_ptr,
    out_h_ptr,
    stride_h1_m,
    stride_moe_m,
    stride_res_m,
    stride_h_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm_2, combination with h1, post_ff_norm, and residual add
    into h for MoE with PLE.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    moe = tl.load(moe_out_ptr + m * stride_moe_m + offs, mask=mask, other=0.0)
    moe_f32 = moe.to(tl.float32)
    var_m = tl.sum(moe_f32 * moe_f32, axis=0) / H
    r_m = tl.rsqrt(var_m + eps)
    normed_moe = (moe_f32 * r_m).to(tl.bfloat16)

    w_post2 = tl.load(w_post2_ptr + offs, mask=mask, other=0.0)
    normed_moe = normed_moe * w_post2

    h1 = tl.load(h1_ptr + m * stride_h1_m + offs, mask=mask, other=0.0)
    comb = h1 + normed_moe

    comb_f32 = comb.to(tl.float32)
    var_c = tl.sum(comb_f32 * comb_f32, axis=0) / H
    r_c = tl.rsqrt(var_c + eps)
    normed_c = (comb_f32 * r_c).to(tl.bfloat16)

    w_post_ff = tl.load(w_post_ff_ptr + offs, mask=mask, other=0.0)
    normed_c = normed_c * w_post_ff

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_c + res
    tl.store(out_h_ptr + m * stride_h_m + offs, h, mask=mask)


@triton.jit
def _fused_post_ff_norm_add_scalar_cross_kernel(
    mlp_out_ptr,
    residual_ptr,
    post_ff_w_ptr,
    layer_scalar_ptr,
    next_w_ptr,
    out_normed_ptr,
    out_res_ptr,
    stride_mlp_m,
    stride_res_m,
    stride_normed_m,
    stride_out_res_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm + residual add + layer_scalar + next_input_layernorm.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs, mask=mask, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_mlp = (mlp_f32 * r).to(tl.bfloat16)

    post_w = tl.load(post_ff_w_ptr + offs, mask=mask, other=0.0)
    normed_mlp = normed_mlp * post_w

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_mlp + res

    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    res_next = (h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_res_ptr + m * stride_out_res_m + offs, res_next, mask=mask)

    res_next_f32 = res_next.to(tl.float32)
    var_next = tl.sum(res_next_f32 * res_next_f32, axis=0) / H
    r_next = tl.rsqrt(var_next + eps)
    normed_next = (res_next_f32 * r_next).to(tl.bfloat16)

    next_w = tl.load(next_w_ptr + offs, mask=mask, other=1.0)
    normed_next = normed_next * next_w
    tl.store(out_normed_ptr + m * stride_normed_m + offs, normed_next, mask=mask)


@triton.jit
def _fused_post_ple_norm_add_scalar_cross_kernel(
    ple_proj_ptr,
    h_ptr,
    post_ple_w_ptr,
    layer_scalar_ptr,
    next_w_ptr,
    out_normed_ptr,
    out_res_ptr,
    stride_proj_m,
    stride_h_m,
    stride_normed_m,
    stride_out_res_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ple_norm(ple_proj) + h + layer_scalar + next_input_layernorm.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    proj = tl.load(ple_proj_ptr + m * stride_proj_m + offs, mask=mask, other=0.0)
    proj_f32 = proj.to(tl.float32)
    var = tl.sum(proj_f32 * proj_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_proj = (proj_f32 * r).to(tl.bfloat16)

    w = tl.load(post_ple_w_ptr + offs, mask=mask, other=0.0)
    normed_proj = normed_proj * w

    h = tl.load(h_ptr + m * stride_h_m + offs, mask=mask, other=0.0)
    final_h = normed_proj + h

    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    res_next = (final_h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_res_ptr + m * stride_out_res_m + offs, res_next, mask=mask)

    res_next_f32 = res_next.to(tl.float32)
    var_next = tl.sum(res_next_f32 * res_next_f32, axis=0) / H
    r_next = tl.rsqrt(var_next + eps)
    normed_next = (res_next_f32 * r_next).to(tl.bfloat16)

    next_w = tl.load(next_w_ptr + offs, mask=mask, other=1.0)
    normed_next = normed_next * next_w
    tl.store(out_normed_ptr + m * stride_normed_m + offs, normed_next, mask=mask)


@triton.jit
def _fused_post_attn_add_moe_prenorms_kernel(
    attn_out_ptr,
    residual_ptr,
    post_w_ptr,
    pre_ff_w_ptr,
    pre_moe_w_ptr,
    router_scale_ptr,
    out_pre_ff_ptr,
    out_moe_in_ptr,
    out_router_in_ptr,
    out_res_ptr,
    stride_attn_m,
    stride_res_m,
    stride_pre_ff_m,
    stride_moe_in_m,
    stride_router_in_m,
    stride_out_res_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_attn_norm, residual add, pre_ff_norm, pre_ff2_norm,
    and router_norm & scale.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # 1. Post attn norm
    attn = tl.load(attn_out_ptr + m * stride_attn_m + offs, mask=mask, other=0.0)
    attn_f32 = attn.to(tl.float32)
    var1 = tl.sum(attn_f32 * attn_f32, axis=0) / H
    r1 = tl.rsqrt(var1 + eps)
    normed_attn = (attn_f32 * r1).to(tl.bfloat16)

    post_w = tl.load(post_w_ptr + offs, mask=mask, other=0.0)
    normed_attn = normed_attn * post_w

    # 2. Residual add in registers
    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    new_res = normed_attn + res
    tl.store(out_res_ptr + m * stride_out_res_m + offs, new_res, mask=mask)

    # 3. Compute shared variance once for pre_ff, moe_in, and router_in
    new_res_f32 = new_res.to(tl.float32)
    var2 = tl.sum(new_res_f32 * new_res_f32, axis=0) / H
    r2 = tl.rsqrt(var2 + eps)
    normed_res = (new_res_f32 * r2).to(tl.bfloat16)

    # 3a. pre_ff
    pre_ff_w = tl.load(pre_ff_w_ptr + offs, mask=mask, other=0.0)
    pre_ff = normed_res * pre_ff_w
    tl.store(out_pre_ff_ptr + m * stride_pre_ff_m + offs, pre_ff, mask=mask)

    # 3b. moe_in
    pre_moe_w = tl.load(pre_moe_w_ptr + offs, mask=mask, other=0.0)
    moe_in = normed_res * pre_moe_w
    tl.store(out_moe_in_ptr + m * stride_moe_in_m + offs, moe_in, mask=mask)

    # 3c. router_in (preprocessed with folded scale = scale * root_size)
    router_scale = tl.load(router_scale_ptr + offs, mask=mask, other=0.0)
    router_in = normed_res * router_scale
    tl.store(out_router_in_ptr + m * stride_router_in_m + offs, router_in, mask=mask)


@triton.jit
def _fused_moe_combine_norm_add_scalar_cross_kernel(
    h1_ptr,
    moe_out_ptr,
    residual_ptr,
    w_post2_ptr,
    w_post_ff_ptr,
    layer_scalar_ptr,
    next_w_ptr,
    out_normed_ptr,
    out_res_ptr,
    stride_h1_m,
    stride_moe_m,
    stride_res_m,
    stride_normed_m,
    stride_out_res_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff2 + combine + post_ff + residual add + scalar + next_input_norm.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    moe = tl.load(moe_out_ptr + m * stride_moe_m + offs, mask=mask, other=0.0)
    moe_f32 = moe.to(tl.float32)
    var_m = tl.sum(moe_f32 * moe_f32, axis=0) / H
    r_m = tl.rsqrt(var_m + eps)
    normed_moe = (moe_f32 * r_m).to(tl.bfloat16)

    w_post2 = tl.load(w_post2_ptr + offs, mask=mask, other=0.0)
    normed_moe = normed_moe * w_post2

    h1 = tl.load(h1_ptr + m * stride_h1_m + offs, mask=mask, other=0.0)
    comb = h1 + normed_moe

    comb_f32 = comb.to(tl.float32)
    var_c = tl.sum(comb_f32 * comb_f32, axis=0) / H
    r_c = tl.rsqrt(var_c + eps)
    normed_c = (comb_f32 * r_c).to(tl.bfloat16)

    w_post_ff = tl.load(w_post_ff_ptr + offs, mask=mask, other=0.0)
    normed_c = normed_c * w_post_ff

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_c + res

    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    res_next = (h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_res_ptr + m * stride_out_res_m + offs, res_next, mask=mask)

    res_next_f32 = res_next.to(tl.float32)
    var_next = tl.sum(res_next_f32 * res_next_f32, axis=0) / H
    r_next = tl.rsqrt(var_next + eps)
    normed_next = (res_next_f32 * r_next).to(tl.bfloat16)

    next_w = tl.load(next_w_ptr + offs, mask=mask, other=1.0)
    normed_next = normed_next * next_w
    tl.store(out_normed_ptr + m * stride_normed_m + offs, normed_next, mask=mask)


@triton.jit
def _fused_moe_consolidated_epilogue_kernel(
    mlp_out_ptr,
    moe_out_ptr,
    residual_ptr,
    w_post1_ptr,
    w_post2_ptr,
    w_post_ff_ptr,
    layer_scalar_ptr,
    next_w_ptr,
    out_normed_ptr,
    out_res_ptr,
    stride_mlp_m,
    stride_moe_m,
    stride_res_m,
    stride_normed_m,
    stride_out_res_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
    HAS_NEXT_NORM: tl.constexpr,
):
    """Fuses post_ff1(mlp) + post_ff2(moe) + combine + post_ff + residual + scalar
    (+ optional next norm).

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs, mask=mask, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var1 = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r1 = tl.rsqrt(var1 + eps)
    normed_mlp = (mlp_f32 * r1).to(tl.bfloat16)
    w_post1 = tl.load(w_post1_ptr + offs, mask=mask, other=0.0)
    h1 = normed_mlp * w_post1

    moe = tl.load(moe_out_ptr + m * stride_moe_m + offs, mask=mask, other=0.0)
    moe_f32 = moe.to(tl.float32)
    var_m = tl.sum(moe_f32 * moe_f32, axis=0) / H
    r_m = tl.rsqrt(var_m + eps)
    normed_moe = (moe_f32 * r_m).to(tl.bfloat16)
    w_post2 = tl.load(w_post2_ptr + offs, mask=mask, other=0.0)
    normed_moe = normed_moe * w_post2

    comb = h1 + normed_moe

    comb_f32 = comb.to(tl.float32)
    var_c = tl.sum(comb_f32 * comb_f32, axis=0) / H
    r_c = tl.rsqrt(var_c + eps)
    normed_c = (comb_f32 * r_c).to(tl.bfloat16)
    w_post_ff = tl.load(w_post_ff_ptr + offs, mask=mask, other=0.0)
    normed_c = normed_c * w_post_ff

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    h = normed_c + res

    scalar = tl.load(layer_scalar_ptr).to(tl.float32)
    res_next = (h.to(tl.float32) * scalar).to(tl.bfloat16)
    tl.store(out_res_ptr + m * stride_out_res_m + offs, res_next, mask=mask)

    if HAS_NEXT_NORM:
        res_next_f32 = res_next.to(tl.float32)
        var_next = tl.sum(res_next_f32 * res_next_f32, axis=0) / H
        r_next = tl.rsqrt(var_next + eps)
        normed_next = (res_next_f32 * r_next).to(tl.bfloat16)

        next_w = tl.load(next_w_ptr + offs, mask=mask, other=1.0)
        normed_next = normed_next * next_w
        tl.store(out_normed_ptr + m * stride_normed_m + offs, normed_next, mask=mask)


@triton.jit
def _fused_post_ff_norm_add_ple_gate_gelu_kernel(
    mlp_out_ptr,
    residual_ptr,
    post_ff_w_ptr,
    ple_in_ptr,
    w_gate_ptr,
    out_h_ptr,
    out_gated_ptr,
    stride_mlp_m,
    stride_res_m,
    stride_h_m,
    stride_ple_m,
    stride_ple_d,
    stride_gated_m,
    stride_wg_k,
    stride_wg_p,
    H: tl.constexpr,
    PLE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses post_ff_norm + residual add + PLE gate GEMM + GELU tanh mul.

    Grid: (M,)
    """
    m = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs_h, mask=mask_h, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_mlp = (mlp_f32 * r).to(tl.bfloat16)

    post_w = tl.load(post_ff_w_ptr + offs_h, mask=mask_h, other=0.0)
    res = tl.load(residual_ptr + m * stride_res_m + offs_h, mask=mask_h, other=0.0)
    h = normed_mlp * post_w + res
    tl.store(out_h_ptr + m * stride_h_m + offs_h, h, mask=mask_h)

    # Compute gate_raw = h @ w_gate.T in registers
    offs_p = tl.arange(0, BLOCK_P)
    acc = tl.zeros((1, BLOCK_P), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k[:, None] == offs_h[None, :]
        h_k = tl.sum(tl.where(mask_k, h[None, :], 0.0), axis=1)

        w_ptrs = (
            w_gate_ptr + offs_k[:, None] * stride_wg_k + offs_p[None, :] * stride_wg_p
        )
        w_k = tl.load(w_ptrs)
        acc += tl.dot(h_k[None, :], w_k).to(tl.float32)

    acc_1d = tl.reshape(acc, (BLOCK_P,))
    c: tl.constexpr = 0.7978845608028654
    arg = c * (acc_1d + 0.044715 * acc_1d * acc_1d * acc_1d)
    th = tl.extra.cuda.libdevice.tanh(arg)
    gelu_val = 0.5 * acc_1d * (1.0 + th)

    mask_p = offs_p < PLE_DIM
    ple = tl.load(
        ple_in_ptr + m * stride_ple_m + offs_p * stride_ple_d,
        mask=mask_p,
        other=0.0,
    )
    out_gated = (gelu_val * ple.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_gated_ptr + m * stride_gated_m + offs_p, out_gated, mask=mask_p)


@triton.jit
def _fused_ple_proj_norm_add_scalar_cross_kernel(
    gated_ple_ptr,
    h_ptr,
    w_proj_ptr,
    post_ple_w_ptr,
    layer_scalar_ptr,
    next_w_ptr,
    proj_buf_ptr,
    var_acc_ptr,
    done_cnt_ptr,
    out_res_ptr,
    out_normed_ptr,
    stride_gated_m,
    stride_h_m,
    stride_wp_h,
    stride_wp_p,
    stride_out_res_m,
    stride_normed_m,
    H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    BLOCK_H_TILE: tl.constexpr,
    BLOCK_H_FULL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_NEXT_NORM: tl.constexpr,
    eps: tl.constexpr,
):
    """Fused PLE projection GEMM + norm + residual add + scalar + cross-norm.

    Grid: (M, NUM_H_BLOCKS)
    """
    m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_p = tl.arange(0, BLOCK_P)
    offs_h = pid_h * BLOCK_H_TILE + tl.arange(0, BLOCK_H_TILE)

    gated = tl.load(gated_ple_ptr + m * stride_gated_m + offs_p)
    w_ptrs = w_proj_ptr + offs_h[None, :] * stride_wp_h + offs_p[:, None] * stride_wp_p
    w_proj = tl.load(w_ptrs)

    proj_tile = tl.sum(
        gated[:, None].to(tl.float32) * w_proj.to(tl.float32), axis=0
    ).to(tl.bfloat16)
    tl.store(proj_buf_ptr + m * H + offs_h, proj_tile)

    tile_sum_sq = tl.sum(proj_tile.to(tl.float32) * proj_tile.to(tl.float32), axis=0)
    tl.atomic_add(var_acc_ptr + m, tile_sum_sq)

    tl.debug_barrier()

    offs_tile = tl.arange(0, BLOCK_H_TILE)
    is_t0 = offs_tile == 0
    cnt_ptrs = done_cnt_ptr + m + tl.zeros((BLOCK_H_TILE,), dtype=tl.int32)
    prev_cnt = tl.atomic_add(cnt_ptrs, 1, mask=is_t0, sem="acq_rel")
    is_last = tl.sum(tl.where(is_t0, prev_cnt, 0), axis=0) == (NUM_H_BLOCKS - 1)

    if is_last:
        sum_sq = tl.load(var_acc_ptr + m)
        tl.store(var_acc_ptr + m, 0.0)
        tl.store(done_cnt_ptr + m, 0)

        r = tl.rsqrt(sum_sq / H + eps)
        offs_full = tl.arange(0, BLOCK_H_FULL)
        mask_full = offs_full < H

        proj = tl.load(proj_buf_ptr + m * H + offs_full, mask=mask_full, other=0.0)
        proj_norm = (proj.to(tl.float32) * r).to(tl.bfloat16)

        post_ple_w = tl.load(post_ple_w_ptr + offs_full, mask=mask_full, other=0.0)
        h = tl.load(h_ptr + m * stride_h_m + offs_full, mask=mask_full, other=0.0)
        scalar = tl.load(layer_scalar_ptr).to(tl.float32)

        res = ((proj_norm * post_ple_w + h).to(tl.float32) * scalar).to(tl.bfloat16)
        tl.store(out_res_ptr + m * stride_out_res_m + offs_full, res, mask=mask_full)

        if HAS_NEXT_NORM:
            res_f32 = res.to(tl.float32)
            var_next = tl.sum(res_f32 * res_f32, axis=0) / H
            r_next = tl.rsqrt(var_next + eps)
            next_normed = (res_f32 * r_next).to(tl.bfloat16)
            next_w = tl.load(next_w_ptr + offs_full, mask=mask_full, other=0.0)
            next_normed = next_normed * next_w
            tl.store(
                out_normed_ptr + m * stride_normed_m + offs_full,
                next_normed,
                mask=mask_full,
            )


@triton.jit
def _fused_ple_model_proj_norm_combine_kernel(
    model_proj_ptr,
    norm_w_ptr,
    embed_ple_ptr,
    out_ptr,
    proj_scale_ptr,
    input_scale_ptr,
    stride_proj_m,
    stride_proj_l,
    stride_proj_p,
    stride_embed_m,
    stride_embed_l,
    stride_embed_p,
    stride_out_m,
    stride_out_l,
    stride_out_p,
    P: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_EMBED: tl.constexpr,
    eps: tl.constexpr,
):
    """Fuses (RMSNorm(model_proj * proj_scale, norm_w, eps) + embed_ple) * input_scale.

    Grid: (M, num_layers)
    """
    m = tl.program_id(0)
    layer_idx = tl.program_id(1)

    offs_p = tl.arange(0, BLOCK_P)
    mask = offs_p < P

    proj_scale = tl.load(proj_scale_ptr).to(tl.float32)

    x = tl.load(
        model_proj_ptr
        + m * stride_proj_m
        + layer_idx * stride_proj_l
        + offs_p * stride_proj_p,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    x = x * proj_scale
    var = tl.sum(x * x, axis=0) / P
    r = tl.rsqrt(var + eps)
    w = tl.load(norm_w_ptr + offs_p, mask=mask, other=0.0).to(tl.float32)
    normed = (x * r) * w

    if HAS_EMBED:
        input_scale = tl.load(input_scale_ptr).to(tl.float32)
        embed = tl.load(
            embed_ple_ptr
            + m * stride_embed_m
            + layer_idx * stride_embed_l
            + offs_p * stride_embed_p,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = ((normed + embed) * input_scale).to(tl.bfloat16)
    else:
        out = normed.to(tl.bfloat16)

    tl.store(
        out_ptr + m * stride_out_m + layer_idx * stride_out_l + offs_p * stride_out_p,
        out,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# TorchDynamo Custom Op Implementations & Fakes
# ---------------------------------------------------------------------------


def _gemma4_fused_qkv_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    is_kv_shared_layer: bool = False,
    is_k_eq_v: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_2d = qkv.reshape(-1, qkv.shape[-1])
    positions_1d = positions.flatten()
    if cos_sin_cache.device != qkv_2d.device:
        cos_sin_cache = cos_sin_cache.to(qkv_2d.device)
    if cos_sin_cache.dtype != qkv_2d.dtype:
        cos_sin_cache = cos_sin_cache.to(qkv_2d.dtype)
    M = qkv_2d.shape[0]

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    if M == 0:
        out_q = torch.empty(0, q_size, dtype=qkv.dtype, device=qkv.device)
        out_k = torch.empty(0, kv_size, dtype=qkv.dtype, device=qkv.device)
        out_v = torch.empty(0, kv_size, dtype=qkv.dtype, device=qkv.device)
        return out_q, out_k, out_v

    out_q = torch.empty(M, q_size, dtype=qkv.dtype, device=qkv.device)

    if is_kv_shared_layer:
        grid = (num_heads, M)
        if qkv_2d.shape[-1] >= q_size + 2 * kv_size:
            out_k = qkv_2d[:, q_size : q_size + kv_size]
            out_v = qkv_2d[:, q_size + kv_size : q_size + 2 * kv_size]
            stride_out_k_m = out_k.stride(0)
            stride_out_v_m = out_v.stride(0)
        else:
            out_k = torch.empty((0, kv_size), dtype=qkv.dtype, device=qkv.device)
            out_v = torch.empty((0, kv_size), dtype=qkv.dtype, device=qkv.device)
            stride_out_k_m = 0
            stride_out_v_m = 0
    elif is_k_eq_v:
        grid = (num_heads + num_kv_heads, M)
        out_k = torch.empty(M, kv_size, dtype=qkv.dtype, device=qkv.device)
        out_v = torch.empty(M, kv_size, dtype=qkv.dtype, device=qkv.device)
        stride_out_k_m = out_k.stride(0)
        stride_out_v_m = out_v.stride(0)
    else:
        grid = (num_heads + 2 * num_kv_heads, M)
        out_k = torch.empty(M, kv_size, dtype=qkv.dtype, device=qkv.device)
        out_v = torch.empty(M, kv_size, dtype=qkv.dtype, device=qkv.device)
        stride_out_k_m = out_k.stride(0)
        stride_out_v_m = out_v.stride(0)

    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)

    _fused_qkv_norm_rope_kernel[grid](
        qkv_2d,
        positions_1d,
        cos_sin_cache,
        q_weight,
        k_weight,
        out_q,
        out_k,
        out_v,
        qkv_2d.stride(0),
        cos_sin_cache.stride(0),
        out_q.stride(0),
        stride_out_k_m,
        stride_out_v_m,
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        BLOCK_HALF=block_half,
        eps=eps,
        is_kv_shared=is_kv_shared_layer,
        is_k_eq_v=is_k_eq_v,
        num_warps=2 if block_half <= 128 else 4,
    )

    if qkv.dim() == 3:
        out_q = out_q.reshape(qkv.shape[0], qkv.shape[1], q_size)
        if not is_kv_shared_layer or qkv_2d.shape[-1] >= q_size + 2 * kv_size:
            out_k = out_k.reshape(qkv.shape[0], qkv.shape[1], kv_size)
            out_v = out_v.reshape(qkv.shape[0], qkv.shape[1], kv_size)
        else:
            out_k = torch.empty((0, 0, kv_size), dtype=qkv.dtype, device=qkv.device)
            out_v = torch.empty((0, 0, kv_size), dtype=qkv.dtype, device=qkv.device)

    return out_q, out_k, out_v


def _gemma4_fused_qkv_norm_rope_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    is_kv_shared_layer: bool = False,
    is_k_eq_v: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    if is_kv_shared_layer:
        if qkv.shape[-1] >= q_size + 2 * kv_size:
            if qkv.dim() == 3:
                B, S, _ = qkv.shape
                return (
                    torch.empty(B, S, q_size, dtype=qkv.dtype, device=qkv.device),
                    qkv[:, :, q_size : q_size + kv_size],
                    qkv[:, :, q_size + kv_size : q_size + 2 * kv_size],
                )
            M = qkv.shape[0]
            return (
                torch.empty(M, q_size, dtype=qkv.dtype, device=qkv.device),
                qkv[:, q_size : q_size + kv_size],
                qkv[:, q_size + kv_size : q_size + 2 * kv_size],
            )
        if qkv.dim() == 3:
            B, S, _ = qkv.shape
            return (
                torch.empty(B, S, q_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty(0, 0, kv_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty(0, 0, kv_size, dtype=qkv.dtype, device=qkv.device),
            )
        M = qkv.shape[0]
        return (
            torch.empty(M, q_size, dtype=qkv.dtype, device=qkv.device),
            torch.empty(0, kv_size, dtype=qkv.dtype, device=qkv.device),
            torch.empty(0, kv_size, dtype=qkv.dtype, device=qkv.device),
        )
    if qkv.dim() == 3:
        B, S, _ = qkv.shape
        return (
            torch.empty(B, S, q_size, dtype=qkv.dtype, device=qkv.device),
            torch.empty(B, S, kv_size, dtype=qkv.dtype, device=qkv.device),
            torch.empty(B, S, kv_size, dtype=qkv.dtype, device=qkv.device),
        )
    M = qkv.shape[0]
    return (
        torch.empty(M, q_size, dtype=qkv.dtype, device=qkv.device),
        torch.empty(M, kv_size, dtype=qkv.dtype, device=qkv.device),
        torch.empty(M, kv_size, dtype=qkv.dtype, device=qkv.device),
    )


def _gemma4_fused_post_attn_add_pre_ff_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor,
    pre_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = attn_out.shape
    attn_2d = attn_out.reshape(-1, attn_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = attn_2d.shape

    if M == 0:
        return torch.empty_like(attn_out), torch.empty_like(residual)

    out_pre_ff = torch.empty_like(attn_2d)
    out_res = torch.empty_like(attn_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_attn_add_pre_ff_norm_kernel[(M,)](
        attn_2d,
        res_2d,
        post_w,
        pre_w,
        out_pre_ff,
        out_res,
        attn_2d.stride(0),
        res_2d.stride(0),
        out_pre_ff.stride(0),
        out_res.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_pre_ff = out_pre_ff.view(orig_shape)
        out_res = out_res.view(orig_shape)

    return out_pre_ff, out_res


def _gemma4_fused_post_attn_add_pre_ff_norm_fake(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor,
    pre_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(attn_out), torch.empty_like(attn_out)


def _gemma4_fused_post_ff_norm_add_scalar(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = mlp_2d.shape

    if M == 0:
        return torch.empty_like(mlp_out)

    out = torch.empty_like(mlp_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_ff_norm_add_scalar_kernel[(M,)](
        mlp_2d,
        res_2d,
        post_ff_w,
        layer_scalar,
        out,
        mlp_2d.stride(0),
        res_2d.stride(0),
        out.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out = out.view(orig_shape)
    return out


def _gemma4_fused_post_ff_norm_add_scalar_fake(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(mlp_out)


def _gemma4_fused_post_ff_norm_add(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = mlp_2d.shape

    if M == 0:
        return torch.empty_like(mlp_out)

    out_h = torch.empty_like(mlp_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_ff_norm_add_kernel[(M,)](
        mlp_2d,
        res_2d,
        post_ff_w,
        out_h,
        mlp_2d.stride(0),
        res_2d.stride(0),
        out_h.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_h = out_h.view(orig_shape)
    return out_h


def _gemma4_fused_post_ff_norm_add_fake(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(mlp_out)


def _gemma4_fused_gelu_tanh_mul(
    gate_raw: torch.Tensor,
    per_layer_input: torch.Tensor,
) -> torch.Tensor:
    orig_shape = gate_raw.shape
    gate_2d = gate_raw.reshape(-1, gate_raw.shape[-1])
    M, ple_dim = gate_2d.shape

    if M == 0:
        return torch.empty_like(gate_raw)

    ple_stride_m = per_layer_input.stride(0) if per_layer_input.shape[0] > 1 else 0
    ple_stride_d = per_layer_input.stride(-1)

    out = torch.empty_like(gate_2d)
    block_size = triton.next_power_of_2(ple_dim)

    _fused_gelu_tanh_mul_kernel[(M,)](
        gate_2d,
        per_layer_input,
        out,
        gate_2d.stride(0),
        ple_stride_m,
        ple_stride_d,
        out.stride(0),
        PLE_DIM=ple_dim,
        BLOCK_SIZE=block_size,
    )

    if len(orig_shape) == 3:
        out = out.view(orig_shape)
    return out


def _gemma4_fused_gelu_tanh_mul_fake(
    gate_raw: torch.Tensor,
    per_layer_input: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(gate_raw)


def _gemma4_fused_post_ple_norm_add_scalar(
    ple_proj: torch.Tensor,
    h: torch.Tensor,
    post_ple_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = ple_proj.shape
    proj_2d = ple_proj.reshape(-1, ple_proj.shape[-1])
    h_2d = h.reshape(-1, h.shape[-1])
    M, H = proj_2d.shape

    if M == 0:
        return torch.empty_like(ple_proj)

    out = torch.empty_like(proj_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_ple_norm_add_scalar_kernel[(M,)](
        proj_2d,
        h_2d,
        post_ple_w,
        layer_scalar,
        out,
        proj_2d.stride(0),
        h_2d.stride(0),
        out.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out = out.view(orig_shape)
    return out


def _gemma4_fused_post_ple_norm_add_scalar_fake(
    ple_proj: torch.Tensor,
    h: torch.Tensor,
    post_ple_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(ple_proj)


def _gemma4_fused_mlp_norm_and_moe_prenorm(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    w_post1: torch.Tensor,
    w_pre2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = mlp_2d.shape

    if M == 0:
        return torch.empty_like(mlp_out), torch.empty_like(residual)

    out_h1 = torch.empty_like(mlp_2d)
    out_moe_in = torch.empty_like(mlp_2d)
    block_size = triton.next_power_of_2(H)

    _fused_mlp_norm_and_moe_prenorm_kernel[(M,)](
        mlp_2d,
        res_2d,
        w_post1,
        w_pre2,
        out_h1,
        out_moe_in,
        mlp_2d.stride(0),
        res_2d.stride(0),
        out_h1.stride(0),
        out_moe_in.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_h1 = out_h1.view(orig_shape)
        out_moe_in = out_moe_in.view(orig_shape)
    return out_h1, out_moe_in


def _gemma4_fused_mlp_norm_and_moe_prenorm_fake(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    w_post1: torch.Tensor,
    w_pre2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(mlp_out), torch.empty_like(mlp_out)


def _gemma4_fused_moe_combine_norm_add_scalar(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = h1.shape
    h1_2d = h1.reshape(-1, h1.shape[-1])
    moe_2d = moe_out.reshape(-1, moe_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = h1_2d.shape

    if M == 0:
        return torch.empty_like(h1)

    out = torch.empty_like(h1_2d)
    block_size = triton.next_power_of_2(H)

    _fused_moe_combine_norm_add_scalar_kernel[(M,)](
        h1_2d,
        moe_2d,
        res_2d,
        w_post2,
        w_post_ff,
        layer_scalar,
        out,
        h1_2d.stride(0),
        moe_2d.stride(0),
        res_2d.stride(0),
        out.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out = out.view(orig_shape)
    return out


def _gemma4_fused_moe_combine_norm_add_scalar_fake(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(h1)


def _gemma4_fused_moe_combine_norm_add(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = h1.shape
    h1_2d = h1.reshape(-1, h1.shape[-1])
    moe_2d = moe_out.reshape(-1, moe_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = h1_2d.shape

    if M == 0:
        return torch.empty_like(h1)

    out_h = torch.empty_like(h1_2d)
    block_size = triton.next_power_of_2(H)

    _fused_moe_combine_norm_add_kernel[(M,)](
        h1_2d,
        moe_2d,
        res_2d,
        w_post2,
        w_post_ff,
        out_h,
        h1_2d.stride(0),
        moe_2d.stride(0),
        res_2d.stride(0),
        out_h.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_h = out_h.view(orig_shape)
    return out_h


def _gemma4_fused_moe_combine_norm_add_fake(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(h1)


def _gemma4_fused_post_ff_norm_add_scalar_cross(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = mlp_2d.shape

    if M == 0:
        return torch.empty_like(mlp_out), torch.empty_like(residual)

    out_normed = torch.empty_like(mlp_2d)
    out_res = torch.empty_like(mlp_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_ff_norm_add_scalar_cross_kernel[(M,)](
        mlp_2d,
        res_2d,
        post_ff_w,
        layer_scalar,
        next_w,
        out_normed,
        out_res,
        mlp_2d.stride(0),
        res_2d.stride(0),
        out_normed.stride(0),
        out_res.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_normed = out_normed.view(orig_shape)
        out_res = out_res.view(orig_shape)
    return out_normed, out_res


def _gemma4_fused_post_ff_norm_add_scalar_cross_fake(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(mlp_out), torch.empty_like(mlp_out)


def _gemma4_fused_post_ple_norm_add_scalar_cross(
    ple_proj: torch.Tensor,
    h: torch.Tensor,
    post_ple_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = ple_proj.shape
    proj_2d = ple_proj.reshape(-1, ple_proj.shape[-1])
    h_2d = h.reshape(-1, h.shape[-1])
    M, H = proj_2d.shape

    if M == 0:
        return torch.empty_like(h), torch.empty_like(h)

    out_normed = torch.empty_like(proj_2d)
    out_res = torch.empty_like(proj_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_ple_norm_add_scalar_cross_kernel[(M,)](
        proj_2d,
        h_2d,
        post_ple_w,
        layer_scalar,
        next_w,
        out_normed,
        out_res,
        proj_2d.stride(0),
        h_2d.stride(0),
        out_normed.stride(0),
        out_res.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_normed = out_normed.view(orig_shape)
        out_res = out_res.view(orig_shape)
    return out_normed, out_res


def _gemma4_fused_post_ple_norm_add_scalar_cross_fake(
    ple_proj: torch.Tensor,
    h: torch.Tensor,
    post_ple_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(ple_proj), torch.empty_like(ple_proj)


def _gemma4_fused_post_attn_add_moe_prenorms(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor,
    pre_ff_w: torch.Tensor,
    pre_moe_w: torch.Tensor,
    router_scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_shape = attn_out.shape
    attn_2d = attn_out.reshape(-1, attn_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = attn_2d.shape

    if M == 0:
        return (
            torch.empty_like(attn_out),
            torch.empty_like(attn_out),
            torch.empty_like(attn_out),
            torch.empty_like(residual),
        )

    out_pre_ff = torch.empty_like(attn_2d)
    out_moe_in = torch.empty_like(attn_2d)
    out_router_in = torch.empty_like(attn_2d)
    out_res = torch.empty_like(attn_2d)
    block_size = triton.next_power_of_2(H)

    _fused_post_attn_add_moe_prenorms_kernel[(M,)](
        attn_2d,
        res_2d,
        post_w,
        pre_ff_w,
        pre_moe_w,
        router_scale,
        out_pre_ff,
        out_moe_in,
        out_router_in,
        out_res,
        attn_2d.stride(0),
        res_2d.stride(0),
        out_pre_ff.stride(0),
        out_moe_in.stride(0),
        out_router_in.stride(0),
        out_res.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_pre_ff = out_pre_ff.view(orig_shape)
        out_moe_in = out_moe_in.view(orig_shape)
        out_router_in = out_router_in.view(orig_shape)
        out_res = out_res.view(orig_shape)

    return out_pre_ff, out_moe_in, out_router_in, out_res


def _gemma4_fused_post_attn_add_moe_prenorms_fake(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor,
    pre_ff_w: torch.Tensor,
    pre_moe_w: torch.Tensor,
    router_scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(attn_out),
        torch.empty_like(attn_out),
        torch.empty_like(attn_out),
        torch.empty_like(attn_out),
    )


def _gemma4_fused_moe_combine_norm_add_scalar_cross(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = h1.shape
    h1_2d = h1.reshape(-1, h1.shape[-1])
    moe_2d = moe_out.reshape(-1, moe_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = h1_2d.shape

    if M == 0:
        return torch.empty_like(h1), torch.empty_like(h1)

    out_normed = torch.empty_like(h1_2d)
    out_res = torch.empty_like(h1_2d)
    block_size = triton.next_power_of_2(H)

    _fused_moe_combine_norm_add_scalar_cross_kernel[(M,)](
        h1_2d,
        moe_2d,
        res_2d,
        w_post2,
        w_post_ff,
        layer_scalar,
        next_w,
        out_normed,
        out_res,
        h1_2d.stride(0),
        moe_2d.stride(0),
        res_2d.stride(0),
        out_normed.stride(0),
        out_res.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        out_normed = out_normed.view(orig_shape)
        out_res = out_res.view(orig_shape)
    return out_normed, out_res


def _gemma4_fused_moe_combine_norm_add_scalar_cross_fake(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(h1), torch.empty_like(h1)


def _gemma4_fused_moe_consolidated_epilogue(
    mlp_out: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post1: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    moe_2d = moe_out.reshape(-1, moe_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = mlp_2d.shape

    if M == 0:
        return torch.empty_like(mlp_out), torch.empty_like(mlp_out)

    out_res = torch.empty_like(mlp_2d)
    has_next = next_w is not None
    if has_next:
        out_normed = torch.empty_like(mlp_2d)
        stride_normed = out_normed.stride(0)
    else:
        out_normed = mlp_2d[:0, :0]
        stride_normed = 0
        next_w = mlp_2d.new_ones(H)

    block_size = triton.next_power_of_2(H)

    _fused_moe_consolidated_epilogue_kernel[(M,)](
        mlp_2d,
        moe_2d,
        res_2d,
        w_post1,
        w_post2,
        w_post_ff,
        layer_scalar,
        next_w,
        out_normed,
        out_res,
        mlp_2d.stride(0),
        moe_2d.stride(0),
        res_2d.stride(0),
        stride_normed,
        out_res.stride(0),
        H=H,
        BLOCK_SIZE=block_size,
        eps=eps,
        HAS_NEXT_NORM=has_next,
        num_warps=16 if block_size >= 4096 else (8 if block_size >= 2048 else 4),
    )

    if len(orig_shape) == 3:
        if has_next:
            out_normed = out_normed.view(orig_shape)
        out_res = out_res.view(orig_shape)

    return out_normed, out_res


def _gemma4_fused_moe_consolidated_epilogue_fake(
    mlp_out: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post1: torch.Tensor,
    w_post2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(mlp_out), torch.empty_like(mlp_out)


_PLE_STATIC_MAX_M = 16
_PLE_WORKSPACE_CACHE: dict[
    tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_ple_workspace(M: int, H: int, device: torch.device):
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (dev_idx, H)
    entry = _PLE_WORKSPACE_CACHE.get(key)
    if entry is None or entry[0].device != device or entry[0].shape[0] < M:
        alloc_m = max(M, _PLE_STATIC_MAX_M)
        proj_buf = torch.empty((alloc_m, H), dtype=torch.bfloat16, device=device)
        var_acc = torch.zeros(alloc_m, dtype=torch.float32, device=device)
        done_cnt = torch.zeros(alloc_m, dtype=torch.int32, device=device)
        entry = (proj_buf, var_acc, done_cnt)
        _PLE_WORKSPACE_CACHE[key] = entry
    return entry


def _gemma4_fused_post_ff_norm_add_ple_gate_gelu(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    ple_in: torch.Tensor,
    w_gate: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    ple_2d = ple_in.reshape(-1, ple_in.shape[-1])
    M, H = mlp_2d.shape
    ple_dim = ple_2d.shape[-1]

    if M == 0:
        return torch.empty_like(mlp_out), torch.empty_like(ple_in)

    out_h = torch.empty_like(mlp_2d)
    out_gated = torch.empty_like(ple_2d)

    block_h = triton.next_power_of_2(H)
    _fused_post_ff_norm_add_ple_gate_gelu_kernel[(M,)](
        mlp_2d,
        res_2d,
        post_ff_w,
        ple_2d,
        w_gate,
        out_h,
        out_gated,
        mlp_2d.stride(0),
        res_2d.stride(0),
        out_h.stride(0),
        ple_2d.stride(0) if ple_2d.shape[0] > 1 else 0,
        ple_2d.stride(1) if ple_2d.dim() > 1 else 1,
        out_gated.stride(0),
        w_gate.stride(1),
        w_gate.stride(0),
        H=H,
        PLE_DIM=ple_dim,
        BLOCK_H=block_h,
        BLOCK_K=64,
        BLOCK_P=256,
        eps=eps,
        num_warps=8,
    )
    if len(orig_shape) == 3:
        out_h = out_h.view(orig_shape)
        out_gated = out_gated.view(*orig_shape[:-1], ple_dim)
    return out_h, out_gated


def _gemma4_fused_post_ff_norm_add_ple_gate_gelu_fake(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    ple_in: torch.Tensor,
    w_gate: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(mlp_out), torch.empty_like(ple_in)


def _gemma4_fused_ple_proj_norm_add_scalar_cross(
    gated_ple: torch.Tensor,
    h: torch.Tensor,
    w_proj: torch.Tensor,
    post_ple_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = h.shape
    gated_2d = gated_ple.reshape(-1, gated_ple.shape[-1])
    h_2d = h.reshape(-1, h.shape[-1])
    M, H = h_2d.shape

    if M == 0:
        return torch.empty_like(h), torch.empty_like(h)

    has_next = next_w.numel() > 0
    out_res = torch.empty_like(h_2d)
    out_normed = (
        torch.empty_like(h_2d)
        if has_next
        else torch.empty(0, dtype=h.dtype, device=h.device)
    )

    proj_buf, var_acc, done_cnt = _get_ple_workspace(M, H, h.device)
    block_h_tile = 256
    num_h_blocks = H // block_h_tile
    block_h_full = triton.next_power_of_2(H)

    _fused_ple_proj_norm_add_scalar_cross_kernel[(M, num_h_blocks)](
        gated_2d,
        h_2d,
        w_proj,
        post_ple_w,
        layer_scalar,
        next_w if has_next else h_2d,
        proj_buf,
        var_acc,
        done_cnt,
        out_res,
        out_normed,
        gated_2d.stride(0),
        h_2d.stride(0),
        w_proj.stride(0),
        w_proj.stride(1),
        out_res.stride(0),
        out_normed.stride(0) if has_next else 0,
        H=H,
        NUM_H_BLOCKS=num_h_blocks,
        BLOCK_H_TILE=block_h_tile,
        BLOCK_H_FULL=block_h_full,
        BLOCK_P=256,
        HAS_NEXT_NORM=has_next,
        eps=eps,
        num_warps=8 if block_h_full >= 4096 else 4,
    )
    if len(orig_shape) == 3:
        out_res = out_res.view(orig_shape)
        if has_next:
            out_normed = out_normed.view(orig_shape)
    return out_normed, out_res


def _gemma4_fused_ple_proj_norm_add_scalar_cross_fake(
    gated_ple: torch.Tensor,
    h: torch.Tensor,
    w_proj: torch.Tensor,
    post_ple_w: torch.Tensor,
    layer_scalar: torch.Tensor,
    next_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    has_next = next_w.numel() > 0
    return torch.empty_like(h) if has_next else torch.empty(
        0, dtype=h.dtype, device=h.device
    ), torch.empty_like(h)


def _gemma4_fused_ple_model_proj_norm_combine(
    model_proj: torch.Tensor,
    norm_w: torch.Tensor,
    embed_ple: torch.Tensor,
    proj_scale: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = model_proj.shape
    P = norm_w.shape[-1]
    proj_3d = model_proj.reshape(
        -1, orig_shape[-2] if model_proj.dim() == 3 else orig_shape[-1] // P, P
    )
    M, L, _ = proj_3d.shape
    has_embed = embed_ple.numel() > 0
    embed_3d = (
        embed_ple.reshape(M, L, P)
        if has_embed
        else torch.empty(0, dtype=model_proj.dtype, device=model_proj.device)
    )

    out = torch.empty((M, L, P), dtype=model_proj.dtype, device=model_proj.device)

    if M == 0:
        if model_proj.dim() == 2:
            return out.reshape(orig_shape[0], L, P)
        return (
            out.view(*orig_shape[:-1], L, P)
            if model_proj.dim() > 3
            else out.view(orig_shape)
        )

    _fused_ple_model_proj_norm_combine_kernel[(M, L)](
        proj_3d,
        norm_w,
        embed_3d if has_embed else proj_3d,
        out,
        proj_scale,
        input_scale,
        proj_3d.stride(0),
        proj_3d.stride(1),
        proj_3d.stride(2),
        embed_3d.stride(0) if has_embed else 0,
        embed_3d.stride(1) if has_embed else 0,
        embed_3d.stride(2) if has_embed else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        P=P,
        BLOCK_P=256,
        HAS_EMBED=has_embed,
        eps=eps,
        num_warps=4,
    )
    if model_proj.dim() == 2:
        return out.reshape(orig_shape[0], L, P)
    return (
        out.view(*orig_shape[:-1], L, P)
        if model_proj.dim() > 3
        else out.view(orig_shape)
    )


def _gemma4_fused_ple_model_proj_norm_combine_fake(
    model_proj: torch.Tensor,
    norm_w: torch.Tensor,
    embed_ple: torch.Tensor,
    proj_scale: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    P = norm_w.shape[-1]
    if model_proj.dim() == 2:
        M = model_proj.shape[0]
        L = model_proj.shape[1] // P
        return torch.empty(M, L, P, dtype=model_proj.dtype, device=model_proj.device)
    orig_shape = model_proj.shape
    return torch.empty(
        (*orig_shape[:-1], orig_shape[-1] // P, P),
        dtype=model_proj.dtype,
        device=model_proj.device,
    )


def _register_op(
    op_name: str,
    op_func: Callable[..., Any],
    fake_impl: Callable[..., Any],
    mutates_args: list[str] | None = None,
) -> None:
    try:
        direct_register_custom_op(
            op_name=op_name,
            op_func=op_func,
            mutates_args=mutates_args or [],
            fake_impl=fake_impl,
        )
    except Exception as e:
        logger.debug("Custom op %s registration notice: %s", op_name, e)


# Register all custom ops to vLLM library with fake implementations
_CUSTOM_OPS: list[tuple[str, Callable[..., Any], Callable[..., Any]]] = [
    (
        "gemma4_fused_post_ff_norm_add_ple_gate_gelu",
        _gemma4_fused_post_ff_norm_add_ple_gate_gelu,
        _gemma4_fused_post_ff_norm_add_ple_gate_gelu_fake,
    ),
    (
        "gemma4_fused_ple_proj_norm_add_scalar_cross",
        _gemma4_fused_ple_proj_norm_add_scalar_cross,
        _gemma4_fused_ple_proj_norm_add_scalar_cross_fake,
    ),
    (
        "gemma4_fused_ple_model_proj_norm_combine",
        _gemma4_fused_ple_model_proj_norm_combine,
        _gemma4_fused_ple_model_proj_norm_combine_fake,
    ),
    (
        "gemma4_fused_qkv_norm_rope",
        _gemma4_fused_qkv_norm_rope,
        _gemma4_fused_qkv_norm_rope_fake,
    ),
    (
        "gemma4_fused_post_attn_add_pre_ff_norm",
        _gemma4_fused_post_attn_add_pre_ff_norm,
        _gemma4_fused_post_attn_add_pre_ff_norm_fake,
    ),
    (
        "gemma4_fused_post_ff_norm_add_scalar",
        _gemma4_fused_post_ff_norm_add_scalar,
        _gemma4_fused_post_ff_norm_add_scalar_fake,
    ),
    (
        "gemma4_fused_post_ff_norm_add_scalar_cross",
        _gemma4_fused_post_ff_norm_add_scalar_cross,
        _gemma4_fused_post_ff_norm_add_scalar_cross_fake,
    ),
    (
        "gemma4_fused_post_ff_norm_add",
        _gemma4_fused_post_ff_norm_add,
        _gemma4_fused_post_ff_norm_add_fake,
    ),
    (
        "gemma4_fused_gelu_tanh_mul",
        _gemma4_fused_gelu_tanh_mul,
        _gemma4_fused_gelu_tanh_mul_fake,
    ),
    (
        "gemma4_fused_post_ple_norm_add_scalar",
        _gemma4_fused_post_ple_norm_add_scalar,
        _gemma4_fused_post_ple_norm_add_scalar_fake,
    ),
    (
        "gemma4_fused_post_ple_norm_add_scalar_cross",
        _gemma4_fused_post_ple_norm_add_scalar_cross,
        _gemma4_fused_post_ple_norm_add_scalar_cross_fake,
    ),
    (
        "gemma4_fused_mlp_norm_and_moe_prenorm",
        _gemma4_fused_mlp_norm_and_moe_prenorm,
        _gemma4_fused_mlp_norm_and_moe_prenorm_fake,
    ),
    (
        "gemma4_fused_post_attn_add_moe_prenorms",
        _gemma4_fused_post_attn_add_moe_prenorms,
        _gemma4_fused_post_attn_add_moe_prenorms_fake,
    ),
    (
        "gemma4_fused_moe_combine_norm_add_scalar",
        _gemma4_fused_moe_combine_norm_add_scalar,
        _gemma4_fused_moe_combine_norm_add_scalar_fake,
    ),
    (
        "gemma4_fused_moe_combine_norm_add_scalar_cross",
        _gemma4_fused_moe_combine_norm_add_scalar_cross,
        _gemma4_fused_moe_combine_norm_add_scalar_cross_fake,
    ),
    (
        "gemma4_fused_moe_consolidated_epilogue",
        _gemma4_fused_moe_consolidated_epilogue,
        _gemma4_fused_moe_consolidated_epilogue_fake,
    ),
    (
        "gemma4_fused_moe_combine_norm_add",
        _gemma4_fused_moe_combine_norm_add,
        _gemma4_fused_moe_combine_norm_add_fake,
    ),
]
for op_name, op_func, fake_impl in _CUSTOM_OPS:
    _register_op(op_name, op_func, fake_impl)


# Dispatch references
def _dispatch(op_name: str, fallback_func: Any) -> Any:
    return getattr(torch.ops.vllm, op_name, fallback_func)


# ---------------------------------------------------------------------------
# Public High-Level Entrypoints
# ---------------------------------------------------------------------------


def fused_ple_model_proj_norm_combine(
    model_proj: torch.Tensor,
    norm_weight: torch.Tensor | nn.Module,
    embed_ple: torch.Tensor | None = None,
    num_layers: int = 1,
    proj_scale: float | torch.Tensor = 1.0,
    input_scale: float | torch.Tensor = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Public helper for fused PLE model projection norm + combine."""
    nw = norm_weight.weight if hasattr(norm_weight, "weight") else norm_weight
    if not isinstance(proj_scale, torch.Tensor):
        p_scale = torch.tensor(
            [proj_scale], dtype=torch.float32, device=model_proj.device
        )
    elif proj_scale.device != model_proj.device:
        p_scale = proj_scale.to(device=model_proj.device)
    else:
        p_scale = proj_scale

    if not isinstance(input_scale, torch.Tensor):
        i_scale = torch.tensor(
            [input_scale], dtype=torch.float32, device=model_proj.device
        )
    elif input_scale.device != model_proj.device:
        i_scale = input_scale.to(device=model_proj.device)
    else:
        i_scale = input_scale

    dummy_embed = (
        embed_ple
        if embed_ple is not None
        else torch.empty(0, dtype=model_proj.dtype, device=model_proj.device)
    )
    dispatch_op = _dispatch(
        "gemma4_fused_ple_model_proj_norm_combine",
        _gemma4_fused_ple_model_proj_norm_combine,
    )
    return dispatch_op(model_proj, nw, dummy_embed, p_scale, i_scale, eps=eps)


def fused_qkv_norm_rope(
    *args, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV split + RMSNorm + Neox-style proportional RoPE.

    Flexible signature supports both:
    1. fused_qkv_norm_rope(qkv, positions, cos_sin_cache, q_weight, k_weight,
                           num_heads, num_kv_heads, head_dim, eps=1e-6,
                           is_kv_shared_layer=False)
    2. fused_qkv_norm_rope(q, k, v, positions, q_weight, k_weight,
                           rotary_emb_or_cache, is_kv_shared_layer=False, ...)
    """
    dispatch_op = _dispatch("gemma4_fused_qkv_norm_rope", _gemma4_fused_qkv_norm_rope)

    # Detect calling pattern
    if (
        len(args) >= 3
        and isinstance(args[0], torch.Tensor)
        and isinstance(args[1], torch.Tensor)
        and isinstance(args[2], torch.Tensor)
        and args[0].dim() >= 2
        and args[1].dim() >= 2
        and args[2].dim() >= 2
    ):
        # Pattern 2: separate (q, k, v, positions, ...)
        q, k, v = args[0], args[1], args[2]
        orig_shape = q.shape
        q_2d = q.reshape(-1, q.shape[-1])
        k_2d = k.reshape(-1, k.shape[-1])
        v_2d = v.reshape(-1, v.shape[-1])

        positions = args[3] if len(args) > 3 else kwargs["positions"]
        q_weight = (
            args[4] if len(args) > 4 else kwargs.get("q_weight", kwargs.get("wq"))
        )
        k_weight = (
            args[5] if len(args) > 5 else kwargs.get("k_weight", kwargs.get("wk"))
        )
        rotary_arg = (
            args[6]
            if len(args) > 6
            else kwargs.get("rotary_emb", kwargs.get("cos_sin_cache"))
        )

        assert rotary_arg is not None and q_weight is not None and k_weight is not None
        cos_sin_cache = (
            rotary_arg.cos_sin_cache
            if hasattr(rotary_arg, "cos_sin_cache")
            else rotary_arg
        )
        q_weight = q_weight.weight if hasattr(q_weight, "weight") else q_weight
        k_weight = k_weight.weight if hasattr(k_weight, "weight") else k_weight
        assert q_weight is not None and k_weight is not None

        is_kv_shared_layer = (
            args[7]
            if len(args) > 7
            else kwargs.get("is_kv_shared_layer", kwargs.get("is_kv_shared", False))
        )
        is_k_eq_v = kwargs.get("is_k_eq_v", False)
        eps = kwargs.get("eps", 1e-6)

        head_dim = q_weight.shape[-1]
        num_heads = q_2d.shape[-1] // head_dim
        num_kv_heads = k_2d.shape[-1] // head_dim

        qkv = torch.cat([q_2d, k_2d, v_2d], dim=-1)
        out_q, out_k, out_v = dispatch_op(
            qkv,
            positions,
            cos_sin_cache,
            q_weight,
            k_weight,
            num_heads,
            num_kv_heads,
            head_dim,
            eps=eps,
            is_kv_shared_layer=is_kv_shared_layer,
            is_k_eq_v=is_k_eq_v,
        )
        if len(orig_shape) == 3:
            out_q = out_q.view(orig_shape[0], orig_shape[1], -1)
            out_k = out_k.view(orig_shape[0], orig_shape[1], -1)
            out_v = out_v.view(orig_shape[0], orig_shape[1], -1)
        return out_q, out_k, out_v

    # Pattern 1: (qkv, positions, cos_sin_cache, q_weight, k_weight, ...)
    qkv = args[0]
    positions = args[1] if len(args) > 1 else kwargs["positions"]
    rotary_arg = (
        args[2]
        if len(args) > 2
        else kwargs.get("cos_sin_cache", kwargs.get("rotary_emb"))
    )
    assert rotary_arg is not None
    cos_sin_cache = (
        rotary_arg.cos_sin_cache if hasattr(rotary_arg, "cos_sin_cache") else rotary_arg
    )

    q_weight = args[3] if len(args) > 3 else kwargs.get("q_weight", kwargs.get("wq"))
    k_weight = args[4] if len(args) > 4 else kwargs.get("k_weight", kwargs.get("wk"))
    assert q_weight is not None and k_weight is not None
    q_weight = q_weight.weight if hasattr(q_weight, "weight") else q_weight
    k_weight = k_weight.weight if hasattr(k_weight, "weight") else k_weight
    assert q_weight is not None and k_weight is not None

    head_dim = kwargs.get("head_dim")
    if head_dim is None:
        head_dim = args[7] if len(args) > 7 else q_weight.shape[-1]

    num_heads = kwargs.get("num_heads")
    num_kv_heads = kwargs.get("num_kv_heads")
    if num_heads is None:
        num_heads = args[5] if len(args) > 5 else None
    if num_kv_heads is None:
        num_kv_heads = args[6] if len(args) > 6 else None

    if num_heads is None or num_kv_heads is None:
        raise ValueError(
            "num_heads and num_kv_heads must be provided to fused_qkv_norm_rope"
        )

    eps = kwargs.get("eps", args[8] if len(args) > 8 else 1e-6)
    is_kv_shared_layer = kwargs.get(
        "is_kv_shared_layer",
        kwargs.get("is_kv_shared", args[9] if len(args) > 9 else False),
    )
    is_k_eq_v = kwargs.get("is_k_eq_v", False)

    return dispatch_op(
        qkv,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        num_heads,
        num_kv_heads,
        head_dim,
        eps=eps,
        is_kv_shared_layer=is_kv_shared_layer,
        is_k_eq_v=is_k_eq_v,
    )


def fused_post_attn_add_pre_ff_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor | nn.Module,
    pre_w: torch.Tensor | nn.Module,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuses post_attention_layernorm + residual add + pre_feedforward_layernorm."""
    dispatch_op = _dispatch(
        "gemma4_fused_post_attn_add_pre_ff_norm",
        _gemma4_fused_post_attn_add_pre_ff_norm,
    )
    pw = post_w.weight if hasattr(post_w, "weight") else post_w
    rw = pre_w.weight if hasattr(pre_w, "weight") else pre_w
    return dispatch_op(attn_out, residual, pw, rw, eps=eps)


def fused_mlp_ple_epilogue(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_weight: torch.Tensor | nn.Module,
    per_layer_input: torch.Tensor | None = None,
    per_layer_input_gate: nn.Module | None = None,
    per_layer_projection: nn.Module | None = None,
    post_ple_weight: torch.Tensor | nn.Module | None = None,
    layer_scalar: torch.Tensor | float = 1.0,
    next_norm_weight: torch.Tensor | nn.Module | None = None,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fuses MLP epilogue for dense layers with optional cross-layer norm."""
    pw = post_ff_weight.weight if hasattr(post_ff_weight, "weight") else post_ff_weight
    next_w = (
        next_norm_weight.weight
        if (next_norm_weight is not None and hasattr(next_norm_weight, "weight"))
        else next_norm_weight
    )

    if not isinstance(layer_scalar, torch.Tensor):
        layer_scalar_tensor = torch.tensor(
            [layer_scalar], dtype=torch.float32, device=mlp_out.device
        )
    elif layer_scalar.device != mlp_out.device:
        layer_scalar_tensor = layer_scalar.to(device=mlp_out.device)
    else:
        layer_scalar_tensor = layer_scalar

    if mlp_out.numel() == 0:
        if next_w is not None:
            return torch.empty_like(mlp_out), torch.empty_like(residual)
        return torch.empty_like(mlp_out)

    if (
        per_layer_input is None
        or per_layer_input_gate is None
        or per_layer_projection is None
    ):
        if next_w is not None:
            dispatch_cross = _dispatch(
                "gemma4_fused_post_ff_norm_add_scalar_cross",
                _gemma4_fused_post_ff_norm_add_scalar_cross,
            )
            return dispatch_cross(
                mlp_out, residual, pw, layer_scalar_tensor, next_w, eps=eps
            )
        dispatch_dense = _dispatch(
            "gemma4_fused_post_ff_norm_add_scalar",
            _gemma4_fused_post_ff_norm_add_scalar,
        )
        return dispatch_dense(mlp_out, residual, pw, layer_scalar_tensor, eps=eps)

    M, H = mlp_out.reshape(-1, mlp_out.shape[-1]).shape
    ple_dim = per_layer_input.shape[-1]

    # Use two-stage fused PLE projection for small batches (M <= 16).
    if (
        0 < M <= 16
        and H in (1536, 2560)
        and ple_dim == 256
        and hasattr(per_layer_input_gate, "weight")
        and hasattr(per_layer_projection, "weight")
    ):
        post_ple_w = (
            post_ple_weight.weight
            if (post_ple_weight is not None and hasattr(post_ple_weight, "weight"))
            else post_ple_weight
        )
        if post_ple_w is None:
            post_ple_w = torch.ones(H, dtype=mlp_out.dtype, device=mlp_out.device)

        dispatch_k1 = _dispatch(
            "gemma4_fused_post_ff_norm_add_ple_gate_gelu",
            _gemma4_fused_post_ff_norm_add_ple_gate_gelu,
        )
        h, gated_ple = dispatch_k1(
            mlp_out, residual, pw, per_layer_input, per_layer_input_gate.weight, eps=eps
        )

        dispatch_k2 = _dispatch(
            "gemma4_fused_ple_proj_norm_add_scalar_cross",
            _gemma4_fused_ple_proj_norm_add_scalar_cross,
        )
        dummy_next = (
            next_w
            if next_w is not None
            else torch.empty(0, dtype=mlp_out.dtype, device=mlp_out.device)
        )
        out_normed, out_res = dispatch_k2(
            gated_ple,
            h,
            per_layer_projection.weight,
            post_ple_w,
            layer_scalar_tensor,
            dummy_next,
            eps=eps,
        )
        if next_w is not None:
            return out_normed, out_res
        return out_res

    # Multi-SM PLE pipeline
    dispatch_add = _dispatch(
        "gemma4_fused_post_ff_norm_add", _gemma4_fused_post_ff_norm_add
    )
    h = dispatch_add(mlp_out, residual, pw, eps=eps)

    gate_raw = per_layer_input_gate(h)
    if isinstance(gate_raw, tuple):
        gate_raw = gate_raw[0]

    dispatch_gelu = _dispatch("gemma4_fused_gelu_tanh_mul", _gemma4_fused_gelu_tanh_mul)
    gated_ple = dispatch_gelu(gate_raw, per_layer_input)

    ple_proj = per_layer_projection(gated_ple)
    if isinstance(ple_proj, tuple):
        ple_proj = ple_proj[0]

    post_ple_w = (
        post_ple_weight.weight
        if (post_ple_weight is not None and hasattr(post_ple_weight, "weight"))
        else post_ple_weight
    )
    if post_ple_w is None:
        post_ple_w = torch.ones(
            mlp_out.shape[-1], dtype=mlp_out.dtype, device=mlp_out.device
        )

    if next_w is not None:
        dispatch_cross = _dispatch(
            "gemma4_fused_post_ple_norm_add_scalar_cross",
            _gemma4_fused_post_ple_norm_add_scalar_cross,
        )
        return dispatch_cross(
            ple_proj, h, post_ple_w, layer_scalar_tensor, next_w, eps=eps
        )

    dispatch_epilogue = _dispatch(
        "gemma4_fused_post_ple_norm_add_scalar",
        _gemma4_fused_post_ple_norm_add_scalar,
    )
    return dispatch_epilogue(ple_proj, h, post_ple_w, layer_scalar_tensor, eps=eps)


def fused_mlp_norm_and_moe_prenorm(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    w_post1: torch.Tensor | nn.Module,
    w_pre2: torch.Tensor | nn.Module,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuses post_feedforward_layernorm_1 on MLP out with
    pre_feedforward_layernorm_2 on residual."""
    dispatch_op = _dispatch(
        "gemma4_fused_mlp_norm_and_moe_prenorm",
        _gemma4_fused_mlp_norm_and_moe_prenorm,
    )
    wp1 = w_post1.weight if hasattr(w_post1, "weight") else w_post1
    wp2 = w_pre2.weight if hasattr(w_pre2, "weight") else w_pre2
    return dispatch_op(mlp_out, residual, wp1, wp2, eps=eps)


def fused_post_attn_add_moe_prenorms(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor | nn.Module,
    pre_ff_w: torch.Tensor | nn.Module,
    pre_moe_w: torch.Tensor | nn.Module,
    router_scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuses post_attn_norm, residual add, pre_ff_norm, pre_moe_norm,
    and router_norm & scale."""
    dispatch_op = _dispatch(
        "gemma4_fused_post_attn_add_moe_prenorms",
        _gemma4_fused_post_attn_add_moe_prenorms,
    )
    pw = post_w.weight if hasattr(post_w, "weight") else post_w
    pffw = pre_ff_w.weight if hasattr(pre_ff_w, "weight") else pre_ff_w
    pmoew = pre_moe_w.weight if hasattr(pre_moe_w, "weight") else pre_moe_w
    return dispatch_op(attn_out, residual, pw, pffw, pmoew, router_scale, eps=eps)


def fused_moe_combine_norm_ple_epilogue(
    h1: torch.Tensor,
    moe_out: torch.Tensor,
    residual: torch.Tensor,
    w_post2: torch.Tensor | nn.Module,
    w_post_ff: torch.Tensor | nn.Module,
    w_post1: torch.Tensor | nn.Module | None = None,
    per_layer_input: torch.Tensor | None = None,
    per_layer_input_gate: nn.Module | None = None,
    per_layer_projection: nn.Module | None = None,
    w_post_ple: torch.Tensor | nn.Module | None = None,
    layer_scalar: torch.Tensor | float = 1.0,
    next_norm_weight: torch.Tensor | nn.Module | None = None,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fuses MoE combination sum, post_feedforward_layernorm, residual add,
    and PLE epilogue."""
    wp2 = w_post2.weight if hasattr(w_post2, "weight") else w_post2
    wpff = w_post_ff.weight if hasattr(w_post_ff, "weight") else w_post_ff
    next_w = (
        next_norm_weight.weight
        if (next_norm_weight is not None and hasattr(next_norm_weight, "weight"))
        else next_norm_weight
    )

    if not isinstance(layer_scalar, torch.Tensor):
        layer_scalar_tensor = torch.tensor(
            [layer_scalar], dtype=torch.float32, device=h1.device
        )
    elif layer_scalar.device != h1.device:
        layer_scalar_tensor = layer_scalar.to(device=h1.device)
    else:
        layer_scalar_tensor = layer_scalar

    if h1.numel() == 0:
        if next_w is not None:
            return torch.empty_like(h1), torch.empty_like(h1)
        return torch.empty_like(h1)

    if (
        per_layer_input is None
        or per_layer_input_gate is None
        or per_layer_projection is None
    ):
        if w_post1 is not None:
            # Consolidated epilogue: mlp_out was passed as h1
            wp1 = w_post1.weight if hasattr(w_post1, "weight") else w_post1
            dispatch_cons = _dispatch(
                "gemma4_fused_moe_consolidated_epilogue",
                _gemma4_fused_moe_consolidated_epilogue,
            )
            normed_next, res_next = dispatch_cons(
                h1,
                moe_out,
                residual,
                wp1,
                wp2,
                wpff,
                layer_scalar_tensor,
                next_w,
                eps=eps,
            )
            return (normed_next, res_next) if next_w is not None else res_next

        if next_w is not None:
            dispatch_cross = _dispatch(
                "gemma4_fused_moe_combine_norm_add_scalar_cross",
                _gemma4_fused_moe_combine_norm_add_scalar_cross,
            )
            return dispatch_cross(
                h1, moe_out, residual, wp2, wpff, layer_scalar_tensor, next_w, eps=eps
            )

        dispatch_moe_scalar = _dispatch(
            "gemma4_fused_moe_combine_norm_add_scalar",
            _gemma4_fused_moe_combine_norm_add_scalar,
        )
        return dispatch_moe_scalar(
            h1, moe_out, residual, wp2, wpff, layer_scalar_tensor, eps=eps
        )

    # With PLE:
    if w_post1 is not None:
        wp1 = w_post1.weight if hasattr(w_post1, "weight") else w_post1
        mlp_f32 = h1.to(torch.float32)
        var1 = torch.mean(mlp_f32 * mlp_f32, dim=-1, keepdim=True)
        r1 = torch.rsqrt(var1 + eps)
        actual_h1 = (mlp_f32 * r1).to(h1.dtype) * wp1
    else:
        actual_h1 = h1

    dispatch_moe_add = _dispatch(
        "gemma4_fused_moe_combine_norm_add",
        _gemma4_fused_moe_combine_norm_add,
    )
    h = dispatch_moe_add(actual_h1, moe_out, residual, wp2, wpff, eps=eps)

    gate_raw = per_layer_input_gate(h)
    if isinstance(gate_raw, tuple):
        gate_raw = gate_raw[0]

    dispatch_gelu = _dispatch("gemma4_fused_gelu_tanh_mul", _gemma4_fused_gelu_tanh_mul)
    gated_ple = dispatch_gelu(gate_raw, per_layer_input)

    ple_proj = per_layer_projection(gated_ple)
    if isinstance(ple_proj, tuple):
        ple_proj = ple_proj[0]

    post_ple_w = (
        w_post_ple.weight
        if (w_post_ple is not None and hasattr(w_post_ple, "weight"))
        else w_post_ple
    )
    if post_ple_w is None:
        post_ple_w = torch.ones(
            actual_h1.shape[-1], dtype=actual_h1.dtype, device=actual_h1.device
        )

    if next_w is not None:
        dispatch_cross = _dispatch(
            "gemma4_fused_post_ple_norm_add_scalar_cross",
            _gemma4_fused_post_ple_norm_add_scalar_cross,
        )
        return dispatch_cross(
            ple_proj, h, post_ple_w, layer_scalar_tensor, next_w, eps=eps
        )

    dispatch_epilogue = _dispatch(
        "gemma4_fused_post_ple_norm_add_scalar",
        _gemma4_fused_post_ple_norm_add_scalar,
    )
    return dispatch_epilogue(ple_proj, h, post_ple_w, layer_scalar_tensor, eps=eps)


def prewarm_gemma4_fused_kernels(
    device: torch.device | str,
    dtype: torch.dtype,
    hidden_size: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    ple_dim: int | None = None,
) -> None:
    """Warm up fused Triton kernels before CUDAGraph capture."""
    if not torch.cuda.is_available():
        return
    device = torch.device(device)
    if device.type != "cuda":
        return

    logger.info("Pre-warming Gemma 4 fused Triton kernels for %s %s...", device, dtype)
    try:
        M = 2
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_qkv_size = q_size + 2 * kv_size

        qkv = torch.randn(M, total_qkv_size, dtype=dtype, device=device)
        positions = torch.zeros(M, dtype=torch.int64, device=device)
        cos_sin_cache = torch.ones(8, head_dim, dtype=dtype, device=device)
        wq = torch.ones(head_dim, dtype=dtype, device=device)
        wk = torch.ones(head_dim, dtype=dtype, device=device)

        # Prewarm Kernel 1 (non-shared, shared, Q-only shared, and k_eq_v)
        fused_qkv_norm_rope(
            qkv,
            positions,
            cos_sin_cache,
            wq,
            wk,
            num_heads,
            num_kv_heads,
            head_dim,
            is_kv_shared_layer=False,
        )
        fused_qkv_norm_rope(
            qkv,
            positions,
            cos_sin_cache,
            wq,
            wk,
            num_heads,
            num_kv_heads,
            head_dim,
            is_kv_shared_layer=True,
        )
        q_only = torch.randn(M, q_size, dtype=dtype, device=device)
        fused_qkv_norm_rope(
            q_only,
            positions,
            cos_sin_cache,
            wq,
            wk,
            num_heads,
            num_kv_heads,
            head_dim,
            is_kv_shared_layer=True,
        )
        qk = torch.randn(M, q_size + kv_size, dtype=dtype, device=device)
        fused_qkv_norm_rope(
            qk,
            positions,
            cos_sin_cache,
            wq,
            wk,
            num_heads,
            num_kv_heads,
            head_dim,
            is_k_eq_v=True,
        )

        # Prewarm Kernel 2
        attn_out = torch.randn(M, hidden_size, dtype=dtype, device=device)
        residual = torch.randn(M, hidden_size, dtype=dtype, device=device)
        post_w = torch.ones(hidden_size, dtype=dtype, device=device)
        pre_w = torch.ones(hidden_size, dtype=dtype, device=device)
        next_w = torch.ones(hidden_size, dtype=dtype, device=device)
        fused_post_attn_add_pre_ff_norm(attn_out, residual, post_w, pre_w)

        # Prewarm MoE Prenorms
        router_scale = torch.ones(hidden_size, dtype=dtype, device=device)
        fused_post_attn_add_moe_prenorms(
            attn_out, residual, post_w, pre_w, pre_w, router_scale
        )

        # Prewarm Kernel 3 (Dense non-PLE: standard & cross-layer)
        mlp_out = torch.randn(M, hidden_size, dtype=dtype, device=device)
        layer_scalar = torch.tensor([1.0], dtype=dtype, device=device)
        fused_mlp_ple_epilogue(mlp_out, residual, post_w, layer_scalar=layer_scalar)
        fused_mlp_ple_epilogue(
            mlp_out,
            residual,
            post_w,
            layer_scalar=layer_scalar,
            next_norm_weight=next_w,
        )

        # Prewarm Kernel 3 (Dense PLE if ple_dim is configured)
        if ple_dim and ple_dim > 0:
            ple_in = torch.randn(M, ple_dim, dtype=dtype, device=device)
            gate_raw = torch.randn(M, ple_dim, dtype=dtype, device=device)
            _gemma4_fused_gelu_tanh_mul(gate_raw, ple_in)
            _gemma4_fused_post_ple_norm_add_scalar(
                mlp_out, residual, post_w, layer_scalar
            )
            _gemma4_fused_post_ple_norm_add_scalar_cross(
                mlp_out, residual, post_w, layer_scalar, next_w
            )

            # Prewarm small-batch PLE kernels
            if hidden_size in (1536, 2560) and ple_dim == 256:
                w_gate = torch.randn(ple_dim, hidden_size, dtype=dtype, device=device)
                w_proj = torch.randn(hidden_size, ple_dim, dtype=dtype, device=device)
                h_pw, gated_pw = _gemma4_fused_post_ff_norm_add_ple_gate_gelu(
                    mlp_out, residual, post_w, ple_in, w_gate
                )
                _gemma4_fused_ple_proj_norm_add_scalar_cross(
                    gated_pw, h_pw, w_proj, post_w, layer_scalar, next_w
                )
                _gemma4_fused_ple_proj_norm_add_scalar_cross(
                    gated_pw,
                    h_pw,
                    w_proj,
                    post_w,
                    layer_scalar,
                    torch.empty(0, dtype=dtype, device=device),
                )

            # Prewarm PLE model projection norm and combine
            proj_3d = torch.randn(M, 26, ple_dim, dtype=dtype, device=device)
            embed_3d = torch.randn(M, 26, ple_dim, dtype=dtype, device=device)
            norm_w_ple = torch.ones(ple_dim, dtype=dtype, device=device)
            fused_ple_model_proj_norm_combine(
                proj_3d,
                norm_w_ple,
                embed_3d,
                num_layers=26,
                proj_scale=hidden_size**-0.5,
                input_scale=2.0**-0.5,
            )
            fused_ple_model_proj_norm_combine(
                proj_3d,
                norm_w_ple,
                None,
                num_layers=26,
                proj_scale=hidden_size**-0.5,
                input_scale=2.0**-0.5,
            )

        # Prewarm Kernel 3A & 3B (MoE)
        h1, moe_in = fused_mlp_norm_and_moe_prenorm(mlp_out, residual, post_w, pre_w)
        moe_out = torch.randn(M, hidden_size, dtype=dtype, device=device)
        fused_moe_combine_norm_ple_epilogue(
            h1, moe_out, residual, post_w, pre_w, layer_scalar=layer_scalar
        )
        fused_moe_combine_norm_ple_epilogue(
            mlp_out,
            moe_out,
            residual,
            post_w,
            pre_w,
            w_post1=post_w,
            layer_scalar=layer_scalar,
            next_norm_weight=next_w,
        )

        torch.cuda.synchronize(device)
        logger.info("Gemma 4 fused kernels successfully pre-warmed.")
    except Exception as e:
        logger.warning("Gemma 4 fused kernel pre-warming encountered: %s", e)
