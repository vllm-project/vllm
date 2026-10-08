# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Gemma4 MTP (Multi-Token Prediction) model.

The Gemma4 assistant model is a lightweight decoder that shares KV cache
with the target (backbone) model.  All assistant decoder layers are
KV-shared: they only have Q projections (no K/V projections or norms),
and read K/V from the target model's cache at runtime.

Checkpoint layout (``gemma4_assistant``)::

    model.embed_tokens.*          -- token embeddings
    model.layers.{i}.*            -- decoder layers (Q-only attention + MLP)
    model.norm.*                  -- final RMSNorm
    pre_projection.*              -- Linear(2 * backbone_hidden_size, hidden_size)
    post_projection.*             -- Linear(hidden_size, backbone_hidden_size)
    lm_head.*                     -- language model head (tied to embed_tokens)
    masked_embedding.centroids.*  -- centroid projection (when use_ordered_embeddings)
    masked_embedding.token_ordering -- token-to-centroid mapping buffer
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.gemma4 import gemma4_layer_config
from vllm.triton_utils import HAS_TRITON, tl, triton

from .gemma4 import Gemma4MLP, _get_text_config
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    get_draft_quant_config,
    maybe_prefix,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Fused MTP Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_gather_gemv_kernel(
    hidden_states_ptr,
    lm_head_weight_ptr,
    selected_indices_ptr,
    out_ptr,
    stride_h_t,
    stride_h_h,
    stride_w_v,
    stride_w_h,
    stride_idx_t,
    stride_idx_s,
    stride_out_t,
    stride_out_s,
    T: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_s = tl.program_id(1).to(tl.int64)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    idx_ptrs = selected_indices_ptr + pid_t * stride_idx_t + offs_s * stride_idx_s
    vocab_idxs = tl.load(idx_ptrs, mask=mask_s, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(
            hidden_states_ptr + pid_t * stride_h_t + offs_h * stride_h_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        w_ptrs = (
            lm_head_weight_ptr
            + vocab_idxs[:, None] * stride_w_v
            + offs_h[None, :] * stride_w_h
        )
        mask_2d = mask_s[:, None] & mask_h[None, :]
        w_block = tl.load(w_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        acc += tl.sum(w_block * h_vals[None, :], axis=1)

    out_ptrs = out_ptr + pid_t * stride_out_t + offs_s * stride_out_s
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_s)


def fused_mtp_sparse_gather_gemv(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute sparse dot products einsum('td,tsd->ts') in-register."""
    T, H = hidden_states.shape
    S = selected_indices.shape[1]
    if T == 0 or S == 0:
        return torch.empty(
            (T, S), dtype=hidden_states.dtype, device=hidden_states.device
        )

    if not HAS_TRITON or not hidden_states.is_cuda:
        embeddings = lm_head_weight[selected_indices.reshape(-1)].view(T, S, H)
        return torch.einsum("td,tsd->ts", hidden_states, embeddings)

    if hidden_states.stride(-1) != 1:
        hidden_states = hidden_states.contiguous()
    if lm_head_weight.stride(-1) != 1:
        lm_head_weight = lm_head_weight.contiguous()
    if selected_indices.stride(-1) != 1:
        selected_indices = selected_indices.contiguous()

    out = torch.empty((T, S), dtype=hidden_states.dtype, device=hidden_states.device)
    BLOCK_S = 32
    BLOCK_H = 512

    grid = (T, triton.cdiv(S, BLOCK_S))
    _sparse_gather_gemv_kernel[grid](
        hidden_states,
        lm_head_weight,
        selected_indices,
        out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        lm_head_weight.stride(0),
        lm_head_weight.stride(1),
        selected_indices.stride(0),
        selected_indices.stride(1),
        out.stride(0),
        out.stride(1),
        T=T,
        S=S,
        H=H,
        BLOCK_S=BLOCK_S,
        BLOCK_H=BLOCK_H,
    )
    return out


@triton.jit
def _mtp_q_norm_rope_kernel(
    q_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_weight_ptr,
    out_q_ptr,
    stride_q_m,
    stride_cache_pos,
    stride_out_q_m,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    PASS_THROUGH_DIM: tl.constexpr,
    BLOCK_PASS: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)

    head_off = pid_h * HEAD_DIM
    offs_half = tl.arange(0, BLOCK_HALF)
    half_mask = offs_half < HALF_ROTARY_DIM

    pos = tl.load(positions_ptr + pid_m)
    cos = tl.load(
        cos_sin_cache_ptr + pos * stride_cache_pos + offs_half,
        mask=half_mask,
        other=1.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr + pos * stride_cache_pos + HALF_ROTARY_DIM + offs_half,
        mask=half_mask,
        other=0.0,
    )

    x1 = tl.load(
        q_ptr + pid_m * stride_q_m + head_off + offs_half,
        mask=half_mask,
        other=0.0,
    )
    x2 = tl.load(
        q_ptr + pid_m * stride_q_m + head_off + HALF_ROTARY_DIM + offs_half,
        mask=half_mask,
        other=0.0,
    )

    x1_f32 = x1.to(tl.float32)
    x2_f32 = x2.to(tl.float32)
    sum_sq = tl.sum(x1_f32 * x1_f32 + x2_f32 * x2_f32, axis=0)

    if PASS_THROUGH_DIM > 0:
        offs_pass = tl.arange(0, BLOCK_PASS)
        pass_mask = offs_pass < PASS_THROUGH_DIM
        x_pass = tl.load(
            q_ptr + pid_m * stride_q_m + head_off + ROTARY_DIM + offs_pass,
            mask=pass_mask,
            other=0.0,
        )
        x_pass_f32 = x_pass.to(tl.float32)
        sum_sq += tl.sum(x_pass_f32 * x_pass_f32, axis=0)

    var = sum_sq / HEAD_DIM
    r = tl.rsqrt(var + eps)

    out_dtype = out_q_ptr.dtype.element_ty
    x1_norm = (x1_f32 * r).to(out_dtype)
    x2_norm = (x2_f32 * r).to(out_dtype)

    w1 = tl.load(q_weight_ptr + offs_half, mask=half_mask, other=1.0)
    w2 = tl.load(q_weight_ptr + HALF_ROTARY_DIM + offs_half, mask=half_mask, other=1.0)
    x1_norm = x1_norm * w1
    x2_norm = x2_norm * w2

    cos_f32 = cos.to(tl.float32)
    sin_f32 = sin.to(tl.float32)
    x1_norm_f32 = x1_norm.to(tl.float32)
    x2_norm_f32 = x2_norm.to(tl.float32)
    o1 = (x1_norm_f32 * cos_f32 - x2_norm_f32 * sin_f32).to(out_dtype)
    o2 = (x2_norm_f32 * cos_f32 + x1_norm_f32 * sin_f32).to(out_dtype)

    out_off = pid_m * stride_out_q_m + head_off
    tl.store(out_q_ptr + out_off + offs_half, o1, mask=half_mask)
    tl.store(out_q_ptr + out_off + HALF_ROTARY_DIM + offs_half, o2, mask=half_mask)

    if PASS_THROUGH_DIM > 0:
        offs_pass = tl.arange(0, BLOCK_PASS)
        pass_mask = offs_pass < PASS_THROUGH_DIM
        w_pass = tl.load(
            q_weight_ptr + ROTARY_DIM + offs_pass, mask=pass_mask, other=1.0
        )
        o_pass = ((x_pass_f32 * r).to(out_dtype) * w_pass).to(out_dtype)
        tl.store(out_q_ptr + out_off + ROTARY_DIM + offs_pass, o_pass, mask=pass_mask)


def fused_mtp_q_norm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuses Q unflatten, per-head RMSNorm, NeoX RoPE, and flatten into 1 launch."""
    orig_shape = q.shape
    q_2d = q.reshape(-1, q.shape[-1])
    M, H = q_2d.shape
    if M == 0:
        return torch.empty_like(q)

    rotary_dim = cos_sin_cache.shape[-1]
    if not HAS_TRITON or not q.is_cuda or H > 16384:
        q_heads = q.unflatten(-1, (num_heads, head_dim))
        variance = q_heads.pow(2).mean(-1, keepdim=True)
        q_normed = (q_heads * torch.rsqrt(variance + eps)).to(q.dtype) * q_weight
        q_rot = q_normed[..., :rotary_dim]
        q_pass = q_normed[..., rotary_dim:] if rotary_dim < head_dim else None
        half_dim = rotary_dim // 2
        cos_sin = cos_sin_cache[positions]
        cos = cos_sin[..., :half_dim].unsqueeze(-2)
        sin = cos_sin[..., half_dim:].unsqueeze(-2)
        q1 = q_rot[..., :half_dim]
        q2 = q_rot[..., half_dim:]
        o1 = (q1.float() * cos.float() - q2.float() * sin.float()).to(q.dtype)
        o2 = (q2.float() * cos.float() + q1.float() * sin.float()).to(q.dtype)
        o_rot = torch.cat([o1, o2], dim=-1)
        o = torch.cat([o_rot, q_pass], dim=-1) if q_pass is not None else o_rot
        return o.flatten(-2, -1)

    if q_2d.stride(-1) != 1:
        q_2d = q_2d.contiguous()
    pos_1d = positions.reshape(-1)
    if pos_1d.stride(0) != 1:
        pos_1d = pos_1d.contiguous()
    if cos_sin_cache.stride(-1) != 1:
        cos_sin_cache = cos_sin_cache.contiguous()
    if q_weight.stride(-1) != 1:
        q_weight = q_weight.contiguous()

    out_q = torch.empty_like(q_2d)
    half_rotary_dim = rotary_dim // 2
    pass_through_dim = head_dim - rotary_dim
    BLOCK_HALF = triton.next_power_of_2(half_rotary_dim)
    BLOCK_PASS = triton.next_power_of_2(max(1, pass_through_dim))

    grid = (M, num_heads)
    _mtp_q_norm_rope_kernel[grid](
        q_2d,
        pos_1d,
        cos_sin_cache,
        q_weight,
        out_q,
        q_2d.stride(0),
        cos_sin_cache.stride(0),
        out_q.stride(0),
        num_heads=num_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY_DIM=half_rotary_dim,
        BLOCK_HALF=BLOCK_HALF,
        PASS_THROUGH_DIM=pass_through_dim,
        BLOCK_PASS=BLOCK_PASS,
        eps=eps,
    )
    return out_q.reshape(orig_shape)


@triton.jit
def _mtp_post_attn_add_pre_ff_norm_kernel(
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
    m = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    out_dtype = out_res_ptr.dtype.element_ty

    attn = tl.load(attn_out_ptr + m * stride_attn_m + offs, mask=mask, other=0.0)
    attn_f32 = attn.to(tl.float32)
    var1 = tl.sum(attn_f32 * attn_f32, axis=0) / H
    r1 = tl.rsqrt(var1 + eps)
    normed_attn = (attn_f32 * r1).to(out_dtype)

    post_w = tl.load(post_w_ptr + offs, mask=mask, other=0.0)
    normed_attn = normed_attn * post_w

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    new_res = normed_attn + res
    tl.store(out_res_ptr + m * stride_out_res_m + offs, new_res, mask=mask)

    new_res_f32 = new_res.to(tl.float32)
    var2 = tl.sum(new_res_f32 * new_res_f32, axis=0) / H
    r2 = tl.rsqrt(var2 + eps)
    normed_res = (new_res_f32 * r2).to(out_dtype)

    pre_w = tl.load(pre_w_ptr + offs, mask=mask, other=0.0)
    pre_ff = normed_res * pre_w
    tl.store(out_pre_ff_ptr + m * stride_pre_m + offs, pre_ff, mask=mask)


def fused_mtp_post_attn_add_pre_ff_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_w: torch.Tensor,
    pre_w: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuses post_attention_layernorm + residual add + pre_feedforward_layernorm."""
    orig_shape = attn_out.shape
    attn_2d = attn_out.reshape(-1, attn_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = attn_2d.shape
    if M == 0:
        return torch.empty_like(attn_out), torch.empty_like(residual)

    if not HAS_TRITON or not attn_out.is_cuda or H > 16384:
        var1 = attn_out.pow(2).mean(-1, keepdim=True)
        res = (attn_out * torch.rsqrt(var1 + eps)).to(
            attn_out.dtype
        ) * post_w + residual
        var2 = res.pow(2).mean(-1, keepdim=True)
        pre_ff = (res * torch.rsqrt(var2 + eps)).to(res.dtype) * pre_w
        return pre_ff, res

    if attn_2d.stride(-1) != 1:
        attn_2d = attn_2d.contiguous()
    if res_2d.stride(-1) != 1:
        res_2d = res_2d.contiguous()
    if post_w.stride(-1) != 1:
        post_w = post_w.contiguous()
    if pre_w.stride(-1) != 1:
        pre_w = pre_w.contiguous()

    out_pre_ff = torch.empty_like(attn_2d)
    out_res = torch.empty_like(res_2d)
    BLOCK_SIZE = triton.next_power_of_2(H)
    num_warps = 8 if BLOCK_SIZE >= 4096 else 4

    _mtp_post_attn_add_pre_ff_norm_kernel[(M,)](
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
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps,
        num_warps=num_warps,
    )
    return out_pre_ff.reshape(orig_shape), out_res.reshape(orig_shape)


@triton.jit
def _mtp_post_ff_epilogue_kernel(
    mlp_out_ptr,
    residual_ptr,
    post_ff_w_ptr,
    next_norm_w_ptr,
    out_res_ptr,
    out_normed_ptr,
    layer_scalar_ptr,
    layer_scalar_val,
    stride_mlp_m,
    stride_res_m,
    stride_out_res_m,
    stride_normed_m,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
    HAS_NEXT_NORM: tl.constexpr,
    IS_SCALAR_TENSOR: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    out_dtype = out_res_ptr.dtype.element_ty

    scalar = (
        tl.load(layer_scalar_ptr).to(tl.float32)
        if IS_SCALAR_TENSOR
        else layer_scalar_val
    )

    mlp = tl.load(mlp_out_ptr + m * stride_mlp_m + offs, mask=mask, other=0.0)
    mlp_f32 = mlp.to(tl.float32)
    var = tl.sum(mlp_f32 * mlp_f32, axis=0) / H
    r = tl.rsqrt(var + eps)
    normed_mlp = (mlp_f32 * r).to(out_dtype)

    post_w = tl.load(post_ff_w_ptr + offs, mask=mask, other=0.0)
    normed_mlp = normed_mlp * post_w

    res = tl.load(residual_ptr + m * stride_res_m + offs, mask=mask, other=0.0)
    new_res = ((normed_mlp.to(tl.float32) + res.to(tl.float32)) * scalar).to(out_dtype)
    tl.store(out_res_ptr + m * stride_out_res_m + offs, new_res, mask=mask)

    if HAS_NEXT_NORM:
        new_res_f32 = new_res.to(tl.float32)
        var_next = tl.sum(new_res_f32 * new_res_f32, axis=0) / H
        r_next = tl.rsqrt(var_next + eps)
        normed_next = (new_res_f32 * r_next).to(out_dtype)

        next_w = tl.load(next_norm_w_ptr + offs, mask=mask, other=0.0)
        normed_next = normed_next * next_w
        tl.store(out_normed_ptr + m * stride_normed_m + offs, normed_next, mask=mask)


def fused_mtp_post_ff_epilogue(
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    post_ff_w: torch.Tensor,
    layer_scalar: torch.Tensor | float,
    next_norm_w: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuses post_ff norm + residual add + scalar + optional next norm."""
    orig_shape = mlp_out.shape
    mlp_2d = mlp_out.reshape(-1, mlp_out.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    M, H = mlp_2d.shape
    if M == 0:
        return (
            torch.empty_like(mlp_out),
            torch.empty_like(residual) if next_norm_w is not None else None,
        )

    is_scalar_tensor = isinstance(layer_scalar, torch.Tensor) and layer_scalar.is_cuda
    if is_scalar_tensor:
        layer_scalar_ptr = layer_scalar
        layer_scalar_val = 1.0
    else:
        layer_scalar_ptr = mlp_2d
        layer_scalar_val = (
            float(layer_scalar.item())
            if isinstance(layer_scalar, torch.Tensor)
            else float(layer_scalar)
        )

    if not HAS_TRITON or not mlp_out.is_cuda or H > 16384:
        var = mlp_out.pow(2).mean(-1, keepdim=True)
        scalar = (
            layer_scalar if isinstance(layer_scalar, torch.Tensor) else layer_scalar_val
        )
        res = (
            (mlp_out * torch.rsqrt(var + eps)).to(mlp_out.dtype) * post_ff_w + residual
        ) * scalar
        if next_norm_w is not None:
            var_next = res.pow(2).mean(-1, keepdim=True)
            normed_next = (res * torch.rsqrt(var_next + eps)).to(
                res.dtype
            ) * next_norm_w
            return normed_next, res
        return res, None

    if mlp_2d.stride(-1) != 1:
        mlp_2d = mlp_2d.contiguous()
    if res_2d.stride(-1) != 1:
        res_2d = res_2d.contiguous()
    if post_ff_w.stride(-1) != 1:
        post_ff_w = post_ff_w.contiguous()
    if next_norm_w is not None and next_norm_w.stride(-1) != 1:
        next_norm_w = next_norm_w.contiguous()

    out_res = torch.empty_like(res_2d)
    out_normed = torch.empty_like(mlp_2d) if next_norm_w is not None else mlp_2d
    BLOCK_SIZE = triton.next_power_of_2(H)
    num_warps = 8 if BLOCK_SIZE >= 4096 else 4

    _mtp_post_ff_epilogue_kernel[(M,)](
        mlp_2d,
        res_2d,
        post_ff_w,
        next_norm_w if next_norm_w is not None else post_ff_w,
        out_res,
        out_normed,
        layer_scalar_ptr,
        layer_scalar_val,
        mlp_2d.stride(0),
        res_2d.stride(0),
        out_res.stride(0),
        out_normed.stride(0),
        H=H,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps,
        HAS_NEXT_NORM=next_norm_w is not None,
        IS_SCALAR_TENSOR=is_scalar_tensor,
        num_warps=num_warps,
    )
    if next_norm_w is not None:
        return out_normed.reshape(orig_shape), out_res.reshape(orig_shape)
    return out_res.reshape(orig_shape), None


@triton.jit
def _mtp_embed_scale_cat_kernel(
    raw_embeds_ptr,
    hidden_states_ptr,
    out_ptr,
    normalizer_ptr,
    normalizer_val,
    stride_raw_m,
    stride_hid_m,
    stride_out_m,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NORM_TENSOR: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_chunk = tl.program_id(1).to(tl.int64)

    offs_h = pid_chunk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    out_dtype = out_ptr.dtype.element_ty

    norm = tl.load(normalizer_ptr).to(tl.float32) if IS_NORM_TENSOR else normalizer_val

    raw = tl.load(raw_embeds_ptr + pid_m * stride_raw_m + offs_h, mask=mask, other=0.0)
    scaled = (raw.to(tl.float32) * norm).to(out_dtype)
    tl.store(out_ptr + pid_m * stride_out_m + offs_h, scaled, mask=mask)

    hid = tl.load(
        hidden_states_ptr + pid_m * stride_hid_m + offs_h, mask=mask, other=0.0
    )
    tl.store(out_ptr + pid_m * stride_out_m + H + offs_h, hid.to(out_dtype), mask=mask)


def fused_mtp_embed_scale_cat(
    raw_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    normalizer: torch.Tensor | float,
) -> torch.Tensor:
    """Fuses raw_embeds * normalizer and concat([scaled, hidden_states], dim=-1)."""
    orig_shape = raw_embeds.shape
    raw_2d = raw_embeds.reshape(-1, raw_embeds.shape[-1])
    hid_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    M, H = raw_2d.shape
    out_shape = (*orig_shape[:-1], 2 * H)
    if M == 0:
        return torch.empty(out_shape, dtype=raw_embeds.dtype, device=raw_embeds.device)

    is_norm_tensor = isinstance(normalizer, torch.Tensor) and normalizer.is_cuda
    if is_norm_tensor:
        normalizer_ptr = normalizer
        normalizer_val = 1.0
    else:
        normalizer_ptr = raw_2d
        normalizer_val = (
            float(normalizer.item())
            if isinstance(normalizer, torch.Tensor)
            else float(normalizer)
        )

    if not HAS_TRITON or not raw_embeds.is_cuda or H > 16384:
        norm = normalizer if isinstance(normalizer, torch.Tensor) else normalizer_val
        return torch.cat([raw_embeds * norm, hidden_states], dim=-1)

    if raw_2d.stride(-1) != 1:
        raw_2d = raw_2d.contiguous()
    if hid_2d.stride(-1) != 1:
        hid_2d = hid_2d.contiguous()

    out = torch.empty((M, 2 * H), dtype=raw_embeds.dtype, device=raw_embeds.device)
    BLOCK_H = 1024
    grid = (M, triton.cdiv(H, BLOCK_H))

    _mtp_embed_scale_cat_kernel[grid](
        raw_2d,
        hid_2d,
        out,
        normalizer_ptr,
        normalizer_val,
        raw_2d.stride(0),
        hid_2d.stride(0),
        out.stride(0),
        H=H,
        BLOCK_H=BLOCK_H,
        IS_NORM_TENSOR=is_norm_tensor,
    )
    return out.reshape(out_shape)


class Gemma4MTPMaskedEmbedder(nn.Module):
    """Sparse logit computation via centroid-based vocabulary masking.

    Instead of computing logits against the full vocabulary, projects
    hidden states to centroid scores, selects top-K centroids, and
    computes logits only for the ~top_k * (vocab_size / num_centroids)
    tokens belonging to those centroids.
    """

    token_ordering: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid

        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        self.register_buffer(
            "token_ordering",
            torch.empty(vocab_size, dtype=torch.long),
        )

    def _select_and_score(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Centroid selection + sparse dot product.

        Returns:
            logits: (num_tokens, num_selected) sparse logits.
            indices: (num_tokens, num_selected) corresponding vocab indices.

        """
        num_tokens = hidden_states.shape[0]
        _, top_k_indices = torch.topk(
            self.centroids(hidden_states),
            k=self.centroid_intermediate_top_k,
            dim=-1,
        )
        clusters = self.token_ordering.view(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected = clusters[top_k_indices]
        selected_flat = selected.view(num_tokens, -1)
        logits = fused_mtp_sparse_gather_gemv(
            hidden_states, lm_head_weight, selected_flat
        )
        return logits, selected_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Full-vocab logits with non-selected positions masked to -inf."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        output = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            fill_value=torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.scatter_(-1, indices, logits)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse argmax — returns vocab token IDs without full-vocab tensor."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)


class Gemma4MTPAttention(nn.Module):
    """Q-only attention for Gemma4 MTP layers.

    K/V come from the target model's KV cache via
    ``kv_sharing_target_layer_name`` (set by the proposer after
    model construction).
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.scaling = 1.0

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        # kv_sharing_target_layer_name is set after model construction
        # by Gemma4Proposer._setup_gemma4_kv_sharing().
        self.is_kv_shared_layer = True
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)

        q = fused_mtp_q_norm_rope(
            q,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.q_norm.weight,
            self.num_heads,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
        )

        # Attention reads K/V from the target's cache via KV sharing.
        attn_output = self.attn(q, None, None)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MTPDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        layer_idx = extract_layer_index(prefix)
        layer_config = gemma4_layer_config(config, layer_idx)
        head_dim = layer_config.head_dim
        num_kv_heads = layer_config.num_key_value_heads

        self.self_attn = Gemma4MTPAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
        )

        text_config = _get_text_config(config)
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_activation=text_config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        next_norm_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        pre_ff, residual = fused_mtp_post_attn_add_pre_ff_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.pre_feedforward_layernorm.weight,
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        mlp_out = self.mlp(pre_ff)

        hidden_states, residual = fused_mtp_post_ff_epilogue(
            mlp_out,
            residual,
            self.post_feedforward_layernorm.weight,
            layer_scalar=self.layer_scalar,
            next_norm_w=next_norm_weight,
            eps=self.post_feedforward_layernorm.variance_epsilon,
        )
        return hidden_states, residual


class Gemma4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        text_config = _get_text_config(config)
        quant_config = get_draft_quant_config(vllm_config)
        self.config = text_config
        self.quant_config = quant_config

        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = getattr(
            config, "backbone_hidden_size", self.hidden_size
        )
        self.vocab_size = text_config.vocab_size
        self.num_mtp_layers = text_config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.pre_projection = ColumnParallelLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.pre_projection",
        )

        self.post_projection = RowParallelLinear(
            self.hidden_size,
            self.backbone_hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=quant_config,
            prefix=f"{prefix}.post_projection",
        )

        self.layers = nn.ModuleList(
            Gemma4MTPDecoderLayer(
                text_config,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.norm = RMSNorm(self.hidden_size, eps=text_config.rms_norm_eps)

        # After embedding sharing, embed_tokens is replaced with the
        # target model's backbone-dim embedding.  Scale by
        # sqrt(backbone_hidden_size) to match the target's convention.
        self.register_buffer(
            "normalizer",
            torch.tensor(self.backbone_hidden_size**0.5),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (draft_hidden_states, backbone_hidden_states).

        draft_hidden_states: draft-dim, used by compute_logits via lm_head.
        backbone_hidden_states: backbone-dim, stored in the proposer's
            hidden-state buffer and fed back as input to the next step.
        """
        if inputs_embeds is None:
            assert input_ids is not None
            raw_embeds = self.embed_tokens(input_ids)
            combined = fused_mtp_embed_scale_cat(
                raw_embeds, hidden_states, self.normalizer
            )
        else:
            combined = torch.cat([inputs_embeds, hidden_states], dim=-1)

        hidden_states, _ = self.pre_projection(combined)

        residual = None
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            layer_next_norm_w = (
                self.layers[idx + 1].input_layernorm.weight
                if idx + 1 < num_layers
                else self.norm.weight
            )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                next_norm_weight=layer_next_norm_w,
            )

        if residual is None:
            draft_hidden_states = self.norm(hidden_states)
        else:
            draft_hidden_states = hidden_states

        backbone_hidden_states, _ = self.post_projection(draft_hidden_states)
        return draft_hidden_states, backbone_hidden_states


@support_torch_compile
class Gemma4MTP(nn.Module):
    """Gemma4 Multi-Token Prediction model for speculative decoding.

    forward() returns (draft_hidden_states, backbone_hidden_states).
    The proposer uses draft_hidden_states for compute_logits (via
    the draft-dim lm_head) and backbone_hidden_states for the
    hidden-state feedback buffer.
    """

    has_own_lm_head = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "pre_projection.": "model.pre_projection.",
            "post_projection.": "model.post_projection.",
        },
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        text_config = _get_text_config(config)
        self.quant_config = get_draft_quant_config(vllm_config)
        self.config = config
        self._stable_full_lm_head_weight: torch.Tensor | None = None

        self.model = Gemma4MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "draft_model"),
        )

        # lm_head operates in draft-dim.  Tied to embed_tokens at init
        # so load_weights populates both from a single checkpoint entry.
        # After embedding sharing, lm_head.weight still references the
        # original draft-dim tensor.
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=getattr(text_config, "final_logit_softcapping", None),
        )

        self.masked_embedding: Gemma4MTPMaskedEmbedder | None
        if getattr(config, "use_ordered_embeddings", False):
            num_centroids = getattr(config, "num_centroids", 2048)
            top_k = getattr(config, "centroid_intermediate_top_k", 32)
            self.masked_embedding = Gemma4MTPMaskedEmbedder(
                hidden_size=text_config.hidden_size,
                vocab_size=text_config.vocab_size,
                num_centroids=num_centroids,
                centroid_intermediate_top_k=top_k,
            )
            logger.info(
                "Gemma4 MTP: centroids masking enabled "
                "(num_centroids=%d, top_k=%d, active_tokens=%d/%d).",
                num_centroids,
                top_k,
                top_k * (text_config.vocab_size // num_centroids),
                text_config.vocab_size,
            )
        else:
            self.masked_embedding = None

        draft_cfg = speculative_config.draft_model_config
        gen_cfg = draft_cfg.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None
        # Materialized on-device in load_weights: compute_logits runs under CUDA
        # graph capture in the V2 speculator, where indexing with a Python list
        # would issue an unpinned H2D copy (illegal during capture).
        self._suppress_idx: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )

    def _get_full_lm_head_weight(self) -> torch.Tensor:
        if self._stable_full_lm_head_weight is not None:
            return self._stable_full_lm_head_weight
        assert self.masked_embedding is not None
        lm_head_weight = self.lm_head.weight
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            lm_head_weight = tensor_model_parallel_all_gather(
                lm_head_weight,
                dim=0,
            )
        lm_head_weight = lm_head_weight[: self.masked_embedding.vocab_size]
        if tp_size > 1:
            lm_head_weight = lm_head_weight.contiguous()
            self._stable_full_lm_head_weight = lm_head_weight
        return lm_head_weight

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        if self.masked_embedding is not None:
            logits = self.masked_embedding(
                hidden_states,
                self._get_full_lm_head_weight(),
            )
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self._suppress_idx is not None:
            logits.index_fill_(1, self._suppress_idx, -float("inf"))
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse argmax via centroids masking. Returns token IDs directly."""
        assert self.masked_embedding is not None
        return self.masked_embedding.get_top_tokens(
            hidden_states,
            self._get_full_lm_head_weight(),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._stable_full_lm_head_weight = None
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if self._suppress_token_ids:
            self._suppress_idx = torch.tensor(
                self._suppress_token_ids,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        return loaded
