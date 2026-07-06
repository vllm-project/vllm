# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused QKV split + QK norm + RoPE + paged KV-cache update."""

from functools import cache

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import AttentionType


@cache
def is_aiter_fused_qkv_norm_rope_cache_available() -> bool:
    if (
        not current_platform.is_rocm()
        or not rocm_aiter_ops.is_qwen3_qkv_rope_cache_enabled()
    ):
        return False
    try:
        from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (  # noqa: F401
            fused_qkv_split_qk_norm_rope_cache,
        )
    except ImportError:
        return False
    return True


def can_use_aiter_fused_qkv_norm_rope_cache(
    *,
    attn_output_gate: bool,
    rotary_emb: torch.nn.Module,
    attn: torch.nn.Module,
    text_only: bool,
    attn_type: str,
) -> bool:
    """Return whether the Qwen3.5/Qwen3Next fused AITER path is safe to use."""
    if not is_aiter_fused_qkv_norm_rope_cache_available():
        return False
    if not (
        attn_output_gate
        and text_only
        and attn_type == AttentionType.DECODER
        and getattr(rotary_emb, "is_neox_style", False)
    ):
        return False

    # The kernel handles the NHD/HND paged layouts used by normal ROCm attention
    # and fp8 per-tensor KV cache. Keep less common layouts on the stock path.
    if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
        return False
    if getattr(attn, "head_size", None) != getattr(attn, "head_size_v", None):
        return False
    kv_cache_dtype = getattr(attn, "kv_cache_dtype", "auto")
    if kv_cache_dtype not in ("auto", "fp8", "fp8_e4m3"):
        return False
    if getattr(attn, "sliding_window", None) is not None:
        return False
    if getattr(attn, "has_sink", False):
        return False
    if getattr(attn, "calculate_kv_scales", False):
        return False
    if getattr(attn, "kv_sharing_target_layer_name", None) is not None:
        return False
    if getattr(attn.attn_backend, "forward_includes_kv_cache_update", False):
        return False
    if not hasattr(attn.impl, "_split_kv_cache"):
        return False
    return True


def _get_kv_cache_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError(
            "AITER fused QKV/RoPE/cache expects rank-4 K/V cache tensors "
            f"with matching shapes, got {tuple(key_cache.shape)} and "
            f"{tuple(value_cache.shape)}."
        )
    if key_cache.shape[1] == num_kv_heads and key_cache.shape[3] == head_dim:
        return "HND"
    if key_cache.shape[2] == num_kv_heads and key_cache.shape[3] == head_dim:
        return "NHD"
    raise ValueError(
        "Unsupported KV cache layout for AITER fused QKV/RoPE/cache: "
        f"shape={tuple(key_cache.shape)}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}."
    )


def _gemma_rmsnorm_rope_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    is_neox: bool,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x = (x_float * torch.rsqrt(variance + eps) * (1.0 + weight.float())).to(
        orig_dtype
    )

    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    rotary_dim = cos.shape[-1] * 2
    cos = cos[positions].float()[:, None, :]
    sin = sin[positions].float()[:, None, :]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    if is_neox:
        x1, x2 = x_rot.chunk(2, dim=-1)
    else:
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
    o1 = x1.float() * cos - x2.float() * sin
    o2 = x2.float() * cos + x1.float() * sin
    if is_neox:
        rotated = torch.cat((o1, o2), dim=-1).to(orig_dtype)
    else:
        rotated = torch.stack((o1, o2), dim=-1).flatten(-2).to(orig_dtype)
    return torch.cat((rotated, x_pass), dim=-1)


def qwen3_aiter_fused_qkv_norm_rope_cache_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run AITER fused QKV split + Gemma QK norm + RoPE + KV-cache write."""
    del rotary_dim
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    num_tokens = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    if layer_slot_mapping is None or num_tokens == 0:
        # Profiling/no-cache path: compute the same q/k/gate values without
        # writing KV cache.
        q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
        q_gate = q_gate.view(num_tokens, num_heads, 2 * head_dim)
        q = q_gate[:, :, :head_dim]
        gate = q_gate[:, :, head_dim:].reshape(num_tokens, q_size)
        k = k.view(num_tokens, num_kv_heads, head_dim)
        pos = positions[0] if positions.ndim == 2 else positions
        q = _gemma_rmsnorm_rope_ref(
            q, q_weight, cos_sin_cache, pos, eps, is_neox
        ).reshape(num_tokens, q_size)
        k = _gemma_rmsnorm_rope_ref(
            k, k_weight, cos_sin_cache, pos, eps, is_neox
        ).reshape(num_tokens, kv_size)
        return (
            torch.empty(0, device=qkv.device, dtype=qkv.dtype),
            q,
            k,
            v,
            gate,
        )

    from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
        fused_qkv_split_qk_norm_rope_cache,
    )

    key_cache, value_cache = attn_layer.impl._split_kv_cache(  # type: ignore[attr-defined]
        kv_cache
    )
    if is_quantized_kv_cache(attn_layer.kv_cache_dtype):
        key_cache = key_cache.view(attn_layer.impl.fp8_dtype)  # type: ignore[attr-defined]
        value_cache = value_cache.view(attn_layer.impl.fp8_dtype)  # type: ignore[attr-defined]

    kv_cache_layout = _get_kv_cache_layout(
        key_cache, value_cache, num_kv_heads, head_dim
    )
    pos = positions[0] if positions.ndim == 2 else positions
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    q, gate, k, v = fused_qkv_split_qk_norm_rope_cache(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        pos,
        key_cache,
        value_cache,
        layer_slot_mapping,
        num_heads,
        num_kv_heads,
        head_dim,
        is_neox=is_neox,
        reuse_freqs_front_part=True,
        attn_output_gate=True,
        k_scale=attn_layer._k_scale,
        v_scale=attn_layer._v_scale,
        eps=eps,
        gated_qkv_layout="interleaved",
        kv_cache_layout=kv_cache_layout,
    )

    return (
        torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype),
        q.reshape(num_tokens, q_size),
        k.reshape(num_tokens, kv_size),
        v.reshape(num_tokens, kv_size),
        gate.reshape(num_tokens, q_size),
    )


def qwen3_aiter_fused_qkv_norm_rope_cache_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        layer_name,
        rotary_dim,
        eps,
        is_neox,
    )
    num_tokens = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    return (
        torch.empty(0, device=qkv.device, dtype=qkv.dtype),
        torch.empty((num_tokens, q_size), dtype=qkv.dtype, device=qkv.device),
        torch.empty((num_tokens, kv_size), dtype=qkv.dtype, device=qkv.device),
        torch.empty((num_tokens, kv_size), dtype=qkv.dtype, device=qkv.device),
        torch.empty((num_tokens, q_size), dtype=qkv.dtype, device=qkv.device),
    )


direct_register_custom_op(
    op_name="qwen3_aiter_fused_qkv_norm_rope_cache",
    op_func=qwen3_aiter_fused_qkv_norm_rope_cache_impl,
    mutates_args=[],
    fake_impl=qwen3_aiter_fused_qkv_norm_rope_cache_fake,
)
