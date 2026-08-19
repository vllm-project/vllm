# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused QKV split, QK RMSNorm, RoPE, and paged KV-cache update."""

import torch

from vllm._aiter_ops import (
    is_aiter_fused_qkv_split_qk_norm_rope_cache_available,
    rocm_aiter_ops,
)
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout


def attn_layer_supports_gated_qk_norm_rope_kvcache(
    attn_layer: torch.nn.Module,
) -> bool:
    """Return whether the layer can use the gated QK-norm/RoPE/KV-cache fusion."""
    if not is_aiter_fused_qkv_split_qk_norm_rope_cache_available():
        return False
    if not rocm_aiter_ops.is_enabled():
        return False
    if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
        return False
    if attn_layer.head_size != attn_layer.head_size_v:
        return False
    if attn_layer.head_size <= 0 or (attn_layer.head_size & (attn_layer.head_size - 1)):
        return False
    kv_cache_dtype = getattr(attn_layer, "kv_cache_dtype", "auto")
    if is_quantized_kv_cache(kv_cache_dtype) and kv_cache_dtype not in (
        "fp8",
        "fp8_e4m3",
    ):
        return False
    if getattr(attn_layer, "sliding_window", None) is not None:
        return False
    if getattr(attn_layer, "has_sink", False):
        return False
    if getattr(attn_layer, "calculate_kv_scales", False):
        return False
    if getattr(attn_layer, "kv_sharing_target_layer_name", None) is not None:
        return False
    if getattr(attn_layer.attn_backend, "forward_includes_kv_cache_update", False):
        return False
    return hasattr(attn_layer.impl, "_split_kv_cache")


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
    configured_layout = (
        get_kv_cache_layout() if get_current_vllm_config_or_none() is not None else None
    )
    hnd_matches = key_cache.shape[1] == num_kv_heads and key_cache.shape[3] == head_dim
    nhd_matches = key_cache.shape[2] == num_kv_heads and key_cache.shape[3] == head_dim
    if configured_layout == "HND" and hnd_matches:
        return "HND"
    if configured_layout == "NHD" and nhd_matches:
        return "NHD"
    if configured_layout is None:
        if hnd_matches and not nhd_matches:
            return "HND"
        if nhd_matches and not hnd_matches:
            return "NHD"
    raise ValueError(
        "Unsupported KV cache layout for AITER fused QKV/RoPE/cache: "
        f"shape={tuple(key_cache.shape)}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, configured_layout={configured_layout!r}."
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
    x = (x_float * torch.rsqrt(variance + eps) * (1.0 + weight.float())).to(orig_dtype)

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


def run_gated_qk_norm_rope_kvcache(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run AITER fused QKV split + Gemma QK norm + RoPE + KV-cache write."""
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    num_tokens = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    if layer_slot_mapping is None or num_tokens == 0:
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
        )
        gate = gate.contiguous()
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
        k,
        v.reshape(num_tokens, kv_size),
        gate.reshape(num_tokens, q_size),
    )
