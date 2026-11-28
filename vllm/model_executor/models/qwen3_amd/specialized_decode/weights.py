# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight binding helpers for the Qwen3 RDNA dense specialized decode."""

from __future__ import annotations

import torch

from .backend import _Layer


def _head_dim(config) -> int:
    return getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )


def _row_major_bf16(
    name: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, int],
    keep: list[torch.Tensor],
) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} dtype {tensor.dtype} is not BF16")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} does not match {expected_shape}"
        )
    if tensor.is_contiguous():
        return tensor
    packed = tensor.contiguous()
    keep.append(packed)
    return packed


def build_layer_bindings(owner, qk_norm: bool, keep: list[torch.Tensor]):
    model = owner.model
    config = owner.config
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = _head_dim(config)
    q = config.num_attention_heads * head_dim
    kv = config.num_key_value_heads * head_dim

    layers = (_Layer * len(model.layers))()
    for idx, layer in enumerate(model.layers):
        qkv = _row_major_bf16(
            "qkv_proj.weight",
            layer.self_attn.qkv_proj.weight,
            (q + 2 * kv, hidden_size),
            keep,
        )
        gate_up = _row_major_bf16(
            "gate_up_proj.weight",
            layer.mlp.gate_up_proj.weight,
            (2 * intermediate_size, hidden_size),
            keep,
        )
        o_proj = _row_major_bf16(
            "o_proj.weight",
            layer.self_attn.o_proj.weight,
            (hidden_size, hidden_size),
            keep,
        )
        down_proj = _row_major_bf16(
            "down_proj.weight",
            layer.mlp.down_proj.weight,
            (hidden_size, intermediate_size),
            keep,
        )
        attn_norm = layer.input_layernorm.weight.float().contiguous()
        ffn_norm = layer.post_attention_layernorm.weight.float().contiguous()
        keep.extend([attn_norm, ffn_norm])
        q_norm = k_norm = None
        if qk_norm:
            q_norm = layer.self_attn.q_norm.weight.float().contiguous()
            k_norm = layer.self_attn.k_norm.weight.float().contiguous()
            keep.extend([q_norm, k_norm])
        base = qkv.data_ptr()
        layers[idx] = _Layer(
            base,
            base + q * hidden_size * 2,
            base + (q + kv) * hidden_size * 2,
            o_proj.data_ptr(),
            gate_up.data_ptr(),
            gate_up.data_ptr() + intermediate_size * hidden_size * 2,
            down_proj.data_ptr(),
            attn_norm.data_ptr(),
            ffn_norm.data_ptr(),
            q_norm.data_ptr() if q_norm is not None else None,
            k_norm.data_ptr() if k_norm is not None else None,
        )
    return layers


def final_norm(owner, keep: list[torch.Tensor]) -> torch.Tensor:
    out = owner.model.norm.weight.float().contiguous()
    keep.append(out)
    return out


def rope_cache(owner, keep: list[torch.Tensor]) -> torch.Tensor:
    rope = owner.model.layers[0].self_attn.rotary_emb
    cache = rope.cos_sin_cache
    if cache.dtype != torch.bfloat16 or not cache.is_contiguous():
        cache = cache.to(torch.bfloat16).contiguous()
        keep.append(cache)
    return cache
