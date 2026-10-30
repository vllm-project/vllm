# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[EXPERIMENTAL] RoPE re-anchoring: re-rotate a live request's cached keys by
the constant R(-d) so its position clock can drop by d while every relative
position, and so every attention score, stays the same.

Model-agnostic: each attention layer is re-rotated with its own RoPE
(frequencies, rotary dim, NeoX or GPT-J style), read from the model once at
startup by ``collect_rotary_specs``. Shared by both model runners and their
unit tests.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs


@dataclass(frozen=True, eq=False)
class RotarySpec:
    """How one attention layer rotates its keys."""

    # The layer's own RoPE frequencies (computed in float32 like its forward
    # pass, stored in float64 so d * inv_freq stays exact at large d).
    inv_freq: torch.Tensor
    rotary_dim: int
    is_neox: bool
    # Positions this layer advances per scheduled token. 1 for a decoder;
    # Voxtral's block-pooled audio encoder runs at block_pool_size.
    position_scale: int = 1


# RoPE classes whose angle at position p is exactly p * inv_freq. YaRN scales
# cos and sin by the same mscale, which leaves the rotation itself unchanged.
_STATIC_ROPE_INV_FREQ = {
    RotaryEmbedding: lambda rope: rope._compute_inv_freq(rope.base),
    Llama3RotaryEmbedding: lambda rope: rope._compute_inv_freq(rope.base),
    YaRNScalingRotaryEmbedding: lambda rope: rope._compute_inv_freq(
        rope.scaling_factor
    ),
}


def collect_rotary_specs(
    model: nn.Module, layer_names: Iterable[str]
) -> dict[str, RotarySpec]:
    """Map each attention layer to the RoPE that rotated its cached keys.

    Follows the usual vLLM layout, where the module holding an attention layer
    (any child with a ``layer_name``) also holds its ``rotary_emb``. Raises if a
    layer has no RoPE, or one whose rotation is not a plain function of the
    position (dynamic NTK, M-RoPE, ...).
    """
    owners: dict[str, tuple[nn.Module, Any]] = {}
    for module in model.modules():
        rope = getattr(module, "rotary_emb", None)
        if rope is None:
            continue
        for child in module.children():
            layer_name = getattr(child, "layer_name", None)
            if layer_name is not None:
                owners[layer_name] = (module, rope)

    # Layers sharing a RoPE share one spec, so the worker computes each
    # cos/sin table once per re-anchor instead of once per layer.
    shared: dict[tuple[int, int], RotarySpec] = {}
    specs: dict[str, RotarySpec] = {}
    for layer_name in layer_names:
        if layer_name not in owners:
            raise ValueError(
                f"RoPE re-anchoring: no rotary embedding found for attention "
                f"layer {layer_name!r}."
            )
        module, rope = owners[layer_name]
        get_inv_freq = _STATIC_ROPE_INV_FREQ.get(type(rope))
        if get_inv_freq is None:
            supported = ", ".join(cls.__name__ for cls in _STATIC_ROPE_INV_FREQ)
            raise ValueError(
                f"RoPE re-anchoring does not support {type(rope).__name__} "
                f"(layer {layer_name!r}). Supported: {supported}."
            )
        scale = int(getattr(module, "reanchor_position_scale", 1))
        key = (id(rope), scale)
        if key not in shared:
            shared[key] = RotarySpec(
                inv_freq=get_inv_freq(rope).to("cpu", torch.float64),
                rotary_dim=rope.rotary_dim,
                is_neox=rope.is_neox_style,
                position_scale=scale,
            )
        specs[layer_name] = shared[key]
    return specs


def init_reanchor_rotary_specs(
    model: nn.Module, kv_cache_config: KVCacheConfig
) -> dict[str, RotarySpec]:
    """Startup check for --enable-unbounded-streaming, run once the KV cache
    groups are known: every group must be sliding-window (so re-anchoring can
    drop what fell out of the window) and every layer must have a RoPE it can
    invert. Returns the specs the worker re-rotates with."""
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if getattr(spec, "sliding_window", None) is None:
            raise ValueError(
                "--enable-unbounded-streaming needs every attention layer to "
                f"use a sliding window; layers {group.layer_names[:3]}... use "
                f"{type(spec).__name__}."
            )
    layer_names = [
        name for group in kv_cache_config.kv_cache_groups for name in group.layer_names
    ]
    return collect_rotary_specs(model, layer_names)


def rerotate_cached_keys(
    forward_context: Mapping[str, Any],
    kv_cache_groups: Sequence[Any],
    group_block_ids: Sequence[torch.Tensor],
    d: int,
    specs: Mapping[str, RotarySpec],
) -> None:
    """Re-rotate one request's cached keys by R(-d) in every KV group.

    ``group_block_ids[g]`` is the request's block-table row in KV group ``g``
    (long tensor on the KV cache device). The whole row is re-rotated: leading
    null blocks and this step's freshly allocated blocks hold no live keys, so
    rotating them is a harmless no-op, and R(-d) is position-independent, so a
    uniform rotation of every key in the row is exactly right.
    """
    cos_sin: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for group, block_ids in zip(kv_cache_groups, group_block_ids):
        if block_ids.numel() == 0:
            continue
        for layer_name in group.layer_names:
            layer = forward_context[layer_name]
            kv = layer.kv_cache
            if isinstance(kv, (list, tuple)):
                kv = kv[0]
            # Standardized layer view (B, H, N, 2 * head_size): K is the first
            # head_size channels, V the rest.
            head_size = layer.head_size
            assert kv.dim() == 4 and kv.shape[-1] == 2 * head_size, (
                f"re-anchor: unexpected KV cache view {tuple(kv.shape)} "
                f"for head_size {head_size}"
            )
            assert kv.is_floating_point() and kv.element_size() >= 2, (
                "RoPE re-anchoring requires a non-quantized KV cache"
            )
            spec = specs[layer_name]
            cs = cos_sin.get(id(spec))
            if cs is None:
                cs = inverse_rotary_cos_sin(
                    spec.position_scale * d, spec.inv_freq, kv.dtype, kv.device
                )
                cos_sin[id(spec)] = cs
            key = kv[block_ids, ..., :head_size]
            kv[block_ids, ..., :head_size] = apply_inverse_rotary_cos_sin(
                key, cs[0], cs[1], spec.rotary_dim, spec.is_neox
            )


def inverse_rotary_cos_sin(
    d: int, inv_freq: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of the constant R(-d) re-rotation angles.

    Computed in float64: at d ~ 1e5 a float32 angle accumulates ~1e-2 rad of
    error, which compounds across the cached keys re-rotated at every
    re-anchor.
    """
    angle = float(d) * inv_freq.to(device, torch.float64)
    return torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)


def apply_inverse_rotary_cos_sin(
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
) -> torch.Tensor:
    """Apply R(-d) to ``key`` given precomputed inverse_rotary_cos_sin tables."""
    half = rotary_dim // 2
    rot, passthru = key[..., :rotary_dim], key[..., rotary_dim:]
    if is_neox:
        x1, x2 = rot[..., :half], rot[..., half:]
        out = torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
    else:  # GPT-J interleaved
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        o1, o2 = x1 * cos + x2 * sin, -x1 * sin + x2 * cos
        out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([out, passthru], dim=-1) if passthru.numel() else out


def apply_inverse_rotary(key: torch.Tensor, d: int, spec: RotarySpec) -> torch.Tensor:
    """Re-rotate an already position-rotated key by the constant R(-d).

    cos(-d.f)=cos(d.f), sin(-d.f)=-sin(d.f) -> R(-d)=R(d)^T. ``key`` is
    [..., head_size] with the leading ``spec.rotary_dim`` dims rotary.
    """
    cos, sin = inverse_rotary_cos_sin(d, spec.inv_freq, key.dtype, key.device)
    return apply_inverse_rotary_cos_sin(key, cos, sin, spec.rotary_dim, spec.is_neox)
