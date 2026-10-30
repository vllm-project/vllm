# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Key re-rotation for --enable-unbounded-streaming.

When the scheduler shifts a session's positions down by d, its cached keys are
re-rotated by R(-d). RoPE scores only depend on relative positions, so they do
not change. Each layer uses its own RoPE, read from the model at startup.
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

    # The layer's RoPE frequencies: float32 like its forward pass, stored as
    # float64 for d * inv_freq at large d.
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

    # Layers sharing a RoPE share a spec: cos/sin are computed once per shift.
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
    """Check that every KV cache group uses a sliding window and every layer a
    supported RoPE. Returns the spec of each layer."""
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
    """Apply R(-d) to one request's cached keys in every KV cache group.

    ``group_block_ids[g]`` is the request's block table row in group ``g``. The
    null blocks and the blocks allocated for this step hold no live keys, so
    rotating the whole row is fine.
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
            # Layer view (blocks, heads, block_size, 2 * head_size): K then V.
            head_size = layer.head_size
            assert kv.dim() == 4 and kv.shape[-1] == 2 * head_size, (
                f"unexpected KV cache shape {tuple(kv.shape)} for head_size {head_size}"
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
    """cos/sin of the R(-d) angles. In float64: in float32, d * inv_freq is off
    by ~1e-2 rad at d ~ 1e5."""
    angle = float(d) * inv_freq.to(device, torch.float64)
    return torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)


def apply_inverse_rotary_cos_sin(
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
) -> torch.Tensor:
    """Apply R(-d) to ``key`` with tables from inverse_rotary_cos_sin."""
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
    """Apply R(-d) to ``key`` ([..., head_size], rotary dims first)."""
    cos, sin = inverse_rotary_cos_sin(d, spec.inv_freq, key.dtype, key.device)
    return apply_inverse_rotary_cos_sin(key, cos, sin, spec.rotary_dim, spec.is_neox)
