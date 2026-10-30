# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Key re-rotation for --enable-unbounded-streaming-input.

When the scheduler shifts a session's positions down by d, its cached keys are
re-rotated by R(-d). RoPE scores only depend on relative positions, so they stay
the same apart from rounding. Each layer uses its own RoPE, read from the model
at startup.
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
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVQuantMode,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


@dataclass(frozen=True)
class RotarySpec:
    """How one attention layer rotates its keys."""

    # The layer's float32 frequencies, widened to float64: in float32,
    # d * inv_freq is off by up to 3e-3 rad at d ~ 1e5.
    inv_freq: torch.Tensor
    rotary_dim: int
    is_neox: bool
    # Positions this layer advances per scheduled token. 1 for a decoder;
    # Voxtral's block-pooled audio encoder runs at block_pool_size.
    position_scale: int


# RoPE classes whose angle at position p is p * inv_freq. YaRN scales cos and
# sin by the same mscale, which leaves the rotation itself unchanged.
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
    layer has no RoPE, or one whose angle is not p * inv_freq (M-RoPE, Fourier,
    long RoPE, ...), or one that does not cover its whole key.
    """
    owners: dict[str, tuple[nn.Module, Any, nn.Module]] = {}
    for module in model.modules():
        rope = getattr(module, "rotary_emb", None)
        if rope is None:
            continue
        for child in module.children():
            layer_name = getattr(child, "layer_name", None)
            if layer_name is not None:
                owners[layer_name] = (module, rope, child)

    specs: dict[str, RotarySpec] = {}
    for layer_name in layer_names:
        if layer_name not in owners:
            raise ValueError(
                "--enable-unbounded-streaming-input: no rotary_emb on the parent "
                f"module of attention layer {layer_name!r}."
            )
        module, rope, attn = owners[layer_name]
        get_inv_freq = _STATIC_ROPE_INV_FREQ.get(type(rope))
        if get_inv_freq is None:
            supported = ", ".join(cls.__name__ for cls in _STATIC_ROPE_INV_FREQ)
            raise ValueError(
                "--enable-unbounded-streaming-input does not support "
                f"{type(rope).__name__} (layer {layer_name!r}). "
                f"Supported: {supported}."
            )
        # The rotary dims must be the first dims of the key.
        if rope.head_size != attn.head_size:
            raise ValueError(
                f"--enable-unbounded-streaming-input: layer {layer_name!r} applies "
                "RoPE to part of its keys only."
            )
        # On the device of the forward table: pow differs by 1 ulp across devices.
        with torch.device(rope.cos_sin_cache.device):
            inv_freq = get_inv_freq(rope).to(torch.float64)
        specs[layer_name] = RotarySpec(
            inv_freq=inv_freq,
            rotary_dim=rope.rotary_dim,
            is_neox=rope.is_neox_style,
            position_scale=getattr(module, "reanchor_position_scale", 1),
        )
    return specs


def init_reanchor_rotary_specs(
    model: nn.Module,
    kv_cache_config: KVCacheConfig,
    forward_context: Mapping[str, Any],
) -> dict[str, RotarySpec]:
    """Check that the cached keys of every KV cache layer can be re-rotated in
    place. Returns the spec of each layer."""
    layer_names = []
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                # KV-sharing aliases have no spec of their own.
                spec = spec.kv_cache_specs.get(layer_name, spec.first_spec)
            if not isinstance(spec, SlidingWindowSpec) or isinstance(
                spec, SlidingWindowMLASpec
            ):
                raise ValueError(
                    "--enable-unbounded-streaming-input needs every attention layer "
                    f"to use a sliding window; {layer_name!r} uses "
                    f"{type(spec).__name__}."
                )
            layer = forward_context[layer_name]
            kv = layer.kv_cache
            if (
                spec.kv_quant_mode != KVQuantMode.NONE
                or spec.num_head_slots is not None
                or layer.kv_sharing_target_layer_name is not None
                or kv.dim() != 4
                or kv.shape[0] != kv_cache_config.num_blocks
                or kv.shape[-1] != 2 * layer.head_size
            ):
                raise ValueError(
                    "--enable-unbounded-streaming-input needs an unshared, "
                    "non-quantized KV cache of (blocks, heads, block_size, "
                    f"2 * head_size); layer {layer_name!r}: "
                    f"{spec.kv_quant_mode.name}, {tuple(kv.shape)}."
                )
            layer_names.append(layer_name)
    return collect_rotary_specs(model, layer_names)


def rerotate_cached_keys(
    forward_context: Mapping[str, Any],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    group_block_ids: Sequence[torch.Tensor],
    d: int,
    specs: Mapping[str, RotarySpec],
) -> None:
    """Apply R(-d) to one request's cached keys in every KV cache group.

    ``group_block_ids[g]`` is the request's block table row in group ``g``. The
    null blocks and the blocks allocated for this step hold no live keys, so
    rotating the whole row is fine.
    """
    for group, block_ids in zip(kv_cache_groups, group_block_ids):
        for layer_name in group.layer_names:
            layer = forward_context[layer_name]
            kv, head_size, spec = layer.kv_cache, layer.head_size, specs[layer_name]
            # Layer view (blocks, heads, block_size, 2 * head_size): K then V.
            kv[block_ids, ..., :head_size] = apply_inverse_rotary(
                kv[block_ids, ..., :head_size], spec.position_scale * d, spec
            )


def apply_inverse_rotary(key: torch.Tensor, d: int, spec: RotarySpec) -> torch.Tensor:
    """Apply R(-d) to ``key`` ([..., head_size], rotary dims first)."""
    angle = d * spec.inv_freq
    rotated = ApplyRotaryEmb.forward_static(
        key[..., : spec.rotary_dim], angle.cos(), -angle.sin(), spec.is_neox
    )
    return torch.cat((rotated, key[..., spec.rotary_dim :]), dim=-1)
