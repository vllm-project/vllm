# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Key re-rotation for --enable-unbounded-streaming-input.

When the scheduler shifts a session's positions down by d, its cached keys are
re-rotated by R(-d). RoPE scores only depend on relative positions, so they stay
the same apart from rounding. Each layer uses its own RoPE, read from the model
at startup.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.sched.output import SchedulerOutput
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
    # Positions this layer advances per scheduled token: 1, or the pool size of a
    # block-pooled layer (see AttentionSpec.tokens_per_state).
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
    model: nn.Module, position_scales: Mapping[str, int]
) -> dict[str, RotarySpec]:
    """Map each attention layer to the RoPE that rotated its cached keys.

    Follows the usual vLLM layout, where the module holding an attention layer
    (any child with a ``layer_name``) also holds its ``rotary_emb``. Raises if a
    layer has no RoPE, or one whose angle is not p * inv_freq (M-RoPE, Fourier,
    long RoPE, ...), or one built for another head size.
    """
    owners: dict[str, tuple[Any, nn.Module]] = {}
    for module in model.modules():
        rope = getattr(module, "rotary_emb", None)
        if rope is None:
            continue
        for child in module.children():
            layer_name = getattr(child, "layer_name", None)
            if layer_name is not None:
                owners[layer_name] = (rope, child)

    specs: dict[str, RotarySpec] = {}
    for layer_name, position_scale in position_scales.items():
        if layer_name not in owners:
            raise ValueError(
                "--enable-unbounded-streaming-input: no rotary_emb on the parent "
                f"module of attention layer {layer_name!r}."
            )
        rope, attn = owners[layer_name]
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
                f"--enable-unbounded-streaming-input: layer {layer_name!r} has head "
                f"size {attn.head_size}, its RoPE {rope.head_size}."
            )
        # On the device of the forward table: pow differs by 1 ulp across devices.
        with torch.device(rope.cos_sin_cache.device):
            inv_freq = get_inv_freq(rope).to(torch.float64)
        specs[layer_name] = RotarySpec(
            inv_freq=inv_freq,
            rotary_dim=rope.rotary_dim,
            is_neox=rope.is_neox_style,
            position_scale=position_scale,
        )
    return specs


def init_reanchor_rotary_specs(
    model: nn.Module,
    kv_cache_config: KVCacheConfig,
    forward_context: Mapping[str, Any],
) -> dict[str, RotarySpec]:
    """Check that the cached keys of every KV cache layer can be re-rotated in
    place. Returns the spec of each layer."""
    position_scales = {}
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
            position_scale = 1 / Fraction(spec.tokens_per_state)
            if (
                spec.kv_quant_mode != KVQuantMode.NONE
                or spec.num_head_slots is not None
                or position_scale.denominator != 1
                or layer.kv_sharing_target_layer_name is not None
                or kv.dim() != 4
                or kv.shape[0] != kv_cache_config.num_blocks
                or kv.shape[-1] != 2 * layer.head_size
            ):
                raise ValueError(
                    "--enable-unbounded-streaming-input needs an unshared, "
                    "non-quantized KV cache of (blocks, heads, block_size, "
                    f"2 * head_size); layer {layer_name!r}: "
                    f"{spec.kv_quant_mode.name}, {tuple(kv.shape)}, "
                    f"tokens_per_state {spec.tokens_per_state}."
                )
            position_scales[layer_name] = int(position_scale)
    return collect_rotary_specs(model, position_scales)


def rerotate_shifted_sessions(
    scheduler_output: SchedulerOutput,
    forward_context: Mapping[str, Any],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    specs: Mapping[str, RotarySpec],
    device: torch.device,
) -> None:
    """Re-rotate the cached keys of the sessions the scheduler shifted down in
    this step. A shifted session is always among the new requests."""
    reanchor_reqs = scheduler_output.reanchor_reqs
    assert reanchor_reqs is not None
    for new_req in scheduler_output.scheduled_new_reqs:
        d = reanchor_reqs.get(new_req.req_id)
        if d is not None:
            group_block_ids = [
                async_tensor_h2d(
                    [i for i in ids if i != NULL_BLOCK_ID], device, dtype=torch.long
                )
                for ids in new_req.block_ids
            ]
            rerotate_cached_keys(
                forward_context, kv_cache_groups, group_block_ids, d, specs
            )


def rerotate_cached_keys(
    forward_context: Mapping[str, Any],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    group_block_ids: Sequence[torch.Tensor],
    d: int,
    specs: Mapping[str, RotarySpec],
) -> None:
    """Apply R(-d) to one request's cached keys in every KV cache group.

    ``group_block_ids[g]`` holds the request's blocks in group ``g``. The blocks
    allocated for this step hold no live keys yet, so rotating them is fine.
    """
    for group, block_ids in zip(kv_cache_groups, group_block_ids, strict=True):
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
    # In fp32: cos and sin rounded to bf16 would give every key the same error, in
    # angle and in scale, at each frequency.
    rotated = ApplyRotaryEmb.forward_static(
        key[..., : spec.rotary_dim],
        angle.cos(),
        -angle.sin(),
        spec.is_neox,
        enable_fp32_compute=True,
    )
    return torch.cat((rotated, key[..., spec.rotary_dim :]), dim=-1)
