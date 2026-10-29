# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[EXPERIMENTAL] RoPE re-anchoring key re-rotation, shared between both model
runners (V1 and V2 ``_reanchor_requests``) and their unit tests
(benchmarks/voxtral_realtime/test_reanchor_math.py) so the tests exercise the
*production* op rather than a hand-copied duplicate.

Kept dependency-light (torch only, no worker graph) so the benchmark test can
import it without pulling the GPU model runner.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

# Voxtral/Ministral rope_theta. Startup validation (VllmConfig) rejects any model
# whose text/audio rope_theta != 1e6, that carries a real RoPE scaling, or whose
# decoder/encoder head_dim is not the validated 128-NeoX / 64-GPT-J pairing that
# rerotate_cached_keys keys off.
ROPE_THETA = 1.0e6


def audio_position_factor(hf_config: Any) -> int:
    """How many audio encoder positions one decoder position spans.

    The audio encoder's RoPE positions run at this factor x the decoder (whisper
    positions are expanded by it), so its constant shift is factor * D. It lives
    on audio_config (the Mistral adapter writes it there, not top-level).
    Startup validation rejects a non-derivable factor, so the trailing ``or 1``
    is unreachable on the re-anchor path.
    """
    audio_cfg = getattr(hf_config, "audio_config", None)
    return (
        getattr(audio_cfg, "block_pool_size", None)
        or getattr(audio_cfg, "downsample_factor", None)
        or 1
    )


def rerotate_cached_keys(
    forward_context: Mapping[str, Any],
    kv_cache_groups: Sequence[Any],
    group_block_ids: Sequence[torch.Tensor],
    d: int,
    audio_factor: int,
) -> None:
    """Re-rotate one request's cached keys by R(-d) in every KV group.

    ``group_block_ids[g]`` is the request's block-table row in KV group ``g``
    (long tensor on the KV cache device). The whole row is re-rotated: leading
    null blocks and this step's freshly allocated blocks hold no live keys, so
    rotating them is a harmless no-op, and R(-d) is position-independent, so a
    uniform rotation of every key in the row is exactly right.
    """
    # At most two distinct rotations per re-anchor (decoder 128-dim NeoX shifted
    # by d, audio encoder 64-dim GPT-J shifted by audio_factor * d); compute each
    # cos/sin table once instead of per layer (~60x).
    cos_sin: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for group, block_ids in zip(kv_cache_groups, group_block_ids):
        if block_ids.numel() == 0:
            continue
        for layer_name in group.layer_names:
            layer = forward_context[layer_name]
            kv = layer.kv_cache
            if isinstance(kv, (list, tuple)):
                kv = kv[0]
            # Standardized layer view (B, H, N, 2 * head_size): K is the first
            # head_size channels, V the rest. head_size comes from the layer,
            # not the tensor, and per layer: equal text/audio windows merge the
            # decoder and the audio encoder into one group.
            head_size = layer.head_size
            assert head_size in (64, 128), (
                f"re-anchor: unexpected rotary head_size {head_size}"
            )
            assert kv.dim() == 4 and kv.shape[-1] == 2 * head_size, (
                f"re-anchor: unexpected KV cache view {tuple(kv.shape)} "
                f"for head_size {head_size}"
            )
            assert kv.is_floating_point() and kv.element_size() >= 2, (
                "realtime re-anchoring requires a non-quantized KV cache"
            )
            is_neox = head_size == 128
            shift = d if is_neox else audio_factor * d
            cs = cos_sin.get((shift, head_size))
            if cs is None:
                cs = inverse_rotary_cos_sin(
                    shift, head_size, ROPE_THETA, kv.dtype, kv.device
                )
                cos_sin[(shift, head_size)] = cs
            key = kv[block_ids, ..., :head_size]
            kv[block_ids, ..., :head_size] = apply_inverse_rotary_cos_sin(
                key, cs[0], cs[1], head_size, is_neox
            )


def inverse_rotary_cos_sin(
    d: int, rotary_dim: int, base: float, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of the constant R(-d) re-rotation angles.

    Computed in fp64: at re-anchor distances d ~ 1e5 a fp32 angle accumulates
    ~1e-2 rad of error, which compounds across the cached keys re-rotated at
    every re-anchor. Split out from apply_inverse_rotary so the worker can
    compute the (at most two) tables once per re-anchor instead of per layer.
    """
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float64, device=device)
            / rotary_dim
        )
    )
    angle = float(d) * inv_freq
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


def apply_inverse_rotary(
    key: torch.Tensor, d: int, rotary_dim: int, base: float, is_neox: bool
) -> torch.Tensor:
    """Re-rotate an already position-rotated key by the constant R(-d).

    Used by RoPE re-anchoring (unbounded realtime). cos(-d.f)=cos(d.f),
    sin(-d.f)=-sin(d.f) -> R(-d)=R(d)^T. Validated to ~1e-6 against the vLLM
    rotary convention for both NeoX and GPT-J styles in
    benchmarks/voxtral_realtime/test_reanchor_math.py. ``key`` is
    [..., head_size] with the leading ``rotary_dim`` dims rotary.
    """
    cos, sin = inverse_rotary_cos_sin(d, rotary_dim, base, key.dtype, key.device)
    return apply_inverse_rotary_cos_sin(key, cos, sin, rotary_dim, is_neox)
