# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[EXPERIMENTAL] RoPE re-anchoring key re-rotation, shared between the worker
(GPUModelRunner._reanchor_requests) and its unit test
(benchmarks/voxtral_realtime/test_reanchor_math.py) so the test exercises the
*production* op rather than a hand-copied duplicate.

Kept dependency-light (torch only, no worker graph) so the benchmark test can
import it without pulling the GPU model runner.
"""

import torch


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
