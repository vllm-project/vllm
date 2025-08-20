# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# vllm/_rope_fastpath.py
# Prototype RoPE fast-path helper:
# - Baseline: even/odd indexing
# - Fast-path: treat pairs as complex numbers to reduce indexing/allocs
# NOTE: Not wired into runtime yet. Used by tests/bench to validate approach.

from __future__ import annotations

import torch


def rope_torch_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference RoPE using even/odd indexing.
    Assumes last dim == head_dim and head_dim is even.
    """
    assert q.shape[-1] % 2 == 0 and k.shape[-1] % 2 == 0, (
        "head_dim must be even"
    )

    # Split even/odd lanes.
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Broadcast cos/sin over leading dims as needed.
    q_out[..., ::2] = q1 * cos - q2 * sin
    q_out[..., 1::2] = q1 * sin + q2 * cos
    k_out[..., ::2] = k1 * cos - k2 * sin
    k_out[..., 1::2] = k1 * sin + k2 * cos
    return q_out, k_out


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    # [..., hd] -> [..., hd/2, 2] -> complex
    x = x.view(*x.shape[:-1], -1, 2).contiguous()
    return torch.view_as_complex(x)


def _from_complex(xc: torch.Tensor) -> torch.Tensor:
    # [..., hd/2] (complex) -> [..., hd/2, 2] -> [..., hd]
    xr = torch.view_as_real(xc)
    return xr.reshape(*xr.shape[:-2], -1)


def rope_complex_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prototype fast-path using complex multiply:
    (x_even + i*x_odd) * (cos + i*sin).
    """
    # view_as_complex is not supported for fp16/bf16 on CPU → fall back.
    if q.device.type == "cpu" and q.dtype in (torch.float16, torch.bfloat16):
        return rope_torch_baseline(q, k, cos, sin)

    assert q.shape[-1] % 2 == 0 and k.shape[-1] % 2 == 0, (
        "head_dim must be even"
    )

    # Build complex rotation (cos + i*sin).
    # cos/sin must be broadcastable to the pair dimension.
    rot = torch.complex(cos, sin)

    q_out = _from_complex(_to_complex(q) * rot)
    k_out = _from_complex(_to_complex(k) * rot)
    return q_out, k_out
