# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hadamard helpers for INT4 KV cache quantization."""

from __future__ import annotations

import math

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

_H_CACHE: dict[tuple[int, torch.dtype, str], torch.Tensor] = {}

# This fallback is only intended for realistic INT4 attention head sizes.
_MAX_DENSE_FALLBACK_D = 1 << 15


def _has_hadacore_transform() -> bool:
    return hasattr(torch.ops._C, "hadacore_transform")


def _build_normalized_hadamard_matrix(
    d: int,
    device: torch.device,
) -> torch.Tensor:
    H = torch.ones((1, 1), dtype=torch.float32, device=device)
    while H.shape[0] < d:
        H = torch.cat(
            (
                torch.cat((H, H), dim=1),
                torch.cat((H, -H), dim=1),
            ),
            dim=0,
        )
    return H / math.sqrt(d)


def _get_hadamard_matrix(
    d: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached normalized Hadamard matrix of shape ``(d, d)``.

    The cached matrix follows the same ``H / sqrt(d)`` convention as
    ``hadacore_transform`` so applying the transform twice is the identity.
    """
    if d <= 0:
        raise ValueError(f"hadamard_transform requires positive last dim, got {d}")
    if d & (d - 1):
        raise ValueError(f"hadamard_transform requires power-of-2 last dim, got {d}")
    if d > _MAX_DENSE_FALLBACK_D:
        raise ValueError(
            "hadamard_transform Tier-2 dense fallback is not intended for "
            f"d={d} > {_MAX_DENSE_FALLBACK_D}. Use hadacore_transform on CUDA."
        )

    key = (d, dtype, str(device))
    if key not in _H_CACHE:
        H = _build_normalized_hadamard_matrix(d, device).to(dtype=dtype)
        _H_CACHE[key] = H.contiguous()
    return _H_CACHE[key]


def hadamard_transform(
    x: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply a normalized Hadamard transform.

    Tier-1 uses ``hadacore_transform`` on CUDA. Tier-2 falls back to a cached
    dense matmul on non-CUDA platforms. The fallback is intended for the INT4
    KV cache path and prioritizes correctness over performance.
    """
    if x.numel() == 0:
        return x

    d = x.shape[-1]
    if (
        current_platform.is_cuda()
        and x.device.type == "cuda"
        and _has_hadacore_transform()
    ):
        flat = x.reshape(-1, d)
        if inplace:
            if flat.is_contiguous() and flat.data_ptr() == x.data_ptr():
                ops.hadacore_transform(flat, inplace=True)
                return x
            work = x.contiguous().reshape(-1, d)
            ops.hadacore_transform(work, inplace=True)
            x.copy_(work.view_as(x))
            return x

        work = flat.contiguous().clone()
        ops.hadacore_transform(work, inplace=True)
        return work.view_as(x)

    matmul_dtype = x.dtype if x.is_floating_point() else torch.float32
    H = _get_hadamard_matrix(d, matmul_dtype, x.device)
    flat = x.reshape(-1, d)
    transformed = flat.to(matmul_dtype) @ H
    transformed = transformed.to(x.dtype).view_as(x)

    if inplace:
        x.copy_(transformed)
        return x
    return transformed
