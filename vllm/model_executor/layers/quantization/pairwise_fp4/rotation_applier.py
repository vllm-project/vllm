# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def apply_givens_rotation(
    tensor: torch.Tensor,
    pairs: torch.Tensor,
    angles: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            f"pairs must have shape (N, 2), got {pairs.shape}")
    if angles.ndim != 1:
        raise ValueError(
            f"angles must have shape (N,), got {angles.shape}")
    if pairs.shape[0] != angles.shape[0]:
        raise ValueError(
            f"pairs and angles must have the same length, "
            f"got {pairs.shape[0]} and {angles.shape[0]}")

    out = tensor.clone()

    n = pairs.shape[0]
    if n == 0:
        return out

    if inverse:
        order = range(n - 1, -1, -1)
        sign = -1.0
    else:
        order = range(n)
        sign = 1.0

    for k in order:
        theta = angles[k].float() * sign
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        i = pairs[k, 0].item()
        j = pairs[k, 1].item()

        xi = out[..., i].clone()
        xj = out[..., j].clone()

        out[..., i] = xi * cos_t + xj * sin_t
        out[..., j] = -xi * sin_t + xj * cos_t

    return out
