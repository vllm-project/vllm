# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""PyTorch reference implementation of the composite Hadamard transform."""

from __future__ import annotations

import math

import torch


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply an unnormalized Walsh-Hadamard transform to the last axis."""
    output = x
    stride = 1
    while stride < x.shape[-1]:
        pairs = output.reshape(*x.shape[:-1], x.shape[-1] // (2 * stride), 2, stride)
        left = pairs[..., 0, :]
        right = pairs[..., 1, :]
        output = torch.cat((left + right, left - right), dim=-1).reshape(x.shape)
        stride *= 2
    return output.reshape(x.shape)


def composite_hadamard_torch(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    *,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Apply the composite Hadamard transform using PyTorch operations."""
    hidden_size = x.shape[-1]
    if hidden_size != k * p:
        raise ValueError("hidden_size must equal K * P")

    output = x.to(torch.float32)
    sign = sign.to(device=x.device, dtype=torch.float32)
    hk = hk.to(device=x.device, dtype=torch.float32)

    if not inverse:
        output = output * sign

    output = _fwht(output.reshape(*x.shape[:-1], k, p))
    output = output.transpose(-2, -1)
    output = output @ (hk.mT if inverse else hk)
    output = output.transpose(-2, -1).reshape(x.shape)
    output = output * (1.0 / math.sqrt(hidden_size))

    if inverse:
        output = output * sign
    return output.to(dtype=x.dtype)


__all__ = ["composite_hadamard_torch"]
