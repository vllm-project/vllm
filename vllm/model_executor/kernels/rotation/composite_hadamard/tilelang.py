# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Composite rotation using HadaCore and a TileLang mixing/sign epilogue."""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from .torch import composite_hadamard_torch
from .utils import resolve_hadamard_dim


def composite_hadamard_tilelang(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Apply the composite transform; FP32 uses the PyTorch reference."""
    if x.numel() == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    if x.dtype == torch.float32:
        return composite_hadamard_torch(x, sign, hk, inverse=inverse, k=k, p=p)
    if not 32 <= p <= 32768:
        raise ValueError("HadaCore rotation requires 32 <= P <= 32768")

    from vllm import _custom_ops as ops

    from .warmup import COMPOSITE_HADAMARD_EPILOGUE

    values = x.contiguous()
    if not inverse:
        values = values * sign
    values = ops.hadacore_transform(values.reshape(-1, p), inplace=False)
    values = values.reshape(-1, k * p)
    output = torch.empty_like(values)
    kernel = COMPOSITE_HADAMARD_EPILOGUE(k, p, inverse, x.dtype)
    kernel(values, sign, hk, output)
    return output.reshape(x.shape)


def _composite_hadamard_tilelang_fake(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


direct_register_custom_op(
    op_name="composite_hadamard_tilelang",
    op_func=composite_hadamard_tilelang,
    mutates_args=[],
    fake_impl=_composite_hadamard_tilelang_fake,
)


def composite_hadamard(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    *,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Apply the registered composite Hadamard custom op."""
    resolve_hadamard_dim(x.shape[-1], k=k, p=p)
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("Hadamard supports bfloat16, float16, and float32")
    if hk.shape != (k, k):
        raise ValueError(f"hk must have shape ({k}, {k})")
    return torch.ops.vllm.composite_hadamard_tilelang(
        x,
        sign,
        hk,
        inverse,
        k,
        p,
    )


__all__ = ["composite_hadamard", "composite_hadamard_tilelang"]
