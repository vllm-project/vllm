# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""vLLM custom-op wrapper for the TileLang Hadamard kernel."""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def composite_hadamard_tilelang(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h_gemm: torch.Tensor,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Execute the TileLang Hadamard transform."""
    if x.reshape(-1, x.shape[-1]).shape[0] == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    # Keep TileLang out of the model import path and compile on first use.
    from .tilelang_kernels import composite_hadamard as composite_hadamard_impl

    return composite_hadamard_impl(
        x,
        sign,
        inverse=inverse,
        hk=hk,
        h_gemm=h_gemm,
        k=k,
        p=p,
    )


def _composite_hadamard_tilelang_fake(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h_gemm: torch.Tensor,
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
    h_gemm: torch.Tensor,
    *,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Apply the registered TileLang Hadamard custom op."""
    return torch.ops.vllm.composite_hadamard_tilelang(
        x,
        sign,
        hk,
        h_gemm,
        inverse,
        k,
        p,
    )


__all__ = ["composite_hadamard", "composite_hadamard_tilelang"]
