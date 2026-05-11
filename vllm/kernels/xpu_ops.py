# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

current_platform.import_kernels()


def is_xpu_kernels_found() -> bool:
    from importlib.util import find_spec

    return find_spec("vllm_xpu_kernels") is not None


XPU_KERNELS_SUPPORTED = is_xpu_kernels_found()
"""Kernels in this file are supported if vLLM XPU kernels are installed."""

rms_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None and (
    weight is None or weight.dtype == x.dtype
)


@ir.ops.rms_norm.register_impl(
    "xpu_kernels", supports_args=rms_no_var, supported=XPU_KERNELS_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output


rms_add_no_var_size = (
    lambda x, x_residual, weight, epsilon, variance_size=None: variance_size is None
    and (weight is None or weight.dtype == x.dtype)
)


@ir.ops.fused_add_rms_norm.register_impl(
    "xpu_kernels",
    supports_args=rms_add_no_var_size,
    supported=XPU_KERNELS_SUPPORTED,
    inplace=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)

    assert variance_size is None
    torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
    return x, x_residual


@ir.ops.rotary_embedding.register_impl(
    "xpu_kernels",
    supported=XPU_KERNELS_SUPPORTED,
    inplace=True,
)
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    offsets: Tensor | None = None,
    cos_sin_format: str = "standard",
    inverse: bool = False,
    rope_dim_offset: int = 0,
) -> tuple[Tensor, Tensor]:
    if cos_sin_format == "standard":
        torch.ops._C.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
            rope_dim_offset,
            inverse,
        )
        return query, key

    if cos_sin_format != "deepseek":
        raise ValueError(f"Unsupported cos_sin_format={cos_sin_format!r}")

    return torch.ops.vllm.xpu_ops_deepseek_scaling_rope(
        positions,
        query,
        key,
        offsets,
        cos_sin_cache,
        rotary_dim,
        is_neox_style,
    )


def _rotary_query_only_xpu_supported(
    positions: Tensor,
    query: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    offsets: Tensor | None = None,
    cos_sin_format: str = "standard",
    inverse: bool = False,
    rope_dim_offset: int = 0,
) -> bool:
    # XPU deepseek kernel requires a key tensor; only support standard format
    # for query-only (deepseek falls back to native Python impl).
    return cos_sin_format == "standard" and offsets is None


@ir.ops.rotary_embedding_query_only.register_impl(
    "xpu_kernels",
    supports_args=_rotary_query_only_xpu_supported,
    supported=XPU_KERNELS_SUPPORTED,
    inplace=True,
)
def rotary_embedding_query_only(
    positions: Tensor,
    query: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    offsets: Tensor | None = None,
    cos_sin_format: str = "standard",
    inverse: bool = False,
    rope_dim_offset: int = 0,
) -> Tensor:
    torch.ops._C.rotary_embedding(
        positions,
        query,
        None,
        head_size,
        cos_sin_cache,
        is_neox_style,
        rope_dim_offset,
        inverse,
    )
    return query
