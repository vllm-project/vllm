# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

rms_no_var_size = (
    lambda x, weight, epsilon, variance_size=None: variance_size is None
    and (weight is None or weight.dtype == x.dtype)
)
"""vLLM kernel requires no variance_size override and matching input/weight dtype."""


@ir.ops.rms_norm.register_impl(
    "vllm_c", supports_args=rms_no_var_size, supported=CUDA_ALIKE
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
"""vLLM Kernel does not support variance_size parameter and requires
matching input/weight dtype."""


@ir.ops.fused_add_rms_norm.register_impl(
    "vllm_c",
    supports_args=rms_add_no_var_size,
    supported=CUDA_ALIKE,
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


def rotary_no_offsets(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    offsets: Tensor | None = None,
    cos_sin_format: str = "standard",
) -> bool:
    return offsets is None and cos_sin_format == "standard"


@ir.ops.rotary_embedding.register_impl(
    "vllm_c",
    supports_args=rotary_no_offsets,
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
) -> tuple[Tensor, Tensor]:
    torch.ops._C.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
    )
    return query, key
