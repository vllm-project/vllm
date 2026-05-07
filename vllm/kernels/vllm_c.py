# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

rms_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
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
    # NOTE: 3D inputs are only expected from the Transformers modeling backend,
    # where the 0-th dimension is 1. This check is necessary for ROCm. However,
    # CUDA (A100) was also found to score better performance with this path, so
    # we use it for all CUDA-alike platforms.
    if CUDA_ALIKE and x.dim() > 2:
        original_shape = x.shape
        x = x.view(*original_shape[-2:])
        output = torch.empty_like(x)
        torch.ops._C.rms_norm(output, x, weight, epsilon)
        return output.view(original_shape)

    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output


rms_add_no_var_size = lambda x, x_residual, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
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
    # NOTE: 3D inputs are only expected from the Transformers modeling backend,
    # where the 0-th dimension is 1. This check is necessary for ROCm. However,
    # CUDA (A100) was also found to score better performance with this path, so
    # we use it for all CUDA-alike platforms.
    if CUDA_ALIKE and x.dim() > 2:
        original_shape = x.shape
        x = x.view(*original_shape[-2:])
        x_residual = x_residual.view(*original_shape[-2:])
        torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
        return x.view(original_shape), x_residual.view(original_shape)

    torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
    return x, x_residual
