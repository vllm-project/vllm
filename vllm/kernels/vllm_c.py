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


@ir.ops.gelu_and_mul.register_impl("vllm_c", supported=CUDA_ALIKE)
def gelu_and_mul(x: Tensor, approximate: str = "none") -> Tensor:
    """GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2"""
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if approximate == "none":
        torch.ops._C.gelu_and_mul(out, x)
    else:
        torch.ops._C.gelu_tanh_and_mul(out, x)
    return out


#===================
# GELU Activations
#===================


@ir.ops.gelu_new.register_impl("vllm_c", supported=CUDA_ALIKE)
def gelu_new(x: Tensor) -> Tensor:
    """0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    out = torch.empty_like(x)
    torch.ops._C.gelu_new(out, x)
    return out


@ir.ops.gelu_fast.register_impl("vllm_c", supported=CUDA_ALIKE)
def gelu_fast(x: Tensor) -> Tensor:
    """0.5 * x * (1.0 + tanh(x * 0.7978845608 * (1.0 + 0.044715 * x^2)))"""
    out = torch.empty_like(x)
    torch.ops._C.gelu_fast(out, x)
    return out


@ir.ops.quick_gelu.register_impl("vllm_c", supported=CUDA_ALIKE)
def quick_gelu(x: Tensor) -> Tensor:
    """x * sigmoid(1.702 * x)"""
    out = torch.empty_like(x)
    torch.ops._C.gelu_quick(out, x)
    return out
