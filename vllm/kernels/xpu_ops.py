# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.ir.ops.quant import get_fp8_min_max, make_group_quant_scales
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


_xpu_kernels_group_quant_args = (
    lambda x, group_shape, column_major, use_ue8m0, fp8_dtype, scale_alignment=1: (
        x.is_contiguous()
        and x.shape[-1] % group_shape[-1] == 0
        and scale_alignment == 1
    )
)
"""xpu_kernels dynamic_group_quant_fp8 requires a contiguous input tensor with
 hidden dim divisible by group size."""


@ir.ops.dynamic_group_quant_fp8.register_impl(
    "xpu_kernels",
    supports_args=_xpu_kernels_group_quant_args,
    supported=XPU_KERNELS_SUPPORTED,
)
def dynamic_group_quant_fp8(
    x: Tensor,
    group_shape: list[int],
    column_major: bool,
    use_ue8m0: bool,
    fp8_dtype: torch.dtype,
    scale_alignment: int = 1,
) -> tuple[Tensor, Tensor]:
    group_size = group_shape[-1]

    assert x.is_contiguous()
    assert x.shape[-1] % group_size == 0

    x_q = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
    x_s = make_group_quant_scales(x, group_size, column_major, scale_alignment)
    torch.ops._C.per_token_group_fp8_quant(
        x, x_q, x_s, group_size, 1e-10, *get_fp8_min_max(fp8_dtype), use_ue8m0
    )
    if use_ue8m0:
        x_s = x_s.to(torch.float8_e8m0fnu)
    return x_q, x_s
