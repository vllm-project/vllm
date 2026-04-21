# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.ir.ops.quant import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_tma_aligned_size

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

CUDA_ONLY = current_platform.is_cuda()


def make_group_quant_scales(
    x: Tensor,
    group_size: int,
    column_major: bool,
    scale_alignment: int,
) -> Tensor:
    """Allocate the output scale tensor for group FP8 quantization.
    Handles row-major, column-major, and TMA-aligned column-major layouts."""
    if column_major:
        if scale_alignment > 1:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, scale_alignment)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (1, tma_aligned_m)
            return torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            return torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        return torch.empty(shape, device=x.device, dtype=torch.float32)


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


_vllm_c_static_quant_fp8_args = (
    lambda x, scale, fp8_dtype, num_token_padding=None: x.ndim == 2
)
"""vllm_c static_quant_fp8 requires a 2D input tensor."""


@ir.ops.static_quant_fp8.register_impl(
    "vllm_c", supports_args=_vllm_c_static_quant_fp8_args, supported=CUDA_ALIKE
)
def static_quant_fp8(
    x: Tensor,
    scale: Tensor,
    fp8_dtype: torch.dtype,
    num_token_padding: int | None = None,
) -> Tensor:
    shape = x.shape
    if num_token_padding:
        shape = (max(num_token_padding, x.shape[0]),) + x.shape[1:]
    output = torch.empty(shape, device=x.device, dtype=fp8_dtype)
    torch.ops._C.static_scaled_fp8_quant(output, x, scale, None)
    return output


_vllm_c_static_group_quant_fp8_args = (
    lambda x, scale, fp8_dtype, num_token_padding=None: x.ndim == 2
)
"""vllm_c static_group_quant_fp8 requires a 2D input tensor."""


@ir.ops.static_group_quant_fp8.register_impl(
    "vllm_c", supports_args=_vllm_c_static_group_quant_fp8_args, supported=CUDA_ALIKE
)
def static_group_quant_fp8(
    x: Tensor,
    scale: Tensor,
    fp8_dtype: torch.dtype,
    num_token_padding: int | None = None,
) -> Tensor:
    shape = x.shape
    if num_token_padding:
        shape = (max(num_token_padding, x.shape[0]),) + x.shape[1:]
    output = torch.empty(shape, device=x.device, dtype=fp8_dtype)
    torch.ops._C.static_scaled_fp8_quant(output, x, scale, None)
    return output


_vllm_c_dynamic_quant_fp8_args = (
    lambda x, per_token, fp8_dtype, scale_ub=None, num_token_padding=None: (
        x.ndim == 2 and (per_token or scale_ub is None)
    )
)
"""vllm_c dynamic_quant_fp8 requires a 2D input tensor."""


@ir.ops.dynamic_quant_fp8.register_impl(
    "vllm_c", supports_args=_vllm_c_dynamic_quant_fp8_args, supported=CUDA_ALIKE
)
def dynamic_quant_fp8(
    x: Tensor,
    per_token: bool,
    fp8_dtype: torch.dtype,
    scale_ub: Tensor | None = None,
    num_token_padding: int | None = None,
) -> tuple[Tensor, Tensor]:
    shape = x.shape
    if num_token_padding:
        shape = (max(num_token_padding, x.shape[0]),) + x.shape[1:]
    output = torch.empty(shape, device=x.device, dtype=fp8_dtype)
    if per_token:
        scale = torch.empty((shape[0], 1), device=x.device, dtype=torch.float32)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, x, scale, scale_ub)
    else:
        scale = torch.empty(1, device=x.device, dtype=torch.float32)
        torch.ops._C.dynamic_scaled_fp8_quant(output, x, scale)
    return output, scale


_vllm_c_group_quant_args = (
    lambda x, group_shape, column_major, use_ue8m0, fp8_dtype, scale_alignment=1: (
        x.is_contiguous() and x.shape[-1] % group_shape[-1] == 0
    )
)
"""vllm_c dynamic_group_quant_fp8 requires a contiguous input tensor with
 hidden dim divisible by group size."""


@ir.ops.dynamic_group_quant_fp8.register_impl(
    "vllm_c", supports_args=_vllm_c_group_quant_args, supported=CUDA_ONLY
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
        x,
        x_q,
        x_s,
        group_size,
        1e-10,
        *get_fp8_min_max(fp8_dtype),
        use_ue8m0,
        column_major,
        scale_alignment > 1,
    )
    return x_q, x_s
