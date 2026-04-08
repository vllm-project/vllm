# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_tma_aligned_size, is_deep_gemm_e8m0_used

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

CUDA_ONLY = current_platform.is_cuda()

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()


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


rms_no_var_size = lambda x, weight, epsilon, variance_size=None: variance_size is None
"""vLLM kernel does not support variance_size parameter."""


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


_vllm_c_group_quant_args = (
    lambda x, group_shape, column_major, use_ue8m0, scale_alignment=1: (
        x.is_contiguous() and x.shape[-1] % group_shape[-1] == 0
    )
)


def _group_quant_fp8_packed(
    x: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """DeepGEMM packed UE8M0 path: 4 exponent bytes packed per int32,
    stored with TMA-aligned column-major strides."""
    hidden_dim = x.shape[-1]
    mn = x.numel() // hidden_dim
    num_groups_per_row = hidden_dim // group_size
    k_num_packed_sf_k = (num_groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4
    x_q = torch.empty_like(x, dtype=_FP8_DTYPE)
    x_s = torch.empty_strided(
        (mn, k_num_packed_sf_k),
        (1, tma_aligned_mn),
        device=x.device,
        dtype=torch.int32,
    )
    torch.ops._C.per_token_group_fp8_quant_packed(
        x, x_q, x_s, group_size, 1e-10, _FP8_MIN, _FP8_MAX
    )
    return x_q, x_s


@ir.ops.dynamic_group_quant_fp8.register_impl(
    "vllm_c", supports_args=_vllm_c_group_quant_args, supported=CUDA_ONLY
)
def dynamic_group_quant_fp8(
    x: Tensor,
    group_shape: list[int],
    column_major: bool,
    use_ue8m0: bool,
    scale_alignment: int = 1,
) -> tuple[Tensor, Tensor]:
    group_size = group_shape[-1]

    assert x.is_contiguous()
    assert x.shape[-1] % group_size == 0

    if use_ue8m0 and column_major and scale_alignment > 1 and is_deep_gemm_e8m0_used():
        return _group_quant_fp8_packed(x, group_size)

    x_q = torch.empty(x.shape, device=x.device, dtype=_FP8_DTYPE)
    x_s = make_group_quant_scales(x, group_size, column_major, scale_alignment)
    torch.ops._C.per_token_group_fp8_quant(
        x,
        x_q,
        x_s,
        group_size,
        1e-10,
        _FP8_MIN,
        _FP8_MAX,
        use_ue8m0,
        column_major,
        scale_alignment > 1,
    )
    return x_q, x_s
