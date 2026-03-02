# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.platform_utils import get_cu_count
from vllm.utils.torch_utils import direct_register_custom_op

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def rocm_per_tensor_float_w8a8_scaled_mm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if (
        A.shape[0] <= 4
        and B.shape[0] % 16 == 0  # M TODO: needed?
        and B.shape[1] % 16 == 0  # K
        and ((bias is None) or (bias.dtype == out_dtype))
    ):
        output = ops.wvSplitKQ(
            B.t(),
            A,
            out_dtype,
            As,
            Bs,
            get_cu_count(),
            bias,
        )
    # Fallback
    else:
        output = torch._scaled_mm(
            A,
            B,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs,
            bias=bias,
        )
    return output


def rocm_per_tensor_float_w8a8_scaled_mm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return A.new_empty((*A.shape[:-1], B.shape[1]), dtype=out_dtype)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_per_tensor_float_w8a8_scaled_mm_impl",
        op_func=rocm_per_tensor_float_w8a8_scaled_mm_impl,
        fake_impl=rocm_per_tensor_float_w8a8_scaled_mm_fake,
    )


class ROCmFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        from vllm.platforms.rocm import on_mi3xx

        if not on_mi3xx():
            return False, "requires MI3xx."

        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "requires VLLM_ROCM_USE_SKINNY_GEMM to be enabled."

        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            config.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = (
            config.weight_quant_key.scale.group_shape.is_per_tensor()
        )

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return False, "requires per tensor activation and weight scales."

        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
            A, B, out_dtype, As, Bs, bias
        )
        return torch.narrow(output, 0, 0, A.shape[0]).view(*output_shape)
