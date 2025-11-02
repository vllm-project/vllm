# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearQuantStrategy,
)
from .utils import apply_weights_fp8


def rocm_per_tensor_float_w8a8_scaled_mm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if (
        A.shape[0] == 1
        and B.shape[1] % 16 == 0
        and ((bias is None) or (bias.dtype == out_dtype))
    ):
        output = ops.wvSplitKQ(
            B.t(),
            A,
            out_dtype,
            As,
            Bs,
            current_platform.get_cu_count(),
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


def rocm_per_tensor_float_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    output = torch.ops.vllm.rocm_per_tensor_w8a8_scaled_mm_impl(
        A, B, out_dtype, As, Bs, bias
    )
    return torch.narrow(output, 0, 0, A.shape[0]).view(*output_shape)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_per_tensor_float_w8a8_scaled_mm_impl",
        op_func=rocm_per_tensor_float_w8a8_scaled_mm_impl,
        fake_impl=rocm_per_tensor_float_w8a8_scaled_mm_fake,
    )


class ROCmScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def get_ouput_padding(self) -> int | None:
        return None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # TODO: check if this causes an issue on non-ROCM platforms
        from vllm.platforms.rocm import on_mi3xx

        per_tensor_activation_scales = c.activation_group_shape.is_per_tensor()
        per_tensor_weight_scales = (
            c.weight_quant_strategy == ScaledMMLinearQuantStrategy.TENSOR
        )

        if not current_platform.is_rocm():
            return (
                False,
                "ROCmScaledMMLinearFP8Kernel is supported " + "on ROCm platforms Only.",
            )
        if not on_mi3xx():
            return (
                False,
                "ROCmScaledMMLinearFP8Kernel is supported "
                + "on MI3xx architures only.",
            )
        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return (
                False,
                "VLLM_ROCM_USE_SKINNY_GEMM must be enabled "
                + "to use ROCmScaledMMLinearKernel.",
            )

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return (
                False,
                "ROCmScaledMMLinearKernel requires "
                + "per tensor activation and weight scales.",
            )
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)
        return apply_weights_fp8(
            rocm_per_tensor_float_w8a8_scaled_mm,
            self.quant_fp8,
            w,
            x,
            w_s,
            x_s,
            bias,
            x_s_ub,
            self.config.out_dtype,
        )
