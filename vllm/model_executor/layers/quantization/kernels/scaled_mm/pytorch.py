# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from packaging import version

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def torch_per_tensor_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list,
) -> torch.Tensor:
    output = torch._scaled_mm(
        A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
    )
    # A fix for discrepancy in scaled_mm which returns tuple
    # for torch < 2.5 and a single value in torch >= 2.5
    if type(output) is tuple and len(output) == 2:
        output = output[0]

    return torch.narrow(output, 0, 0, output_shape[0]).view(*output_shape)


def torch_row_wise_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list,
) -> torch.Tensor:
    #  Note:
    #  For now it has only been validated on ROCm platform.
    #  fp8 rowwise scaling in torch._scaled_mm is introduced in
    #  https://github.com/pytorch/pytorch/pull/144432 using
    #  hipBLASLt and ROCm 6.3, which only exists in torch 2.7 and above.
    #
    #  For CUDA platform please validate if the torch._scaled_mm supports
    #  rowwise scaled GEMM before using it

    # Fused GEMM_DQ Rowwise GEMM
    output = torch._scaled_mm(
        A,
        B,
        out_dtype=out_dtype,
        scale_a=As,
        scale_b=Bs.t(),
        bias=bias,
    )

    output = torch.narrow(output, 0, 0, output_shape[0])
    output = output.view(*output_shape)
    return output


def torch_channelwise_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list,
) -> torch.Tensor:
    # Use unfused DQ due to limitations with scaled_mm

    # Symmetric quantized GEMM by definition computes the following:
    #   C = (s_x * X) (s_w * W) + bias
    # This is equivalent to dequantizing the weights and activations
    # before applying a GEMM.
    #
    # In order to compute quantized operands, a quantized kernel
    # will rewrite the above like so:
    #   C = s_w * s_x * (X * W) + bias
    #
    # For the scaled_mm fallback case, we break this down, since it
    # does not support s_w being a vector.

    # Input scaling factors are no longer optional in _scaled_mm starting
    # from pytorch 2.5. Allocating a dummy tensor to pass as scales
    dummy_tensor = torch.ones(1, dtype=torch.float32, device=A.device)

    # GEMM
    # This computes C = (X * W).
    # Output in fp32 to allow subsequent ops to happen in-place
    output = torch._scaled_mm(
        A,
        B,
        scale_a=dummy_tensor,
        scale_b=dummy_tensor,
        out_dtype=torch.float32,
    )
    # A fix for discrepancy in scaled_mm which returns tuple
    # for torch < 2.5 and a single value in torch >= 2.5
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    # Unpad (undo num_token_padding)
    output = torch.narrow(output, 0, 0, output_shape[0])
    x_scale = torch.narrow(As, 0, 0, output_shape[0])

    # DQ
    # C = sw * sx * (X * W) + bias
    output = output * x_scale * Bs.t()
    if bias is not None:
        output = output + bias
    return output.to(out_dtype).view(*output_shape)


class TorchScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    Base class for FP8 linear kernels using Torch.
    Each subclass represents a kernel variant for
    specific device capabilities and torch versions,
    so we split them up and implement
    get_min_capability() separately for each.
    """

    def get_ouput_padding(self) -> int | None:
        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        # We also don't pad when using torch.compile,
        # as it breaks with dynamic shapes.
        vllm_config = get_current_vllm_config().compilation_config
        pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE
        output_padding = 17 if pad_output else None
        return output_padding


class PerTensorTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return (
                False,
                "PerTensorTorchScaledMMLinearKernel requires "
                + "per tensor activation and weight scales.",
            )
        return True, None

    def get_scaled_mm_func(self) -> Callable[..., torch.Tensor]:
        return torch_per_tensor_w8a8_scaled_mm


class RowWiseTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 94

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if per_tensor_activation_scales or per_tensor_weight_scales:
            return (
                False,
                "RowWiseTorchScaledMMLinearKernel cannot be used with "
                + "per tensor activation and weight scales.",
            )

        if not current_platform.is_rocm():
            return (
                False,
                "RowWiseTorchScaledMMLinearKernel is only supported "
                + "on ROCm platforms.",
            )

        if not version.parse(torch.__version__) >= version.parse("2.7"):
            return (
                False,
                "RowWiseTorchScaledMMLinearKernel requires " + "pytorch version >=2.7.",
            )

        return True, None

    def get_scaled_mm_func(self) -> Callable[..., torch.Tensor]:
        return torch_row_wise_w8a8_scaled_mm


class ChannelWiseTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if per_tensor_activation_scales and per_tensor_weight_scales:
            return (
                False,
                "ChannelWiseTorchScaledMMLinearKernel cannot be used with "
                + "per tensor activation and weight scales.",
            )

        return True, None

    def get_scaled_mm_func(self) -> Callable[..., torch.Tensor]:
        return torch_channelwise_w8a8_scaled_mm
