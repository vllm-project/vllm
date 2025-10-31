# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from packaging import version

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearQuantStrategy,
)
from .utils import apply_weights_fp8

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def maybe_create_device_identity():
    # Allocate dummy ones tensor for torch._scaled_mm
    global TORCH_DEVICE_IDENTITY
    if TORCH_DEVICE_IDENTITY is None:
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32)


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

    return torch.narrow(output, 0, 0, A.shape[0]).view(*output_shape)


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
    # Note: Callers of this function should check USE_ROWWISE_TORCH_SCALED_MM
    #  when using it.
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

    output = torch.narrow(output, 0, 0, A.shape[0])
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

    # GEMM
    # This computes C = (X * W).
    # Output in fp32 to allow subsequent ops to happen in-place
    output = torch._scaled_mm(
        A,
        B,
        scale_a=TORCH_DEVICE_IDENTITY,
        scale_b=TORCH_DEVICE_IDENTITY,
        out_dtype=torch.float32,
    )
    # A fix for discrepancy in scaled_mm which returns tuple
    # for torch < 2.5 and a single value in torch >= 2.5
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    # Unpad (undo num_token_padding)
    output = torch.narrow(output, 0, 0, A.shape[0])
    x_scale = torch.narrow(As, 0, 0, A.shape[0])

    # DQ
    # C = sw * sx * (X * W) + bias
    output = output * x_scale * Bs.t()
    if bias is not None:
        output = output + bias
    return output.to(out_dtype).view(*output_shape)


class TorchScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def get_ouput_padding(self) -> int | None:
        vllm_config = get_current_vllm_config().compilation_config
        pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE
        output_padding = 17 if pad_output else None
        return output_padding


class PerTensorTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = c.activation_group_shape.is_per_tensor()
        per_tensor_weight_scales = (
            c.weight_quant_strategy == ScaledMMLinearQuantStrategy.TENSOR
        )

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return (
                False,
                "PerTensorTorchScaledMMLinearKernel requires "
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
            torch_per_tensor_w8a8_scaled_mm,
            self.quant_fp8,
            w,
            x,
            w_s,
            x_s,
            bias,
            x_s_ub,
            self.config.out_dtype,
        )


class RowWiseTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 94

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = c.activation_group_shape.is_per_tensor()
        per_tensor_weight_scales = (
            c.weight_quant_strategy == ScaledMMLinearQuantStrategy.TENSOR
        )

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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)
        return apply_weights_fp8(
            torch_row_wise_w8a8_scaled_mm,
            self.quant_fp8,
            w,
            x,
            w_s,
            x_s,
            bias,
            x_s_ub,
            self.config.out_dtype,
        )


class ChannelWiseTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 94

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = c.activation_group_shape.is_per_tensor()
        per_tensor_weight_scales = (
            c.weight_quant_strategy == ScaledMMLinearQuantStrategy.TENSOR
        )

        if per_tensor_activation_scales and per_tensor_weight_scales:
            return (
                False,
                "ChannelWiseTorchScaledMMLinearKernel cannot be used with "
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
            torch_channelwise_w8a8_scaled_mm,
            self.quant_fp8,
            w,
            x,
            w_s,
            x_s,
            bias,
            x_s_ub,
            self.config.out_dtype,
        )
