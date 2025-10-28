# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from packaging import version

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
    ScaledMMLinearQuantStrategy,
)

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


class TorchScaledMMLinearKernel(ScaledMMLinearKernel):
    def __init__(
        self, c: ScaledMMLinearLayerConfig, layer_mapping_function: Callable
    ) -> None:
        vllm_config = get_current_vllm_config().compilation_config
        pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE

        output_padding = 17 if pad_output else None

        self.quant_fp8 = QuantFP8(
            static=c.is_static_input_scheme,
            group_shape=GroupShape.PER_TENSOR,
            num_token_padding=output_padding,
        )
        super().__init__(c, layer_mapping_function)

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return


class PerTensorTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        assert c.activation_group_shape is not None
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
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.
        (w, w_s, x_s), _ = self.layer_mapping_function(layer)
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])

        out_dtype = self.config.out_dtype
        out_dtype = x.dtype if out_dtype is None else out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != current_platform.fp8_dtype():
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
            )
        output_shape = [*x_2d_q.shape[:-1], w.shape[1]]
        return torch_per_tensor_w8a8_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )


class RowWiseTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 94

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        assert c.activation_group_shape is not None

        per_tensor_activation_scales = c.activation_group_shape.is_per_tensor()
        per_tensor_weight_scales = (
            c.weight_quant_strategy == ScaledMMLinearQuantStrategy.TENSOR
        )

        if per_tensor_activation_scales and per_tensor_weight_scales:
            return (
                False,
                "RowWiseTorchScaledMMLinearKernel cannot be used with "
                + "per tensor activation and weight scales.",
            )

        if not current_platform.is_rocm():
            return (
                False,
                "RowWiseTorchScaledMMLinearKernel is only supported "
                + "in ROCm platforms.",
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
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.
        (w, w_s, x_s), _ = self.layer_mapping_function(layer)
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])

        out_dtype = self.config.out_dtype
        out_dtype = x.dtype if out_dtype is None else out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != current_platform.fp8_dtype():
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
            )
        output_shape = [*x_2d_q.shape[:-1], w.shape[1]]
        return torch_row_wise_w8a8_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )


class ChannelWiseTorchScaledMMLinearKernel(TorchScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 94

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        assert c.activation_group_shape is not None

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
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.
        (w, w_s, x_s), _ = self.layer_mapping_function(layer)
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])

        out_dtype = self.config.out_dtype
        out_dtype = x.dtype if out_dtype is None else out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != current_platform.fp8_dtype():
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
            )
        output_shape = [*x_2d_q.shape[:-1], w.shape[1]]
        return torch_channelwise_w8a8_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )
