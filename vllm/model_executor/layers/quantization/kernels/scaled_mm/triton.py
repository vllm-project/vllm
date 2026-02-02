# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class TritonInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike():
            return True, None
        return False, "requires ROCm or CUDA."

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not config.input_symmetric:
            return False, "supports symmetric input only."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        params = self._get_layer_params(layer)

        replace_parameter(
            layer,
            params.WEIGHT,
            torch.nn.Parameter(params.weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Triton kernel supports only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = params.weight_scale
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            params.WEIGHT_SCALE,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        i_s = params.input_scale
        i_s_name = params.INPUT_SCALE
        i_zp_name = params.INPUT_ZERO_POINT
        if self.config.is_static_input_scheme:
            assert i_s is not None
            replace_parameter(
                layer,
                i_s_name,
                torch.nn.Parameter(i_s.max(), requires_grad=False),
            )
            setattr(layer, i_zp_name, None)
        else:
            setattr(layer, i_s_name, None)
            setattr(layer, i_zp_name, None)

        setattr(layer, params.AZP_ADJ, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)

        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), params.input_scale, params.input_zero_point, symmetric=True
        )

        assert x_zp is None, "Triton kernel only supports symmetric quantization"

        return self.apply_scaled_mm(
            A=x_q,
            B=params.weight,
            As=x_s,
            Bs=params.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
            output_shape=[],
        )

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
        return triton_scaled_mm(
            A,
            B,
            scale_a=As,
            scale_b=Bs,
            out_dtype=out_dtype,
            bias=bias,
        )
