# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from ..base import (
    FP8Params,
    Int8Params,
    MMLinearKernel,
    MMLinearLayerConfig,
)


@dataclass
class Int8ScaledMMLinearLayerConfig(MMLinearLayerConfig):
    # TODO: Change to QuantKey like FP8ScaledMMLinearLayerConfig
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool


@dataclass
class FP8ScaledMMLinearLayerConfig(MMLinearLayerConfig):
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    out_dtype: torch.dtype | None


class FP8ScaledMMLinearKernel(
    MMLinearKernel[FP8ScaledMMLinearLayerConfig, FP8Params], ABC
):
    def __init__(
        self,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> None:
        super().__init__(config)

        act_scale_descriptor = config.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )
        self.fp8_dtype = current_platform.fp8_dtype()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> FP8Params:
        return FP8Params.from_layer(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        fp8_dtype = self.fp8_dtype
        maybe_out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)

        w = params.weight
        w_s = params.weight_scale
        x_s = params.input_scale
        x_s_ub = params.input_scale_ub

        #   ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_s computed from x.
        #   If static, layer.input_scale is scalar and x_s is input_scale.
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], w.shape[1]]
        out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != fp8_dtype:
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
                x_s_ub,
            )
        return self.apply_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )

    @abstractmethod
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
        raise NotImplementedError


class Int8ScaledMMLinearKernel(
    MMLinearKernel[Int8ScaledMMLinearLayerConfig, Int8Params], ABC
):
    def process_weights_after_loading(self, layer):
        # These processors are common between, cutlass, triton, aiter.
        params = self._get_layer_params(layer)
        config = self.config
        # WEIGHT
        replace_parameter(
            layer,
            params.WEIGHT,
            torch.nn.Parameter(params.weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = params.weight_scale
        if is_fused_module and not config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            params.WEIGHT_SCALE,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if config.is_static_input_scheme:
            input_scale = params.input_scale
            i_s_name = params.INPUT_SCALE
            i_zp_name = params.INPUT_ZERO_POINT

            if config.input_symmetric:
                assert input_scale is not None
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
                setattr(layer, i_zp_name, None)

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> Int8Params:
        return Int8Params.from_layer(layer)
