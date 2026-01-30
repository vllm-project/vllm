# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from typing_extensions import Self

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.model_executor.utils import replace_parameter

from ...utils.fp8_utils import process_fp8_weight_block_strategy
from ..base import (
    CustomKernelConfig,
    FP8Params,
    MMLinearKernel,
)


@dataclass
class Fp8BlockMMScaledConfig(CustomKernelConfig):
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    out_dtype: torch.dtype | None


@dataclass
class FP8BlockParams(FP8Params):
    weight_scale_inv: torch.Tensor
    weight_scale: torch.Tensor | None

    WEIGHT_SCALE_INV: ClassVar[str] = "weight_scale_inv"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Self:
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale_inv=getattr(layer, cls.WEIGHT_SCALE_INV),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE, None),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )


class Fp8BlockScaledMMKernel(MMLinearKernel[Fp8BlockMMScaledConfig, FP8BlockParams]):
    def __init__(self, config: Fp8BlockMMScaledConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.weight_group_shape = config.weight_quant_key.scale.group_shape
        self.input_quant = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )

    @classmethod
    def can_implement(cls, config):
        return True, None

    @classmethod
    @abstractmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMKernel"]]:
        raise NotImplementedError

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> FP8BlockParams:
        return FP8BlockParams.from_layer(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        params = self._get_layer_params(layer)
        print("--- process params ---")
        print(params)
        weight, weight_scale_inv = process_fp8_weight_block_strategy(
            params.weight,
            params.weight_scale_inv,
        )

        replace_parameter(layer, params.WEIGHT, weight.data)
        replace_parameter(layer, params.WEIGHT_SCALE_INV, weight_scale_inv.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale_inv = params.weight_scale_inv
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]
        output_dtype = x.dtype

        q_input, input_scale = self.input_quant(input_2d, input_scale, scale_up)

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            out_dtype=output_dtype,
            As=input_scale,
            Bs=weight_scale_inv,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=output_dtype).view(*output_shape)

    @abstractmethod
    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
