# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from typing_extensions import Self

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.utils import replace_parameter

from ...utils.fp8_utils import process_fp8_weight_block_strategy
from ..base import (
    FP8Params,
    MMLinearKernel,
)
from .ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig


@dataclass
class FP8BlockParams(FP8Params):
    weight_scale_inv: torch.Tensor
    weight_scale: torch.Tensor

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


class Fp8BlockScaledMMLinearKernel(
    MMLinearKernel[FP8ScaledMMLinearLayerConfig, FP8BlockParams], ABC
):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.weight_group_shape = config.weight_quant_key.scale.group_shape
        self.input_quant_op = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        act_quant_key = config.activation_quant_key
        if act_quant_key.scale.static:
            return (
                False,
                "Only dynamic per token group activation quantization is supported.",
            )

        return True, None

    @classmethod
    @abstractmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMLinearKernel"]]:
        """
        Returns fallback kernel implementations when this kernel cannot handle
        the computation or is not supported.

        This method enables static and dynamic kernel dispatching:

        1. **Static dispatching** (at initialization):
           - Called during kernel init to select the best fallback kernel
           - First supported kernel from the list is instantiated and stored
           - Example: CudaFp8BlockScaledMMKernel tries [Cutlass, Triton]
             and selects first passing is_supported() (see cuda.py:62-64)

        2. **Dynamic dispatching** (at runtime):
           - Pre-selected fallback used when runtime conditions aren't met
           - Example: CudaFp8BlockScaledMMKernel tries FlashInfer/DeepGEMM,
             then falls back to default_fallback_kernel (cuda.py:110)

        Fallback chains form a priority hierarchy:
        - CudaFp8BlockScaledMMKernel → [Cutlass, Triton]
        - CutlassFp8BlockScaledMMKernel → [Triton]
        - TritonFp8BlockScaledMMKernel → [itself] (last resort)
        - AiterFp8BlockScaledMMKernel → [Triton]
        - FlashInferFp8DeepGEMMDynamicBlockScaledKernel → [DeepGemm,
          Cutlass, Triton]

        Returns:
            List of kernel classes in priority order. System iterates and
            uses first kernel passing is_supported().
        """
        raise NotImplementedError

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> FP8BlockParams:
        return FP8BlockParams.from_layer(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        params = self._get_layer_params(layer)
        weight, weight_scale_inv = process_fp8_weight_block_strategy(
            params.weight,
            params.weight_scale_inv,
        )

        replace_parameter(layer, params.WEIGHT, weight.data)
        replace_parameter(layer, params.WEIGHT_SCALE_INV, weight_scale_inv.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        maybe_out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale_inv = params.weight_scale_inv
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]
        out_dtype = input_2d.dtype if maybe_out_dtype is None else maybe_out_dtype

        q_input, input_scale = self.input_quant_op(input_2d, input_scale, scale_up)

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            out_dtype=out_dtype,
            As=input_scale,
            Bs=weight_scale_inv,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=out_dtype).view(*output_shape)

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
