# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch
import os
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from .ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(layer, "weight", layer.weight.data.t())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

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
        pass


class XPUFP8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        self.apply_input_quant = os.environ.get("XPU_FP8_BLOCK_SCALED_MM_INPUT_QUANT", "1") == "1"

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUFP8BlockScaledMM only support on XPU"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Transpose weight back to col major for xpu kernels
        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )
        weight_scale = weight_scale.t().contiguous()
        replace_parameter(layer, params.WEIGHT, params.weight.data.t())
        replace_parameter(layer, scale_attr_name, weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[1]]

        if self.apply_input_quant:
            q_input, input_scale = self.quant_fp8(
                input_2d, input_scale, scale_up, use_triton=self.use_triton
            )
        else:
            q_input = input_2d
            # Provide a concrete placeholder so apply_block_scaled_mm args are
            # always Tensors. Subclasses with apply_input_quant=False must not
            # use As in apply_block_scaled_mm.
            input_scale = (
                input_scale if input_scale is not None else input_2d.new_ones(1)
            )

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            As=input_scale,
            Bs=weight_scale,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=out_dtype).view(*output_shape)
    
    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # if not self.apply_input_quant:
        #     return torch.ops._xpu_C.fp8_gemm_w8a16(A, B, Bs, None)
        print(f"input shape: {A.shape}, weight shape: {B.shape}, input scale shape: {As.shape}, weight scale shape: {Bs.shape}")
        return torch.ops._xpu_C.fp8_gemm(A, B, self.config.out_dtype, As, Bs, None)
