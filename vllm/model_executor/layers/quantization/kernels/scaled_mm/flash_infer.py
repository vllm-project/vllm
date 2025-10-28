# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm, has_flashinfer

from .ScaledMMLinearKernel import (
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
    ScaledMMLinearQuantStrategy,
)


def flashinfer_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return flashinfer_scaled_fp8_mm(
        A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
    )


class FlashInferScaledMMLinearKernel(ScaledMMLinearKernel):
    def __init__(
        self, c: ScaledMMLinearLayerConfig, layer_mapping_function: Callable
    ) -> None:
        self.quant_fp8 = QuantFP8(
            static=c.is_static_input_scheme,
            group_shape=GroupShape.PER_TENSOR,
            num_token_padding=None,
        )
        super().__init__(c, layer_mapping_function)

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = c.activation_group_shape.is_per_tensor()
        per_tensor_weight_scales = (
            c.weight_quant_strategy == ScaledMMLinearQuantStrategy.TENSOR
        )

        if not current_platform.is_cuda():
            return (
                False,
                "FlashInferScaledMMLinearKernel is supported "
                + "on CUDA platforms Only.",
            )

        if not has_flashinfer():
            return (
                False,
                "FlashInferScaledMMLinearKernel requires "
                + "FlashInfer to be installed.",
            )
        if not has_flashinfer():
            return (
                False,
                "FlashInferScaledMMLinearKernel requires "
                + "FlashInfer to be installed.",
            )

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return (
                False,
                "FlashInferScaledMMLinearKernel requires "
                + "per tensor activation and weight scales.",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

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

        return flashinfer_w8a8_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )
