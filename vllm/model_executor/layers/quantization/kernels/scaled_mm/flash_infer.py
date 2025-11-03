# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm, has_flashinfer

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearQuantStrategy,
)
from .utils import apply_weights_fp8


def flashinfer_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list,
) -> torch.Tensor:
    return flashinfer_scaled_fp8_mm(
        A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
    )


class FlashInferScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def get_ouput_padding(self) -> int | None:
        return None

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)
        return apply_weights_fp8(
            flashinfer_w8a8_scaled_mm,
            self.quant_fp8,
            w,
            x,
            w_s,
            x_s,
            bias,
            x_s_ub,
            self.config.out_dtype,
        )
