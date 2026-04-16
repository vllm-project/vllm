# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    MXFP4_BLOCK_SIZE,
    MXFP4_SCALE_DTYPE,
    dq_mxfp4_torch,
)

from .Mxfp4LinearKernel import Mxfp4LinearKernel, Mxfp4LinearLayerConfig


class EmulationMxfp4LinearKernel(Mxfp4LinearKernel):
    """Software emulation fallback for MXFP4 (dequant to BF16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight_packed.data  # [N, K]
        N, K = weight.shape
        scale_k = K * 2 // MXFP4_BLOCK_SIZE

        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        del layer.weight_packed

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        weight_scale = layer.weight_scale
        if weight_scale.dtype != MXFP4_SCALE_DTYPE:
            raise ValueError(
                f"Emulation backend requires {MXFP4_SCALE_DTYPE} "
                f"weight_scale dtype, got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"Emulation backend requires 2D weight_scale, "
                f"got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )
        weight_hf = dq_mxfp4_torch(layer.weight, layer.weight_scale, out_dtype)
        output = torch.nn.functional.linear(x, weight_hf, bias)
        return output.to(x.dtype)
