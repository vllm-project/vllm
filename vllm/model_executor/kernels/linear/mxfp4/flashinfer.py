# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    swizzle_mxfp4_scales,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

_MXFP4_GROUP_SIZE = 32


class FlashInferMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 W4A4 GEMM via FlashInfer CUTLASS (SM100+)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.has_device_capability(100) and has_flashinfer_cutedsl():
            return True, None
        return False, "FlashInfer + >=sm_100 (Blackwell) required"

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        N, scale_K = layer.weight_scale.shape
        K = scale_K * _MXFP4_GROUP_SIZE

        # swizzle pads N to the next multiple of 128 for CUTLASS tiling
        padded_N = ((N + 127) // 128) * 128
        layer.weight_scale = Parameter(
            swizzle_mxfp4_scales(layer.weight_scale.data, N, K).reshape(padded_N, -1),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.flashinfer import (
            flashinfer_mxfp4_quantize,
            flashinfer_scaled_fp4_mm,
        )

        weight = layer.weight
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        x_2d = x.reshape(-1, x.shape[-1])

        x_fp4, x_scale = flashinfer_mxfp4_quantize(x_2d)
        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            weight,
            x_scale,
            layer.weight_scale,
            alpha=None,
            out_dtype=x.dtype,
            backend="cute-dsl",
            block_size=_MXFP4_GROUP_SIZE,
            use_nvfp4=False,
        )

        if bias is not None:
            out = out + bias
        return out.view(out_shape)
