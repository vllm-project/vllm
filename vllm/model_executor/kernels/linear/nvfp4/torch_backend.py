# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    _apply_nvfp4_linear_torch,
    swizzle_blockscale,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class TorchNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via torch._scaled_mm (requires PyTorch >= 2.8)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.utils.torch_utils import is_torch_equal_or_newer

        if not is_torch_equal_or_newer("2.8"):
            return False, "torch backend requires PyTorch >= 2.8"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(
            layer.weight.data.view(torch.float4_e2m1fn_x2), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        # torch._scaled_mm applies blockscales internally before returning,
        # so fp16 output can overflow before the external alpha correction.
        assert output_dtype != torch.float16, (
            "TORCH nvfp4 backend does not support float16 — use bfloat16"
        )

        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]

        return _apply_nvfp4_linear_torch(
            x=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_global_scale_inv=layer.input_global_scale_inv,
            alpha=layer.alpha,
            output_dtype=output_dtype,
            output_shape=output_shape,
            output_size=output_size,
            bias=bias,
        )
