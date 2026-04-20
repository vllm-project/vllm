# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.utils.import_utils import has_fbgemm_gpu

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class FbgemmNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FBGEMM."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if has_fbgemm_gpu():
            return True, None
        return False, "fbgemm_gpu required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        swizzled = swizzle_blockscale(layer.weight_scale.data)
        layer.weight_scale = torch.nn.Parameter(
            swizzled.view(-1).view(torch.uint8), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import fbgemm_gpu  # noqa: F401 - registers torch.ops.fbgemm.*

        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="fbgemm",
        )

        out = torch.ops.fbgemm.f4f4bf16(
            x_fp4,
            layer.weight,
            x_blockscale.view(-1).view(torch.uint8),
            layer.weight_scale,
            layer.alpha,
            use_mx=False,
        ).to(output_dtype)

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
