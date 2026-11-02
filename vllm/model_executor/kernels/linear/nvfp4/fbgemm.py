# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.utils.import_utils import has_fbgemm_gpu

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class FbgemmNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FBGEMM."""

    input_quant_backend_name = "fbgemm"

    def input_quant_key(self) -> QuantKey:
        return kNvfp4Dynamic

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
        replace_parameter(layer, "weight_scale", swizzled.view(-1).view(torch.uint8))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import fbgemm_gpu  # noqa: F401 - registers torch.ops.fbgemm.*

        output_size = layer.output_size_per_partition
        qa = self.prepare_input(layer, x)

        out = torch.ops.fbgemm.f4f4bf16(
            qa.data,
            layer.weight,
            qa.scale.view(-1).view(torch.uint8),
            layer.weight_scale,
            layer.alpha,
            use_mx=False,
        ).to(qa.orig_dtype)

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*qa.orig_shape[:-1], output_size)
