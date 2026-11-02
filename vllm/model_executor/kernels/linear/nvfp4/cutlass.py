# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    scaled_fp4_quant,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    nvfp4_weight_padding_bytes,
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class CutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the vLLM CUTLASS kernel."""

    def input_quant_key(self) -> QuantKey:
        return kNvfp4Dynamic

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if cutlass_fp4_supported():
            return True, None
        return False, "CUTLASS FP4 kernels not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(
            layer,
            "weight_scale",
            swizzle_blockscale(layer.weight_scale.data),
        )
        padded_weight, _ = pad_nvfp4_weight_for_cutlass(layer.weight.data)
        replace_parameter(layer, "weight", padded_weight)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        weights_padding_bytes = nvfp4_weight_padding_bytes(layer)

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            x_fp4 = qa.data.reshape(-1, qa.data.shape[-1])
            x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_bytes)
            x_blockscale = qa.scale
            output_dtype = qa.orig_dtype
            output_shape = [*qa.orig_shape[:-1], output_size]
        else:
            assert isinstance(x, torch.Tensor)
            output_dtype = x.dtype
            output_shape = [*x.shape[:-1], output_size]
            x_fp4, x_blockscale = scaled_fp4_quant(
                x,
                layer.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend="cutlass",
                padded_n=x.shape[-1] + weights_padding_bytes * 2,
            )

        out = cutlass_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
