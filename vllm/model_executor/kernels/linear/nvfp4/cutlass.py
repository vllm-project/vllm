# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from ..base import partial_sum_prescaled, store_partial_sum_prescaled
from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class CutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the vLLM CUTLASS kernel."""

    supports_partial_sum_dtype = True

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        # The SM120 GEMM can add the bias in its epilogue.
        self.bias_epilogue = current_platform.is_device_capability_family(120)

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
        store_partial_sum_prescaled(layer, "partial_sum_alpha", layer.alpha)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        qa = self.prepare_input(layer, x)
        output_dtype = qa.orig_dtype

        # FP16 partial sums of RowParallelLinear, prescaled by 2^-k.
        alpha = layer.alpha
        partial_sum_dtype = getattr(layer, "partial_sum_dtype", None)
        if partial_sum_dtype is not None:
            output_dtype = partial_sum_dtype
            if layer.partial_sum_scale != 1.0:
                alpha = partial_sum_prescaled(layer, "partial_sum_alpha", alpha)
                if bias is not None:
                    bias = bias * layer.partial_sum_scale
        if bias is not None:
            bias = bias.to(output_dtype)

        # A padded N leaves the bias shorter than the GEMM output.
        fuse_bias = (
            bias is not None
            and self.bias_epilogue
            and layer.weight.shape[0] == output_size
        )
        out = cutlass_scaled_fp4_mm(
            qa.data,
            layer.weight,
            qa.scale,
            layer.weight_scale,
            alpha,
            output_dtype,
            bias=bias if fuse_bias else None,
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None and not fuse_bias:
            out = out + bias
        return out.view(*qa.orig_shape[:-1], output_size)
