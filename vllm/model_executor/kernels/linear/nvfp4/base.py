# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    nvfp4_weight_padding_bytes,
    pad_nvfp4_activation_for_cutlass,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
)

from ..base import LinearKernel


@dataclass(slots=True)
class NvFp4LinearLayerConfig:
    """Configuration for an NVFP4 linear layer.

    All NVFP4 layers share the same structure: packed uint8 weights (2 FP4 values per
    byte), FP8-E4M3 per-block weight scales (group size 16), and scalar global
    scales for both weights and activations. `act_dtype` is the activation
    dtype of weight-only layers, when the caller knows it.
    """

    act_dtype: torch.dtype | None = None


class NvFp4LinearKernel(LinearKernel[NvFp4LinearLayerConfig]):
    """Base class for NVFP4 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order,calling `is_supported` and `can_implement` to find the best
    match for the current hardware.
    """

    def prepare_input(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
    ) -> QuantizedActivation:
        """Return 2D packed NVFP4 data with swizzled 128x4 block scales.

        K is zero-padded to match the padded weight. Prequantized inputs are
        validated and padded without modifying the caller's activation.
        """
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is None:
            assert isinstance(x, torch.Tensor)
            data, scale = scaled_fp4_quant(
                x,
                layer.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                padded_n=layer.weight.shape[1] * 2,
            )
            return QuantizedActivation(data, scale, x.dtype, x.shape, kNvfp4Dynamic)

        data = qa.data.reshape(-1, qa.data.shape[-1])
        assert data.shape[-1] == layer.input_size_per_partition // 2, (
            f"Expected {layer.input_size_per_partition // 2} packed NVFP4 bytes "
            f"per row, got {data.shape[-1]}"
        )
        data = pad_nvfp4_activation_for_cutlass(data, nvfp4_weight_padding_bytes(layer))
        return replace(qa, data=data)
