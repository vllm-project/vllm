# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 visible math with a fixed K32 accumulation order for precision runs."""

import torch
from torch.nn.parameter import Parameter

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.batch_invariant import (
    block32_scaled_mm as batched_block32_mm,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def supports_ds41_mxfp8_bi() -> bool:
    """Limit recipe-specific K32 arithmetic to DS4.1 on SM100."""
    config = get_current_vllm_config_or_none()
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and config is not None
        and config.model_config is not None
        and config.model_config.hf_text_config.model_type == "deepseek_v41"
    )


def block32_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Consume E4M3 codes/E8M0 bytes using FP8 products and FP32 accumulation.

    Each K32 product is accumulated before the next block, independently of
    token count, backend autotuning and the physical scale layout.
    """
    return batched_block32_mm(a, a_scale, weight, weight_scale, output_dtype)


class BatchInvariantMxfp8LinearKernel(Mxfp8LinearKernel):
    """Fixed-block FP8 GEMM; retain quantized weights through reload."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        supported = supports_ds41_mxfp8_bi() and (
            compute_capability is None or 100 <= compute_capability < 110
        )
        return supported, "requires DS4.1 on CUDA SM100"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        n, k = c.weight_shape
        supported = c.bmm_batch_size is None and n > 0 and k > 0 and k % 32 == 0
        return supported, "requires an ungrouped positive shape with K divisible by 32"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        n, k = weight.shape
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data[:n, : k // 32].contiguous(), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype, shape = x.dtype, x.shape
        k = layer.weight.shape[-1]
        codes, scales = mxfp8_e4m3_quantize(
            x.reshape(-1, k), is_sf_swizzled_layout=False
        )
        output = block32_scaled_mm(
            codes, scales, layer.weight, layer.weight_scale, dtype
        )
        if bias is not None:
            output = output + bias
        return output.reshape(*shape[:-1], layer.weight.shape[0])
