# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def uses_flat_mxfp8_swizzled_scale(quant_config: FusedMoEQuantConfig) -> bool:
    return quant_config.quant_dtype == "mxfp8" and quant_config.is_scale_swizzled


def swizzle_scale_after_alltoall(
    scale: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    num_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    if not quant_config.is_scale_swizzled:
        return scale

    if quant_config.quant_dtype == "nvfp4":
        scale = scale.reshape(-1, scale.shape[-1])
        if scale.element_size() == 1:
            scale = scale.view(torch.uint8)
        return nvfp4_block_scale_interleave(scale)

    if quant_config.quant_dtype == "mxfp8":
        scale = scale.reshape(-1, scale.shape[-1])
        assert scale.size(0) == num_tokens, (
            f"mxfp8 scale has {scale.size(0)} rows, but num_tokens={num_tokens}"
        )
        scale_k = scale.shape[-1] * MXFP8_BLOCK_SIZE
        assert scale_k >= hidden_size, (
            f"mxfp8 scale covers K={scale_k}, but hidden_size={hidden_size}"
        )
        return swizzle_mxfp8_scale(scale, M=num_tokens, K=scale_k)

    return scale
