# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared quantized-dispatch helpers for the DeepEP prepare/finalize impls."""

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)


def quantize_before_dispatch(
    quant_config: FusedMoEQuantConfig, defer_input_quant: bool
) -> bool:
    """
    Do quantized dispatch for blockfp8 and mxfp8, unless the
    subsequent moe kernel requires bf16 inputs.
    """
    if defer_input_quant:
        return False
    return quant_config.is_block_quantized or quant_config.quant_dtype == "mxfp8"


def pack_mxfp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack row-major [M, K/32] UE8M0 scales into [M, K/128] int32.

    DeepEP moves scale factors in 4-byte units, so 1-byte UE8M0 scales must be
    packed 4-per-int32.
    """
    assert scale.dtype == torch.uint8 and scale.ndim == 2, (
        f"expected 2D uint8 mxfp8 scales, got {scale.shape} {scale.dtype}"
    )
    assert scale.size(1) % 4 == 0, (
        f"mxfp8 dispatch needs hidden_size % {MXFP8_BLOCK_SIZE * 4} == 0, "
        f"got {scale.size(1)} scale columns"
    )
    return scale.contiguous().view(torch.int32)


def unpack_mxfp8_scale(
    scale: torch.Tensor, hidden_size: int, is_scale_swizzled: bool
) -> torch.Tensor:
    """Inverse of `pack_mxfp8_scale`, restoring the expert kernel's layout.

    TRTLLM consumes the row-major [M, K/32] scales as-is; CUTLASS wants them
    swizzled into F8_128x4, which can only happen here because the swizzle
    interleaves scales across a 128-row tile (i.e. across tokens).
    """
    scale = scale.contiguous().view(torch.uint8)
    if is_scale_swizzled:
        scale = swizzle_mxfp8_scale(scale, M=scale.size(0), K=hidden_size)
    return scale
