# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for NVFP4 + FlashInfer fused-MoE path"""
from __future__ import annotations

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                         FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
    FlashInferCutlassMoEPrepareAndFinalize)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

__all__ = [
    "is_flashinfer_fp4_cutlass_moe_available",
    "reorder_w1w3_to_w3w1",
    "build_flashinfer_fp4_cutlass_moe_prepare_finalize",
]


def is_flashinfer_fp4_cutlass_moe_available() -> bool:
    """Return ``True`` when FlashInfer CUTLASS NV-FP4 kernels can be used."""
    return (envs.VLLM_USE_FLASHINFER_MOE_FP4
            and has_flashinfer_cutlass_fused_moe()
            and current_platform.is_cuda()
            and current_platform.is_device_capability(100))


def reorder_w1w3_to_w3w1(weight: torch.Tensor,
                         scale: torch.Tensor,
                         dim: int = -2) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
    size = weight.size(dim)
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
    half = size // 2

    w1, w3 = weight.split(half, dim=dim)
    s1, s3 = scale.split(half, dim=dim)

    return (torch.cat([w3, w1],
                      dim=dim).contiguous(), torch.cat([s3, s1],
                                                       dim=dim).contiguous())


def build_flashinfer_fp4_cutlass_moe_prepare_finalize(
        moe: FusedMoEConfig) -> mk.FusedMoEPrepareAndFinalize:
    """Create a FlashInfer CUTLASS fused-MoE prepare finalize kernel"""
    use_dp = moe.moe_parallel_config.dp_size > 1
    return FlashInferCutlassMoEPrepareAndFinalize(use_dp)


def select_nvfp4_gemm_impl(
    moe: FusedMoEConfig,
    moe_quant_config: FusedMoEQuantConfig,
    allow_flashinfer: bool,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    """Return a GEMM *experts* implementation for NV-FP4 fused-MoE layers"""

    if allow_flashinfer:
        return FlashInferExperts(
            out_dtype=moe.in_dtype,
            quant_config=moe_quant_config,
            ep_rank=moe.moe_parallel_config.ep_rank,
            ep_size=moe.moe_parallel_config.ep_size,
            tp_rank=moe.moe_parallel_config.tp_rank,
            tp_size=moe.moe_parallel_config.tp_size,
        )

    # native cutlass experts currently don't support DP; TP case won't call this
    raise ValueError(
        "CutlassExpertsFp4 doesn't support DP. Use flashinfer CUTLASS "
        "Fused MoE backend instead (set VLLM_USE_FLASHINFER_MOE_FP4=1)")
