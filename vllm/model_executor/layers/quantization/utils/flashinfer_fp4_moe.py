# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for NVFP4 + FlashInfer fused-MoE path"""
from __future__ import annotations

from typing import Optional

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts, is_valid_flashinfer_cutlass_fused_moe)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
    FlashInferCutlassMoEPrepareAndFinalize)
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = [
    "is_flashinfer_fp4_cutlass_moe_available",
    "reorder_w1w3_to_w3w1",
    "build_flashinfer_fp4_cutlass_moe_kernel",
    "flashinfer_fp4_cutlass_moe_forward",
]


def is_flashinfer_fp4_cutlass_moe_available() -> bool:
    """Return ``True`` when FlashInfer CUTLASS NV-FP4 kernels can be used."""
    return (envs.VLLM_USE_FLASHINFER_MOE_FP4 and current_platform.is_cuda()
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


def build_flashinfer_fp4_cutlass_moe_kernel(
    moe_parallel_config: FusedMoEParallelConfig, ) -> mk.FusedMoEModularKernel:
    """Create *and return* a FlashInfer CUTLASS fused-MoE modular kernel"""
    experts = FlashInferExperts(
        use_nvfp4_w4a4=True,
        use_dp=moe_parallel_config.dp_size > 1,
        ep_rank=moe_parallel_config.ep_rank,
        ep_size=moe_parallel_config.ep_size,
        tp_rank=moe_parallel_config.tp_rank,
        tp_size=moe_parallel_config.tp_size,
    )
    logger.debug_once("FlashInferExperts (util)")
    return mk.FusedMoEModularKernel(
        FlashInferCutlassMoEPrepareAndFinalize(quant_dtype=torch.uint8),
        experts,
    )


def flashinfer_fp4_cutlass_moe_forward(
    fused_experts: mk.FusedMoEModularKernel,
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Common forward wrapper for FlashInfer NV-FP4 fused-MoE"""

    assert is_valid_flashinfer_cutlass_fused_moe(
        x, layer.w13_weight,
        layer.w2_weight), ("FlashInfer CUTLASS fused-MoE not applicable!")

    a1_gscale = layer.w13_input_scale_quant
    a2_gscale = layer.w2_input_scale_quant

    extra_expert_args = {
        "g1_alphas": layer.g1_alphas,
        "g2_alphas": layer.g2_alphas,
        # Avoid confusion with a1_scale and a2_scale
        # where are batch size related.
        "a1_gscale": a1_gscale,
        "a2_gscale": a2_gscale,
        "out_dtype": x.dtype,
    }
    extra_prepare_args = {
        "use_dp": layer.dp_size > 1,
        "local_tokens": x.shape[0],
        "a1_gscale": a1_gscale,
    }
    extra_finalize_args = {
        "use_dp": layer.dp_size > 1,
        "local_tokens": x.shape[0],
    }

    return fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,  # TODO(shuw): fix later, now output is high prec
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=layer.w13_blockscale_swizzled,
        w2_scale=layer.w2_blockscale_swizzled,
        apply_router_weight_on_input=apply_router_weight_on_input,
        extra_expert_args=extra_expert_args,
        extra_prepare_args=extra_prepare_args,
        extra_finalize_args=extra_finalize_args,
    )


def select_nvfp4_gemm_impl(
        allow_flashinfer: bool,
        moe,  # FusedMoEConfig
        logger):
    """Return a GEMM *experts* implementation for NV-FP4 fused-MoE layers"""

    # lazy import
    from vllm.distributed import get_ep_group

    all2all_manager = get_ep_group().device_communicator.all2all_manager
    assert all2all_manager is not None

    if allow_flashinfer:
        flashinfer_backend = envs.VLLM_FLASHINFER_MOE_BACKEND
        if flashinfer_backend != "throughput":
            raise ValueError(
                f"Only throughput backend is supported for FlashInferExperts, "
                f"but got {flashinfer_backend}.")
        logger.debug_once(
            "Initializing FlashInferExperts with throughput backend.")
        return FlashInferExperts(
            use_nvfp4_w4a4=True,
            use_dp=moe.moe_parallel_config.dp_size > 1,
            ep_rank=moe.moe_parallel_config.ep_rank,
            ep_size=moe.moe_parallel_config.ep_size,
            tp_rank=moe.moe_parallel_config.tp_rank,
            tp_size=moe.moe_parallel_config.tp_size,
        )

    # native cutlass experts currently don't support DP; TP case won't call this
    raise ValueError(
        "CutlassExpertsFp4 doesn't support DP. Use flashinfer CUTLASS "
        "Fused MoE backend instead (set VLLM_USE_FLASHINFER_MOE_FP4=1)")
