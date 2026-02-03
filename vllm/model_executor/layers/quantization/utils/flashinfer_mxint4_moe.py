# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for MxInt4 + FlashInfer fused-MoE path"""

import functools

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

__all__ = [
    "prepare_static_weights_for_trtllm_mxint4_moe",
    "flashinfer_trtllm_mxint4_moe",
    "is_flashinfer_mxint4_moe_available",
]

logger = init_logger(__name__)


@functools.cache
def is_flashinfer_mxint4_moe_available() -> bool:
    """Return `True` when FlashInfer MxInt4 kernels can be used."""
    return (
        envs.VLLM_USE_FLASHINFER_MOE_INT4
        and has_flashinfer_trtllm_fused_moe()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    )


def prepare_static_weights_for_trtllm_mxint4_moe(
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Prepare MxInt4 weights for TRT-LLM kernel.

    Input:
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size//8] int32
            (checkpoint uint4b8 packed) or uint8 (already packed signed int4)
        gemm1_scales: [num_experts, 2*intermediate_size, hidden_size//32] bf16
        gemm2_weights: [num_experts, hidden_size, intermediate_size//8] int32
            (checkpoint uint4b8 packed) or uint8 (already packed signed int4)
        gemm2_scales: [num_experts, hidden_size, intermediate_size//32] bf16

    Returns:
        Dict with keys 'gemm1_weights', 'gemm1_scales', 'gemm2_weights',
            'gemm2_scales' containing shuffled/packed tensors ready for kernel
    """
    from flashinfer import block_scale_interleave
    from flashinfer.fused_moe import (
        convert_to_block_layout,
    )
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        reorder_w1w3_to_w3w1,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        convert_packed_uint4b8_to_signed_int4_inplace,
    )

    device = gemm1_weights.device
    assert gemm1_weights.ndim == 3, (
        f"Expected a 3D gemm1_weights tensor, got {gemm1_weights.shape}"
    )
    assert gemm1_scales.ndim == 3, (
        f"Expected a 3D gemm1_scales tensor, got {gemm1_scales.shape}"
    )
    assert gemm2_weights.ndim == 3, (
        f"Expected a 3D gemm2_weights tensor, got {gemm2_weights.shape}"
    )
    assert gemm2_scales.ndim == 3, (
        f"Expected a 3D gemm2_scales tensor, got {gemm2_scales.shape}"
    )

    # Convert checkpoint format (uint4b8 in int32) to signed int4
    # Checkpoint stores INT4 as unsigned [0, 15], kernel expects signed [-8, 7]
    if gemm1_weights.dtype == torch.int32 and gemm2_weights.dtype == torch.int32:
        convert_packed_uint4b8_to_signed_int4_inplace(gemm1_weights)
        convert_packed_uint4b8_to_signed_int4_inplace(gemm2_weights)

    gemm1_weights, gemm1_scales = reorder_w1w3_to_w3w1(
        gemm1_weights, gemm1_scales, dim=-2
    )

    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    num_experts = gemm1_weights.shape[0]

    # Convert quantized weights to proper formats -
    gemm1_weights_mxint4 = gemm1_weights.view(torch.uint8)
    assert gemm1_scales.dtype == torch.bfloat16
    gemm2_weights_mxint4 = gemm2_weights.view(torch.uint8)
    assert gemm2_scales.dtype == torch.bfloat16

    epilogue_tile_m = 128
    gemm1_weights_mxint4_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_mxint4_shuffled = []
    gemm2_scales_shuffled = []

    for i in range(num_experts):
        # Calculate the permute indices for the following:
        # 1. Reorder rows of W1 and scales for fused gated activation
        # 2. Shuffle weights and scaling factors for transposed mma output
        # for both w3_w1 and w2 weights and scale factors
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_mxint4[i],
            epilogue_tile_m,
        )
        gemm1_weights_shuffled = gemm1_weights_mxint4[i][
            permute_indices.to(gemm1_weights.device)
        ].contiguous()
        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales[i],
            epilogue_tile_m,
            num_elts_per_sf=32,
        ).to(device)
        gemm1_scales_shuffled.append(
            block_scale_interleave(gemm1_scales[i][permute_sf_indices].contiguous())
        )

        permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_mxint4[i],
            epilogue_tile_m,
        )
        gemm2_weights_shuffled = gemm2_weights_mxint4[i][
            permute_indices.to(gemm2_weights.device)
        ].contiguous()

        permute_sf_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales[i],
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_shuffled.append(
            block_scale_interleave(
                gemm2_scales[i][permute_sf_indices.to(gemm2_scales.device)].contiguous()
            )
        )

        block_k = 128
        gemm1_weights_shuffled = convert_to_block_layout(
            gemm1_weights_shuffled.view(torch.uint8), block_k
        )
        gemm2_weights_shuffled = convert_to_block_layout(
            gemm2_weights_shuffled.view(torch.uint8), block_k
        )

        gemm1_weights_mxint4_shuffled.append(gemm1_weights_shuffled)
        gemm2_weights_mxint4_shuffled.append(gemm2_weights_shuffled)

    gemm1_weights_mxint4_shuffled = torch.stack(gemm1_weights_mxint4_shuffled)
    gemm2_weights_mxint4_shuffled = torch.stack(gemm2_weights_mxint4_shuffled)
    gemm1_scales_shuffled = torch.stack(gemm1_scales_shuffled).view(torch.bfloat16)
    gemm2_scales_shuffled = torch.stack(gemm2_scales_shuffled).view(torch.bfloat16)
    return {
        "gemm1_weights": gemm1_weights_mxint4_shuffled,
        "gemm1_scales": gemm1_scales_shuffled,
        "gemm2_weights": gemm2_weights_mxint4_shuffled,
        "gemm2_scales": gemm2_scales_shuffled,
    }


def flashinfer_trtllm_mxint4_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    global_num_experts: int,
    top_k: int,
    intermediate_size_per_partition: int,
    local_num_experts: int,
    ep_rank: int = 0,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    routing_method_type: int | None = None,
) -> torch.Tensor:
    """
    Apply FlashInfer TensorRT-LLM MxInt4 MoE kernel.

    Args:
        x: Input hidden states. dtype: bfloat16
        router_logits: Router logits for expert selection. dtype: bfloat16/float32
        w13_weight_packed: Packed gate+up weights. dtype: uint8
        w13_weight_scale: Scales for gate+up weights. dtype: bfloat16
        w2_weight_packed: Packed down weights. dtype: uint8
        w2_weight_scale: Scales for down weights. dtype: bfloat16
        global_num_experts: Total number of experts across all ranks
        top_k: Number of experts to select per token
        intermediate_size_per_partition: Intermediate size per partition
        local_num_experts: Number of experts on this rank
        ep_rank: Expert parallelism rank (default: 0)
        num_expert_group: Number of expert groups (default: None -> 0)
        topk_group: Top-k within groups (default: None -> 0)
        e_score_correction_bias: Optional routing bias. dtype: bfloat16
        routing_method_type: FlashInfer RoutingMethodType enum value

    Returns:
        Output tensor from MoE layer. dtype: same as x (bfloat16)
    """
    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe

    assert x.dtype == torch.bfloat16, f"x dtype must be bfloat16, got {x.dtype}"
    assert w13_weight_packed.dtype == torch.uint8, (
        f"w13_weight_packed dtype must be uint8, got {w13_weight_packed.dtype}"
    )
    assert w13_weight_scale.dtype == torch.bfloat16, (
        f"w13_weight_scale dtype must be bfloat16, got {w13_weight_scale.dtype}"
    )
    assert w2_weight_packed.dtype == torch.uint8, (
        f"w2_weight_packed dtype must be uint8, got {w2_weight_packed.dtype}"
    )
    assert w2_weight_scale.dtype == torch.bfloat16, (
        f"w2_weight_scale dtype must be bfloat16, got {w2_weight_scale.dtype}"
    )

    routing_bias = None
    if e_score_correction_bias is not None:
        routing_bias = e_score_correction_bias.to(torch.bfloat16)

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    out = trtllm_mxint4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=x,
        gemm1_weights=w13_weight_packed.data,
        gemm1_weights_scale=w13_weight_scale.data,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2_weight_packed.data,
        gemm2_weights_scale=w2_weight_scale.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group if num_expert_group is not None else 0,
        topk_group=topk_group if topk_group is not None else 0,
        intermediate_size=intermediate_size_per_partition,
        local_expert_offset=ep_rank * local_num_experts,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type,
        enable_pdl=None,
        output=None,
        tune_max_num_tokens=8192,
    ).to(x.dtype)

    return out
