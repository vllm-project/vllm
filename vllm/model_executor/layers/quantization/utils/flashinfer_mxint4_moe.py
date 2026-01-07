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


@functools.lru_cache(maxsize=1)
def is_flashinfer_mxint4_moe_available() -> bool:
    """Return `True` when FlashInfer MxInt4 kernels can be used."""
    use_flashinfer_mxint4_moe = (
        envs.VLLM_USE_FLASHINFER_MOE_INT4
        and has_flashinfer_trtllm_fused_moe()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    )
    logger.debug_once(
        f"Using FlashInfer MxInt4 MoE: {use_flashinfer_mxint4_moe}", scope="local"
    )
    return use_flashinfer_mxint4_moe


def prepare_static_weights_for_trtllm_mxint4_moe(
    gemm1_weights,
    gemm1_scales,
    gemm2_weights,
    gemm2_scales,
):
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
        Tuple of shuffled/packed weights and scales ready for kernel
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
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Apply FlashInfer TensorRT-LLM MxInt4 MoE kernel.

    Args:
        layer: MoE layer with mxint4 weights and BF16 scales
        x: Input tensor (bf16)
        router_logits: Router logits for expert selection

    Returns:
        Output tensor from MoE layer
    """
    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe

    assert x.dtype == torch.bfloat16
    assert layer.w13_weight_packed.dtype == torch.uint8, (
        f"w13_weight_packed dtype: {layer.w13_weight_packed.dtype}"
    )
    assert layer.w13_weight_scale.dtype == torch.bfloat16, (
        f"w13_weight_scale dtype: {layer.w13_weight_scale.dtype}"
    )
    assert layer.w2_weight_packed.dtype == torch.uint8, (
        f"w2_weight_packed dtype: {layer.w2_weight_packed.dtype}"
    )
    assert layer.w2_weight_scale.dtype == torch.bfloat16, (
        f"w2_weight_scale dtype: {layer.w2_weight_scale.dtype}"
    )

    routing_bias = None
    if layer.e_score_correction_bias is not None:
        routing_bias = layer.e_score_correction_bias.to(torch.bfloat16)

    if layer.routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    out = trtllm_mxint4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=x,
        gemm1_weights=layer.w13_weight_packed.data,
        gemm1_weights_scale=layer.w13_weight_scale.data,
        gemm1_alpha=layer.gemm1_alpha.data if hasattr(layer, "gemm1_alpha") else None,
        gemm1_beta=layer.gemm1_beta.data if hasattr(layer, "gemm1_beta") else None,
        gemm1_clamp_limit=layer.gemm1_clamp_limit.data
        if hasattr(layer, "gemm1_clamp_limit")
        else None,
        gemm2_weights=layer.w2_weight_packed.data,
        gemm2_weights_scale=layer.w2_weight_scale.data,
        num_experts=layer.global_num_experts,
        top_k=layer.top_k,
        n_group=layer.num_expert_group if layer.num_expert_group is not None else 0,
        topk_group=layer.topk_group if layer.topk_group is not None else 0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=layer.routing_method_type,
        enable_pdl=None,
        output=None,
        tune_max_num_tokens=8192,
    ).to(x.dtype)

    return out
