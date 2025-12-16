# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for NVFP4 + FlashInfer fused-MoE path"""

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
    create_flashinfer_prepare_finalize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    has_flashinfer_cutlass_fused_moe,
)

__all__ = [
    "is_flashinfer_fp4_cutlass_moe_available",
    "is_flashinfer_fp4_cutedsl_moe_available",
    "reorder_w1w3_to_w3w1",
    "build_flashinfer_fp4_cutlass_moe_prepare_finalize",
]


def is_flashinfer_fp4_cutlass_moe_available() -> bool:
    """Return `True` when FlashInfer CUTLASS NV-FP4 kernels can be used."""
    return (
        envs.VLLM_USE_FLASHINFER_MOE_FP4
        and has_flashinfer_cutlass_fused_moe()
        and current_platform.is_cuda()
        and current_platform.has_device_capability(100)
    )


def is_flashinfer_fp4_cutedsl_moe_available() -> bool:
    """Return ``True`` when FlashInfer CUTEDSL NV-FP4 kernels can be used."""
    return (
        envs.VLLM_USE_FLASHINFER_MOE_FP4
        and has_flashinfer_cutedsl_grouped_gemm_nt_masked()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    )


def reorder_w1w3_to_w3w1(
    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
    size = weight.size(dim)
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
    half = size // 2

    w1, w3 = weight.split(half, dim=dim)
    s1, s3 = scale.split(half, dim=dim)

    return (
        torch.cat([w3, w1], dim=dim).contiguous(),
        torch.cat([s3, s1], dim=dim).contiguous(),
    )


def build_flashinfer_fp4_cutlass_moe_prepare_finalize(
    moe: FusedMoEConfig,
) -> mk.FusedMoEPrepareAndFinalize:
    """Create a FlashInfer CUTLASS fused-MoE prepare finalize kernel"""
    use_dp = moe.moe_parallel_config.dp_size > 1
    enable_alltoallv = moe.moe_parallel_config.all2all_backend == "flashinfer_all2allv"
    return create_flashinfer_prepare_finalize(
        use_dp=use_dp, use_nvfp4=True, enable_alltoallv=enable_alltoallv
    )


def select_nvfp4_gemm_impl(
    moe: FusedMoEConfig,
    moe_quant_config: FusedMoEQuantConfig,
    allow_flashinfer: bool,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    """Return a GEMM *experts* implementation for NV-FP4 fused-MoE layers"""

    if allow_flashinfer:
        if envs.VLLM_FLASHINFER_MOE_BACKEND == "masked_gemm":
            return FlashInferCuteDSLExperts(
                out_dtype=moe.in_dtype,
                quant_config=moe_quant_config,
            )
        elif envs.VLLM_FLASHINFER_MOE_BACKEND == "throughput":
            return FlashInferExperts(
                out_dtype=moe.in_dtype,
                quant_config=moe_quant_config,
                ep_rank=moe.moe_parallel_config.ep_rank,
                ep_size=moe.moe_parallel_config.ep_size,
                tp_rank=moe.moe_parallel_config.tp_rank,
                tp_size=moe.moe_parallel_config.tp_size,
                use_dp=moe.moe_parallel_config.dp_size > 1,
            )

    # native cutlass experts currently don't support DP; TP case won't call this
    raise ValueError(
        "CutlassExpertsFp4 doesn't support DP. Use flashinfer CUTLASS "
        "Fused MoE backend instead (set VLLM_USE_FLASHINFER_MOE_FP4=1)"
    )


def prepare_static_weights_for_trtllm_fp4_moe(
    # args_dequant,
    # args,
    gemm1_weights,
    gemm2_weights,
    gemm1_scales_linear_fp4_bytes,
    gemm2_scales_linear_fp4_bytes,
    hidden_size,
    intermediate_size,
    num_experts,
):
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    """Prepare quantized weights for kernel (done offline with weights)."""
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(num_experts, hidden_size, intermediate_size // 16)  # fp8 scaling factors

    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(num_experts):
        # Calculate the permute indices for the following:
        # 1. Reorder rows of W1 and scales for fused gated activation
        # 2. Shuffle weights and scaling factors for transposed mma output
        # for both w3_w1 and w2 weights and scale factors
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm1_weights_fp4_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm1_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_fp4_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    )


def flashinfer_trtllm_fp4_moe(
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    num_expert_group: int | None,
    topk_group: int | None,
    custom_routing_function: object | None,
    e_score_correction_bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply FlashInfer TensorRT-LLM FP4 MoE kernel.

    Args:
        layer: The MoE layer with weights and scales
        x: Input tensor
        router_logits: Router logits for expert selection
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts across all ranks
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group
        custom_routing_function: Custom routing function (e.g., Llama4)
        e_score_correction_bias: Optional routing bias correction

    Returns:
        Output tensor from the MoE layer
    """
    import flashinfer

    from vllm.model_executor.models.llama4 import Llama4MoE

    # Quantize input to FP4
    a1_gscale = layer.w13_input_scale_quant
    (hidden_states_fp4, hidden_states_scale_linear_fp4) = flashinfer.fp4_quantize(
        x,
        a1_gscale,
        is_sf_swizzled_layout=False,
    )

    # Determine routing method type
    use_llama4_routing = custom_routing_function is Llama4MoE.custom_routing_function
    routing_method_type = layer.routing_method_type
    if use_llama4_routing:
        routing_method_type = flashinfer.RoutingMethodType.Llama4

    # Prepare routing bias
    routing_bias = e_score_correction_bias
    if routing_bias is not None:
        routing_bias = routing_bias.to(torch.bfloat16)

    router_logits = (
        router_logits.to(torch.float32)
        if routing_method_type == RoutingMethodType.DeepSeekV3
        else router_logits
    )

    # Call TRT-LLM FP4 block-scale MoE kernel
    out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale_linear_fp4.view(
            torch.float8_e4m3fn
        ).flatten(),
        gemm1_weights=layer.w13_weight.data,
        gemm1_weights_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer.w2_weight.data,
        gemm2_weights_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=layer.g1_scale_c.data,
        output1_scale_gate_scalar=layer.g1_alphas.data,
        output2_scale_scalar=layer.g2_alphas.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group if num_expert_group is not None else 0,
        topk_group=topk_group if topk_group is not None else 0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        tile_tokens_dim=None,
        routing_method_type=routing_method_type,
        do_finalize=True,
    )[0]

    return out


def flashinfer_trtllm_fp4_routed_moe(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    global_num_experts: int,
) -> torch.Tensor:
    """
    Apply FlashInfer TensorRT-LLM FP4 MoE kernel. Uses packed
    input top k expert indices and scores rather than computing
    top k expert indices from scores.

    Args:
        layer: The MoE layer with weights and scales
        x: Input tensor
        topk_ids: Ids of selected experts
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts across all ranks

    Returns:
        Output tensor from the MoE layer
    """
    import flashinfer

    # Pack top k ids and expert weights into a single int32 tensor, as
    # required by TRT-LLM
    packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    # Quantize input to FP4
    a1_gscale = layer.w13_input_scale_quant
    (hidden_states_fp4, hidden_states_scale_linear_fp4) = flashinfer.fp4_quantize(
        x,
        a1_gscale,
        is_sf_swizzled_layout=False,
    )

    # Call TRT-LLM FP4 block-scale MoE kernel
    out = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed_tensor,
        routing_bias=None,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale_linear_fp4.view(
            torch.float8_e4m3fn
        ).flatten(),
        gemm1_weights=layer.w13_weight.data,
        gemm1_weights_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer.w2_weight.data,
        gemm2_weights_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=layer.g1_scale_c.data,
        output1_scale_gate_scalar=layer.g1_alphas.data,
        output2_scale_scalar=layer.g2_alphas.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=0,
        topk_group=0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        tile_tokens_dim=None,
        routing_method_type=1,
        do_finalize=True,
    )[0]

    return out
