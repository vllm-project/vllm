# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.router.fused_topk_bias_router import (  # noqa: E501
    FusedTopKBiasRouter,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)

# Routers DSv4 hw-agnostic doesn't vendor: AiterSharedRoutedFusedMoERouter
# (ROCm AITER fused-shared), CustomRoutingRouter (user-supplied),
# GroupedTopKRouter (DeepSeek V2/V3 style grouped top-k),
# RoutingSimulatorRouter (test/benchmark), ZeroExpertRouter (zero-expert
# layouts). The corresponding selection branches in
# ``create_fused_moe_router`` raise NotImplementedError below; OOT
# vendor plugins re-add them via their own subclass / factory.


def create_fused_moe_router(
    # common parameters
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    # grouped topk parameters
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    # grouped topk + fused topk bias parameters
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # custom routing parameters
    custom_routing_function: Callable | None = None,
    # eplb parameters
    eplb_state: EplbLayerState | None = None,
    # zero expert parameters
    zero_expert_type: str | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. ZeroExpertRouter - if zero_expert_type is not None
    3. GroupedTopKRouter - if use_grouped_topk is True
    4. CustomRoutingRouter - if custom_routing_function is not None
    5. FusedTopKBiasRouter - if e_score_correction_bias is not None
    6. AiterSharedRoutedFusedMoERouter - if num_fused_shared_experts > 0
    7. FusedTopKRouter - default fallback

    Common arguments:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        renormalize: Whether to renormalize the routing weights
        routing_method_type: Optional explicit routing method type

    Grouped topk arguments:
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group (for grouped routing)
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        num_fused_shared_experts: Number of fused shared experts (for ROCm AITER)

    Grouped topk and fused topk bias arguments:
        routed_scaling_factor: Scaling factor for routed weights
        e_score_correction_bias: Optional bias correction for expert scores

    Custom routing arguments:
        custom_routing_function: Optional custom routing function

    EPLB arguments:
        eplb_state: Optional EplbLayerState, None when EPLB is disabled.

    Zero expert arguments:
        zero_expert_type: Type of zero expert (e.g. identity). If not None,
            creates a ZeroExpertRouter.
        num_logical_experts: Number of real (non-zero) experts. Required when
            zero_expert_type is not None.

    Hash Indices Table:
        Used to map input_ids to experts, need for Deepseek V4

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """

    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        raise NotImplementedError(
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY (RoutingSimulatorRouter) "
            "is not vendored on the DSv4 hw-agnostic FusedMoE path."
        )

    if zero_expert_type is not None:
        raise NotImplementedError(
            "ZeroExpertRouter is not vendored on the DSv4 hw-agnostic "
            "FusedMoE path. DSv4 doesn't use zero-expert layouts."
        )

    if use_grouped_topk:
        raise NotImplementedError(
            "GroupedTopKRouter is not vendored on the DSv4 hw-agnostic "
            "FusedMoE path. DSv4 uses sqrtsoftplus / hash MoE / noaux_tc "
            "routing (FusedTopKRouter / FusedTopKBiasRouter); use the "
            "upstream FusedMoE for grouped top-k models."
        )

    if custom_routing_function is not None:
        raise NotImplementedError(
            "CustomRoutingRouter is not vendored on the DSv4 hw-agnostic FusedMoE path."
        )

    assert scoring_func in ["sigmoid", "softmax", "sqrtsoftplus"]

    if e_score_correction_bias is not None or hash_indices_table is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func=scoring_func,
            hash_indices_table=hash_indices_table,
        )

    if num_fused_shared_experts > 0:
        raise NotImplementedError(
            "AiterSharedRoutedFusedMoERouter (ROCm AITER fused-shared) is "
            "not vendored on the DSv4 hw-agnostic FusedMoE path. Disable "
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS for this layer."
        )

    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )
