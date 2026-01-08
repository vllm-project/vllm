# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.fused_topk_router import FusedTopKRouter
from vllm.model_executor.layers.fused_moe.grouped_topk_router import GroupedTopKRouter
from vllm.model_executor.layers.fused_moe.routing_simulator_router import (
    RoutingSimulatorRouter,
)


def create_fused_moe_router(
    top_k: int,
    global_num_experts: int,
    eplb_state: EplbLayerState,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
    enable_eplb: bool = False,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    routing_method_type: RoutingMethodType | None = None,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. GroupedTopKRouter - if use_grouped_topk is True
    3. FusedTopKBiasRouter - if e_score_correction_bias is not None
    4. CustomRoutingRouter - if custom_routing_function is not None
    5. FusedTopKRouter - default fallback

    Args:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        eplb_state: EPLB (Expert Parallelism Load Balancing) state
        renormalize: Whether to renormalize the routing weights
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group (for grouped routing)
        custom_routing_function: Optional custom routing function
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        routed_scaling_factor: Scaling factor for routed weights
        e_score_correction_bias: Optional bias correction for expert scores
        num_fused_shared_experts: Number of fused shared experts (for ROCm AITER)
        enable_eplb: Whether EPLB is enabled
        indices_type_getter: Function to get the desired indices dtype
        routing_method_type: Optional explicit routing method type

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """

    # Priority 1: Check if routing simulation is enabled via environment variable
    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        return RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    # Priority 2: Check if grouped top-k routing is requested
    if use_grouped_topk:
        if num_expert_group is None or topk_group is None:
            raise ValueError(
                "num_expert_group and topk_group must be provided when "
                "use_grouped_topk is True"
            )
        return GroupedTopKRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
            routing_method_type=routing_method_type,
        )

    # Priority 3: Check if bias correction is provided
    if e_score_correction_bias is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    # Priority 4: Check if custom routing function is provided
    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    # Priority 5: Default to standard fused top-k routing
    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        indices_type_getter=indices_type_getter,
    )
