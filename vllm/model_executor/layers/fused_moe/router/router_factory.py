# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter,
)

EMPTY_EPLB_STATE: EplbLayerState = EplbLayerState()


def create_fused_moe_router(
    # common parameters
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    # grouped topk parameters
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    # grouped topk + fused topk bias parameters
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # custom routing paramaters
    custom_routing_function: Callable | None = None,
    # eplb parameters
    enable_eplb: bool = False,
    eplb_state: EplbLayerState = EMPTY_EPLB_STATE,
    capture: Callable[[torch.tensor], None] | None = None,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. GroupedTopKRouter - if use_grouped_topk is True
    3. CustomRoutingRouter - if custom_routing_function is not None
    4. FusedTopKBiasRouter - if e_score_correction_bias is not None
    5. FusedTopKRouter - default fallback

    Common arguments:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        renormalize: Whether to renormalize the routing weights
        indices_type_getter: Function to get the desired indices dtype
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
        enable_eplb: Whether EPLB is enabled
        eplb_state: EPLB (Expert Parallelism Load Balancing) state

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """
    router: BaseRouter

    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        router = RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        # TODO(bnell): this is temporary until select_experts is
        # separated from apply.
        router.capture = capture
        return router

    if use_grouped_topk:
        assert custom_routing_function is None
        if num_expert_group is None or topk_group is None:
            raise ValueError(
                "num_expert_group and topk_group must be provided when "
                "use_grouped_topk is True"
            )
        router = GroupedTopKRouter(
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
        )
        router.capture = capture
        return router

    if custom_routing_function is not None:
        router = CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        router.capture = capture
        return router

    if e_score_correction_bias is not None:
        router = FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        router.capture = capture
        return router

    router = FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
        enable_eplb=enable_eplb,
        indices_type_getter=indices_type_getter,
    )
    router.capture = capture
    return router
