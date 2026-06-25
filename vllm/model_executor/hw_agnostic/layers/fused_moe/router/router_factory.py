# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router selection for the hw-agnostic FusedMoE.

Vendor-specific routers (AITER fused-shared, RoutingSimulator, ZeroExpert,
GroupedTopK, custom) are out of scope. Selection here picks between
``FusedTopKBiasRouter`` (DSv4-style sqrtsoftplus / hash MoE / sigmoid+bias)
and ``FusedTopKRouter`` (plain softmax/sigmoid top-k).
"""

from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.hw_agnostic.layers.fused_moe.router.fused_moe_router import (  # noqa: E501
    FusedMoERouter,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.router.fused_topk_bias_router import (  # noqa: E501
    FusedTopKBiasRouter,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.router.fused_topk_router import (  # noqa: E501
    FusedTopKRouter,
)


def create_fused_moe_router(
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    eplb_state: EplbLayerState | None = None,
    zero_expert_type: str | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> FusedMoERouter:
    """Construct the appropriate ``FusedMoERouter`` subclass.

    Returns ``FusedTopKBiasRouter`` when an ``e_score_correction_bias`` or
    ``hash_indices_table`` is set (DSv4-style sqrtsoftplus / hash MoE),
    otherwise ``FusedTopKRouter``.
    """

    if envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY:
        raise NotImplementedError(
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is not supported on "
            "the hw-agnostic FusedMoE path."
        )
    if zero_expert_type is not None:
        raise NotImplementedError(
            "ZeroExpertRouter is not supported on the hw-agnostic FusedMoE path."
        )
    if use_grouped_topk:
        raise NotImplementedError(
            "GroupedTopKRouter is not supported on the hw-agnostic FusedMoE path."
        )
    if custom_routing_function is not None:
        raise NotImplementedError(
            "CustomRoutingRouter is not supported on the hw-agnostic FusedMoE path."
        )

    assert scoring_func in ("sigmoid", "softmax", "sqrtsoftplus")

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

    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )
