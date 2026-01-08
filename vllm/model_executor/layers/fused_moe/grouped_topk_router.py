# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from functools import partial

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk,
)


class GroupedTopKRouter(BaseRouter):
    """Router using grouped top-k routing (e.g., DeepSeekV2/V3)."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        num_expert_group: int,
        topk_group: int,
        renormalize: bool = True,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_fused_shared_experts: int = 0,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
        routing_method_type: RoutingMethodType | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_fused_shared_experts = num_fused_shared_experts

        # Determine routing method type
        if routing_method_type is not None:
            self._routing_method_type = routing_method_type
        elif scoring_func == "sigmoid":
            self._routing_method_type = RoutingMethodType.DeepSeekV3
        else:
            self._routing_method_type = RoutingMethodType.TopK

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self._routing_method_type

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using grouped top-k."""

        def valid_grouping() -> bool:
            # Check if num_experts is greater than num_expert_group
            # and is divisible by num_expert_group
            num_experts = router_logits.shape[-1]
            if num_experts <= self.num_expert_group:
                return False
            return num_experts % self.num_expert_group == 0

        if not valid_grouping():
            raise ValueError(
                f"Invalid grouping: num_experts={router_logits.shape[-1]}, "
                f"num_expert_group={self.num_expert_group}"
            )

        # Select grouped_topk implementation
        if rocm_aiter_ops.is_fused_moe_enabled():
            if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                assert self.num_fused_shared_experts == 0
            grouped_topk_impl = partial(
                rocm_aiter_grouped_topk,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
        else:
            grouped_topk_impl = grouped_topk

        topk_weights, topk_ids = grouped_topk_impl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )

        return topk_weights, topk_ids
