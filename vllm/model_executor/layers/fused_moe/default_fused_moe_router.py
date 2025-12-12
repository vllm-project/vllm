# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from functools import partial

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingSimulator
from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from .fused_moe import eplb_map_to_physical_and_record
else:

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids


from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk,
)


class DefaultFusedMoERouter(FusedMoERouter):
    def __init__(
        self,
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
    ):
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_fused_shared_experts = num_fused_shared_experts
        self.enable_eplb = enable_eplb
        self.eplb_state = eplb_state
        self.indices_type_getter = indices_type_getter

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        # ToDo: Better logic to determine the routing method type
        if routing_method_type is not None:
            self._routing_method_type: RoutingMethodType = routing_method_type
        else:
            if scoring_func == "sigmoid":
                if self.use_grouped_topk:
                    self._routing_method_type = RoutingMethodType.DeepSeekV3
                elif self.top_k == 1:
                    self._routing_method_type = RoutingMethodType.Llama4
            elif self.scoring_func == "softmax":
                self._routing_method_type = (
                    RoutingMethodType.Renormalize
                    if not self.renormalize
                    else RoutingMethodType.RenormalizeNaive
                )
            else:
                self._routing_method_type = RoutingMethodType.TopK

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self._routing_method_type

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids)
                (tuple[torch.Tensor, torch.Tensor]):
                The weights, expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        if self.enable_eplb:
            if self.eplb_state.expert_load_view is None:
                raise ValueError("enable_eplb=True requiere expert_load_view != None")
            if self.eplb_state.logical_to_physical_map is None:
                raise ValueError(
                    "enable_eplb=True requiere logical_to_physical_map != None"
                )
            if self.eplb_state.logical_replica_count is None:
                raise ValueError(
                    "enable_eplb=True requiere logical_replica_count != None"
                )

        def valid_grouping() -> bool:
            # Check if num_experts is greater than num_expert_group
            # and is divisible by num_expert_group
            assert self.num_expert_group is not None
            num_experts = router_logits.shape[-1]
            if num_experts <= self.num_expert_group:
                return False
            return num_experts % self.num_expert_group == 0

        indices_type = (
            self.indices_type_getter() if self.indices_type_getter is not None else None
        )

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=self.top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        elif self.use_grouped_topk and valid_grouping():
            assert self.topk_group is not None
            assert self.num_expert_group is not None
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
        elif self.e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        elif self.custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )

        if self.enable_eplb:
            assert self.eplb_state.expert_load_view is not None
            assert self.eplb_state.logical_to_physical_map is not None
            assert self.eplb_state.logical_replica_count is not None
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=self.eplb_state.expert_load_view,
                logical_to_physical_map=self.eplb_state.logical_to_physical_map,
                logical_replica_count=self.eplb_state.logical_replica_count,
            )

        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None

        return topk_weights, topk_ids
