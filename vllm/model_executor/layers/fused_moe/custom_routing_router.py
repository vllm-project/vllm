# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
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


class CustomRoutingRouter(FusedMoERouter):
    """Router using a custom user-provided routing function."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        custom_routing_function: Callable,
        renormalize: bool = True,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.custom_routing_function = custom_routing_function
        self.renormalize = renormalize
        self.eplb_state = eplb_state
        self.enable_eplb = enable_eplb
        self.indices_type_getter = indices_type_getter
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Unspecified

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

        indices_type = (
            self.indices_type_getter() if self.indices_type_getter is not None else None
        )

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

        # Compute zero expert result if needed
        if (
            self.zero_expert_num is not None
            and self.zero_expert_num > 0
            and self.zero_expert_type is not None
            and self.global_num_experts is not None
        ):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=self.global_num_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None

        return topk_weights, topk_ids, zero_expert_result
