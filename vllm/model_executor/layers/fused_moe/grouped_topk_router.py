# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from functools import partial

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.fused_moe import (
    grouped_topk,
    zero_experts_compute_triton,
)
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk,
)
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


class GroupedTopKRouter(FusedMoERouter):
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
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
        routing_method_type: RoutingMethodType | None = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_fused_shared_experts = num_fused_shared_experts
        self.eplb_state = eplb_state
        self.enable_eplb = enable_eplb
        self.indices_type_getter = indices_type_getter
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

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

        indices_type = (
            self.indices_type_getter() if self.indices_type_getter is not None else None
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
