# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    zero_experts_compute_triton,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)


class ZeroExpertRouter(BaseRouter):
    """Router that handles zero expert computation as part of routing.

    Routes over all experts (real + zero) using full e_score_correction_bias.
    Computes zero expert identity contributions as a side effect during routing.
    Remaps zero expert IDs to real expert ID 0 (with weight 0) so downstream
    MoE computation can ignore them.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        e_score_correction_bias: torch.Tensor,
        num_logical_experts: int,
        zero_expert_type: str,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.e_score_correction_bias = e_score_correction_bias
        self.num_logical_experts = num_logical_experts
        self.zero_expert_type = zero_expert_type
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self._zero_expert_output: torch.Tensor | None = None

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=True,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing with full bias, compute zero expert output,
        mask zero expert IDs."""
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            e_score_correction_bias=self.e_score_correction_bias.data,
            topk=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
            indices_type=indices_type,
        )

        if self.routed_scaling_factor != 1.0:
            topk_weights *= self.routed_scaling_factor

        # Compute zero expert output using pre-EPLB topk_ids/weights.
        # zero_experts_compute_triton modifies its inputs in-place, so
        # pass clones.
        self._zero_expert_output = zero_experts_compute_triton(
            expert_indices=topk_ids.clone(),
            expert_scales=topk_weights.clone(),
            num_experts=self.num_logical_experts,
            zero_expert_type=self.zero_expert_type,
            hidden_states=hidden_states,
        )

        # Mask zero expert entries: remap zero expert IDs to 0 with weight 0
        # so downstream MoE computation ignores them.
        zero_mask = topk_ids >= self.num_logical_experts
        topk_ids[zero_mask] = 0
        topk_weights[zero_mask] = 0.0

        return topk_weights, topk_ids

    @property
    def zero_expert_output(self) -> torch.Tensor | None:
        """Retrieve and clear the zero expert output."""
        output = self._zero_expert_output
        self._zero_expert_output = None
        return output
