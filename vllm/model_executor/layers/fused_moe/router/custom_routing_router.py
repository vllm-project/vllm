# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


class CustomRoutingRouter(BaseRouter):
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
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.custom_routing_function = custom_routing_function
        self.renormalize = renormalize

    @property
    def routing_method_type(self) -> RoutingMethodType:
        from vllm.model_executor.models.llama4 import Llama4MoE

        # NOTE: FLASHINFER_TRTLLM support the Llama4 router.
        if self.custom_routing_function == Llama4MoE.custom_routing_function:
            return RoutingMethodType.Llama4
        return RoutingMethodType.Custom

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using the custom routing function."""
        topk_weights, topk_ids = self.custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )

        return topk_weights.to(torch.float32), topk_ids
