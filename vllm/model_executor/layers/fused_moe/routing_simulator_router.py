# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingSimulator


class RoutingSimulatorRouter(BaseRouter):
    """Router that uses routing simulation strategies for testing/debugging."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
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

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Simulated

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use routing simulator to compute routing."""
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=routing_strategy,
            top_k=self.top_k,
            indices_type=indices_type,
        )
        return topk_weights, topk_ids
