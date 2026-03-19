# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter


class MemoizingRouter(FusedMoERouter):
    def __init__(self, router: FusedMoERouter):
        self.router = router

    def set_capture_fn(
        self,
        capture_fn: Callable[[torch.Tensor], None] | None,
    ) -> None:
        self.router.set_capture_fn(capture_fn)
        self.results: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self.router.routing_method_type

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.results is None:
            self.results = self.router.select_experts(hidden_states, router_logits)
        return self.results
