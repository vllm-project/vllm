# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


# TODO: add eplb stuff here
class FusedMoERouter(ABC):
    @property
    @abstractmethod
    def enable_eplb(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType:
        raise NotImplementedError

    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError
