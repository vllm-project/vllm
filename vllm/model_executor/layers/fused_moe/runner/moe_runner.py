# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


class MoERunner(ABC):
    """
    Abstract base class for Mixture of Experts (MoE) runners.

    This class defines the interface that all MoE runner implementations must follow.
    MoE runners are responsible for executing the forward pass of MoE layers, handling
    expert routing, and managing tensor parallel operations.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def quant_method(self) -> FusedMoEMethodBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def shared_experts(self) -> SharedExperts | None:
        raise NotImplementedError

    @abstractmethod
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_internal_router(self) -> bool:
        raise NotImplementedError
