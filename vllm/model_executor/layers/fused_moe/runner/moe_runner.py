# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod

import torch

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


class MoERunner(torch.nn.Module):
    """
    Abstract base class for Mixture of Experts (MoE) runners.

    This class defines the interface that all MoE runner implementations must follow.
    MoE runners are responsible for executing the forward pass of MoE layers, handling
    expert routing, and managing tensor parallel operations.
    """

    def __init__(self):
        super().__init__()
        # HACK
        self._already_called_process_weights_after_loading = True

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
