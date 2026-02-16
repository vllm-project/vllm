# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch


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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def must_reduce_shared_expert_outputs(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def maybe_all_reduce_tensor_model_parallel(
        self,
        final_hidden_states: torch.Tensor,
    ):
        raise NotImplementedError
