# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


class FusedMoERouter(ABC):
    """
    FusedMoERouter is an abstract class that provides a 'select_experts'
    method that is used for routing hidden states based on router logits.
    """

    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType:
        raise NotImplementedError

    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        raise NotImplementedError
