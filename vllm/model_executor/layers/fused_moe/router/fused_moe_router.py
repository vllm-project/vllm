# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


class FusedMoERouter(ABC):
    """
    FusedMoERouter is an abstract class that provides a 'select_experts'
    method that is used for routing hidden states based on router logits.
    """

    @abstractmethod
    def set_capture_fn(
        self,
        capture_fn: Callable[[torch.Tensor], None] | None,
    ) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType:
        raise NotImplementedError

    @abstractmethod
    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
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

        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            input_ids=input_ids,
        )

        # Get routing replay buffer from persistent attribute
        # (set by bind_routing_capture_to_model during capturer init)
        routing_replay_out = getattr(self, "_routing_replay_out", None)

        # Write routing data for non-monolithic path (Triton, etc.)
        if routing_replay_out is not None:
            routing_replay_out[: topk_ids.shape[0]].copy_(topk_ids.to(torch.int16))

        return topk_weights, topk_ids
