# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


class FusedMoERouter(ABC):
    """
    FusedMoERouter is an abstract class that provides a 'select_experts'
    method that is used for routing hidden states based on router logits.
    """

    def __init__(self, eplb_state: EplbLayerState | None = None):
        self._routing_replay_out: torch.Tensor | None = None
        self.eplb_state = eplb_state

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
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """

        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            topk_indices_dtype=topk_indices_dtype,
            input_ids=input_ids,
        )

        # Write routing data for non-monolithic path (Triton, etc.)
        # (set by bind_routing_capture_to_model during capturer init)
        if self._routing_replay_out is not None:
            self._routing_replay_out[: topk_ids.shape[0]].copy_(
                topk_ids.to(torch.int16)
            )

        return topk_weights, topk_ids
