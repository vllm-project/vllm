# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


class FusedMoERouter(ABC):
    """FusedMoERouter is an abstract class that provides a 'select_experts'
    method that is used for routing hidden states based on router logits.
    """

    def __init__(self, eplb_state: EplbLayerState | None = None):
        self._routing_replay_out: torch.Tensor | None = None
        self.eplb_state = eplb_state
        # Deepseek V4.1 vision checkpoints carry a second per-expert routing
        # bias for image sentinel tokens; attached by model code when present.
        self.bias_vl: torch.Tensor | None = None
        self.image_sentinel_lo: int = 0

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
        """Route the input hidden states to the top-k experts based on the
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
            topk_indices_dtype=topk_indices_dtype,
            input_ids=input_ids,
        )
        self._record_routing(topk_ids)
        return topk_weights, topk_ids

    def _record_routing(self, topk_ids: torch.Tensor) -> None:
        # Write routing data for non-monolithic path (Triton, etc.)
        # (set by bind_routing_capture_to_model during capturer init)
        if self._routing_replay_out is not None:
            self._routing_replay_out[: topk_ids.shape[0]].copy_(
                topk_ids.to(torch.int16)
            )

    # A router may own the gate GEMM and route straight from hidden states,
    # which lets it fuse the GEMM with expert selection. The MoE runner binds
    # its gate and skips computing router logits when this is supported.

    def bind_gate(self, gate: torch.nn.Module) -> None:  # noqa: B027
        """Offer the MoE gate to a router that can fuse it into routing.

        A no-op for routers that cannot; the runner keeps the gate GEMM.
        """

    def can_select_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> bool:
        """Whether select_experts_from_hidden_states can route this batch."""
        return False

    def select_experts_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """select_experts with the gate GEMM computed by the router itself."""
        raise NotImplementedError
