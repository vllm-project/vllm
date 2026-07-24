# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


class BaseRouter(FusedMoERouter):
    """
    Base router class that provides common functionality for all router implementations.

    This class implements the template method pattern where select_experts() handles
    common pre-processing and post-processing, delegating the actual routing logic
    to the abstract _compute_routing() method.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState | None = None,
    ):
        """
        Args:
            top_k: Number of experts to select per token
            global_num_experts: Total number of experts
            eplb_state: Optional EPLBLayerState for load balancing
        """
        super().__init__(eplb_state=eplb_state)
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.capture_fn: Callable[[torch.Tensor], None] | None = None

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """Set a capture callback for logical routed expert IDs."""
        self.capture_fn = capture_fn

    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            if eplb_state.expert_load_view is None:
                raise ValueError("EPLB requires expert_load_view != None")
            if eplb_state.logical_to_physical_map is None:
                raise ValueError("EPLB requires logical_to_physical_map != None")
            if eplb_state.logical_replica_count is None:
                raise ValueError("EPLB requires logical_replica_count != None")
            if eplb_state.should_record_tensor is None:
                raise ValueError("EPLB requires should_record_tensor != None")
            if eplb_state.num_unpadded_tokens_tensors is None:
                raise ValueError("EPLB requires num_unpadded_tokens_tensors != None")
            if eplb_state.map_and_record is None:
                raise ValueError("EPLB requires map_and_record != None")

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Apply EPLB mapping to convert logical expert IDs to physical expert IDs."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_to_physical_map is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            assert eplb_state.num_unpadded_tokens_tensors is not None
            assert eplb_state.map_and_record is not None
            return eplb_state.map_and_record(
                topk_ids,
                eplb_state.logical_to_physical_map,
                eplb_state.logical_replica_count,
                eplb_state.expert_load_view,
                eplb_state.should_record_tensor,
                eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()],
            )
        return topk_ids

    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """Convert topk_ids to the desired dtype if needed."""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids

    @abstractmethod
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the actual routing logic.

        This method must be implemented by subclasses to provide the specific
        routing algorithm (e.g., grouped_topk, fused_topk, custom routing, etc.).

        Args:
            hidden_states: Input hidden states
            router_logits: Router logits for expert selection
            indices_type: Desired dtype for expert indices (may be None)

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        raise NotImplementedError

    def _select_experts(
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

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Calls _compute_routing() to get topk_weights and topk_ids
        3. Applies EPLB mapping if enabled
        4. Converts indices dtype if needed

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Step 2: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 3: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 4: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, topk_indices_dtype)

        return topk_weights, topk_ids
