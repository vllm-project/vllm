# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)


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
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        # TODO(bnell): Once the MK is constructed at layer init time, we
        # can make this a plain value instead of a callback.
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        """
        Note: the indices dtype might not be available at router construction
        time, so we need to supply a callback to get it at runtime.  This is
        because the indices type is supplied by modular kernels which are
        created after MoE layer/router construction.
        """
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.eplb_state = eplb_state
        self.enable_eplb = enable_eplb
        self.indices_type_getter = indices_type_getter
        self.capture_fn: Callable[[torch.Tensor], None] | None = None

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """Set a capture callback for logical routed expert IDs."""
        self.capture_fn = capture_fn

    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.enable_eplb:
            if self.eplb_state.expert_load_view is None:
                raise ValueError("enable_eplb=True requires expert_load_view != None")
            if self.eplb_state.logical_to_physical_map is None:
                raise ValueError(
                    "enable_eplb=True requires logical_to_physical_map != None"
                )
            if self.eplb_state.logical_replica_count is None:
                raise ValueError(
                    "enable_eplb=True requires logical_replica_count != None"
                )

    def _get_indices_type(self) -> torch.dtype | None:
        """Get the desired indices dtype from the getter function."""
        return (
            self.indices_type_getter() if self.indices_type_getter is not None else None
        )

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

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Gets indices type
        3. Calls _compute_routing() to get topk_weights and topk_ids
        4. Applies EPLB mapping if enabled
        5. Converts indices dtype if needed

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

        # Step 2: Get indices type.
        indices_type = self._get_indices_type()

        # Step 3: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 4: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, indices_type)

        return topk_weights, topk_ids
