# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class ZeroExpertFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of zero experts.
    Zero experts perform identity operations (scaled pass-through) instead
    of full MLP computations.

    This class uses memoization to avoid redundant routing computation:
    routing is computed once and reused for both zero expert computation
    and the main FusedMoE forward pass.
    """

    def __init__(
        self,
        zero_expert_num: int = 0,
        zero_expert_type: str | None = None,
        router: nn.Module | None = None,
        **kwargs,
    ):
        # ZeroExpertFusedMoE manages its own custom_routing_function for memoization
        assert (
            "custom_routing_function" not in kwargs
            or kwargs.get("custom_routing_function") is None
        ), (
            "ZeroExpertFusedMoE does not support external custom_routing_function. "
            "It manages its own for routing memoization."
        )

        # Automatically slice router's e_score_correction_bias to only include
        # real experts (not zero_experts) for the base FusedMoE.
        # The full bias will be used temporarily in forward() for routing.
        if (
            router is not None
            and hasattr(router, "e_score_correction_bias")
            and "num_experts" in kwargs
        ):
            num_real_experts = kwargs["num_experts"]
            router_bias = router.e_score_correction_bias
            user_bias = kwargs.get("e_score_correction_bias")

            # Use router's bias if:
            # 1. User didn't provide bias, or
            # 2. User provided full bias (same size as router)
            if user_bias is None or user_bias.shape[0] == router_bias.shape[0]:
                kwargs["e_score_correction_bias"] = router_bias[:num_real_experts]

        super().__init__(**kwargs)
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self._router = router  # Full router (includes zero experts)

        # Memoization state for routing results
        self._memoized_topk_weights: torch.Tensor | None = None
        self._memoized_topk_ids: torch.Tensor | None = None

        # Create custom_routing_function to reuse memoized routing results
        def custom_routing_function(hidden_states, gating_output, topk, renormalize):
            """Return memoized `topk_weights` and `topk_ids`."""
            if self._memoized_topk_weights is None or self._memoized_topk_ids is None:
                raise RuntimeError(
                    "ZeroExpertFusedMoE: routing results not memoized. "
                    "Call select_experts first to compute routing."
                )
            return self._memoized_topk_weights, self._memoized_topk_ids

        self.custom_routing_function = custom_routing_function

    def _compute_zero_expert_result(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute zero expert results using pre-computed routing."""
        if (
            self.zero_expert_num is None
            or self.zero_expert_num <= 0
            or self.zero_expert_type is None
        ):
            return None

        return zero_experts_compute_triton(
            expert_indices=topk_ids.clone(),
            expert_scales=topk_weights.clone(),
            num_experts=self.logical_num_experts,
            zero_expert_type=self.zero_expert_type,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,  # Full logits including zero experts
    ) -> torch.Tensor:
        """
        Forward pass with zero expert support and routing memoization.

        Args:
            hidden_states: Input hidden states
            router_logits: Full router logits (including zero experts)

        Returns:
            Combined output from real experts and zero experts
        """
        # Temporarily override e_score_correction_bias to use full bias
        # (including zero experts) for routing computation
        original_bias = self.e_score_correction_bias
        if self._router is not None:
            self.e_score_correction_bias = self._router.e_score_correction_bias

        # Compute routing once (using full logits to include zero experts)
        # This ensures zero experts can be properly identified in topk_ids
        # select_experts now returns (topk_weights, topk_ids, zero_expert_result)
        topk_weights, topk_ids, zero_expert_result = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,  # Full logits (includes zero experts)
        )

        # Restore original bias
        self.e_score_correction_bias = original_bias

        # Memoize routing results for reuse in super().forward()
        self._memoized_topk_weights = topk_weights
        self._memoized_topk_ids = topk_ids

        # Slice router_logits for real experts only
        router_logits_sliced = router_logits[..., : self.logical_num_experts]

        # Compute real expert results (will reuse memoized routing via
        # custom_routing_function)
        fused_out = super().forward(
            hidden_states=hidden_states,
            router_logits=router_logits_sliced,
        )

        # Combine results
        if zero_expert_result is not None:
            fused_out = fused_out + zero_expert_result

        # Clear memoization after use
        self._memoized_topk_weights = None
        self._memoized_topk_ids = None

        return fused_out
