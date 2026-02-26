# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)


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
        zero_expert_num: int,
        zero_expert_type: str,
        e_score_correction_bias: torch.Tensor | None = None,
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

        # Slice e_score_correction_bias to only include real experts
        # (not zero experts) for the base FusedMoE. The full bias will be
        # used temporarily in forward() for routing.
        if e_score_correction_bias is not None and "num_experts" in kwargs:
            num_real_experts = kwargs["num_experts"]
            user_bias = kwargs.get("e_score_correction_bias")

            if (
                user_bias is None
                or user_bias.shape[0] == e_score_correction_bias.shape[0]
            ):
                kwargs["e_score_correction_bias"] = e_score_correction_bias[
                    :num_real_experts
                ]

        # Create memoizing routing function BEFORE super().__init__() so it
        # gets passed to create_fused_moe_router, which will create a
        # CustomRoutingRouter that uses it. The closure captures `self` and
        # accesses memoization state at call time (not definition time).
        def custom_routing_function(hidden_states, gating_output, topk, renormalize):
            """Return memoized `topk_weights` and `topk_ids`."""
            if self._memoized_topk_weights is None or self._memoized_topk_ids is None:
                raise RuntimeError(
                    "ZeroExpertFusedMoE: routing results not memoized. "
                    "Call select_experts first to compute routing."
                )
            return self._memoized_topk_weights, self._memoized_topk_ids

        kwargs["custom_routing_function"] = custom_routing_function

        # FusedMoE no longer accepts zero_expert_num/zero_expert_type.
        # We handle zero experts ourselves in forward().
        super().__init__(**kwargs)
        # Store the actual zero_expert_num and zero_expert_type for our own use
        self._actual_zero_expert_num = zero_expert_num
        self._actual_zero_expert_type = zero_expert_type
        # Full e_score_correction_bias (includes zero experts)
        self._full_e_score_correction_bias = e_score_correction_bias

        # Expose zero_expert_num and zero_expert_type as attributes for
        # compatibility with quantization methods that check these attributes
        self.zero_expert_num = 0
        self.zero_expert_type = None

        # Memoization state for routing results
        self._memoized_topk_weights: torch.Tensor | None = None
        self._memoized_topk_ids: torch.Tensor | None = None

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route with full e_score_correction_bias (including zero experts).

        This bypasses self.router (which is a memoizing CustomRoutingRouter)
        to perform the actual routing computation with the full bias.
        """
        if self._full_e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self._full_e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
                scoring_func=self.scoring_func,
            )
        else:
            from vllm.model_executor.layers.fused_moe import fused_topk

            topk_weights, topk_ids = fused_topk(
                hidden_states, router_logits, self.top_k, self.renormalize
            )

        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids

    def _compute_zero_expert_result(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute zero expert results using pre-computed routing."""
        if (
            self._actual_zero_expert_num is None
            or self._actual_zero_expert_num <= 0
            or self._actual_zero_expert_type is None
        ):
            return None

        return zero_experts_compute_triton(
            expert_indices=topk_ids.clone(),
            expert_scales=topk_weights.clone(),
            num_experts=self.logical_num_experts,
            zero_expert_type=self._actual_zero_expert_type,
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
        # Compute routing with full logits (including zero experts) so that
        # zero experts can be properly identified in topk_ids.
        # This bypasses self.router (the memoizing CustomRoutingRouter) and
        # performs routing directly with the full e_score_correction_bias.
        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # Compute zero expert result if needed
        zero_expert_result = self._compute_zero_expert_result(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        # Memoize routing results for reuse in super().forward()
        self._memoized_topk_weights = topk_weights
        self._memoized_topk_ids = topk_ids

        # Slice router_logits for real experts only
        router_logits_sliced = router_logits[..., : self.logical_num_experts]

        # Compute real expert results (will reuse memoized routing via
        # custom_routing_function)
        # zero_expert_num is already 0, so FusedMoE won't handle zero experts
        fused_out = super().forward(
            hidden_states=hidden_states,
            router_logits=router_logits_sliced,
        )

        # Combine results
        # Both zero_expert_result and fused_out are computed from the same
        # hidden_states, so they should be on the same device.
        if zero_expert_result is not None:
            fused_out = fused_out + zero_expert_result

        # Clear memoization after use
        self._memoized_topk_weights = None
        self._memoized_topk_ids = None

        return fused_out
