# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

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
        zero_expert_num: int,
        zero_expert_type: str,
        router: nn.Module,
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
        if hasattr(router, "e_score_correction_bias") and "num_experts" in kwargs:
            num_real_experts = kwargs["num_experts"]
            router_bias = router.e_score_correction_bias
            user_bias = kwargs.get("e_score_correction_bias")

            # Use router's bias if:
            # 1. User didn't provide bias, or
            # 2. User provided full bias (same size as router)
            if user_bias is None or user_bias.shape[0] == router_bias.shape[0]:
                kwargs["e_score_correction_bias"] = router_bias[:num_real_experts]

        # FusedMoE no longer accepts zero_expert_num/zero_expert_type.
        # We handle zero experts ourselves in forward().
        super().__init__(**kwargs)
        # Store the actual zero_expert_num and zero_expert_type for our own use
        self._actual_zero_expert_num = zero_expert_num
        self._actual_zero_expert_type = zero_expert_type
        self._router = router  # Full router (includes zero experts)

        # Expose zero_expert_num and zero_expert_type as attributes for
        # compatibility with quantization methods that check these attributes
        self.zero_expert_num = 0
        self.zero_expert_type = None

        # Memoization state for routing results
        self._memoized_topk_weights: torch.Tensor | None = None
        self._memoized_topk_ids: torch.Tensor | None = None
        self._memoized_offset: int | None = None

        # Create custom_routing_function to reuse memoized routing results
        def custom_routing_function(hidden_states, gating_output, topk, renormalize):
            """Return memoized `topk_weights` and `topk_ids`."""
            return self._select_experts_from_memo(hidden_states, gating_output)

        self.custom_routing_function = custom_routing_function

    @contextmanager
    def _temporarily_set_attrs(self, target: object, **attrs):
        """
        Temporarily set attributes on a target object and restore them.

        This bypasses nn.Module.__setattr__ to avoid Dynamo tracing issues.
        When PyTorch Dynamo traces the forward pass, it cannot handle
        nn.Module.__setattr__ calls (which include parameter registration logic),
        resulting in "Unsupported" errors. Using object.__setattr__ directly
        sets the attribute without triggering nn.Module's custom __setattr__,
        allowing Dynamo to trace the code successfully.
        """
        originals = {key: getattr(target, key) for key in attrs}
        try:
            for key, value in attrs.items():
                object.__setattr__(target, key, value)
            yield
        finally:
            for key, value in originals.items():
                object.__setattr__(target, key, value)

    def _select_experts_from_memo(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,  # Unused, kept for signature parity
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._memoized_topk_weights is None or self._memoized_topk_ids is None:
            raise RuntimeError(
                "ZeroExpertFusedMoE: routing results not memoized. "
                "Call select_experts first to compute routing."
            )
        if self._memoized_offset is None:
            return self._memoized_topk_weights, self._memoized_topk_ids

        start = self._memoized_offset
        end = start + hidden_states.size(0)
        total = self._memoized_topk_weights.size(0)
        if end > total:
            raise RuntimeError("ZeroExpertFusedMoE: memoized routing size mismatch.")
        self._memoized_offset = end
        return (
            self._memoized_topk_weights[start:end],
            self._memoized_topk_ids[start:end],
        )

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
        # Prepare temporary attribute overrides for routing computation
        temp_router_attrs = {}
        if self._router is not None and hasattr(self.router, "e_score_correction_bias"):
            temp_router_attrs["e_score_correction_bias"] = (
                self._router.e_score_correction_bias
            )

        # Compute routing with temporary attributes
        # Pass full router_logits (including zero experts) so that zero experts
        # can be properly identified in topk_ids
        with self._temporarily_set_attrs(self.router, **temp_router_attrs):
            topk_weights_full, topk_ids_full = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,  # Full logits (includes zero experts)
            )

        # Compute zero expert result if needed
        zero_expert_result = self._compute_zero_expert_result(
            hidden_states=hidden_states,
            topk_weights=topk_weights_full,
            topk_ids=topk_ids_full,
        )

        # Filter zero experts for real expert computation
        topk_weights = topk_weights_full
        topk_ids = topk_ids_full
        actual_zero_expert_num = self._actual_zero_expert_num
        if actual_zero_expert_num is not None and actual_zero_expert_num > 0:
            zero_mask = topk_ids_full >= self.logical_num_experts
            topk_weights = topk_weights_full.masked_fill(zero_mask, 0.0)
            topk_ids = torch.where(
                zero_mask,
                torch.zeros_like(topk_ids_full),
                topk_ids_full,
            )

        # Memoize routing results for reuse in super().forward()
        self._memoized_topk_weights = topk_weights
        self._memoized_topk_ids = topk_ids
        self._memoized_offset = 0

        # Slice router_logits for real experts only
        router_logits_sliced = router_logits[..., : self.logical_num_experts]

        # Compute real expert results (will reuse memoized routing via
        # custom_routing_function)
        # zero_expert_num is already 0, so FusedMoE won't handle zero experts
        with self._temporarily_set_attrs(
            self.router,
            select_experts=self._select_experts_from_memo,
        ):
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
        self._memoized_offset = None

        return fused_out
