# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)


class ZeroExpertFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of zero experts.
    Zero experts perform identity operations (scaled pass-through) instead
    of full MLP computations.

    Uses a ZeroExpertRouter as the layer's main router. The router handles
    routing over all experts (real + zero) with the full e_score_correction_bias,
    computes zero expert contributions as a side effect, and remaps zero expert
    IDs so downstream MoE computation only processes real experts.
    """

    def __init__(
        self,
        zero_expert_num: int,
        zero_expert_type: str,
        e_score_correction_bias: torch.Tensor | None = None,
        **kwargs,
    ):
        assert (
            "custom_routing_function" not in kwargs
            or kwargs.get("custom_routing_function") is None
        ), (
            "ZeroExpertFusedMoE does not support external custom_routing_function. "
            "Routing is handled by ZeroExpertRouter."
        )

        assert "router" not in kwargs or kwargs.get("router") is None, (
            "ZeroExpertFusedMoE creates its own ZeroExpertRouter. Do not pass a router."
        )

        # Remove custom_routing_function from kwargs if present
        kwargs.pop("custom_routing_function", None)
        kwargs.pop("router", None)

        # Slice e_score_correction_bias to only include real experts
        # for the base FusedMoE router factory (which we'll replace anyway).
        num_real_experts = kwargs["num_experts"]
        if e_score_correction_bias is not None:
            user_bias = kwargs.get("e_score_correction_bias")
            if (
                user_bias is None
                or user_bias.shape[0] == e_score_correction_bias.shape[0]
            ):
                kwargs["e_score_correction_bias"] = e_score_correction_bias[
                    :num_real_experts
                ]

        super().__init__(**kwargs)

        # Replace the factory-created router with our ZeroExpertRouter.
        # Uses self.eplb_state created by super().__init__() so EPLB state
        # is shared between the layer and the router.
        self.router = ZeroExpertRouter(
            top_k=self.top_k,
            global_num_experts=self.global_num_experts,
            eplb_state=self.eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            num_logical_experts=self.logical_num_experts,
            zero_expert_type=zero_expert_type,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            enable_eplb=self.enable_eplb,
            indices_type_getter=lambda: self.quant_method.topk_indices_dtype,
        )

        # Update routing_method_type to match the new router
        self.routing_method_type = self.router.routing_method_type

        # Re-init runner with the new router
        self.runner = self._init_runner()

        # Expose zero_expert_num=0 and zero_expert_type=None for
        # compatibility with quantization methods that check these attributes.
        # The actual zero expert handling is done by ZeroExpertRouter.
        self.zero_expert_num = 0
        self.zero_expert_type = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with zero expert support.

        The ZeroExpertRouter handles routing with full logits (including zero
        experts), computes zero expert contributions internally, and returns
        masked topk_ids suitable for real expert MoE computation.

        Args:
            hidden_states: Input hidden states
            router_logits: Full router logits (including zero experts)

        Returns:
            Combined output from real experts and zero experts
        """
        # The router handles full logits internally: routes over all experts
        # (real + zero), computes zero expert output, masks zero expert IDs.
        fused_out = super().forward(hidden_states, router_logits)

        # Retrieve zero expert output computed during routing
        zero_expert_output = self.router.zero_expert_output
        if zero_expert_output is not None:
            fused_out = fused_out + zero_expert_output

        return fused_out
