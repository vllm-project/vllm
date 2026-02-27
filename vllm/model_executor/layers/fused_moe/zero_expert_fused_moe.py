# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


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

        # Pass the full e_score_correction_bias (real + zero experts) and
        # zero_expert_type through to the router factory, which will create
        # a ZeroExpertRouter.
        num_real_experts = kwargs["num_experts"]
        if e_score_correction_bias is not None:
            kwargs["e_score_correction_bias"] = e_score_correction_bias
        kwargs["zero_expert_type"] = zero_expert_type

        super().__init__(**kwargs)

        # Fix self.e_score_correction_bias to only cover real experts,
        # for compatibility with monolithic kernels that read it directly.
        if e_score_correction_bias is not None:
            self.e_score_correction_bias = e_score_correction_bias[:num_real_experts]

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
