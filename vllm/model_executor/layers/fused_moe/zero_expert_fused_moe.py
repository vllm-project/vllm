# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.triton_utils import tl, triton


@triton.jit
def _compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


def zero_experts_compute_triton(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the contribution of zero experts.

    Args:
        expert_indices: Top-k expert indices selected by router.
        expert_scales: Corresponding router weights.
        num_experts: Number of real experts.
        zero_expert_type: Currently only "identity" is supported.
        hidden_states: Token hidden states prior to expert dispatch.
    """
    if zero_expert_type != "identity":
        raise ValueError(
            f"Unsupported zero_expert_type={zero_expert_type!r}. "
            "Currently only identity zero experts are supported."
        )

    zero_expert_mask = expert_indices < num_experts
    zero_expert_scales = expert_scales.clone()
    zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices[normal_expert_mask] = 0
    expert_scales[normal_expert_mask] = 0.0

    output = torch.zeros_like(hidden_states).to(hidden_states.device)
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)

    # Use cdiv to handle non-divisible hidden_dim
    grid = lambda meta: (num_tokens * triton.cdiv(hidden_dim, meta["BLOCK_SIZE"]),)
    _compute_identity_kernel[grid](
        top_k=expert_indices.size(-1),
        hidden_states_ptr=hidden_states,
        expert_scales_ptr=zero_expert_scales,
        num_tokens=num_tokens,
        output_ptr=output,
        hidden_dim=hidden_dim,
        scales_stride=zero_expert_scales.stride(0),
        BLOCK_SIZE=256,
    )

    return output


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
        if self.zero_expert_num <= 0 or self.zero_expert_type is None:
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
            # Align dimensions if needed
            final_dim = fused_out.size(-1)
            zero_dim = zero_expert_result.size(-1)

            if final_dim != zero_dim:
                if final_dim < zero_dim:
                    fused_out = torch.nn.functional.pad(
                        fused_out,
                        (0, zero_dim - final_dim),
                        mode="constant",
                        value=0.0,
                    )
                else:
                    zero_expert_result = torch.nn.functional.pad(
                        zero_expert_result,
                        (0, final_dim - zero_dim),
                        mode="constant",
                        value=0.0,
                    )

            # Verify alignment succeeded
            if fused_out.size(-1) != zero_expert_result.size(-1):
                raise RuntimeError(
                    f"[ZeroExpertFusedMoE] Shape mismatch after alignment: "
                    f"fused_out.shape={fused_out.shape} "
                    f"(last_dim={fused_out.size(-1)}), "
                    f"zero_expert_result.shape={zero_expert_result.shape} "
                    f"(last_dim={zero_expert_result.size(-1)})"
                )

            fused_out = fused_out + zero_expert_result

        # Clear memoization after use
        self._memoized_topk_weights = None
        self._memoized_topk_ids = None

        return fused_out
