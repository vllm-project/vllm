# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk,
)
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform

#
# TopK softmax (e.g. Mixtral)
#


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    """Custom op for TopK softmax."""
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Dispatch TopK softmax function based on the platform and flags."""
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Out-of place wrapper for Fused TopK softmax."""
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    topk_func = dispatch_topk_func(use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled())
    topk_weights, topk_ids = topk_func(
        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
    )

    return topk_weights, topk_ids, token_expert_indices


#
# Biased-TopK softmax (e.g. Llama-4)
#


def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Topk-softmax with expert score correction bias."""
    n_routed_experts = gating_output.shape[-1]
    scores = gating_output.softmax(dim=-1)
    scores_for_choice = scores.view(
        -1, n_routed_experts
    ) + e_score_correction_bias.unsqueeze(0)

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_is_batch_invariant()
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[1]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)


#
# Grouped-TopK softmax (e.g. DeepSeekV2)
#


def fused_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        # Fully fused kernel path for sigmoid
        topk_values, topk_indices = ops.grouped_topk(
            gating_output,  # raw logits
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            1,  # scoring_func=1 for sigmoid
        )
    elif scoring_func == "softmax":
        # Apply softmax in Python, then use fused kernel
        # TODO: Add support for softmax in kernel
        scores = torch.softmax(gating_output, dim=-1)
        topk_values, topk_indices = ops.grouped_topk(
            scores,  # pre-computed scores
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            0,  # scoring_func=0 (no activation, scores already computed)
        )
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # Fused kernel outputs float32 values and int32 indices directly
    return topk_values, topk_indices


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and current_platform.is_cuda()
        and num_expert_group <= 32
        and topk <= 32
        and e_score_correction_bias is not None
    ):
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_is_batch_invariant()
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# --8<-- [start:grouped_topk]
@CustomOp.register("grouped_topk")
class GroupedTopk(CustomOp):
    """GroupedTopk used by the Deepseek-V2 and Deepseek-V3 model."""

    # --8<-- [end:grouped_topk]

    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        num_fused_shared_experts: int = 0,
    ) -> None:
        super().__init__()
        self.native_impl = grouped_topk
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.native_impl(
            hidden_states,
            gating_output,
            self.topk,
            self.renormalize,
            self.num_expert_group,
            self.topk_group,
            self.scoring_func,
            self.routed_scaling_factor,
            e_score_correction_bias,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(
            hidden_states, gating_output, e_score_correction_bias
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rocm_aiter_ops.is_fused_moe_enabled():
            if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                assert self.num_fused_shared_experts == 0
            return rocm_aiter_grouped_topk(
                hidden_states,
                gating_output,
                self.topk,
                self.renormalize,
                self.num_expert_group,
                self.topk_group,
                self.scoring_func,
                self.routed_scaling_factor,
                e_score_correction_bias,
                self.num_fused_shared_experts,
            )
        else:
            return self.forward_native(
                hidden_states, gating_output, e_score_correction_bias
            )
