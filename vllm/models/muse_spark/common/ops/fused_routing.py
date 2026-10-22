# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for MuseSpark1x's top-k router, one program per token."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _normalized_scores(
    gating_scores,
    expert_mask,
    NORMALIZATION_MODE: tl.constexpr,
):
    """Sigmoid (mode 0) or numerically stable softmax (mode 1)."""
    if NORMALIZATION_MODE == 0:
        return tl.sigmoid(gating_scores)
    else:
        max_score = tl.max(tl.where(expert_mask, gating_scores, -float("inf")), axis=0)
        exp_scores = tl.exp(gating_scores - max_score)
        exp_scores = tl.where(expert_mask, exp_scores, 0.0)
        return exp_scores / tl.sum(exp_scores, axis=0)


@triton.jit
def _select_one(
    scores_for_topk,
    normalized_scores,
    selected_mask,
    expert_mask,
    expert_offs,
):
    """One top-k step: argmax over unselected experts; returns the *unbiased* score."""
    masked_scores = tl.where(selected_mask, -float("inf"), scores_for_topk)
    masked_scores = tl.where(expert_mask, masked_scores, -float("inf"))

    max_idx = tl.argmax(masked_scores, axis=0)
    is_selected = expert_offs == max_idx
    selected_score = tl.sum(tl.where(is_selected, normalized_scores, 0.0), axis=0)

    return max_idx, selected_score, selected_mask | is_selected


@triton.jit
def _routing_kernel(
    gating_output_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    is_padding,
    is_padding_stride,
    num_tokens,
    num_experts,
    stride_gm,
    stride_gn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    router_output_multiplier,
    route_scale,
    division_eps,
    TOPK: tl.constexpr,
    TOPK_PADDED: tl.constexpr,
    NUM_EXPERTS_PADDED: tl.constexpr,
    NORMALIZATION_MODE: tl.constexpr,
    HAS_MULTIPLIER: tl.constexpr,
    HAS_EPS: tl.constexpr,
) -> None:
    """One program per token; constexpr ``TOPK`` keeps the selection in registers."""
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    expert_offs = tl.arange(0, NUM_EXPERTS_PADDED)
    expert_mask = expert_offs < num_experts

    gating_ptr = gating_output_ptr + token_id * stride_gm
    gating_scores = tl.load(
        gating_ptr + expert_offs * stride_gn, mask=expert_mask, other=-float("inf")
    )
    if HAS_MULTIPLIER:
        gating_scores = gating_scores * router_output_multiplier

    normalized_scores = _normalized_scores(
        gating_scores, expert_mask, NORMALIZATION_MODE
    )

    bias_values = tl.load(bias_ptr + expert_offs, mask=expert_mask, other=0.0)
    scores_for_topk = normalized_scores + bias_values

    slots = tl.arange(0, TOPK_PADDED)
    slot_mask = slots < TOPK
    selected_mask = tl.zeros([NUM_EXPERTS_PADDED], dtype=tl.int1)
    ids = tl.zeros([TOPK_PADDED], dtype=tl.int32)
    scores = tl.zeros([TOPK_PADDED], dtype=tl.float32)
    score_sum = 0.0

    for k in tl.static_range(TOPK):
        max_idx, selected_score, selected_mask = _select_one(
            scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
        )
        slot = slots == k
        ids = tl.where(slot, max_idx.to(tl.int32), ids)
        scores = tl.where(slot, selected_score, scores)
        score_sum = score_sum + selected_score

    if HAS_EPS:
        score_sum = score_sum + division_eps
    inv_sum_scaled = route_scale / score_sum
    if is_padding is not None:
        padding = tl.load(is_padding + token_id * is_padding_stride)
        ids = tl.where(padding, -1, ids)
        inv_sum_scaled = tl.where(padding, 0.0, inv_sum_scaled)

    tl.store(
        topk_ids_ptr + token_id * stride_im + slots * stride_ik, ids, mask=slot_mask
    )
    tl.store(
        topk_weights_ptr + token_id * stride_wm + slots * stride_wk,
        scores * inv_sum_scaled,
        mask=slot_mask,
    )


def _padded_num_experts(num_experts: int) -> int:
    """Next power of two >= ``num_experts``, at least 8."""
    return 1 << max(3, (num_experts - 1).bit_length())


def fused_musespark1x_routing(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    router_output_multiplier: float | None = None,
    router_score_normalization: str = "sigmoid",
    router_score_division_eps: float | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    *,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MuseSpark1x's router in one kernel.

    Passing both destinations keeps the call allocation-free (graph-capturable).
    """
    assert gating_output.dim() == 2, "gating_output must be 2D"
    assert bias.is_contiguous(), "bias must be contiguous"

    num_tokens, num_experts = gating_output.shape
    if is_padding is not None:
        assert is_padding.shape == (num_tokens,) and is_padding.dtype == torch.bool
    assert bias.shape[0] == num_experts, "bias size must match num_experts"

    if router_score_normalization == "sigmoid":
        normalization_mode = 0
    elif router_score_normalization == "softmax":
        normalization_mode = 1
    else:
        raise ValueError(
            f"Invalid router_score_normalization: {router_score_normalization}. "
            "Expected 'sigmoid' or 'softmax'."
        )

    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )
    assert topk_weights.shape[0] >= num_tokens and topk_weights.shape[1] >= topk, (
        "topk_weights buffer too small"
    )
    assert topk_ids.shape[0] >= num_tokens and topk_ids.shape[1] >= topk, (
        "topk_ids buffer too small"
    )
    weights_view = topk_weights[:num_tokens, :topk]
    ids_view = topk_ids[:num_tokens, :topk]

    _routing_kernel[(num_tokens,)](
        gating_output,
        bias if bias.dtype == torch.float32 else bias.float(),
        weights_view,
        ids_view,
        is_padding,
        is_padding.stride(0) if is_padding is not None else 0,
        num_tokens,
        num_experts,
        gating_output.stride(0),
        gating_output.stride(1),
        weights_view.stride(0),
        weights_view.stride(1),
        ids_view.stride(0),
        ids_view.stride(1),
        router_output_multiplier if router_output_multiplier is not None else 1.0,
        route_scale,
        router_score_division_eps if router_score_division_eps is not None else 0.0,
        TOPK=topk,
        TOPK_PADDED=triton.next_power_of_2(topk),
        NUM_EXPERTS_PADDED=_padded_num_experts(num_experts),
        NORMALIZATION_MODE=normalization_mode,
        HAS_MULTIPLIER=router_output_multiplier is not None,
        HAS_EPS=router_score_division_eps is not None,
        # Keep Muse's repeated 256-expert selections within one warp.
        num_warps=1 if num_experts == 256 or current_platform.is_rocm() else 4,
    )
    return weights_view, ids_view
