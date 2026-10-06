# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EMA load-penalized logit adjustment for MoE routing.

Rather than correcting overflow after top-k selection, this module adjusts
router logits before top-k so the router naturally avoids overloaded EP
ranks.  A per-rank exponential moving average of token-slot counts drives a
soft penalty subtracted from expert logits on heavy ranks.

Usage::

    VLLM_MOE_BALANCE_ROUTING_EMA=1 vllm serve ...

Optional tuning::

    VLLM_MOE_BALANCE_ROUTING_EMA_ALPHA=0.1   # EMA decay (higher = faster response)
    VLLM_MOE_BALANCE_ROUTING_EMA_LAMBDA=2.0  # penalty strength
"""

import torch


def build_expert_to_rank(
    num_experts: int,
    ep_size: int,
    placement: str,
    device: torch.device,
) -> torch.Tensor:
    """Return a ``[num_experts]`` int64 tensor mapping expert id to EP rank.

    Handles non-multiples of ``ep_size`` correctly for both placement
    strategies, matching the logic in ``determine_expert_map``.

    Args:
        num_experts: Total number of logical routed experts.
        ep_size: Number of EP ranks.
        placement: ``"linear"`` (contiguous blocks) or ``"round_robin"``
            (interleaved).
        device: Target device for the returned tensor.

    Returns:
        ``expert_to_rank[e]`` = the EP rank that holds expert ``e``.

    """
    if placement == "round_robin":
        return torch.arange(num_experts, dtype=torch.int64, device=device) % ep_size

    # "linear": rank r holds a contiguous block starting at
    # r*base + min(r, remainder).  Use cumsum + searchsorted to handle
    # non-multiples without per-expert Python arithmetic.
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    rank_sizes = torch.full((ep_size,), base, dtype=torch.int64, device=device)
    rank_sizes[:remainder] += 1
    # rank_boundaries[r] = exclusive end index of rank r's expert block.
    rank_boundaries = rank_sizes.cumsum(0)
    experts = torch.arange(num_experts, dtype=torch.int64, device=device)
    # right=True: rank_boundaries[i] > expert_id → first rank whose block
    # ends strictly after this expert, i.e. the rank that owns it.
    return torch.searchsorted(rank_boundaries.contiguous(), experts, right=True)


def adjust_logits_ema(
    router_logits: torch.Tensor,
    ema_load: torch.Tensor,
    prev_topk_ids: torch.Tensor | None,
    expert_to_rank: torch.Tensor,
    num_experts: int,
    alpha: float = 0.1,
    lambda_: float = 2.0,
) -> torch.Tensor:
    """Return load-penalty-adjusted logits and update ema_load in-place.

    Penalty for expert e:
        λ * max(0, ema_load[rank(e)] / fair_share - 1)

    where fair_share = ema_load.mean().  Only over-loaded ranks are
    penalised; under-loaded ranks receive no bonus, so the total logit
    ordering is only locally perturbed near overloaded ranks.

    The EMA is updated from prev_topk_ids (the previous step's routing)
    before the penalty is computed, so the adjustment always reflects
    up-to-date load history without any CPU-GPU synchronisation.

    Shared-expert columns (IDs >= num_experts) are excluded from load
    counting via the is_routed mask, matching the behaviour of
    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS.

    router_logits may be wider than num_experts (AITER shared-expert
    fusion appends a shared-gate column; EPLB adds redundant-expert
    columns).  The penalty is built over the logical experts only and
    zero-padded to router_logits.shape[-1].

    No CPU-GPU synchronisation is performed.  The function returns
    router_logits unchanged during torch.compile tracing and CUDA-graph
    capture.

    Args:
        router_logits: Raw gate output ``[T, logit_width]`` where
            ``logit_width >= num_experts``.
        ema_load: Per-rank EMA of token-slot counts ``[ep_size]``, float32.
            Updated in-place from ``prev_topk_ids`` before the penalty is
            applied.
        prev_topk_ids: Top-k expert indices ``[T, K]`` or
            ``[T, K + n_shared]`` from the previous forward pass, used to
            update ``ema_load``.  ``None`` on the very first call (EMA stays
            at its initial value of zero — no penalty applied).
        expert_to_rank: ``[num_experts]`` int64 tensor mapping each logical
            expert id to its EP rank, pre-built by ``build_expert_to_rank``
            for the correct placement strategy.
        num_experts: Total number of logical routed experts.
        alpha: EMA smoothing factor.  Higher values respond faster to load
            changes but are noisier.  Range ``(0, 1]``.
        lambda_: Penalty strength.  Higher values enforce tighter balance at
            the cost of more logit distortion.

    Returns:
        Adjusted logits with the same shape, dtype and device as
        ``router_logits``.

    """
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return router_logits

    ep_size = ema_load.shape[0]
    if ep_size <= 1:
        return router_logits

    device = router_logits.device

    # --- Update EMA from previous step's routing ---
    if prev_topk_ids is not None:
        ids = prev_topk_ids.to(torch.int64)
        is_routed = (ids >= 0) & (ids < num_experts)
        safe_ids = ids.clamp(0, num_experts - 1)
        # Map each token-expert slot to its EP rank using the pre-built table.
        flat_ranks = expert_to_rank[safe_ids.flatten()]
        valid_flat = is_routed.flatten().to(torch.int32)
        rank_counts = torch.zeros(ep_size, dtype=torch.int32, device=device)
        rank_counts.scatter_add_(0, flat_ranks.to(torch.int64), valid_flat)
        ema_load.mul_(1.0 - alpha).add_(rank_counts.float() * alpha)

    # --- Compute per-expert penalty from current EMA ---
    # fair_share = average load per rank; clamp to 1 to avoid div-by-zero
    # on the very first step when ema_load is still zero.
    fair_share = ema_load.mean().clamp(min=1.0)              # scalar tensor
    excess = (ema_load / fair_share - 1.0).clamp(min=0.0)   # [ep_size]

    penalty = excess[expert_to_rank] * lambda_               # [num_experts]

    # Zero-pad to the full logit width when extra columns are present
    # (AITER shared-expert gate or EPLB redundant experts).
    logit_width = router_logits.shape[-1]
    if logit_width > num_experts:
        penalty_full = penalty.new_zeros(logit_width)
        penalty_full[:num_experts] = penalty
        penalty = penalty_full

    return router_logits - penalty.to(router_logits.dtype).unsqueeze(0)
