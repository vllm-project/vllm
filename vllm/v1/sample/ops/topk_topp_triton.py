# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Combined Top-K and Top-P Triton kernels.

These kernels apply top-k filtering first, then top-p on the remaining values.
This is more efficient than sorting the entire vocabulary.

Algorithm:
1. Find k-th largest logit using binary search → top-k threshold
2. Mask logits below threshold, compute softmax (only k values contribute)
3. Find probability threshold for top-p using binary search
4. Apply final mask

Complexity: O(vocab_size * (k_iters + p_iters)) where iters ≈ 16-20
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _topk_topp_kernel(
    # Input/output
    logits_ptr,
    # Parameters per row
    k_ptr,
    p_ptr,
    # Dimensions
    logits_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    # Mask value
    mask_value: tl.constexpr,
    # Algorithm parameters
    BLOCK_SIZE: tl.constexpr,
    K_ITERS: tl.constexpr,
    P_ITERS: tl.constexpr,
    # Feature flags (when False, use default values instead of loading)
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
):
    """
    Combined top-k and top-p masking kernel.

    Applies top-k first (by logit value), then top-p (by probability).
    Optimized to skip softmax computation when p >= 1.0.
    """
    row_idx = tl.program_id(0)

    if TOPK_ENABLED:
        k = tl.load(k_ptr + row_idx)
        apply_topk = k < vocab_size
    else:
        # Default: keep all (no top-k filtering)
        k = vocab_size
        apply_topk = False

    if TOPP_ENABLED:
        p = tl.load(p_ptr + row_idx)
        apply_topp = p < 1.0
    else:
        # Default: keep all (no top-p filtering)
        p = 1.0
        apply_topp = False

    # Early exit if nothing to do
    if (not apply_topk) and (not apply_topp):
        return

    row_ptr = logits_ptr + row_idx * logits_stride

    # =========================================================================
    # Phase 1: Find top-k threshold using binary search on logits
    # OPTIMIZATION: Fuse min/max finding with first binary search iteration
    # by counting values > 0 during min/max pass (saves 1 memory pass)
    # =========================================================================

    topk_threshold = float("-inf")

    if apply_topk:
        # Fused pass: find min/max AND count values > 0 (first binary search step)
        max_logit = float("-inf")
        min_logit = float("inf")
        count_above_zero = tl.zeros([1], dtype=tl.int32)

        for i in range(0, vocab_size, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
            max_logit = tl.maximum(max_logit, tl.max(vals))
            vals_min = tl.where(mask, vals, float("inf"))
            min_logit = tl.minimum(min_logit, tl.min(vals_min))
            # Count values > 0 (fused first binary search iteration)
            count_above_zero += tl.sum((vals > 0.0).to(tl.int32))

        # Use count_above_zero to set initial bounds (equivalent to first iteration)
        # If count_above_zero >= k, the k-th largest is > 0, so raise lo to 0
        # Otherwise, the k-th largest is <= 0, so lower hi to 0
        if tl.sum(count_above_zero) >= k:
            lo = 0.0
            hi = max_logit
        else:
            lo = min_logit
            hi = 0.0

        # Continue with remaining K_ITERS-1 binary search iterations
        for _ in range(K_ITERS - 1):
            mid = (lo + hi) * 0.5
            count_gt = tl.zeros([1], dtype=tl.int32)
            for i in range(0, vocab_size, BLOCK_SIZE):
                offs = i + tl.arange(0, BLOCK_SIZE)
                mask = offs < vocab_size
                vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
                count_gt += tl.sum((vals > mid).to(tl.int32))
            if tl.sum(count_gt) >= k:
                lo = mid
            else:
                hi = mid

        # Refine to exact k-th largest value.
        # After binary search: lo < k-th value <= hi (approximately).
        # Find the actual logit values at these boundaries.
        count_gt_lo = tl.zeros([1], dtype=tl.int32)
        min_above_lo = float("inf")
        max_at_or_below_hi = float("-inf")
        for i in range(0, vocab_size, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
            count_gt_lo += tl.sum((vals > lo).to(tl.int32))
            vals_above_lo = tl.where(vals > lo, vals, float("inf"))
            min_above_lo = tl.minimum(min_above_lo, tl.min(vals_above_lo))
            vals_at_or_below_hi = tl.where(vals <= hi, vals, float("-inf"))
            max_at_or_below_hi = tl.maximum(
                max_at_or_below_hi, tl.max(vals_at_or_below_hi)
            )

        if tl.sum(count_gt_lo) == k:
            topk_threshold = min_above_lo
        else:
            topk_threshold = max_at_or_below_hi

    # =========================================================================
    # If no top-p, apply top-k mask and return early
    # =========================================================================

    if not apply_topp:
        for i in range(0, vocab_size, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
            result = tl.where(vals >= topk_threshold, vals, mask_value)
            tl.store(row_ptr + offs, result, mask=mask)
        return

    # =========================================================================
    # Phase 2: Compute softmax using online softmax (single pass)
    # =========================================================================
    # Online softmax computes max and exp_sum in one pass by rescaling
    # the running sum when a new max is found.
    #
    # Key insight: We need to handle the case where softmax_max is -inf
    # (no valid values seen yet). In this case, -inf - (-inf) = nan,
    # so we must skip blocks with no valid values.

    softmax_max = float("-inf")
    exp_sum = 0.0

    for i in range(0, vocab_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
        # Apply top-k mask
        vals = tl.where(vals >= topk_threshold, vals, float("-inf"))

        # Find block max
        block_max = tl.max(vals)

        # Skip blocks with no valid values (all -inf)
        # This avoids nan from -inf - (-inf)
        if block_max > float("-inf"):
            # Update running max and rescale sum if needed
            new_max = tl.maximum(softmax_max, block_max)

            # Rescale previous sum: sum * exp(old_max - new_max)
            # When softmax_max is -inf (first valid block), exp(-inf - finite) = 0,
            # which is correct since exp_sum starts at 0.
            exp_sum = exp_sum * tl.exp(softmax_max - new_max)
            softmax_max = new_max

            # Add current block's contribution (normalized by new max)
            exp_sum += tl.sum(tl.exp(vals - softmax_max))

    log_exp_sum = tl.log(exp_sum)

    # =========================================================================
    # Phase 3: Find top-p threshold using binary search on probabilities
    # OPTIMIZATION: Fuse min/max finding with first binary search iteration
    # by computing prob mass > 0.5 during min/max pass (saves 1 memory pass)
    # =========================================================================

    # Fused pass: find min/max log-probs AND sum probs > 0.5 (first iteration)
    max_log_prob = float("-inf")
    min_log_prob = float("inf")
    log_half = -0.6931471805599453  # log(0.5)
    prob_sum_above_half = 0.0

    for i in range(0, vocab_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))

        # Only consider top-k values
        is_topk = vals >= topk_threshold

        # log_prob = logit - softmax_max - log(exp_sum)
        log_probs = vals - softmax_max - log_exp_sum

        log_probs_masked = tl.where(is_topk, log_probs, float("-inf"))
        max_log_prob = tl.maximum(max_log_prob, tl.max(log_probs_masked))

        log_probs_for_min = tl.where(is_topk & mask, log_probs, float("inf"))
        min_log_prob = tl.minimum(min_log_prob, tl.min(log_probs_for_min))

        # Sum probability mass above 0.5 (fused first binary search iteration)
        probs = tl.exp(log_probs)
        above_half = (log_probs > log_half) & is_topk
        prob_sum_above_half += tl.sum(tl.where(above_half, probs, 0.0))

    # Use prob_sum_above_half to set initial bounds (equivalent to first iteration)
    if prob_sum_above_half >= p:
        lo_lp = log_half
        hi_lp = max_log_prob
    else:
        lo_lp = min_log_prob
        hi_lp = log_half

    # Continue with remaining P_ITERS-1 binary search iterations
    for _ in range(P_ITERS - 1):
        mid_lp = (lo_lp + hi_lp) * 0.5

        # Sum probabilities strictly > mid_lp
        prob_sum_gt = 0.0
        for i in range(0, vocab_size, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))

            is_topk = vals >= topk_threshold
            log_probs = vals - softmax_max - log_exp_sum
            probs = tl.exp(log_probs)

            # Only sum probs that are strictly > threshold and in top-k
            above_threshold = (log_probs > mid_lp) & is_topk
            prob_sum_gt += tl.sum(tl.where(above_threshold, probs, 0.0))

        # If sum of probs strictly above mid >= p, raise threshold
        if prob_sum_gt >= p:
            lo_lp = mid_lp
        else:
            hi_lp = mid_lp

    # Refine to exact threshold using combined approach (same as top-k).
    # After binary search: prob_sum(> lo_lp) >= p, prob_sum(> hi_lp) < p.
    # Count how many distinct log-probs are > lo_lp to determine which refinement.
    count_gt_lo_lp = tl.zeros([1], dtype=tl.int32)
    min_lp_above_lo = float("inf")
    max_lp_at_or_below_hi = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))

        is_topk = vals >= topk_threshold
        log_probs = vals - softmax_max - log_exp_sum

        above_lo = is_topk & (log_probs > lo_lp)
        count_gt_lo_lp += tl.sum(above_lo.to(tl.int32))

        lp_above_lo = tl.where(above_lo, log_probs, float("inf"))
        min_lp_above_lo = tl.minimum(min_lp_above_lo, tl.min(lp_above_lo))

        at_or_below_hi = is_topk & (log_probs <= hi_lp)
        lp_at_or_below_hi = tl.where(at_or_below_hi, log_probs, float("-inf"))
        max_lp_at_or_below_hi = tl.maximum(
            max_lp_at_or_below_hi, tl.max(lp_at_or_below_hi)
        )

    # For top-p, use min if there are values > lo, otherwise use max.
    # This handles edge cases where lo/hi converge to the same side.
    if tl.sum(count_gt_lo_lp) > 0 and min_lp_above_lo < float("inf"):
        topp_log_threshold = min_lp_above_lo
    else:
        topp_log_threshold = max_lp_at_or_below_hi

    # =========================================================================
    # Phase 4: Apply combined mask
    # =========================================================================

    for i in range(0, vocab_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))

        # Apply top-k mask
        keep = vals >= topk_threshold

        # Apply top-p mask
        log_probs = vals - softmax_max - log_exp_sum
        keep = keep & (log_probs >= topp_log_threshold)

        result = tl.where(keep, vals, mask_value)
        tl.store(row_ptr + offs, result, mask=mask)


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Apply combined top-k and top-p masking using Triton.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).

    Args:
        logits: [n, vocab_size] float32 tensor, modified in-place
        k: [n] int32 tensor of top-k values per row, or None to disable top-k
        p: [n] float32 tensor of top-p values per row (0 to 1),
            or None to disable top-p
        mask_value: Value for masked positions (default: -inf)

    Returns:
        The logits tensor (modified in-place)
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert logits.is_cuda

    n, vocab_size = logits.shape

    topk_enabled = k is not None
    topp_enabled = p is not None

    if n == 0 or not (topk_enabled or topp_enabled):
        return logits

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == n and k.is_cuda
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # Dummy pointer (won't be read)

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == n and p.is_cuda
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # Dummy pointer (won't be read)

    BLOCK_SIZE = 1024
    # K_ITERS must be large enough to distinguish adjacent logit values.
    # With randn logits (range ~8), 20 iterations gives precision ~8/2^19 ≈ 1.5e-5
    K_ITERS = 18
    P_ITERS = 14

    _topk_topp_kernel[(n,)](
        logits,
        k_ptr,
        p_ptr,
        logits_stride=logits.stride(0),
        vocab_size=vocab_size,
        mask_value=mask_value,
        BLOCK_SIZE=BLOCK_SIZE,
        K_ITERS=K_ITERS,
        P_ITERS=P_ITERS,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
    )

    return logits
