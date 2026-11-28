# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

# The feature is a log-odds, log(q / (1 - q)), which a row carrying all of its
# mass on one token sends to infinity. Clamp it symmetrically.
_MAX_LOG_ODDS = 40.0

# Cold-start coefficients: the median slope and intercept fitted online across
# DeepSeek-V4-Flash, Muse-Glimmer-30B, MiMo-V2.5-Pro, Inkling and Kimi-K2.5.
# Log-loss is flat near the optimum, so one starting point serves all of them
# and a real fit replaces it within a refit or two.
_INIT_SLOPE = 0.5
_INIT_BIAS = -0.2


@triton.jit(do_not_specialize=["num_reqs"])
def _accumulate_kernel(
    info_ptr,
    grad_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    features_ptr,
    features_stride,
    predecessor_surprise_ptr,
    pred_ptr,
    pred_stride,
    counts_ptr,
    surprise_counts_ptr,
    num_reqs,
    BLOCK_R: tl.constexpr,
):
    step = tl.program_id(0).to(tl.int64)
    req_block = tl.arange(0, BLOCK_R)
    req_mask = req_block < num_reqs
    req_state_block = tl.load(idx_mapping_ptr + req_block, mask=req_mask, other=0).to(
        tl.int64
    )

    req_mask = req_mask & (req_state_block >= 0)

    num_sampled = tl.load(num_sampled_ptr + req_block, mask=req_mask, other=0).to(
        tl.int64
    )
    # Subtract bonus token.
    num_accepted = tl.maximum(num_sampled - 1, 0)
    num_rejected = tl.load(num_rejected_ptr + req_block, mask=req_mask, other=0).to(
        tl.int64
    )
    num_admitted = num_accepted + num_rejected

    feature = tl.load(
        features_ptr + req_state_block * features_stride + step,
        mask=req_mask,
        other=0.0,
    ).to(tl.float32)
    pred = tl.load(
        pred_ptr + req_state_block * pred_stride + step,
        mask=req_mask,
        other=0.0,
    ).to(tl.float32)

    # The accepted tokens and the first rejected draft token (if one exists) are
    # "observed" and given labels.
    observed = req_mask & (step <= num_accepted) & (step < num_admitted)
    label = tl.where(step < num_accepted, 1.0, 0.0)
    w = tl.where(observed, pred * (1.0 - pred), 0.0)
    resid = tl.where(observed, label - pred, 0.0)

    # This block's contribution to the normal equations for design row
    # x = [feature, predecessor surprise, eₖ]. Shared-slope moments are
    # partials that _refit_kernel sums across positions.
    feature = tl.where(observed, feature, 0.0)
    a_k = tl.sum(w * feature * feature, axis=0)
    b_k = tl.sum(w * feature, axis=0)
    c_k = tl.sum(w, axis=0)
    g0_k = tl.sum(resid * feature, axis=0)
    g1_k = tl.sum(resid, axis=0)
    count = tl.sum(tl.where(observed, 1.0, 0.0), axis=0)
    tl.store(info_ptr + step * 6 + 0, tl.load(info_ptr + step * 6 + 0) + a_k)
    tl.store(info_ptr + step * 6 + 1, tl.load(info_ptr + step * 6 + 1) + b_k)
    tl.store(info_ptr + step * 6 + 2, tl.load(info_ptr + step * 6 + 2) + c_k)
    tl.store(grad_ptr + step * 3 + 0, tl.load(grad_ptr + step * 3 + 0) + g0_k)
    tl.store(grad_ptr + step * 3 + 1, tl.load(grad_ptr + step * 3 + 1) + g1_k)
    extra = tl.load(
        predecessor_surprise_ptr + req_state_block * features_stride + step,
        mask=req_mask,
        other=0.0,
    )
    extra = tl.where(observed, extra, 0.0)
    a01 = tl.sum(w * feature * extra, axis=0)
    a11 = tl.sum(w * extra * extra, axis=0)
    b1 = tl.sum(w * extra, axis=0)
    g_extra = tl.sum(resid * extra, axis=0)
    tl.store(info_ptr + step * 6 + 3, tl.load(info_ptr + step * 6 + 3) + a01)
    tl.store(info_ptr + step * 6 + 4, tl.load(info_ptr + step * 6 + 4) + a11)
    tl.store(info_ptr + step * 6 + 5, tl.load(info_ptr + step * 6 + 5) + b1)
    tl.store(grad_ptr + step * 3 + 2, tl.load(grad_ptr + step * 3 + 2) + g_extra)
    tl.store(counts_ptr + step, tl.load(counts_ptr + step) + count)
    surprise_count = tl.sum(tl.where(observed & (extra > 0.0), 1.0, 0.0), axis=0)
    tl.store(
        surprise_counts_ptr + step,
        tl.load(surprise_counts_ptr + step) + surprise_count,
    )


@triton.jit
def _refit_kernel(
    slope_ptr,
    surprise_slope_ptr,
    intercepts_ptr,
    info_ptr,
    info_row_stride,
    grad_ptr,
    grad_row_stride,
    counts_ptr,
    surprise_counts_ptr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    L2: tl.constexpr,
    DAMPING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k = tl.arange(0, BLOCK)
    mask = k < NUM_SPECULATIVE_STEPS

    # Load the stats accumulated over this round to solve (XᵀWX + λI) Δθ = Xᵀr.
    a_k = tl.load(info_ptr + k * info_row_stride + 0, mask=mask, other=0.0)
    b_k = tl.load(info_ptr + k * info_row_stride + 1, mask=mask, other=0.0)
    c_k = tl.load(info_ptr + k * info_row_stride + 2, mask=mask, other=0.0) + L2
    a = tl.sum(a_k, axis=0) + L2
    g0_k = tl.load(grad_ptr + k * grad_row_stride + 0, mask=mask, other=0.0)
    g1_k = tl.load(grad_ptr + k * grad_row_stride + 1, mask=mask, other=0.0)
    g0 = tl.sum(g0_k, axis=0)
    n = tl.load(counts_ptr + k, mask=mask, other=0.0)
    n_surprise = tl.load(surprise_counts_ptr + k, mask=mask, other=0.0)

    # Load the current shared slope and per-position intercepts.
    w = tl.load(slope_ptr)
    bias = tl.load(intercepts_ptr + k, mask=mask, other=0.0)

    # Eliminate the diagonal intercept block to obtain a 2x2 Schur complement
    # for the two shared slopes. Recover the intercept steps below.
    a01 = tl.load(info_ptr + k * info_row_stride + 3, mask=mask, other=0.0)
    a11 = tl.load(info_ptr + k * info_row_stride + 4, mask=mask, other=0.0)
    b1 = tl.load(info_ptr + k * info_row_stride + 5, mask=mask, other=0.0)
    g_extra = tl.load(grad_ptr + k * grad_row_stride + 2, mask=mask, other=0.0)
    m00 = a - tl.sum(b_k * b_k / c_k, axis=0)
    m01 = tl.sum(a01 - b_k * b1 / c_k, axis=0)
    m11 = tl.sum(a11 - b1 * b1 / c_k, axis=0) + L2
    v0 = g0 - tl.sum(b_k * g1_k / c_k, axis=0)
    v1 = tl.sum(g_extra - b1 * g1_k / c_k, axis=0)
    det = m00 * m11 - m01 * m01
    step_w = (v0 * m11 - v1 * m01) / det
    step_extra = (v1 * m00 - v0 * m01) / det

    # The peak slope learns from every observation; the surprise slope only
    # learns from nonzero predecessor surprises.
    total_n = tl.sum(n, axis=0)
    surprise_n = tl.sum(n_surprise, axis=0)
    step_w *= total_n / (total_n + DAMPING)
    step_extra *= surprise_n / (surprise_n + DAMPING)
    # Rounding can make the Schur determinant nonpositive. Reject those steps,
    # non-finite steps, and steps that would drive the peak slope negative.
    step_w = tl.where(
        (det > 0.0) & (tl.abs(step_w) < float("inf")) & (w + step_w >= 0.0),
        step_w,
        0.0,
    )
    step_extra = tl.where(
        (det > 0.0) & (tl.abs(step_extra) < float("inf")), step_extra, 0.0
    )
    tl.store(surprise_slope_ptr, tl.load(surprise_slope_ptr) + step_extra)
    # Update the shared slope.
    new_w = w + step_w

    # Solve for the per-position intercept deltas, Δβₖ, in the same system.
    step_bias = (g1_k - b_k * step_w - b1 * step_extra) / c_k
    # Damp by the position's sample count.
    step_bias *= n / (n + DAMPING)
    # Mask out non-finite steps.
    step_bias = tl.where(tl.abs(step_bias) < float("inf"), step_bias, 0.0)
    # Update the per-position intercepts.
    new_bias = bias + step_bias

    tl.store(slope_ptr, new_w)
    tl.store(intercepts_ptr + k, new_bias, mask=mask)

    # Start the round clean by clearing the per-round statistics.
    tl.store(info_ptr + k * info_row_stride + 0, 0.0, mask=mask)
    tl.store(info_ptr + k * info_row_stride + 1, 0.0, mask=mask)
    tl.store(info_ptr + k * info_row_stride + 2, 0.0, mask=mask)
    tl.store(grad_ptr + k * grad_row_stride + 0, 0.0, mask=mask)
    tl.store(grad_ptr + k * grad_row_stride + 1, 0.0, mask=mask)
    for col in tl.static_range(3, 6):
        tl.store(info_ptr + k * info_row_stride + col, 0.0, mask=mask)
    tl.store(grad_ptr + k * grad_row_stride + 2, 0.0, mask=mask)
    tl.store(counts_ptr + k, 0.0, mask=mask)
    tl.store(surprise_counts_ptr + k, 0.0, mask=mask)


@triton.jit
def _max_and_excl_sumexp(logits):
    """Block max, and its sumexp with one copy of that max left out.

    The excluding sum is what keeps the feature accurate. Forming the full sum
    and subtracting the max's own e^(m-m) = 1 term afterwards is catastrophic
    cancellation: the complement mass is the whole quantity of interest, and it
    is gone as soon as it falls below an fp32 ulp of 1.0 (~6e-8), which a
    confident drafter reaches routinely.
    """
    block_max = tl.max(logits, axis=0)
    # Excluding by value zeroes every entry tied for the max, so add back all
    # but one: the others are distinct tokens and belong to the complement.
    is_max = logits == block_max
    excl_sumexp = tl.sum(tl.where(is_max, 0.0, tl.exp(logits - block_max)), axis=0)
    excl_sumexp += tl.sum(is_max.to(tl.float32), axis=0) - 1.0
    # An all--inf block (vocab padding) ties with itself throughout, leaving a
    # spurious BLOCK_SIZE - 1 above; it holds no tokens, so it contributes none.
    return block_max, tl.where(block_max > float("-inf"), excl_sumexp, 0.0)


@triton.jit(do_not_specialize=["num_tokens"])
def _local_max_sumexp_kernel(
    local_max_ptr,
    local_max_stride,
    local_excl_sumexp_ptr,
    local_excl_sumexp_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    idx_mapping_stride,
    temperature_ptr,
    num_tokens,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= num_tokens:
        return

    block_idx = tl.program_id(1)
    req_state_idx = tl.load(idx_mapping_ptr + token_idx * idx_mapping_stride).to(
        tl.int64
    )
    if req_state_idx < 0:
        # Cudagraph-padded requests carry -1. Write an empty partial so the
        # buffer holds nothing uninitialized; _predict_kernel skips these rows.
        tl.store(
            local_max_ptr + token_idx * local_max_stride + block_idx, float("-inf")
        )
        tl.store(
            local_excl_sumexp_ptr + token_idx * local_excl_sumexp_stride + block_idx,
            0.0,
        )
        return

    # Draft logits are pre-temperature, and acceptance is decided on the scaled
    # distribution the drafter actually sampled from, so scale first. A request
    # with temp == 0 drafts greedily, so leave its logits alone rather than dividing
    # by zero, matching what the sampler does.
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    temp = tl.where(temp > 0.0, temp, 1.0)

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size
    logits = (
        tl.load(
            logits_ptr + token_idx * logits_stride + offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / temp
    )
    block_max, block_excl_sumexp = _max_and_excl_sumexp(logits)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, block_max)
    tl.store(
        local_excl_sumexp_ptr + token_idx * local_excl_sumexp_stride + block_idx,
        block_excl_sumexp,
    )


@triton.jit
def _acceptance_probability(weight, feature, bias, surprise_weight, surprise):
    base = weight * feature + bias
    return tl.sigmoid(
        tl.where(
            (surprise_weight == 0.0) | (surprise == 0.0),
            base,
            base + surprise_weight * surprise,
        )
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _predict_kernel(
    features_ptr,
    features_stride,
    pred_ptr,
    pred_stride,
    conf_ptr,
    conf_stride,
    slope_ptr,
    intercepts_ptr,
    local_max_ptr,
    local_max_stride,
    local_excl_sumexp_ptr,
    local_excl_sumexp_stride,
    idx_mapping_ptr,
    idx_mapping_stride,
    step_ptr,
    surprise_ptr,
    predecessor_surprise_ptr,
    surprise_slope_ptr,
    sampling_logits_ptr,
    sampling_stride,
    sampled_tokens_ptr,
    temperature_ptr,
    sampling_vocab_size,
    num_tokens,
    vocab_num_blocks,
    per_token_step: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MAX_LOG_ODDS: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= num_tokens:
        return

    req_state_idx = tl.load(idx_mapping_ptr + token_idx * idx_mapping_stride).to(
        tl.int64
    )
    if req_state_idx < 0:
        # Cudagraph-padded requests carry -1. Skip them so that they don't
        # scatter garbage over a live request's features.
        return

    # A per-row step means the speculator sampled the whole block in one pass, so
    # each request owns NUM_SPECULATIVE_STEPS consecutive rows; a step shared by
    # every row means one row per request. Batch position, unlike req_state_idx,
    # is only meaningful for the step that computed it, and is what adaptive
    # verification indexes by.
    if per_token_step:
        step = tl.load(step_ptr + token_idx).to(tl.int64)
        batch_idx = token_idx // NUM_SPECULATIVE_STEPS
    else:
        step = tl.load(step_ptr).to(tl.int64)
        batch_idx = token_idx

    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + token_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    excl_sumexps = tl.load(
        local_excl_sumexp_ptr + token_idx * local_excl_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )

    # The feature, logit(q(m*)) for the row's max draft logit, m*, is:
    #   q(x) = e^(x - m*) / Σ_z e^(z - m*)
    #   logit(q(m*)) = log(q(m*) / (1 - q(m*)))
    #                = -log(Σ_{z≠m*} e^(z - m*)).
    # It scores the draft distribution, not the token drawn from it, so it is
    # a function of the prefix alone. Trimming a draft therefore never
    # conditions on the token being trimmed, and rejection sampling stays
    # lossless. Scoring the drawn token instead would select on the proposal
    # and bias the emitted distribution toward the drafter's confident modes.

    # Rescale each block's partial to the global max and sum. Every block but
    # the one holding the global max has to put its own local max back, since
    # that is a different token and so is genuinely part of the complement; it
    # re-enters as e^(m_i - m*), the scale itself. A block that merely ties the
    # global max is one of those, and contributes in full.
    #
    # Every term is added, never subtracted, and none exceeds the total, so the
    # complement keeps fp32 relative accuracy however small it is -- which is
    # the point, since the feature is its logarithm and is read down to e^-40.
    global_max, best_block = tl.max(maxes, axis=0, return_indices=True)
    scale = tl.where(blocks_mask, tl.exp(maxes - global_max), 0.0)
    complement_sumexp = tl.sum(
        excl_sumexps * scale + tl.where(blocks == best_block, 0.0, scale), axis=0
    )

    # Shifted by m*, the numerator is e^(m* - m*) = 1, so the feature is just the
    # negated complement log-sum-exponential. A genuinely one-hot row sends it to
    # infinity, so clamp to the min/max log-odds.
    feature = -tl.log(complement_sumexp)
    feature = tl.minimum(tl.maximum(feature, -MAX_LOG_ODDS), MAX_LOG_ODDS)
    tl.store(features_ptr + req_state_idx * features_stride + step, feature)

    # The log probability ratio is just a temperature-scaled logit gap.
    # Cache the draw for the next position, never for its own score.
    surprise = 0.0
    if sampled_tokens_ptr is not None:
        token = tl.load(sampled_tokens_ptr + token_idx).to(tl.int64)
        temp = tl.load(temperature_ptr + req_state_idx)
        if (
            temp > 0
            and token >= 0
            and token < sampling_vocab_size
            and tl.abs(global_max) < float("inf")
        ):
            sampled = tl.load(sampling_logits_ptr + token_idx * sampling_stride + token)
            gap = global_max - sampled.to(tl.float32) / temp
            surprise = tl.where(tl.abs(gap) < float("inf"), tl.maximum(0.0, gap), 0.0)
    row = req_state_idx * features_stride
    if per_token_step and sampled_tokens_ptr is not None:
        tl.store(surprise_ptr + row + step, surprise)
        return  # A separate launch reads the preceding row's surprise.
    # Sequential drafting has already published the previous position.
    weight = tl.load(slope_ptr)
    bias = tl.load(intercepts_ptr + step)
    extra = 0.0
    if sampled_tokens_ptr is not None:
        extra = tl.load(
            surprise_ptr + row + step - 1,
            mask=(step > 0) & (tl.load(temperature_ptr + req_state_idx) > 0),
            other=0.0,
        )
    tl.store(surprise_ptr + row + step, surprise)
    tl.store(predecessor_surprise_ptr + row + step, extra)
    prob = _acceptance_probability(
        weight, feature, bias, tl.load(surprise_slope_ptr), extra
    )
    tl.store(pred_ptr + req_state_idx * pred_stride + step, prob)
    tl.store(conf_ptr + batch_idx * conf_stride + step, prob)


@triton.jit
def _predict_causal_block(
    peak_ptr,
    surprise_ptr,
    extra_ptr,
    pred_ptr,
    conf_ptr,
    idx_ptr,
    idx_stride,
    feature_stride,
    conf_stride,
    slope_ptr,
    surprise_slope_ptr,
    bias_ptr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    req = tl.load(idx_ptr + batch * K * idx_stride).to(tl.int64)
    if req < 0:
        return
    step = tl.arange(0, BLOCK)
    valid = step < K
    base = req * feature_stride
    feature = tl.load(peak_ptr + base + step, valid, 0.0)
    extra = tl.load(surprise_ptr + base + step - 1, valid & (step > 0), 0.0)
    bias = tl.load(bias_ptr + step, step < K, 0.0)
    p = _acceptance_probability(
        tl.load(slope_ptr), feature, bias, tl.load(surprise_slope_ptr), extra
    )
    tl.store(extra_ptr + base + step, extra, valid)
    tl.store(pred_ptr + base + step, p, valid)
    tl.store(conf_ptr + batch * conf_stride + step, p, valid)


class OnlineAcceptanceEstimator:
    """Predicts per-position acceptance, and calibrates itself while serving.

    Lifecycle per step, driven by `DraftModelSpeculator`:

    1. `step` folds the previous step's drafts, now graded by the target,
       into the IRLS accumulators, and periodically solves for new coefficients,
       independently but identically on every rank.
    2. `predict` runs inside the captured draft graph, turning this step's
       draft logits into acceptance probabilities for adaptive verification.

    Trimming starts immediately: survival is a running product of sigmoids, so it
    decreases with draft position whatever the coefficients are, and an unfitted
    estimator degrades to uniform-depth truncation rather than to anything harmful.
    """

    # Refit every this many steps, accumulating samples in between.
    REFIT_INTERVAL = 100
    # Newton steps are damped by n / (n + DAMPING_OBSERVATIONS), so a round
    # carrying this many observations moves a parameter half of a full step. It
    # replaces a hard minimum-sample gate: a data-poor position keeps learning,
    # just slowly, instead of freezing until it crosses a threshold. Lower values
    # track a drifting workload faster, higher ones are steadier on thin data.
    DAMPING_OBSERVATIONS = 50.0
    # Ridge on the Newton solve.
    L2 = 1e-3

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ):
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self._steps_since_refit = 0
        self._refits = 0
        self._accumulate_block_r = max(256, triton.next_power_of_2(max_num_reqs))

        # The learned coefficients used for predicting acceptance.
        self.slope = torch.full((1,), _INIT_SLOPE, dtype=torch.float32, device=device)
        self.intercepts = torch.full(
            (num_speculative_steps,), _INIT_BIAS, dtype=torch.float32, device=device
        )
        # Holds logit(max q), which is used as the feature for the logistic.
        # Stored in stable slots keyed by persistent request-state index.
        self.features = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )
        # Predictions and predecessor features are retained until verification.
        self.predictions = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )
        self.surprise = torch.zeros_like(self.features)
        self.predecessor_surprise = torch.zeros_like(self.features)
        self.surprise_slope = torch.zeros_like(self.slope)

        # Per-position contributions to XᵀWX and Xᵀ(Y-P), for a design
        # row [f, s, eₖ] with peak log-odds f and predecessor surprise s.
        # Info columns: [w*f*f, w*f, w, w*f*s, w*s*s, w*s].
        # Gradient columns: [r*f, r, r*s], where w=p(1-p), r=y-p.
        self.info = torch.zeros(
            num_speculative_steps, 6, dtype=torch.float32, device=device
        )
        self.grad = torch.zeros(
            num_speculative_steps, 3, dtype=torch.float32, device=device
        )
        # Number of graded drafts per-position in the current round, zeroed after
        # each refit. _refit_kernel uses this to damp coefficient updates by:
        # n / (n + DAMPING_OBSERVATIONS)
        self.counts = torch.zeros(
            num_speculative_steps, dtype=torch.float32, device=device
        )
        self.surprise_counts = torch.zeros_like(self.counts)

    def step(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        # Accumulate the previous step's graded drafts into the IRLS statistics.
        num_reqs = idx_mapping.shape[0]
        _accumulate_kernel[(self.num_speculative_steps,)](
            self.info,
            self.grad,
            idx_mapping,
            num_sampled,
            num_rejected,
            self.features,
            self.features.stride(0),
            self.predecessor_surprise,
            self.predictions,
            self.predictions.stride(0),
            self.counts,
            self.surprise_counts,
            num_reqs,
            BLOCK_R=self._accumulate_block_r,
            num_warps=1 if self._accumulate_block_r <= 256 else 4,
        )

        self._steps_since_refit += 1
        if self._steps_since_refit < self.REFIT_INTERVAL:
            return
        self._steps_since_refit = 0

        # Fit the coefficients to the accumulated statistics gathered over the
        # course of the last REFIT_INTERVAL steps.
        self._refit()
        self._refits += 1

    def warmup(self) -> None:
        """Drop warmup statistics and compile the refit kernel. A refit on
        zeroed statistics leaves the coefficients unchanged."""
        self.info.zero_()
        self.grad.zero_()
        self.counts.zero_()
        self.surprise_counts.zero_()
        self._steps_since_refit = 0
        self._refit()

    def _refit(self) -> None:
        _refit_kernel[(1,)](
            self.slope,
            self.surprise_slope,
            self.intercepts,
            self.info,
            self.info.stride(0),
            self.grad,
            self.grad.stride(0),
            self.counts,
            self.surprise_counts,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            L2=self.L2,
            DAMPING=self.DAMPING_OBSERVATIONS,
            BLOCK=triton.next_power_of_2(self.num_speculative_steps),
        )

    def predict(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
        confidence_probs: torch.Tensor,
        temperature: torch.Tensor,
        sampled_tokens: torch.Tensor | None = None,
        sampling_logits: torch.Tensor | None = None,
    ) -> None:
        # Reduced-vocab drafters may sample in target-vocabulary space.
        # Keep the peak reduction in draft space and look up the draw in its
        # actual sampling vocabulary.
        num_tokens, vocab_size = logits.shape
        VOCAB_BLOCK_SIZE = 4096
        num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        local_max = torch.empty(
            num_tokens, num_blocks, dtype=torch.float32, device=self.device
        )
        local_excl_sumexp = torch.empty_like(local_max)
        _local_max_sumexp_kernel[(num_tokens, num_blocks)](
            local_max,
            local_max.stride(0),
            local_excl_sumexp,
            local_excl_sumexp.stride(0),
            logits,
            logits.stride(0),
            idx_mapping,
            idx_mapping.stride(0),
            temperature,
            num_tokens,
            vocab_size,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        )
        sampling_logits = logits if sampling_logits is None else sampling_logits
        _predict_kernel[(num_tokens,)](
            self.features,
            self.features.stride(0),
            self.predictions,
            self.predictions.stride(0),
            confidence_probs,
            confidence_probs.stride(0),
            self.slope,
            self.intercepts,
            local_max,
            local_max.stride(0),
            local_excl_sumexp,
            local_excl_sumexp.stride(0),
            idx_mapping,
            idx_mapping.stride(0),
            draft_step,
            self.surprise,
            self.predecessor_surprise,
            self.surprise_slope,
            sampling_logits,
            sampling_logits.stride(0),
            sampled_tokens,
            temperature,
            sampling_logits.shape[1],
            num_tokens,
            num_blocks,
            per_token_step=draft_step.dim() > 0,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            MAX_LOG_ODDS=_MAX_LOG_ODDS,
            PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        )

        if draft_step.dim() > 0 and sampled_tokens is not None:
            k = self.num_speculative_steps
            _predict_causal_block[(num_tokens // k,)](
                self.features,
                self.surprise,
                self.predecessor_surprise,
                self.predictions,
                confidence_probs,
                idx_mapping,
                idx_mapping.stride(0),
                self.features.stride(0),
                confidence_probs.stride(0),
                self.slope,
                self.surprise_slope,
                self.intercepts,
                K=k,
                BLOCK=triton.next_power_of_2(k),
            )
