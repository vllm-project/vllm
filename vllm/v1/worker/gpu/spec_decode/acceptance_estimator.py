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

# A refit window with near-zero Fisher information -- either because it
# genuinely had few observations, or because predictions are already
# saturated so the IRLS weight p(1-p) vanishes even on real samples --
# divides a normal-sized gradient by a near-zero Hessian, bounded only by
# L2. Floor the information terms before dividing, and clip the resulting
# step, so a single refit can't take an arbitrarily large jump.
_C_FLOOR = 1.0
_MAX_STEP_SLOPE = 10.0
_MAX_STEP_BIAS = 50.0

# The floor above bounds a single step, but a persistently one-directional
# gradient still drifts, unbounded, refit after refit: this is separation
# (Albert & Anderson, 1984) -- once a position's samples in a window are
# (quasi-)perfectly predicted by the current coefficients, its own Fisher
# information collapses toward zero regardless of how many real samples
# it saw, and nothing reads that as a reason to stop trusting the step.
# Track, per position and separately for the shared slope, how many
# *consecutive* refits its raw (pre-floor) Fisher information has sat
# below STUCK_THRESHOLD. Once that streak reaches STUCK_LIMIT, discard the
# computed step and reset that one coefficient to its own cold-start
# default instead -- the same value (and justification) used at server
# startup, just applied mid-stream once persistence is confirmed. A single
# low-information refit does not trigger this: the counter clears the
# moment the position sees real information again.
_STUCK_THRESHOLD = 0.1
_STUCK_LIMIT = 3.0


@triton.jit(do_not_specialize=["num_reqs"])
def _accumulate_kernel(
    info_ptr,
    grad_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    features_ptr,
    features_stride,
    pred_ptr,
    pred_stride,
    counts_ptr,
    num_reqs,
    BLOCK_R: tl.constexpr,
):
    step = tl.program_id(0).to(tl.int64)
    req_block = tl.arange(0, BLOCK_R)
    req_mask = req_block < num_reqs
    req_state_block = tl.load(idx_mapping_ptr + req_block, mask=req_mask, other=0).to(
        tl.int64
    )

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
    # x = [feature, eₖ]: the XᵀWX entries aₖ, bₖ, cₖ and the Xᵀr entries g₀ₖ, g₁ₖ.
    # aₖ and g₀ₖ are partials of scalars shared by every position, which
    # _refit_kernel sums.
    a_k = tl.sum(w * feature * feature, axis=0)
    b_k = tl.sum(w * feature, axis=0)
    c_k = tl.sum(w, axis=0)
    g0_k = tl.sum(resid * feature, axis=0)
    g1_k = tl.sum(resid, axis=0)
    count = tl.sum(tl.where(observed, 1.0, 0.0), axis=0)
    tl.store(info_ptr + step * 3 + 0, tl.load(info_ptr + step * 3 + 0) + a_k)
    tl.store(info_ptr + step * 3 + 1, tl.load(info_ptr + step * 3 + 1) + b_k)
    tl.store(info_ptr + step * 3 + 2, tl.load(info_ptr + step * 3 + 2) + c_k)
    tl.store(grad_ptr + step * 2 + 0, tl.load(grad_ptr + step * 2 + 0) + g0_k)
    tl.store(grad_ptr + step * 2 + 1, tl.load(grad_ptr + step * 2 + 1) + g1_k)
    tl.store(counts_ptr + step, tl.load(counts_ptr + step) + count)


@triton.jit
def _refit_kernel(
    slope_ptr,
    intercepts_ptr,
    info_ptr,
    info_row_stride,
    grad_ptr,
    grad_row_stride,
    counts_ptr,
    slope_stuck_ptr,
    pos_stuck_ptr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    L2: tl.constexpr,
    DAMPING: tl.constexpr,
    C_FLOOR: tl.constexpr,
    MAX_STEP_SLOPE: tl.constexpr,
    MAX_STEP_BIAS: tl.constexpr,
    STUCK_THRESHOLD: tl.constexpr,
    STUCK_LIMIT: tl.constexpr,
    INIT_SLOPE: tl.constexpr,
    INIT_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k = tl.arange(0, BLOCK)
    mask = k < NUM_SPECULATIVE_STEPS

    # Load the stats accumulated over this round to solve (XᵀWX + λI) Δθ = Xᵀr.
    a_k = tl.load(info_ptr + k * info_row_stride + 0, mask=mask, other=0.0)
    b_k = tl.load(info_ptr + k * info_row_stride + 1, mask=mask, other=0.0)
    raw_c_k = tl.load(info_ptr + k * info_row_stride + 2, mask=mask, other=0.0)
    # Floor the information terms by a constant, independent of sample
    # count, before dividing -- this is what actually bounds the step; see
    # _C_FLOOR above. The *raw* (pre-floor) values are kept for the stuck
    # check below, since flooring would otherwise hide the exact signal
    # that check needs.
    c_k = tl.maximum(raw_c_k, C_FLOOR) + L2
    raw_a = tl.sum(a_k, axis=0)
    a = tl.maximum(raw_a, C_FLOOR) + L2
    g0_k = tl.load(grad_ptr + k * grad_row_stride + 0, mask=mask, other=0.0)
    g1_k = tl.load(grad_ptr + k * grad_row_stride + 1, mask=mask, other=0.0)
    g0 = tl.sum(g0_k, axis=0)
    n = tl.load(counts_ptr + k, mask=mask, other=0.0)

    # Load the current shared slope and per-position intercepts.
    w = tl.load(slope_ptr)
    bias = tl.load(intercepts_ptr + k, mask=mask, other=0.0)

    # Solve for the shared slope delta, Δω, in the following arrowhead system:
    #   ⎡ a+λ   b₀     b₁    …  bₙ₋₁    ⎤ ⎡  Δω   ⎤   ⎡  g₀   ⎤
    #   ⎢ b₀    c₀+λ                    ⎥ ⎢  Δβ₀  ⎥   ⎢  g₁₀  ⎥
    #   ⎢ b₁           c₁+λ             ⎥ ⎢  Δβ₁  ⎥ = ⎢  g₁₁  ⎥
    #   ⎢ ⋮                  ⋱          ⎥ ⎢  ⋮    ⎥   ⎢  ⋮    ⎥
    #   ⎣ bₙ₋₁                  cₙ₋₁+λ   ⎦ ⎣ Δβₙ₋₁ ⎦   ⎣ g₁ₙ₋₁ ⎦
    #               XᵀWX + λI                Δθ        Xᵀ(Y-P)
    shrink = b_k * b_k / c_k
    coupling = b_k * g1_k / c_k
    step_w = (g0 - tl.sum(coupling, axis=0)) / (a - tl.sum(shrink, axis=0))

    # The slope learns from every position, so it is damped by the round's
    # total sample count.
    total_n = tl.sum(n, axis=0)
    step_w *= total_n / (total_n + DAMPING)
    # Mask out NaNs and steps that would drive the slope negative.
    step_w = tl.where((step_w == step_w) & (w + step_w >= 0.0), step_w, 0.0)
    # Clip the step's magnitude before applying it.
    step_w = tl.minimum(tl.maximum(step_w, -MAX_STEP_SLOPE), MAX_STEP_SLOPE)
    updated_w = w + step_w

    # Circuit breaker for the shared slope: if its raw aggregate
    # information has sat near-zero for STUCK_LIMIT consecutive refits,
    # reset it to the cold-start default instead of nudging it further.
    prev_slope_stuck = tl.load(slope_stuck_ptr)
    slope_is_stuck = raw_a < STUCK_THRESHOLD
    new_slope_stuck = tl.where(slope_is_stuck, prev_slope_stuck + 1.0, 0.0)
    slope_should_reset = new_slope_stuck >= STUCK_LIMIT
    new_w = tl.where(slope_should_reset, INIT_SLOPE, updated_w)
    tl.store(slope_stuck_ptr, tl.where(slope_should_reset, 0.0, new_slope_stuck))

    # Solve for the per-position intercept deltas, Δβₖ, in the same system.
    step_bias = (g1_k - b_k * step_w) / c_k
    # Damp by the position's sample count.
    step_bias *= n / (n + DAMPING)
    # Mask out NaNs.
    step_bias = tl.where(step_bias == step_bias, step_bias, 0.0)
    # Same per-position clip, before applying.
    step_bias = tl.minimum(tl.maximum(step_bias, -MAX_STEP_BIAS), MAX_STEP_BIAS)
    updated_bias = bias + step_bias

    # Circuit breaker per position: same idea as the slope above, but
    # keyed on this position's own raw (pre-floor) Fisher information.
    prev_pos_stuck = tl.load(pos_stuck_ptr + k, mask=mask, other=0.0)
    pos_is_stuck = raw_c_k < STUCK_THRESHOLD
    new_pos_stuck = tl.where(pos_is_stuck, prev_pos_stuck + 1.0, 0.0)
    pos_should_reset = new_pos_stuck >= STUCK_LIMIT
    new_bias = tl.where(pos_should_reset, INIT_BIAS, updated_bias)
    tl.store(
        pos_stuck_ptr + k,
        tl.where(pos_should_reset, 0.0, new_pos_stuck),
        mask=mask,
    )

    tl.store(slope_ptr, new_w)
    tl.store(intercepts_ptr + k, new_bias, mask=mask)

    # Start the round clean by clearing the per-round statistics.
    tl.store(info_ptr + k * info_row_stride + 0, 0.0, mask=mask)
    tl.store(info_ptr + k * info_row_stride + 1, 0.0, mask=mask)
    tl.store(info_ptr + k * info_row_stride + 2, 0.0, mask=mask)
    tl.store(grad_ptr + k * grad_row_stride + 0, 0.0, mask=mask)
    tl.store(grad_ptr + k * grad_row_stride + 1, 0.0, mask=mask)
    tl.store(counts_ptr + k, 0.0, mask=mask)


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

    # Predict the acceptance probability from the feature and current coefficients.
    weight = tl.load(slope_ptr)
    bias = tl.load(intercepts_ptr + step)
    prob = tl.sigmoid(weight * feature + bias)
    tl.store(pred_ptr + req_state_idx * pred_stride + step, prob)
    tl.store(conf_ptr + batch_idx * conf_stride + step, prob)


class OnlineAcceptanceEstimator:
    """Predicts per-position acceptance, and calibrates itself while serving.

    Lifecycle per step, driven by ``DraftModelSpeculator``:

    1. ``step`` folds the previous step's drafts, now graded by the target,
       into the IRLS accumulators, and periodically solves for new coefficients,
       independently but identically on every rank.
    2. ``predict`` runs inside the captured draft graph, turning this step's
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
        # Holds the acceptance predictions made at draft time, computed as:
        #   p = σ(ω * f + βₖ)
        # where f is feature (logit(max q)) at draft position k, ω is the learned
        # shared slope, and βₖ is the learned per-position intercept. Stored in
        # stable slots keyed by the persistent request-state index.
        self.predictions = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )

        # Per-round Newton-IRLS statistics, cleared after each refit. For observation
        # i the design row is xᵢ = [fᵢ, eₖ], the IRLS weight is wᵢ = pᵢ(1-pᵢ), and the
        # residual is yᵢ - pᵢ, where yᵢ is the label (1 for accepted, 0 for rejected).
        # XᵀWX is then the negative Hessian of the Bernoulli log-likelihood loss and
        # Xᵀ(Y-P) is its gradient, so the Newton step for θ = [ω, β₀ … βₙ₋₁] is the
        # the solution of (XᵀWX + λI) Δθ = Xᵀ(Y-P).
        #
        # Columns [aₖ, bₖ, cₖ] are each position's contribution to the XᵀWX arrowhead:
        #   ⎡ a    b₀   b₁ … bₙ₋₁ ⎤
        #   ⎢ b₀   c₀             ⎥
        #   ⎢ b₁        c₁        ⎥
        #   ⎢ ⋮            ⋱      ⎥
        #   ⎣ bₙ₋₁           cₙ₋₁ ⎦
        # bₖ and cₖ are per-position, but a = Σₖ aₖ is one scalar shared by all of
        # them.
        self.info = torch.zeros(
            num_speculative_steps, 3, dtype=torch.float32, device=device
        )
        # Columns [g₀ₖ, g₁ₖ] are each position's contribution to the Xᵀ(Y-P) vector:
        #   ⎡ g₀    ⎤
        #   ⎢ g₁₀   ⎥
        #   ⎢ g₁₁   ⎥
        #   ⎢  ⋮    ⎥
        #   ⎣ g₁ₙ₋₁ ⎦
        # As above, g₁ₖ is per-position and g₀ = Σₖ g₀ₖ is shared.
        self.grad = torch.zeros(
            num_speculative_steps, 2, dtype=torch.float32, device=device
        )
        # Number of graded drafts per-position in the current round, zeroed after
        # each refit. _refit_kernel uses this to damp coefficient updates by:
        # n / (n + DAMPING_OBSERVATIONS)
        self.counts = torch.zeros(
            num_speculative_steps, dtype=torch.float32, device=device
        )
        # Consecutive-stuck-refit counters for the circuit breaker: one
        # scalar for the shared slope, one per position. See _STUCK_LIMIT.
        self.slope_stuck = torch.zeros(1, dtype=torch.float32, device=device)
        self.pos_stuck = torch.zeros(
            num_speculative_steps, dtype=torch.float32, device=device
        )

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
            self.predictions,
            self.predictions.stride(0),
            self.counts,
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
        self._steps_since_refit = 0
        self._refit()

    def _refit(self) -> None:
        _refit_kernel[(1,)](
            self.slope,
            self.intercepts,
            self.info,
            self.info.stride(0),
            self.grad,
            self.grad.stride(0),
            self.counts,
            self.slope_stuck,
            self.pos_stuck,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            L2=self.L2,
            DAMPING=self.DAMPING_OBSERVATIONS,
            C_FLOOR=_C_FLOOR,
            MAX_STEP_SLOPE=_MAX_STEP_SLOPE,
            MAX_STEP_BIAS=_MAX_STEP_BIAS,
            STUCK_THRESHOLD=_STUCK_THRESHOLD,
            STUCK_LIMIT=_STUCK_LIMIT,
            INIT_SLOPE=_INIT_SLOPE,
            INIT_BIAS=_INIT_BIAS,
            BLOCK=triton.next_power_of_2(self.num_speculative_steps),
        )

    def predict(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
        confidence_probs: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
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
            num_tokens,
            num_blocks,
            per_token_step=draft_step.dim() > 0,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            MAX_LOG_ODDS=_MAX_LOG_ODDS,
            PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        )
