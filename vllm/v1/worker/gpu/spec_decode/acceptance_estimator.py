# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online per-position acceptance estimation for adaptive verification.

Adaptive verification needs, for every drafted token, the probability that the
target will accept it. DSpark checkpoints carry a trained confidence head for
this; no other speculator does. This module estimates it instead from the shape
of the draft distribution the speculator already computes, and calibrates that
estimate at runtime against acceptance actually observed during serving.

The feature is the top-2 logit margin ``m1 - m2``: how decisively the drafter
preferred the token it drafted. It needs only max reductions -- no partition
function -- which makes it the one statistic obtainable without a second pass
over the vocab. Empirically it carries essentially all of the available signal
(held-out AUC 0.884 versus 0.889 for a four-feature model on DeepSeek-V4-Flash-
DSpark, against 0.839 for that checkpoint's trained confidence head).

A per-position logistic maps margin to acceptance probability::

    p_k = sigmoid(weight * margin + bias[k])

One slope shared across draft positions and an intercept per position, fit by
Newton-IRLS from sufficient statistics accumulated on device. Sharing the slope
is what makes the deep positions usable: they see too few observations per round
to fit two parameters, but an intercept alone is well determined, and the slope
-- how decisively margin maps to acceptance -- has no reason to vary with depth.
Steps are damped in proportion to the observations behind them, so a position
the drafts rarely reach drifts slowly rather than lurching. Observations are
never stored: each is folded straight into an arrowhead information matrix and
its score, so the entire learning state is a few dozen floats regardless of
traffic.
"""

import os

import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Stands in for "no logit here": the running-max initializer, the fill for
# out-of-range vocab lanes, and the floor for genuine -inf entries. Finite rather
# than -inf so an all-padding block yields 0 rather than (-inf) - (-inf) = NaN,
# far below any real logit, and well inside fp32 and bf16 range so subtracting it
# cannot overflow. The other Triton top-k kernels use the same value.
_MIN_LOGIT = -1.0e30
# A row with a single finite entry would emit an unbounded margin.
_MAX_MARGIN = 40.0
# Floor on -log q before the Gumbel transform, so a draw with q indistinguishable
# from 1 lands at a finite feature value instead of -inf.
_MIN_NEG_LOG_Q = 1e-3


@triton.jit
def _accumulate_kernel(
    info_ptr,
    grad_ptr,
    totals_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    margins_ptr,
    margins_stride,
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
    req_state_block = tl.maximum(req_state_block, 0)

    # num_sampled is accepted + 1 bonus.
    num_sampled = tl.load(num_sampled_ptr + req_block, mask=req_mask, other=0).to(
        tl.int64
    )
    num_accepted = tl.maximum(num_sampled - 1, 0)
    num_rejected = tl.load(num_rejected_ptr + req_block, mask=req_mask, other=0).to(
        tl.int64
    )
    num_admitted = num_accepted + num_rejected

    margin = tl.load(
        margins_ptr + req_state_block * margins_stride + step, mask=req_mask, other=0.0
    ).to(tl.float32)
    pred = tl.load(
        pred_ptr + req_state_block * pred_stride + step,
        mask=req_mask,
        other=0.0,
    ).to(tl.float32)

    observed = req_mask & (step <= num_accepted) & (step < num_admitted)
    mask = tl.where(observed, 1.0, 0.0)
    label = tl.where(step < num_accepted, 1.0, 0.0)
    w = pred * (1.0 - pred) * mask
    resid = (label - pred) * mask

    # Weighted normal-equation pieces for the design row x = [margin, e_k]: the
    # information matrix XtWX and the score Xt(y - p). The slope is shared across
    # positions, so XtWX is an arrowhead whose slope-slope entry sum(w*margin^2)
    # and slope score sum(resid*margin) are single totals; only the coupling
    # sum(w*margin), the intercept block sum(w), and the intercept score
    # sum(resid) are per position. Keeping those two as scalars rather than per
    # position is what lets _refit_kernel skip the per-position 2x2 solves.
    xtwx_01 = tl.sum(w * margin, axis=0)
    xtwx_11 = tl.sum(w, axis=0)
    xtr_1 = tl.sum(resid, axis=0)
    count = tl.sum(mask, axis=0)
    # This program owns position `step`, so these are plain read-modify-writes.
    tl.store(info_ptr + step * 2 + 0, tl.load(info_ptr + step * 2 + 0) + xtwx_01)
    tl.store(info_ptr + step * 2 + 1, tl.load(info_ptr + step * 2 + 1) + xtwx_11)
    tl.store(grad_ptr + step, tl.load(grad_ptr + step) + xtr_1)
    tl.store(counts_ptr + step, tl.load(counts_ptr + step) + count)
    # The totals are shared by every program, so they need real atomics. Their
    # summation order varies between runs, which perturbs the coefficients in the
    # last bits only; ranks are reconciled by the all-reduce in `step` either way.
    tl.atomic_add(totals_ptr + 0, tl.sum(w * margin * margin, axis=0), sem="relaxed")
    tl.atomic_add(totals_ptr + 1, tl.sum(resid * margin, axis=0), sem="relaxed")


@triton.jit
def _refit_kernel(
    coef_ptr,
    coef_stride,
    info_ptr,
    info_row_stride,
    grad_ptr,
    totals_ptr,
    counts_ptr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    L2: tl.constexpr,
    DAMPING: tl.constexpr,
    INV_TP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One Newton-IRLS step for a shared slope and per-position intercepts.

    The design row is x = [margin, e_k], so XtWX is an arrowhead: a dense first
    row and column for the slope over a diagonal intercept block. Eliminating the
    intercept rows (a Schur complement) collapses it to a scalar equation for the
    slope, which is why no matrix solve appears below.

    Each step is damped by n / (n + DAMPING), the fraction of a full Newton step
    a parameter with n observations is allowed to take. A full step is only
    justified where the local quadratic is accurate, which it is not on a few
    dozen samples; undamped, such a round can throw a coefficient far past the
    optimum and take many rounds to walk back.
    """
    k = tl.arange(0, BLOCK)
    mask = k < NUM_SPECULATIVE_STEPS

    # Per position: the slope/intercept coupling sum(w*margin) and the intercept
    # block sum(w), plus the intercept score sum(y - p).
    b = tl.load(info_ptr + k * info_row_stride + 0, mask=mask, other=0.0)
    c = tl.load(info_ptr + k * info_row_stride + 1, mask=mask, other=0.0) + L2
    g1 = tl.load(grad_ptr + k, mask=mask, other=0.0)
    n = tl.load(counts_ptr + k, mask=mask, other=0.0)
    # Shared by every position: the slope's own information and score. The ridge
    # lands once on the slope and once per intercept, which is the L2 penalty for
    # the NUM_SPECULATIVE_STEPS + 1 parameters actually being fit.
    a = tl.load(totals_ptr + 0) + L2
    g0 = tl.load(totals_ptr + 1)

    w = tl.load(coef_ptr + k, mask=mask, other=0.0)
    bias = tl.load(coef_ptr + coef_stride + k, mask=mask, other=0.0)

    # Profiling the intercepts out of the arrowhead leaves
    #     step_w = (g0 - sum_k b_k*g1_k/c_k) / (a - sum_k b_k^2/c_k),
    # the slope's Newton step once each position's intercept has absorbed what it
    # can. Every position contributes in proportion to its own information, so a
    # data-poor position barely moves the slope while still receiving it -- which
    # is what gives the deep positions a usable slope at all: they see too few
    # observations per round to fit two parameters, but an intercept alone is
    # well determined. The ridge keeps c_k >= L2, and Cauchy-Schwarz gives
    # b_k^2 <= a_k*c_k, so the denominator is >= L2 and never degenerate.
    shrink = tl.where(mask, b * b / c, 0.0)
    coupling = tl.where(mask, b * g1 / c, 0.0)
    den = a - tl.sum(shrink, axis=0)
    step_w = (g0 - tl.sum(coupling, axis=0)) / den

    # The slope learns from every position, so it is damped by the round's total
    # sample count rather than any single position's.
    total_n = tl.sum(tl.where(mask, n, 0.0), axis=0)
    step_w = step_w * total_n / (total_n + DAMPING)
    w_now = tl.sum(tl.where(k == 0, w, 0.0), axis=0)  # shared: read once
    # Mask out NaNs and steps that would drive the slope negative.
    w_ok = (step_w == step_w) & (w_now + step_w >= 0.0)
    step_w = tl.where(w_ok, step_w, 0.0)
    new_w = tl.where(mask, w_now + step_w, 0.0)

    # Intercept conditioned on the pooled slope, damped by its own count: a
    # position nobody reached this round has n = 0 and so holds still.
    step_b = (g1 - b * step_w) / c * n / (n + DAMPING)
    new_b = tl.where(mask & (step_b == step_b), bias + step_b, bias)

    # The slope is shared, but it is stored once per position so that `predict`
    # can index weight and bias by draft step alike.
    tl.store(coef_ptr + k, new_w * INV_TP, mask=mask)
    tl.store(coef_ptr + coef_stride + k, new_b * INV_TP, mask=mask)

    # Start the next round clean: each refit fits its own window, which is also
    # what lets the estimator track workload drift.
    tl.store(info_ptr + k * info_row_stride + 0, 0.0, mask=mask)
    tl.store(info_ptr + k * info_row_stride + 1, 0.0, mask=mask)
    tl.store(grad_ptr + k, 0.0, mask=mask)
    tl.store(counts_ptr + k, 0.0, mask=mask)
    tl.store(totals_ptr + tl.arange(0, 2), tl.zeros((2,), tl.float32))


@triton.jit
def _predict_kernel(
    margins_ptr,
    margins_stride,
    pred_ptr,
    pred_stride,
    conf_ptr,
    conf_stride,
    coef_ptr,
    coef_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    idx_mapping_stride,
    step_ptr,
    tokens_ptr,
    num_tokens,
    vocab_size,
    per_token_step: tl.constexpr,
    USE_LOG_Q: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MIN_LOGIT: tl.constexpr,
    MAX_MARGIN: tl.constexpr,
    MIN_NEG_LOG_Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= num_tokens:
        return

    req_state_idx = tl.load(idx_mapping_ptr + token_idx * idx_mapping_stride).to(
        tl.int64
    )
    if req_state_idx < 0:
        # Cudagraph-padded requests carry -1. Skip them so that they don't
        # scatter garbage over a live request's margins.
        return

    # Get the top-2 logits for this token.
    logits_row = logits_ptr + token_idx * logits_stride
    # Each lane keeps its own top two and the lanes are combined once at the
    # end, rather than reducing the block to a scalar pair every iteration.
    lane = tl.arange(0, BLOCK_SIZE)
    lane_max1 = tl.full((BLOCK_SIZE,), MIN_LOGIT, tl.float32)
    lane_max2 = tl.full((BLOCK_SIZE,), MIN_LOGIT, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE):
        block = start + lane
        mask = block < vocab_size
        block_logits = tl.load(logits_row + block, mask=mask, other=MIN_LOGIT).to(
            tl.float32
        )
        block_logits = tl.maximum(block_logits, MIN_LOGIT)
        lane_max2 = tl.maximum(lane_max2, tl.minimum(lane_max1, block_logits))
        lane_max1 = tl.maximum(lane_max1, block_logits)
    max1 = tl.max(lane_max1, axis=0)
    tie_indices = tl.argmax(lane_max1, axis=0, tie_break_left=True)
    max2 = tl.max(tl.where(lane == tie_indices, lane_max2, lane_max1), axis=0)

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

    if USE_LOG_Q:
        # Probabilistic drafting accepts with min(1, p(x)/q(x)) for the token it
        # actually drew, so the drafted token's own log-probability is the term in
        # the mechanism; the top-2 margin describes the mode, which is often not
        # the token drawn. log q = logit(x) - logsumexp, computed with max1 as the
        # shift for stability, so it needs one more pass over the row.
        lane_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
        for start in tl.range(0, vocab_size, BLOCK_SIZE):
            block = start + lane
            mask = block < vocab_size
            block_logits = tl.load(logits_row + block, mask=mask, other=MIN_LOGIT).to(
                tl.float32
            )
            lane_sum += tl.where(mask, tl.exp(block_logits - max1), 0.0)
        log_sum_exp = max1 + tl.log(tl.sum(lane_sum, axis=0))
        token = tl.load(tokens_ptr + token_idx).to(tl.int64)
        sampled_logit = tl.load(logits_row + token).to(tl.float32)
        # log q is <= 0; clamp the tail so one astronomically unlikely draw cannot
        # dominate the IRLS fit.
        log_q = tl.maximum(sampled_logit - log_sum_exp, -MAX_MARGIN)
        # q goes on the Gumbel scale rather than being used directly. The drafter
        # samples by argmax over logits + Gumbel noise, so acceptance is an
        # extremum event, and -log(-log q) is the Gumbel quantile of q: the scale
        # on which that noise is additive and a linear predictor is meaningful.
        # It is monotone increasing, so the fitted slope stays positive, and it
        # resolves the mass that piles up at log q ~ 0, which a term linear in
        # log q cannot -- there the acceptance rate climbs steeply inside a
        # sliver of a range whose tail runs to -MAX_MARGIN.
        feature = -tl.log(tl.maximum(-log_q, MIN_NEG_LOG_Q))
    else:
        # Greedy drafting takes the mode, so how decisively the drafter preferred
        # it is the natural confidence signal.
        feature = tl.minimum(max1 - max2, MAX_MARGIN)
    tl.store(margins_ptr + req_state_idx * margins_stride + step, feature)

    # Predict the acceptance probability from the feature and current coefficients.
    weight = tl.load(coef_ptr + step)
    bias = tl.load(coef_ptr + coef_stride + step)
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

    For the first few refits the estimator reports ``needs_full_verification``;
    callers verify whole draft blocks during that window so the labels it learns
    from are not censored by its own trimming.
    """

    # Adaptive verification is skipped for this many refits before the estimator
    # is considered trained enough to trim drafts.
    NUM_WARMUP_REFITS = int(
        os.getenv("VLLM_ACCEPTANCE_ESTIMATOR_NUM_WARMUP_REFITS", "3")
    )
    # After warmup, refit every this many steps, accumulating samples in between.
    REFIT_INTERVAL = 100
    # Newton steps are damped by n / (n + DAMPING_OBSERVATIONS), so a round
    # carrying this many observations moves a parameter half of a full step. It
    # replaces a hard minimum-sample gate: a data-poor position keeps learning,
    # just slowly, instead of freezing until it crosses a threshold. Lower values
    # track a drifting workload faster, higher ones are steadier on thin data.
    DAMPING_OBSERVATIONS = 50.0
    # Ridge on the Newton solve.
    L2 = 1e-3
    # Log per-position sample counts and calibration gap at every refit. Costs a
    # device sync per refit, so it is off unless explicitly asked for.
    DEBUG_CALIBRATION = bool(int(os.getenv("VLLM_ACCEPTANCE_ESTIMATOR_DEBUG", "0")))

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        use_log_q: bool = False,
    ):
        self.num_speculative_steps = num_speculative_steps
        # Which draft-confidence feature the logistic is fitted on: the drafted
        # token's log-probability under probabilistic drafting, the top-2 logit
        # margin under greedy. See _predict_kernel.
        self.use_log_q = use_log_q
        self.device = device
        self._steps_since_refit = 0
        self._refits = 0
        self._tp_size = (
            get_tp_group().world_size if model_parallel_is_initialized() else 1
        )

        # Coefficients, read inside the captured graph: update in place, never
        # reallocate. A zero slope with a bias matching a plausible acceptance
        # rate makes the initial estimate uniform, which still lets the cost
        # model size the budget.
        # Packed as [weight, bias] rows so the cross-rank all-reduce is a single
        # collective, and predict() a single pointer.
        self.coefficients = torch.zeros(
            2, num_speculative_steps, dtype=torch.float32, device=device
        )
        self.coefficients[1].fill_(1.5)

        # Difference between the top two logits, which is used as the feature for
        # the logistic. Stored in stable slots keyed by persistent request-state index.
        self.margins = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )
        # Predictions made at draft time, in the same stable slots as the margins
        # they came from, because the label that grades them only arrives on the
        # next step, by which point the batch has been reordered. Reused as the
        # IRLS weight then.
        self.predictions = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )

        # Per-round Newton-IRLS statistics, cleared after each refit. With design
        # row x = [margin, e_k], info accumulates w*x*x^T and grad (y - p)*x.
        # Sharing one slope across positions makes that an arrowhead matrix, so
        # only the coupling sum(w*margin) and the intercept block sum(w) are
        # per position, alongside the intercept score sum(y - p).
        self.info = torch.zeros(
            num_speculative_steps, 2, dtype=torch.float32, device=device
        )
        self.grad = torch.zeros(
            num_speculative_steps, dtype=torch.float32, device=device
        )
        # The arrowhead's shared entries: the slope's own information
        # sum(w*margin^2) and its score sum((y - p)*margin), summed over every
        # position rather than kept per position.
        self.totals = torch.zeros(2, dtype=torch.float32, device=device)
        # Per-round observation counts, zeroed after each refit. the cumulative
        # tally decides when the estimate is trustworthy enough to trim on.
        self.counts = torch.zeros(
            num_speculative_steps, dtype=torch.float32, device=device
        )

    @property
    def needs_full_verification(self) -> bool:
        """Whether callers must still verify whole draft blocks."""
        return self._refits < self.NUM_WARMUP_REFITS

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
            self.totals,
            idx_mapping,
            num_sampled,
            num_rejected,
            self.margins,
            self.margins.stride(0),
            self.predictions,
            self.predictions.stride(0),
            self.counts,
            num_reqs,
            BLOCK_R=triton.next_power_of_2(max(num_reqs, 1)),
        )

        self._steps_since_refit += 1
        if self._steps_since_refit < self.REFIT_INTERVAL:
            return
        self._steps_since_refit = 0

        if self.DEBUG_CALIBRATION:
            # grad accumulates sum(label - pred) over the round and counts
            # accumulates the number of observations, so their ratio is the
            # signed calibration gap: positive means the round under-predicted
            # acceptance, negative means it over-predicted. Both are already on
            # hand, so the diagnostic needs no extra accumulator.
            n = self.counts.tolist()
            gap = (self.grad / self.counts.clamp(min=1.0)).tolist()
            w = self.coefficients[0].tolist()
            b = self.coefficients[1].tolist()
            logger.info(
                "acceptance calib refit=%d | %s",
                self._refits,
                " | ".join(
                    f"k{k}: n={n[k]:.0f} gap={gap[k]:+.3f} w={w[k]:.3f} b={b[k]:.2f}"
                    f" damp={n[k] / (n[k] + self.DAMPING_OBSERVATIONS):.2f}"
                    for k in range(self.num_speculative_steps)
                ),
            )

        # Fit the coefficients to the accumulated statistics gathered over the
        # course of the last REFIT_INTERVAL steps.
        _refit_kernel[(1,)](
            self.coefficients,
            self.coefficients.stride(0),
            self.info,
            self.info.stride(0),
            self.grad,
            self.totals,
            self.counts,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            L2=self.L2,
            DAMPING=self.DAMPING_OBSERVATIONS,
            INV_TP=1.0 / self._tp_size,
            BLOCK=triton.next_power_of_2(self.num_speculative_steps),
        )
        if self._tp_size > 1:
            # All-reduce so that all ranks hold identical coefficients. _refit_kernel
            # already scaled them by 1/tp_size, so this sum is the mean.
            self.coefficients.copy_(tensor_model_parallel_all_reduce(self.coefficients))

        self._refits += 1
        if self._refits == self.NUM_WARMUP_REFITS:
            logger.info(
                "Acceptance estimator fitted after %d steps. Adaptive "
                "verification is now active.",
                self._refits * self.REFIT_INTERVAL,
            )

    def predict(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
        confidence_probs: torch.Tensor,
        draft_tokens: torch.Tensor | None = None,
    ) -> None:
        if self.use_log_q:
            assert draft_tokens is not None, (
                "log q feature needs the drafted tokens; score after sampling"
            )
        num_tokens, vocab_size = logits.shape
        _predict_kernel[(num_tokens,)](
            self.margins,
            self.margins.stride(0),
            self.predictions,
            self.predictions.stride(0),
            confidence_probs,
            confidence_probs.stride(0),
            self.coefficients,
            self.coefficients.stride(0),
            logits,
            logits.stride(0),
            idx_mapping,
            # DSpark scores one draft position at a time, so its mapping arrives
            # as a strided column of the (req, step) sample mapping.
            idx_mapping.stride(0),
            draft_step,
            draft_tokens,
            num_tokens,
            vocab_size,
            per_token_step=draft_step.dim() > 0,
            USE_LOG_Q=self.use_log_q,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            MIN_LOGIT=_MIN_LOGIT,
            MAX_MARGIN=_MAX_MARGIN,
            MIN_NEG_LOG_Q=_MIN_NEG_LOG_Q,
            BLOCK_SIZE=8192,
            num_warps=8,
        )
