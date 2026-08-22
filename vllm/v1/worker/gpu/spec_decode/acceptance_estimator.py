# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online per-position acceptance estimation for adaptive verification.

Adaptive verification needs, for every drafted token, the probability that the
target will accept it. DSpark checkpoints carry a trained confidence head for
this; no other speculator does. This module estimates it instead from the shape
of the draft distribution the speculator already computes, and calibrates that
estimate at runtime against acceptance actually observed during serving.

The feature is ``logit(q)`` for the drafted token: its temperature-scaled logit
minus the log-sum-exp of every *other* token in the row. That is the
top-1-vs-rest generalization of a top-2 logit margin -- the same quantity taken
over the whole vocabulary rather than just the runner-up -- and unlike a margin
it describes the token actually drawn, which under probabilistic drafting is
often not the mode. Held out on DeepSeek-V4-Flash-DSpark at temperature 1.0 it
reaches AUC 0.847, against 0.823 for that checkpoint's trained confidence head
and 0.805 for the margin.

Computing it needs the partition function, so the normalizer is reduced in two
stages: ``_local_max_sumexp_kernel`` takes a per-(token, vocab block) max and
sum of exponentials, and ``_predict_kernel`` folds those partials into a single
log-sum-exp, dropping the drawn token from the block that holds it. Excluding it
there rather than forming ``log q - log(1 - q)`` keeps the result exact as q
approaches 1.

A per-position logistic maps the feature to acceptance probability::

    p_k = sigmoid(weight * feature + bias[k])

One slope shared across draft positions and an intercept per position, fit by
Newton-IRLS from sufficient statistics accumulated on device. Sharing the slope
is what makes the deep positions usable: they see too few observations per round
to fit two parameters, but an intercept alone is well determined, and the slope
-- how decisively the feature maps to acceptance -- has no reason to vary with depth.
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
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_max_and_sumexp,
)

logger = init_logger(__name__)

# A row carrying all of its mass on the drawn token would emit an unbounded
# feature, so logit(q) is clamped to this at both ends.
_MAX_MARGIN = 40.0
# Vocabulary block for the two-stage normalizer. Matches the draft sampler and
# the rejection sampler, which reduce the same rows the same way.
_REDUCTION_BLOCK_SIZE = 1024


@triton.jit
def _accumulate_kernel(
    info_ptr,
    grad_ptr,
    totals_ptr,
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

    observed = req_mask & (step <= num_accepted) & (step < num_admitted)
    mask = tl.where(observed, 1.0, 0.0)
    label = tl.where(step < num_accepted, 1.0, 0.0)
    w = pred * (1.0 - pred) * mask
    resid = (label - pred) * mask

    # Weighted normal-equation pieces for the design row x = [feature, e_k]: the
    # information matrix XtWX and the score Xt(y - p). The slope is shared across
    # positions, so XtWX is an arrowhead whose slope-slope entry sum(w*feature^2)
    # and slope score sum(resid*feature) are single totals; only the coupling
    # sum(w*feature), the intercept block sum(w), and the intercept score
    # sum(resid) are per position. Keeping those two as scalars rather than per
    # position is what lets _refit_kernel skip the per-position 2x2 solves.
    xtwx_01 = tl.sum(w * feature, axis=0)
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
    tl.atomic_add(totals_ptr + 0, tl.sum(w * feature * feature, axis=0), sem="relaxed")
    tl.atomic_add(totals_ptr + 1, tl.sum(resid * feature, axis=0), sem="relaxed")


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

    The design row is x = [feature, e_k], so XtWX is an arrowhead: a dense first
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

    # Per position: the slope/intercept coupling sum(w*feature) and the intercept
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
def _local_max_sumexp_kernel(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    idx_mapping_stride,
    temperature_ptr,
    num_tokens,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-(token, vocab block) max and sum of exponentials of the draft logits.

    One program per vocabulary block rather than one per row, so the reduction
    spreads across the machine instead of looping inside a single program.
    ``_predict_kernel`` folds the partials into one normalizer.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    if token_idx >= num_tokens:
        return
    max_out = local_max_ptr + token_idx * local_max_stride + block_idx
    sumexp_out = local_sumexp_ptr + token_idx * local_sumexp_stride + block_idx

    req_state_idx = tl.load(idx_mapping_ptr + token_idx * idx_mapping_stride).to(
        tl.int64
    )
    if req_state_idx < 0:
        # Cudagraph-padded requests carry -1. Write an empty partial so that the
        # combine sees nothing rather than a previous step's values.
        tl.store(max_out, float("-inf"))
        tl.store(sumexp_out, 0.0)
        return

    # Draft logits are pre-temperature, and acceptance is decided on the scaled
    # distribution the drafter actually sampled from, so scale first. A request
    # with temp == 0 drafts greedily; leave its logits alone rather than dividing
    # by zero, which is what the sampler does.
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
    block_max, block_sumexp = _compute_max_and_sumexp(logits)
    tl.store(max_out, block_max)
    tl.store(sumexp_out, block_sumexp)


@triton.jit
def _predict_kernel(
    features_ptr,
    features_stride,
    pred_ptr,
    pred_stride,
    conf_ptr,
    conf_stride,
    coef_ptr,
    coef_stride,
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    idx_mapping_stride,
    step_ptr,
    tokens_ptr,
    temperature_ptr,
    num_tokens,
    vocab_num_blocks,
    per_token_step: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MAX_MARGIN: tl.constexpr,
    RED_BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """Turn this step's draft logits into an acceptance probability.

    The feature is logit(q) for the drafted token: its scaled logit minus the
    log-sum-exp of every *other* token in the row. That is the top-1-vs-rest
    generalization of a top-2 logit feature -- the same quantity, over the whole
    vocabulary instead of the runner-up -- and it describes the token actually
    drawn, which under probabilistic drafting is often not the mode.

    Excluding the drawn token from its own block's partial keeps this exact
    where log q - log(1 - q) would cancel as q approaches 1.
    """
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

    token = tl.load(tokens_ptr + token_idx).to(tl.int64)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    temp = tl.where(temp > 0.0, temp, 1.0)
    sampled = (
        tl.load(logits_ptr + token_idx * logits_stride + token).to(tl.float32) / temp
    )

    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + token_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + token_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    # Drop the drawn token from the block that holds it, leaving the mass of
    # everything else. Only that one partial changes, so precision is lost only
    # where the token dominates its own block -- and there the feature is past
    # MAX_MARGIN already.
    token_block = (token // RED_BLOCK_SIZE).to(tl.int32)
    sumexps = tl.where(
        blocks == token_block, sumexps - tl.exp(sampled - maxes), sumexps
    )
    global_max = tl.max(maxes, axis=0)
    rest = tl.sum(
        tl.where(blocks_mask, sumexps * tl.exp(maxes - global_max), 0.0), axis=0
    )
    # Rounding can leave this a hair below zero. log(0) sends the feature to
    # +inf, which the clamp turns into MAX_MARGIN -- where a row carrying all of
    # its mass on the drawn token belongs.
    rest = tl.maximum(rest, 0.0)
    feature = sampled - global_max - tl.log(rest)
    feature = tl.minimum(tl.maximum(feature, -MAX_MARGIN), MAX_MARGIN)
    tl.store(features_ptr + req_state_idx * features_stride + step, feature)

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

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ):
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        # Scratch for the two-stage normalizer, allocated on the first predict
        # once the vocabulary size is known. See _reduction_buffers.
        self._local_max: torch.Tensor | None = None
        self._local_sumexp: torch.Tensor | None = None
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
        self.features = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )
        # Predictions made at draft time, in the same stable slots as the features
        # they came from, because the label that grades them only arrives on the
        # next step, by which point the batch has been reordered. Reused as the
        # IRLS weight then.
        self.predictions = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )

        # Per-round Newton-IRLS statistics, cleared after each refit. With design
        # row x = [feature, e_k], info accumulates w*x*x^T and grad (y - p)*x.
        # Sharing one slope across positions makes that an arrowhead matrix, so
        # only the coupling sum(w*feature) and the intercept block sum(w) are
        # per position, alongside the intercept score sum(y - p).
        self.info = torch.zeros(
            num_speculative_steps, 2, dtype=torch.float32, device=device
        )
        self.grad = torch.zeros(
            num_speculative_steps, dtype=torch.float32, device=device
        )
        # The arrowhead's shared entries: the slope's own information
        # sum(w*feature^2) and its score sum((y - p)*feature), summed over every
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
            self.features,
            self.features.stride(0),
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

    def _reduction_buffers(self, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Scratch for the per-(token, vocab block) normalizer partials.

        Allocated on the first call, once the vocabulary size is known, and
        reused after: ``predict`` runs inside a captured graph, so a later
        allocation would not be part of it. Rows cover the widest launch, one
        per (request, draft step).
        """
        if self._local_max is None:
            rows = self.features.shape[0] * self.num_speculative_steps
            self._local_max = torch.empty(
                rows, num_blocks, dtype=torch.float32, device=self.device
            )
            self._local_sumexp = torch.empty_like(self._local_max)
        assert self._local_max.shape[1] == num_blocks, (
            f"vocabulary size changed after the first predict: "
            f"{self._local_max.shape[1]} != {num_blocks} blocks"
        )
        return self._local_max, self._local_sumexp

    def predict(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
        confidence_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        num_tokens, vocab_size = logits.shape
        num_blocks = triton.cdiv(vocab_size, _REDUCTION_BLOCK_SIZE)
        local_max, local_sumexp = self._reduction_buffers(num_blocks)
        # DSpark scores one draft position at a time, so its mapping arrives as a
        # strided column of the (req, step) sample mapping.
        idx_mapping_stride = idx_mapping.stride(0)
        _local_max_sumexp_kernel[(num_tokens, num_blocks)](
            local_max,
            local_max.stride(0),
            local_sumexp,
            local_sumexp.stride(0),
            logits,
            logits.stride(0),
            idx_mapping,
            idx_mapping_stride,
            temperature,
            num_tokens,
            vocab_size,
            BLOCK_SIZE=_REDUCTION_BLOCK_SIZE,
        )
        _predict_kernel[(num_tokens,)](
            self.features,
            self.features.stride(0),
            self.predictions,
            self.predictions.stride(0),
            confidence_probs,
            confidence_probs.stride(0),
            self.coefficients,
            self.coefficients.stride(0),
            local_max,
            local_max.stride(0),
            local_sumexp,
            local_sumexp.stride(0),
            logits,
            logits.stride(0),
            idx_mapping,
            idx_mapping_stride,
            draft_step,
            draft_tokens,
            temperature,
            num_tokens,
            num_blocks,
            per_token_step=draft_step.dim() > 0,
            NUM_SPECULATIVE_STEPS=self.num_speculative_steps,
            MAX_MARGIN=_MAX_MARGIN,
            RED_BLOCK_SIZE=_REDUCTION_BLOCK_SIZE,
            PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        )
