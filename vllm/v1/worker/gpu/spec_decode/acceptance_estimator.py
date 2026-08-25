# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
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

# The feature is a log-odds, log(q / (1 - q)), which a row carrying all of its
# mass on one token sends to infinity. Clamp it symmetrically.
_MAX_LOG_ODDS = 40.0


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
        tl.store(local_sumexp_ptr + token_idx * local_sumexp_stride + block_idx, 0.0)
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
    block_max, block_sumexp = _compute_max_and_sumexp(logits)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, block_max)
    tl.store(
        local_sumexp_ptr + token_idx * local_sumexp_stride + block_idx, block_sumexp
    )


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
    sumexps = tl.load(
        local_sumexp_ptr + token_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )

    # The feature, logit(q(m)) for the row's most likely token m, is:
    #   q(m) = e^m / Σ_z e^z
    #   logit(q(m)) = log(q(m) / (1 - q(m)))
    #               = m - log(Σ_{z≠m} e^z).
    # It scores the draft *distribution*, not the token drawn from it, so it is
    # a function of the prefix alone. Trimming a draft therefore never
    # conditions on the token being trimmed, and rejection sampling stays
    # lossless. Scoring the drawn token instead would select on the proposal
    # and bias the emitted distribution toward the drafter's confident modes.
    global_max = tl.max(maxes, axis=0)

    # Rescale each block's partial to the global max and sum. The most likely
    # token contributes exactly e^(m - m) = 1, so removing it leaves the sum
    # over every other token.
    total_sumexp = tl.sum(
        tl.where(blocks_mask, sumexps * tl.exp(maxes - global_max), 0.0), axis=0
    )
    complement_sumexp = tl.maximum(total_sumexp - 1.0, 0.0)

    # Shifted by m, the numerator is e^0 = 1, so the feature is just the
    # negated complement log-sum-exponential. Clamp to the min/max log-odds.
    feature = -tl.log(complement_sumexp)
    feature = tl.minimum(tl.maximum(feature, -MAX_LOG_ODDS), MAX_LOG_ODDS)
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
    NUM_WARMUP_REFITS = envs.VLLM_ACCEPTANCE_ESTIMATOR_NUM_WARMUP_REFITS
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
        # The trailing dimension is rounded up to a multiple of 4 floats so the
        # cross-rank all-reduce below is legal: FlashInfer's fused all-reduce,
        # the default since #52998, requires it, and num_speculative_steps
        # generally is not (5 for DeepSeek-V4-Flash). Every kernel masks its
        # accesses to the real columns, so the padding stays zero and adds
        # nothing to the sum.
        self.coefficients = torch.zeros(
            2, -(-num_speculative_steps // 4) * 4, dtype=torch.float32, device=device
        )
        self.coefficients[1, :num_speculative_steps].fill_(1.5)

        # Holds logit(q_sampled), which is used as the feature for the logistic.
        # Stored in stable slots keyed by persistent request-state index.
        self.features = torch.zeros(
            max_num_reqs, num_speculative_steps, dtype=torch.float32, device=device
        )
        # Predictions made at draft time, in the same stable slots as the features
        # they came from, because the label that grades them only arrives on the
        # next step, by which point the batch has been reordered. Used to derive
        # the IRLS weight and residual.
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
        # Number of graded drafts per position in the current round, zeroed at each
        # refit. _refit_kernel damps by n / (n + DAMPING_OBSERVATIONS): every intercept
        # by its own count, the shared slope by the round's total.
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
        local_sumexp = torch.empty_like(local_max)
        _local_max_sumexp_kernel[(num_tokens, num_blocks)](
            local_max,
            local_max.stride(0),
            local_sumexp,
            local_sumexp.stride(0),
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
            self.coefficients,
            self.coefficients.stride(0),
            local_max,
            local_max.stride(0),
            local_sumexp,
            local_sumexp.stride(0),
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
