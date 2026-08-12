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

    p_k = sigmoid(weight[k] * margin + bias[k])

Two parameters per draft position, fit by Newton-IRLS from sufficient statistics
accumulated on device. Observations are never stored: each is folded straight
into a 2x2 information matrix and a 2-vector gradient per position, so the
entire learning state is a few dozen floats regardless of traffic.
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

# Logits are floored here rather than at -inf: a vocab shard that is entirely
# padding would otherwise make the running-max rescale compute (-inf) - (-inf).
_NEG = -1.0e30
# A row with a single finite entry would emit an unbounded margin.
_MAX_MARGIN = 40.0


@triton.jit
def _top2_margin_kernel(
    logits_ptr,
    logits_row_stride,
    slot_ptr,
    step_ptr,
    out_ptr,
    out_row_stride,
    num_tokens,
    vocab_size,
    per_token_step: tl.constexpr,
    NEG: tl.constexpr,
    MAX_MARGIN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Top-2 logit margin per token, one streaming pass over the vocab.

    Scatters into ``out[slot, step]`` so the result is keyed by persistent
    request-state index, matching how draft state is carried across steps.
    """
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return
    row = logits_ptr + token * logits_row_stride

    m1 = NEG
    m2 = NEG
    for start in tl.range(0, vocab_size, BLOCK_SIZE):
        block = start + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(row + block, mask=mask, other=NEG).to(tl.float32)
        logits = tl.where(mask, tl.maximum(logits, NEG), NEG)

        b1 = tl.max(logits, axis=0)
        # A tied maximum means the margin is genuinely zero, so count duplicates
        # rather than masking one out by index.
        num_at_max = tl.sum((logits == b1).to(tl.int32), axis=0)
        b2_below = tl.max(tl.where(logits < b1, logits, NEG), axis=0)
        b2 = tl.where(num_at_max > 1, b1, b2_below)

        m2 = tl.where(b1 > m1, tl.maximum(m1, b2), tl.maximum(m2, b1))
        m1 = tl.maximum(m1, b1)

    slot = tl.maximum(tl.load(slot_ptr + token).to(tl.int64), 0)
    if per_token_step:
        step = tl.load(step_ptr + token).to(tl.int64)
    else:
        step = tl.load(step_ptr).to(tl.int64)
    tl.store(out_ptr + slot * out_row_stride + step, tl.minimum(m1 - m2, MAX_MARGIN))


@triton.jit
def _observe_kernel(
    idx_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    margin_ptr,
    pred_ptr,
    slot_row_stride,
    info_ptr,
    grad_ptr,
    counts_ptr,
    num_reqs,
    BLOCK_R: tl.constexpr,
):
    """Fold one step's graded drafts into the per-position IRLS statistics.

    One program per draft position, so each owns its accumulator slots outright
    and needs no atomics. Written as a single kernel because the elementwise
    formulation costs ~36 launches per step, which at these tensor sizes is all
    dispatch overhead and no arithmetic.
    """
    step = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    active = r < num_reqs

    slot = tl.load(idx_ptr + r, mask=active, other=0).to(tl.int64)
    slot = tl.maximum(slot, 0)
    # num_sampled is accepted + 1 bonus; num_rejected is admitted - accepted.
    num_accepted = tl.load(num_sampled_ptr + r, mask=active, other=0).to(tl.int64) - 1
    num_accepted = tl.maximum(num_accepted, 0)
    num_admitted = num_accepted + tl.load(
        num_rejected_ptr + r, mask=active, other=0
    ).to(tl.int64)

    off = slot * slot_row_stride + step
    margin = tl.load(margin_ptr + off, mask=active, other=0.0).to(tl.float32)
    pred = tl.load(pred_ptr + off, mask=active, other=0.0).to(tl.float32)

    # Observed only while the chain reached this position and the position was
    # actually verified; that conditioning is what the survival product wants.
    #
    # num_admitted also covers slot recycling, so no separate validity bit is
    # needed: a newly admitted request has nothing to verify yet, because drafts
    # must be proposed before they can be graded, and by the time it does the
    # draft step has already overwritten the slot's margins with its own.
    observed = active & (step <= num_accepted) & (step < num_admitted)
    m = tl.where(observed, 1.0, 0.0)
    label = tl.where(step < num_accepted, 1.0, 0.0)
    w = pred * (1.0 - pred) * m
    resid = (label - pred) * m

    tl.store(
        info_ptr + step * 4 + 0,
        tl.load(info_ptr + step * 4 + 0) + tl.sum(w * margin * margin, axis=0),
    )
    cross = tl.sum(w * margin, axis=0)
    tl.store(info_ptr + step * 4 + 1, tl.load(info_ptr + step * 4 + 1) + cross)
    tl.store(info_ptr + step * 4 + 2, tl.load(info_ptr + step * 4 + 2) + cross)
    tl.store(
        info_ptr + step * 4 + 3, tl.load(info_ptr + step * 4 + 3) + tl.sum(w, axis=0)
    )
    tl.store(
        grad_ptr + step * 2 + 0,
        tl.load(grad_ptr + step * 2 + 0) + tl.sum(resid * margin, axis=0),
    )
    tl.store(
        grad_ptr + step * 2 + 1,
        tl.load(grad_ptr + step * 2 + 1) + tl.sum(resid, axis=0),
    )
    tl.store(counts_ptr + step, tl.load(counts_ptr + step) + tl.sum(m, axis=0))


def compute_top2_margin(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    draft_step: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Scatter per-token top-2 logit margins into ``out[slot, step]``.

    Args:
        logits: [num_tokens, vocab] draft logits for this call. Autoregressive
            speculators pass one position for every request; block speculators
            pass every (request, position) pair flattened.
        idx_mapping: [num_tokens] persistent request-state index per token.
        draft_step: scalar or [num_tokens] draft position per token, exactly as
            handed to ``gumbel_sample`` as ``logits_cache_col``.
        out: [max_num_reqs, num_steps] float32 destination.
    """
    num_tokens, vocab_size = logits.shape
    _top2_margin_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        idx_mapping,
        draft_step,
        out,
        out.stride(0),
        num_tokens,
        vocab_size,
        per_token_step=draft_step.dim() > 0,
        NEG=_NEG,
        MAX_MARGIN=_MAX_MARGIN,
        # Tuned on GB200 at [448, 129280] fp32: 51us / 4570 GB/s, i.e. bandwidth
        # bound, versus 61us at 4096 and 55us at 16384.
        BLOCK_SIZE=8192,
        num_warps=8,
    )


class OnlineAcceptanceEstimator:
    """Predicts per-position acceptance, and calibrates itself while serving.

    Lifecycle per step, driven by ``DraftModelSpeculator``:

    1. ``step`` folds the previous step's drafts, now graded by the target,
       into the IRLS accumulators, and periodically solves for new coefficients,
       independently but identically on every rank.
    2. ``predict`` runs inside the captured draft graph, turning this step's
       margins into acceptance probabilities for adaptive verification.

    For the first few refits the estimator reports ``needs_full_verification``;
    callers verify whole draft blocks during that window so the labels it learns
    from are not censored by its own trimming.
    """

    REFIT_INTERVAL = 100
    # Full blocks are verified for this many refits before trimming starts.
    # Gating on a step count rather than an observation count keeps the decision
    # host-side: reading accumulated counts would need a sync on the serving
    # path. Held-out AUC saturates around 250 observations per position, which
    # even the rarest position clears well inside this window at any useful
    # concurrency.
    WARMUP_REFITS = int(os.getenv("VLLM_AE_WARMUP_REFITS", "3"))
    # A position with fewer observations than this in a round is left alone
    # rather than fit from noise.
    MIN_ROUND_OBSERVATIONS = 50
    # Ridge on the 2x2 solve.
    L2 = 1e-3

    def __init__(
        self,
        max_num_reqs: int,
        num_steps: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
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
        self.weight = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.bias = torch.full((num_steps,), 1.5, dtype=torch.float32, device=device)

        # Per-slot state carried from the draft step to the verify step.
        self.slot_margin = torch.zeros(
            max_num_reqs, num_steps, dtype=torch.float32, device=device
        )
        # Prediction made at draft time: gathered into batch order for the
        # verification manager, and reused as the IRLS weight when the label
        # arrives on the next step.
        self.slot_pred = torch.zeros(
            max_num_reqs, num_steps, dtype=torch.float32, device=device
        )

        # Sufficient statistics. Design row is [margin, 1].
        self.info = torch.zeros(num_steps, 2, 2, dtype=torch.float32, device=device)
        self.grad = torch.zeros(num_steps, 2, dtype=torch.float32, device=device)
        # Per-round observation counts, zeroed at each refit; the cumulative
        # tally decides when the estimate is trustworthy enough to trim on.
        self.counts = torch.zeros(num_steps, dtype=torch.float32, device=device)

        self._steps = torch.arange(num_steps, device=device)

    @property
    def needs_full_verification(self) -> bool:
        """Whether callers must still verify whole draft blocks."""
        return self._refits < self.WARMUP_REFITS

    def step(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        _observe_kernel[(self.num_steps,)](
            idx_mapping,
            num_sampled,
            num_rejected,
            self.slot_margin,
            self.slot_pred,
            self.slot_margin.stride(0),
            self.info,
            self.grad,
            self.counts,
            num_reqs,
            BLOCK_R=triton.next_power_of_2(max(num_reqs, 1)),
        )

        self._steps_since_refit += 1
        if self._steps_since_refit < self.REFIT_INTERVAL:
            return
        self._steps_since_refit = 0

        # One Newton step per position from this round alone. Curvature and
        # gradient must describe the same data at the same coefficients: a
        # gradient carried across a step points where we have already gone, and
        # stepping on it again overshoots.
        a = self.info[:, 0, 0] + self.L2
        b = self.info[:, 0, 1]
        c = self.info[:, 1, 1] + self.L2
        det = a * c - b * b
        g0, g1 = self.grad[:, 0], self.grad[:, 1]
        step_w = (c * g0 - b * g1) / det
        step_b = (a * g1 - b * g0) / det
        ok = torch.isfinite(det) & (det.abs() > 1e-12)
        ok &= torch.isfinite(step_w) & torch.isfinite(step_b)
        # A round too thin to fit -- a rare deep position at low concurrency --
        # keeps the coefficients it has rather than lurching on noise.
        ok &= self.counts >= self.MIN_ROUND_OBSERVATIONS
        new_w = torch.where(ok, self.weight + step_w, self.weight)
        new_b = torch.where(ok, self.bias + step_b, self.bias)
        # A negative slope would say a more decisive draft is less likely to be
        # accepted; treat it as a bad fit rather than trusting it.
        keep = new_w >= 0.0
        new_w = torch.where(keep, new_w, self.weight)
        new_b = torch.where(keep, new_b, self.bias)

        if self._tp_size > 1:
            # Ranks must hold identical coefficients: they set the budget, hence
            # the token count, hence the cudagraph shape and which collectives
            # every rank issues. Disagreement is not a slightly worse estimate,
            # it is a hang, and the accumulators are observed to drift apart
            # even though the draft logits feeding them are bit-identical.
            #
            # A device all-reduce in stream order, unconditional and at the same
            # point on every rank -- unlike broadcast_object, which travels the
            # message queue the executor uses for its own RPCs and deadlocks it.
            averaged = (
                tensor_model_parallel_all_reduce(
                    torch.stack((new_w, new_b)).contiguous()
                )
                / self._tp_size
            )
            new_w, new_b = averaged[0], averaged[1]

        self.weight.copy_(new_w)
        self.bias.copy_(new_b)
        # Start the next round clean: each refit is a fit over its own window,
        # which is also what makes the estimator track workload drift.
        self.info.zero_()
        self.grad.zero_()
        self.counts.zero_()

        self._refits += 1
        if self._refits == self.WARMUP_REFITS:
            logger.info(
                "Acceptance estimator fitted after %d steps; adaptive "
                "verification now trimming.",
                self._refits * self.REFIT_INTERVAL,
            )

    def predict(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
    ) -> None:
        compute_top2_margin(logits, idx_mapping, draft_step, self.slot_margin)
        torch.sigmoid(self.slot_margin * self.weight + self.bias, out=self.slot_pred)

    def gather_acceptance_probs(
        self,
        idx_mapping: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        out[:num_reqs] = self.slot_pred[idx_mapping.long()]
