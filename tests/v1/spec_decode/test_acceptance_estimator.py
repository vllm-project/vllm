# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``predict`` writes one probability into two index spaces at once.

Adaptive verification reads the estimate in this step's batch order, while the
next step's refit reads it in persistent request-state slots. A mix-up between
the two is silent -- every value stays a plausible probability, it just belongs
to another request -- so it is asserted here rather than left to acceptance
metrics to hint at.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for acceptance estimator tests", allow_module_level=True)

MAX_NUM_REQS = 16
NUM_STEPS = 4
VOCAB_SIZE = 2048


def _make_estimator(device: torch.device) -> OnlineAcceptanceEstimator:
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    # Distinct per-position coefficients, so reading the wrong column shows up.
    estimator.coefficients[0] = torch.tensor([0.3, 0.5, 0.7, 0.9], device=device)
    estimator.coefficients[1] = torch.tensor([1.5, 1.0, 0.5, 0.0], device=device)
    return estimator


def _expected_probs(
    estimator: OnlineAcceptanceEstimator,
    logits: torch.Tensor,
    steps: torch.Tensor,
) -> torch.Tensor:
    top2 = logits.float().topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(max=40.0)
    weight, bias = estimator.coefficients[0], estimator.coefficients[1]
    return torch.sigmoid(weight[steps] * margin + bias[steps])


@pytest.mark.parametrize("tokens_per_req", [1, NUM_STEPS])
def test_predict_agrees_across_slot_and_batch_order(tokens_per_req: int):
    """One row per request (per-step drafting) and a whole block at once.

    Which one it is is inferred from whether the draft step is per-row, so the
    two cases differ only in how ``draft_step`` is shaped.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    estimator = _make_estimator(device)
    num_reqs = 5

    # Batch position i holds a shuffled persistent slot, as a served batch would.
    slots = torch.randperm(MAX_NUM_REQS, device=device)[:num_reqs].to(torch.int32)
    if tokens_per_req == 1:
        idx_mapping = slots
        draft_step = torch.tensor(2, device=device, dtype=torch.int32)
        steps = draft_step.expand(num_reqs)
    else:
        idx_mapping = slots.repeat_interleave(NUM_STEPS)
        draft_step = torch.arange(NUM_STEPS, device=device, dtype=torch.int32).repeat(
            num_reqs
        )
        steps = draft_step
    batch_rows = torch.arange(num_reqs, device=device).repeat_interleave(tokens_per_req)
    logits = torch.randn(idx_mapping.shape[0], VOCAB_SIZE, device=device) * 4.0

    confidence_probs = torch.zeros(MAX_NUM_REQS, NUM_STEPS, device=device)
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs)

    expected = _expected_probs(estimator, logits, steps.long())
    torch.testing.assert_close(
        estimator.predictions[idx_mapping.long(), steps.long()],
        expected,
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        confidence_probs[batch_rows, steps.long()], expected, atol=2e-3, rtol=2e-3
    )


def test_predict_skips_cudagraph_padded_rows():
    """Padded rows carry slot -1 and must not land on a live request."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    estimator = _make_estimator(device)
    num_reqs, num_padded = 3, 6

    slots = torch.tensor([7, 2, 11], device=device, dtype=torch.int32)
    idx_mapping = torch.cat(
        [slots, torch.full((num_padded,), -1, device=device, dtype=torch.int32)]
    )
    draft_step = torch.tensor(1, device=device, dtype=torch.int32)
    logits = torch.randn(num_reqs + num_padded, VOCAB_SIZE, device=device) * 4.0

    confidence_probs = torch.zeros(MAX_NUM_REQS, NUM_STEPS, device=device)
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs)

    live = (slots.long(), torch.full((num_reqs,), 1, device=device))
    expected = _expected_probs(estimator, logits[:num_reqs], live[1])
    torch.testing.assert_close(
        estimator.predictions[live], expected, atol=2e-3, rtol=2e-3
    )
    # Nothing outside the live (request, position) cells was touched.
    untouched = torch.ones(MAX_NUM_REQS, NUM_STEPS, dtype=torch.bool, device=device)
    untouched[live] = False
    assert not estimator.predictions[untouched].any()
    assert not confidence_probs[num_reqs:].any()


# A refit takes one undamped Newton step, so it needs a realistic number of
# observations per round to be well conditioned -- a handful of requests can
# send the step far past the optimum. This is the batch a served step sees.
REFIT_NUM_REQS = 512


def _drive_round(estimator, margins, num_accepted, slots):
    """Feed one step's graded drafts through predict's buffers into `step`."""
    estimator.margins[slots.long()] = margins
    estimator.predictions[slots.long()] = torch.sigmoid(
        estimator.coefficients[0] * margins + estimator.coefficients[1]
    )
    estimator.step(
        slots,
        (num_accepted + 1).to(torch.int32),
        (margins.shape[1] - num_accepted).to(torch.int32),
    )


def test_refit_recovers_a_shared_slope_and_per_position_intercepts():
    """The fit must find the logistic that generated the observations.

    One slope is shared across draft positions and each keeps its own intercept,
    so the arrowhead solve is only correct if it recovers both from data whose
    positions differ in level but not in slope.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    true_slope = 0.6
    true_bias = torch.tensor([1.4, 0.9, 0.4, -0.1], device=device)

    estimator = OnlineAcceptanceEstimator(REFIT_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 5
    slots = torch.arange(REFIT_NUM_REQS, device=device, dtype=torch.int32)
    for _ in range(60):
        margins = torch.rand(REFIT_NUM_REQS, NUM_STEPS, device=device) * 8.0
        accepted = torch.rand_like(margins) < torch.sigmoid(
            true_slope * margins + true_bias
        )
        # Verification stops at the first rejection, so a position's label is
        # conditional on every shallower one having been accepted.
        num_accepted = (~accepted).float().argmax(dim=1)
        num_accepted[accepted.all(dim=1)] = NUM_STEPS
        _drive_round(estimator, margins, num_accepted, slots)

    slope, bias = estimator.coefficients[0], estimator.coefficients[1]
    assert torch.allclose(slope, slope[0]), "the slope must be shared"
    torch.testing.assert_close(
        slope[0], torch.tensor(true_slope, device=device), atol=0.1, rtol=0
    )
    torch.testing.assert_close(bias, true_bias, atol=0.3, rtol=0)


def test_refit_survives_a_position_with_no_observations():
    """A draft position nobody reached must not stall the shared slope.

    Its statistics stay zero while the others accumulate, which is the degenerate
    case the arrowhead denominator has to tolerate.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    estimator = OnlineAcceptanceEstimator(REFIT_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 5
    slots = torch.arange(REFIT_NUM_REQS, device=device, dtype=torch.int32)
    before = estimator.coefficients.clone()

    # Every request is rejected at position 1, so positions 2+ are never graded.
    num_accepted = torch.ones(REFIT_NUM_REQS, device=device)
    for _ in range(20):
        margins = torch.rand(REFIT_NUM_REQS, NUM_STEPS, device=device) * 8.0
        _drive_round(estimator, margins, num_accepted, slots)

    assert estimator.coefficients.isfinite().all()
    assert not torch.equal(estimator.coefficients[0], before[0]), "slope never moved"
    # Positions 2 and 3 were never observed, so their intercepts must be untouched.
    torch.testing.assert_close(
        estimator.coefficients[1][2:], before[1][2:], rtol=0, atol=0
    )


def test_step_damping_scales_with_the_evidence():
    """A round moves a coefficient in proportion to the samples behind it.

    The Newton step itself barely changes with how much data a round carries --
    curvature and score both scale with it -- so scaling a round's statistics
    isolates the damping factor n / (n + DAMPING_OBSERVATIONS). The slope/
    intercept coupling is zeroed so the two parameters can be checked apart.
    """
    device = torch.device("cuda")
    damping = OnlineAcceptanceEstimator.DAMPING_OBSERVATIONS
    per_position_n = 20.0

    def refit_with(scale):
        estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
        estimator.info[:, 0] = 0.0  # sum(w*margin): no slope/intercept coupling
        estimator.info[:, 1] = 1.0 * scale  # sum(w)
        estimator.grad[:] = 0.5 * scale  # sum(y - p)
        estimator.totals[0] = 30.0 * scale  # sum(w*margin^2)
        estimator.totals[1] = 2.0 * scale  # sum((y - p)*margin)
        estimator.counts[:] = per_position_n * scale
        estimator._steps_since_refit = estimator.REFIT_INTERVAL - 1
        before = estimator.coefficients.clone()
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        estimator.step(empty, empty, empty)
        delta = estimator.coefficients - before
        return delta[0][0].item(), delta[1][0].item()

    for scale in (1.0, 10.0, 100.0):
        total_n = per_position_n * scale * NUM_STEPS
        slope_delta, bias_delta = refit_with(scale)
        assert slope_delta == pytest.approx(
            (2.0 / 30.0) * total_n / (total_n + damping), rel=1e-3
        )
        assert bias_delta == pytest.approx(
            0.5 * (per_position_n * scale) / (per_position_n * scale + damping),
            rel=1e-3,
        )

    # A round with no observations at all must leave the coefficients alone.
    assert refit_with(0.0) == (0.0, 0.0)
