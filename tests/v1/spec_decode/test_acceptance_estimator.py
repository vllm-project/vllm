# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`predict` writes one probability into two index spaces at once.

Adaptive verification reads the estimate in this step's batch order, while the
next step's refit reads it in persistent request-state slots. A mix-up between
the two is silent -- every value stays a plausible probability, it just belongs
to another request -- so it is asserted here rather than left to acceptance
metrics to hint at.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

from vllm.platforms import current_platform  # noqa: E402
from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (  # noqa: E402
    _MAX_LOG_ODDS,
    OnlineAcceptanceEstimator,
    _accumulate_kernel,
    _refit_kernel,
)

if not current_platform.is_cuda_alike() and not current_platform.is_xpu():
    pytest.skip(
        "CUDA/XPU required for acceptance estimator tests", allow_module_level=True
    )

MAX_NUM_REQS = 16
NUM_STEPS = 4
VOCAB_SIZE = 2048

# predict() reduces the vocabulary in blocks of this width and then combines the
# per-block partials. VOCAB_SIZE is narrower than one block, so tests using it
# never exercise that combine; the multi-block tests below size their vocabulary
# from this instead.
VOCAB_BLOCK_SIZE = 4096


def _make_estimator(device: torch.device) -> OnlineAcceptanceEstimator:
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    # Distinct per-position intercepts, so reading the wrong one shows up.
    estimator.slope.fill_(0.7)
    estimator.intercepts[:] = torch.tensor([1.5, 1.0, 0.5, 0.0], device=device)
    return estimator


def _expected_probs(
    estimator: OnlineAcceptanceEstimator,
    logits: torch.Tensor,
    steps: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """logit(q_max) by a different route than the kernel takes.

    The kernel shifts by the max and drops the unit term; here the argmax is
    masked out and the complement is reduced with logsumexp, so agreement is
    evidence about the value rather than about a shared formula.
    """
    temp = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
    scaled = logits.double() / temp[:, None].double()
    top = scaled.max(dim=-1)
    rest = scaled.scatter(1, top.indices[:, None], float("-inf"))
    feature = (top.values - torch.logsumexp(rest, dim=-1)).clamp(-40.0, 40.0)
    weight, bias = estimator.slope, estimator.intercepts
    return torch.sigmoid(weight.double() * feature + bias[steps].double()).float()


def _reference_feature(logits: torch.Tensor) -> torch.Tensor:
    """logit(q_max) for one row, in fp64, by a different route than the kernel.

    Masks the top token out and reduces the remainder, so `1 + complement` is
    never formed. The kernel instead excludes the top entry inside its own block
    and rescales the other blocks' partials, so agreement is evidence about the
    value rather than about a shared formula.
    """
    scaled = logits.double()
    top = scaled.max(dim=-1, keepdim=True)
    rest = scaled.scatter(-1, top.indices, float("-inf"))
    return -torch.logsumexp(rest - top.values, dim=-1)[0]


def _predict_one_row(
    estimator: OnlineAcceptanceEstimator, logits: torch.Tensor, device
) -> torch.Tensor:
    """Score a single [1, vocab] row at draft position 0 and return its feature."""
    idx_mapping = torch.zeros(1, device=device, dtype=torch.int32)
    draft_step = torch.tensor(0, device=device, dtype=torch.int32)
    temperature = torch.ones(MAX_NUM_REQS, device=device)
    confidence_probs = torch.zeros(MAX_NUM_REQS, NUM_STEPS, device=device)
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs, temperature)
    return estimator.features[0, 0]


@pytest.mark.parametrize("tokens_per_req", [1, NUM_STEPS])
def test_predict_agrees_across_slot_and_batch_order(tokens_per_req: int):
    """One row per request (per-step drafting) and a whole block at once.

    Which one it is is inferred from whether the draft step is per-row, so the
    two cases differ only in how `draft_step` is shaped.
    """
    device = current_platform.device_type
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
    temperature = torch.rand(MAX_NUM_REQS, device=device) + 0.5
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs, temperature)

    expected = _expected_probs(
        estimator, logits, steps.long(), temperature[idx_mapping.long()]
    )
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
    device = current_platform.device_type
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
    temperature = torch.rand(MAX_NUM_REQS, device=device) + 0.5
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs, temperature)

    live = (slots.long(), torch.full((num_reqs,), 1, device=device))
    expected = _expected_probs(
        estimator, logits[:num_reqs], live[1], temperature[slots.long()]
    )
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


def _drive_round(estimator, features, num_accepted, slots):
    """Feed one step's graded drafts through predict's buffers into `step`."""
    estimator.features[slots.long()] = features
    estimator.predictions[slots.long()] = torch.sigmoid(
        estimator.slope * features + estimator.intercepts
    )
    estimator.step(
        slots,
        (num_accepted + 1).to(torch.int32),
        (features.shape[1] - num_accepted).to(torch.int32),
    )


def test_refit_recovers_a_shared_slope_and_per_position_intercepts():
    """The fit must find the logistic that generated the observations.

    One slope is shared across draft positions and each keeps its own intercept,
    so the arrowhead solve is only correct if it recovers both from data whose
    positions differ in level but not in slope.
    """
    device = current_platform.device_type
    torch.manual_seed(0)
    true_slope = 0.6
    true_bias = torch.tensor([1.4, 0.9, 0.4, -0.1], device=device)

    estimator = OnlineAcceptanceEstimator(REFIT_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 5
    slots = torch.arange(REFIT_NUM_REQS, device=device, dtype=torch.int32)
    for _ in range(60):
        features = torch.rand(REFIT_NUM_REQS, NUM_STEPS, device=device) * 8.0
        accepted = torch.rand_like(features) < torch.sigmoid(
            true_slope * features + true_bias
        )
        # Verification stops at the first rejection, so a position's label is
        # conditional on every shallower one having been accepted.
        num_accepted = (~accepted).float().argmax(dim=1)
        num_accepted[accepted.all(dim=1)] = NUM_STEPS
        _drive_round(estimator, features, num_accepted, slots)

    slope, bias = estimator.slope, estimator.intercepts
    torch.testing.assert_close(
        slope[0], torch.tensor(true_slope, device=device), atol=0.1, rtol=0
    )
    torch.testing.assert_close(bias, true_bias, atol=0.3, rtol=0)


def test_refit_survives_a_position_with_no_observations():
    """A draft position nobody reached must not stall the shared slope.

    Its statistics stay zero while the others accumulate, which is the degenerate
    case the arrowhead denominator has to tolerate.
    """
    device = current_platform.device_type
    torch.manual_seed(0)
    estimator = OnlineAcceptanceEstimator(REFIT_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 5
    slots = torch.arange(REFIT_NUM_REQS, device=device, dtype=torch.int32)
    before_slope = estimator.slope.clone()
    before_intercepts = estimator.intercepts.clone()

    # Every request is rejected at position 1, so positions 2+ are never graded.
    num_accepted = torch.ones(REFIT_NUM_REQS, device=device)
    for _ in range(20):
        features = torch.rand(REFIT_NUM_REQS, NUM_STEPS, device=device) * 8.0
        _drive_round(estimator, features, num_accepted, slots)

    assert estimator.slope.isfinite().all()
    assert estimator.intercepts.isfinite().all()
    assert not torch.equal(estimator.slope, before_slope), "slope never moved"
    # Positions 2 and 3 were never observed, so their intercepts must be untouched.
    torch.testing.assert_close(
        estimator.intercepts[2:], before_intercepts[2:], rtol=0, atol=0
    )


def test_refit_rejects_a_non_finite_slope_step():
    """Discard an infinite slope step while still updating valid intercepts."""
    device = current_platform.device_type
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 1
    # These nonnegative moments lose L2 in fp32, making a - sum(b*b/c) zero.
    estimator.info[:] = 2.0**16
    estimator.grad[:, 0] = 2.0**15 + 1.0
    estimator.grad[:, 1] = 2.0**15
    estimator.counts[:] = 2.0**18
    before_slope = estimator.slope.clone()
    before_intercepts = estimator.intercepts.clone()
    expected_intercepts = before_intercepts + (
        estimator.grad[:, 1] / (estimator.info[:, 2] + estimator.L2)
    ) * (estimator.counts / (estimator.counts + estimator.DAMPING_OBSERVATIONS))

    empty = torch.empty(0, dtype=torch.int32, device=device)
    estimator.step(empty, empty, empty)

    torch.testing.assert_close(estimator.slope, before_slope, rtol=0, atol=0)
    torch.testing.assert_close(estimator.intercepts, expected_intercepts)


@pytest.mark.parametrize("gradient", [float("inf"), float("-inf"), float("nan")])
def test_refit_rejects_only_non_finite_intercept_steps(gradient: float):
    """A bad intercept step must not suppress updates at healthy positions."""
    device = current_platform.device_type
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 1
    estimator.info[:, 2] = 1.0
    estimator.grad[:, 1] = 0.5
    estimator.grad[0, 1] = gradient
    estimator.counts[:] = estimator.DAMPING_OBSERVATIONS
    before_slope = estimator.slope.clone()
    expected_intercepts = estimator.intercepts.clone()
    expected_intercepts[1:] += 0.25 / (1.0 + estimator.L2)

    empty = torch.empty(0, dtype=torch.int32, device=device)
    estimator.step(empty, empty, empty)

    torch.testing.assert_close(estimator.slope, before_slope, rtol=0, atol=0)
    torch.testing.assert_close(
        estimator.intercepts[:1], expected_intercepts[:1], rtol=0, atol=0
    )
    torch.testing.assert_close(estimator.intercepts[1:], expected_intercepts[1:])


def test_step_damping_scales_with_the_evidence():
    """A round moves a coefficient in proportion to the samples behind it.

    The Newton step itself barely changes with how much data a round carries --
    curvature and score both scale with it -- so scaling a round's statistics
    isolates the damping factor n / (n + DAMPING_OBSERVATIONS). The slope/
    intercept coupling is zeroed so the two parameters can be checked apart.
    """
    device = current_platform.device_type
    damping = OnlineAcceptanceEstimator.DAMPING_OBSERVATIONS
    per_position_n = 20.0

    def refit_with(scale):
        estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
        estimator.info[:, 1] = 0.0  # sum(w*margin): no slope/intercept coupling
        estimator.info[:, 2] = 1.0 * scale  # sum(w)
        estimator.grad[:, 1] = 0.5 * scale  # sum(y - p)
        # The slope's information and score are accumulated as per-position
        # partials that the refit sums, so spread each total over the positions.
        estimator.info[:, 0] = 30.0 * scale / NUM_STEPS  # sum(w*margin^2)
        estimator.grad[:, 0] = 2.0 * scale / NUM_STEPS  # sum((y - p)*margin)
        estimator.counts[:] = per_position_n * scale
        estimator._steps_since_refit = estimator.REFIT_INTERVAL - 1
        before_slope = estimator.slope.clone()
        before_intercepts = estimator.intercepts.clone()
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        estimator.step(empty, empty, empty)
        return (
            (estimator.slope - before_slope)[0].item(),
            (estimator.intercepts - before_intercepts)[0].item(),
        )

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


@pytest.mark.parametrize("complement_mass", [1e-3, 1e-6, 1e-12])
def test_feature_resolves_a_tiny_complement_mass(complement_mass: float):
    """A confident row's whole signal lives below an fp32 ulp of 1.0.

    The feature is -log of the mass on every token but the top one. Summing all
    of it, including the top token's own unit term, and subtracting 1 afterwards
    cancels that mass away entirely once it drops under ~6e-8 -- and a drafter
    on an easy continuation is far more confident than that. Spread the mass
    over the tail so no single competitor is close to the top token.
    """
    device = current_platform.device_type
    estimator = _make_estimator(device)

    # Top logit at 0; the rest share `complement_mass` of relative mass.
    logits = torch.full(
        (1, VOCAB_SIZE), float(torch.tensor(complement_mass / (VOCAB_SIZE - 1)).log())
    ).to(device)
    logits[0, 0] = 0.0

    feature = _predict_one_row(estimator, logits, device)
    expected = _reference_feature(logits)

    # The tail holds `complement_mass` by construction, so the reference should
    # land on -log of it. Guards the test's own setup, loosely: the logit written
    # into each tail slot is fp32-rounded.
    torch.testing.assert_close(
        expected.item(), -torch.tensor(complement_mass).log().item(), rtol=1e-5, atol=0
    )
    # rtol is ~3 orders looser than fp32 needs here, to stay robust across
    # backends' exp(); the bug this pins produced 40.0 against a true 27.63.
    torch.testing.assert_close(feature.double(), expected, rtol=1e-5, atol=0)


# Three blocks, the last one partial, so the masked tail is covered too.
MULTI_BLOCK_VOCAB = 2 * VOCAB_BLOCK_SIZE + 1536


@pytest.mark.parametrize("max_block", [0, 1, 2])
def test_feature_combines_blocks_around_the_global_max(max_block: int):
    """The top token's block must be found among several, not assumed.

    Each block reduces against its own local max, so the partials are on
    different scales and only become comparable once rescaled by the global
    max. Exactly one block holds the top token and must contribute its
    excluding sum; the rest contribute in full. Pointing that exclusion at the
    wrong block subtracts a token that is not the maximum and leaves the real
    one in, which is wrong by a factor of e^(gap) -- invisible with a
    single-block vocabulary, where the max's block is trivially block 0.

    Mass is spread over every block, in decreasing decades, so a block whose
    partial is dropped or left unrescaled changes the result.
    """
    device = current_platform.device_type
    estimator = _make_estimator(device)
    complement_mass = 1e-9

    weights = torch.zeros(MULTI_BLOCK_VOCAB, dtype=torch.float64)
    for block in range(3):
        start = block * VOCAB_BLOCK_SIZE
        end = min(start + VOCAB_BLOCK_SIZE, MULTI_BLOCK_VOCAB)
        weights[start:end] = 10.0**-block
    top_idx = max_block * VOCAB_BLOCK_SIZE + 7
    weights[top_idx] = 0.0
    weights *= complement_mass / weights.sum()

    logits = weights.log().float()[None, :].to(device)
    logits[0, top_idx] = 0.0

    feature = _predict_one_row(estimator, logits, device)
    expected = _reference_feature(logits)
    # The tail carries `complement_mass` by construction.
    torch.testing.assert_close(
        expected.item(),
        -torch.tensor(complement_mass).log().item(),
        rtol=1e-5,
        atol=0,
    )
    torch.testing.assert_close(feature.double(), expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("num_tied", [2, 3])
@pytest.mark.parametrize("placement", ["same_block", "across_blocks"])
def test_feature_drops_exactly_one_tied_maximum(num_tied: int, placement: str):
    """Tied maxima: remove exactly one, keep the rest.

    The complement is the mass on every token *except the single most likely
    one*, so k tokens at the maximum leave k-1 of them in it, and the feature
    is -log(k-1): 0 for a 2-way tie, -log 2 for a 3-way. Dropping all k instead
    reports a complement of 0 -- a one-hot row -- clamping the feature to
    +MAX_LOG_ODDS, i.e. maximal confidence for a maximally ambiguous row, the
    exact inversion of the truth.

    Both placements are needed because they fail differently, and each is blind
    to the other's bug:

    - `same_block` catches excluding by value (`logits == block_max`)
      without adding the other tied copies back, which zeroes every tied entry
      inside the block at once. Spread across blocks it would hit one entry per
      block and look correct.
    - `across_blocks` catches excluding from every block whose local max ties
      the global max, instead of from the single selected one. Within one block
      there is no second block to wrongly exclude from.

    The floor is low enough that the ~12k other tokens contribute ~1e-22, so the
    surviving ties are effectively the entire complement.
    """
    device = current_platform.device_type
    estimator = _make_estimator(device)

    if placement == "same_block":
        tied = [5 + i for i in range(num_tied)]
    else:
        tied = [block * VOCAB_BLOCK_SIZE + 5 for block in range(num_tied)]
    logits = torch.full((1, MULTI_BLOCK_VOCAB), -60.0, device=device)
    logits[0, tied] = 0.0

    feature = _predict_one_row(estimator, logits, device)
    expected = -torch.tensor(float(num_tied - 1)).log().item()
    # Spelled out before the value check: this is the inversion that matters,
    # and it reports better than "40.0 != -0.69" would.
    assert feature.item() < _MAX_LOG_ODDS, (
        f"{num_tied} tied maxima ({placement}) scored as a one-hot row, so an "
        "ambiguous draft was handed maximal confidence"
    )
    torch.testing.assert_close(feature.item(), expected, rtol=0, atol=1e-5)


def test_feature_ignores_which_token_was_drawn():
    """The score must depend on the draft distribution, not on the draw.

    This is what keeps rejection sampling lossless: if the score read the drawn
    token, trimming a draft would condition on the token being trimmed and tilt
    the emitted distribution toward the drafter's confident modes. `predict`
    takes no token argument, and this pins that -- two rows with identical
    logits must score identically no matter what was sampled from them.
    """
    device = current_platform.device_type
    torch.manual_seed(0)
    estimator = _make_estimator(device)

    row = torch.randn(VOCAB_SIZE, device=device) * 4.0
    logits = row.expand(2, VOCAB_SIZE).contiguous()
    idx_mapping = torch.tensor([0, 1], device=device, dtype=torch.int32)
    draft_step = torch.tensor(0, device=device, dtype=torch.int32)
    temperature = torch.ones(MAX_NUM_REQS, device=device)
    confidence_probs = torch.zeros(MAX_NUM_REQS, NUM_STEPS, device=device)

    estimator.predict(logits, idx_mapping, draft_step, confidence_probs, temperature)
    assert estimator.features[0, 0] == estimator.features[1, 0]

    # And the value is the top token's log-odds, not the top-2 margin.
    scaled = row.double()
    top = scaled.max(dim=-1)
    rest = scaled.scatter(0, top.indices[None], float("-inf"))
    torch.testing.assert_close(
        estimator.features[0, 0].double(),
        (top.values - torch.logsumexp(rest, dim=-1)).clamp(-40.0, 40.0),
        atol=2e-3,
        rtol=2e-3,
    )


def test_warmup_covers_every_batch_size_and_the_refit():
    """After warmup, no batch size and no refit may JIT-compile during serving."""
    device = current_platform.device_type

    def compiled_variants() -> set:
        return {
            (kernel.fn.__name__, key)
            for kernel in (_accumulate_kernel, _refit_kernel)
            for caches in kernel.device_caches.values()
            for key in caches[0]
        }

    estimator = OnlineAcceptanceEstimator(REFIT_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 2
    slots = torch.arange(REFIT_NUM_REQS, device=device, dtype=torch.int32)
    num_accepted = torch.ones(REFIT_NUM_REQS, device=device)
    features = torch.rand(REFIT_NUM_REQS, NUM_STEPS, device=device)
    _drive_round(estimator, features[:1], num_accepted[:1], slots[:1])
    estimator.surprise_counts.fill_(1.0)
    estimator.warmup()
    assert not estimator.surprise_counts.any()
    warmed = compiled_variants()

    for num_reqs in (8, 300):
        _drive_round(
            estimator, features[:num_reqs], num_accepted[:num_reqs], slots[:num_reqs]
        )
    torch.accelerator.synchronize()

    assert estimator._refits == 1
    assert compiled_variants() == warmed


@triton.jit
def _baseline_score(features, slope, intercepts, out, K: tl.constexpr):
    k = tl.arange(0, K)
    tl.store(
        out + k,
        tl.sigmoid(tl.load(slope) * tl.load(features + k) + tl.load(intercepts + k)),
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_predecessor_surprise_is_causal_and_cold_start_matches_baseline(parallel):
    device = current_platform.device_type
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    logits = torch.tensor([[3.0, 1.0, -2.0, -6.0]], device=device)
    slots = torch.full((NUM_STEPS,), 0, device=device, dtype=torch.int32)
    steps = torch.arange(NUM_STEPS, device=device)
    temperature = torch.ones(MAX_NUM_REQS, device=device)
    confidence = torch.empty((1, NUM_STEPS), device=device)

    def score(tokens):
        if parallel:
            estimator.predict(
                logits.repeat(NUM_STEPS, 1),
                slots,
                steps,
                confidence,
                temperature,
                tokens,
            )
        else:
            for k in range(NUM_STEPS):
                estimator.predict(
                    logits,
                    slots[:1],
                    steps[k],
                    confidence,
                    temperature,
                    None if tokens is None else tokens[k : k + 1],
                )
        return confidence.clone()

    draws = torch.tensor([0, 3, 1, 2], device=device)
    cold = score(draws)
    baseline = torch.empty_like(cold)
    _baseline_score[(1,)](
        estimator.features[0],
        estimator.slope,
        estimator.intercepts,
        baseline,
        K=NUM_STEPS,
    )
    torch.testing.assert_close(cold, baseline, rtol=0, atol=0)
    estimator.surprise_slope.fill_(-0.5)
    reference = score(draws)
    expected_surprise = torch.tensor([0.0, 0.0, 9.0, 2.0], device=device)
    torch.testing.assert_close(estimator.predecessor_surprise[0], expected_surprise)
    for k in range(NUM_STEPS):
        changed = draws.clone()
        changed[k:] = (changed[k:] + 1) % logits.shape[1]
        result = score(changed)
        torch.testing.assert_close(
            result[:, : k + 1], reference[:, : k + 1], rtol=0, atol=0
        )
        if k + 1 < NUM_STEPS:
            assert result[0, k + 1] != reference[0, k + 1]

    # No-sample and greedy paths must not inherit a previous round's surprise.
    torch.testing.assert_close(score(None), baseline, rtol=0, atol=0)
    assert not estimator.predecessor_surprise[0].any()
    temperature.zero_()
    torch.testing.assert_close(score(draws), baseline, rtol=0, atol=0)
    assert not estimator.predecessor_surprise[0].any()


@pytest.mark.parametrize("all_rejected", [False, True])
def test_joint_refit_matches_dense_newton_step(all_rejected):
    device = current_platform.device_type
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 1
    rng = np.random.default_rng(12)
    num_reqs = 12
    logits = rng.normal(size=(num_reqs, NUM_STEPS, 8)).astype(np.float32)
    logits[2, 0, 0] = 80.0
    draws = rng.integers(0, 8, size=(num_reqs, NUM_STEPS))
    draws[2, 0] = 1
    temps = np.ones(MAX_NUM_REQS, dtype=np.float32)
    temps[::2] = 0.0
    slots = np.arange(num_reqs)[::-1].copy()
    temperature = torch.tensor(temps, device=device)
    confidence = torch.empty((num_reqs, NUM_STEPS), device=device)
    tensor_slots = torch.tensor(slots, device=device, dtype=torch.int32)
    steps = torch.arange(NUM_STEPS, device=device).repeat(num_reqs)
    estimator.predict(
        torch.tensor(logits.reshape(-1, 8), device=device),
        tensor_slots.repeat_interleave(NUM_STEPS),
        steps,
        confidence,
        temperature,
        torch.tensor(draws.flatten(), device=device),
    )

    # Independent fp64 reference: construct every observed row explicitly.
    scaled = logits.astype(np.float64)
    top = scaled.max(axis=-1)
    rest = scaled.copy()
    np.put_along_axis(rest, scaled.argmax(axis=-1)[..., None], -np.inf, axis=-1)
    peak = np.clip(top - np.logaddexp.reduce(rest, axis=-1), -40.0, 40.0)
    surprise = top - np.take_along_axis(scaled, draws[..., None], axis=-1)[..., 0]
    surprise[temps[slots] == 0] = 0.0
    predecessor = np.concatenate((np.zeros((num_reqs, 1)), surprise[:, :-1]), axis=1)
    np.testing.assert_allclose(
        estimator.predecessor_surprise[tensor_slots].cpu(),
        predecessor,
        rtol=1e-5,
        atol=1e-5,
    )
    accepted = (
        np.zeros(num_reqs, dtype=int)
        if all_rejected
        else np.arange(num_reqs) % (NUM_STEPS + 1)
    )
    admitted = np.minimum(accepted + 1, NUM_STEPS)
    positions = np.arange(NUM_STEPS)[None, :]
    observed = positions < admitted[:, None]
    labels = (positions < accepted[:, None])[observed]
    design = np.concatenate(
        (
            peak[..., None],
            predecessor[..., None],
            np.broadcast_to(np.eye(NUM_STEPS), (num_reqs, NUM_STEPS, NUM_STEPS)),
        ),
        axis=-1,
    )[observed]
    probs = confidence.cpu().numpy().astype(np.float64)[observed]
    hessian = design.T @ ((probs * (1 - probs))[:, None] * design)
    hessian += estimator.L2 * np.eye(NUM_STEPS + 2)
    grad = design.T @ (labels - probs)
    delta = np.linalg.solve(hessian, grad)
    counts = observed.sum(axis=0)
    total = counts.sum()
    damp = estimator.DAMPING_OBSERVATIONS
    surprise_n = np.count_nonzero(observed & (predecessor > 0.0))
    delta[0] *= total / (total + damp)
    delta[1] *= surprise_n / (surprise_n + damp)
    initial = np.r_[
        estimator.slope.cpu(),
        estimator.surprise_slope.cpu(),
        estimator.intercepts.cpu(),
    ]
    if initial[0] + delta[0] < 0:
        delta[0] = 0.0
    # Back-substitution uses the damped/guarded shared slopes, as in production.
    delta[2:] = (grad[2:] - hessian[2:, :2] @ delta[:2]) / hessian.diagonal()[2:]
    delta[2:] *= counts / (counts + damp)
    estimator.step(
        tensor_slots,
        torch.tensor(accepted + 1, device=device, dtype=torch.int32),
        torch.tensor(admitted - accepted, device=device, dtype=torch.int32),
    )
    actual = np.r_[
        estimator.slope.cpu(),
        estimator.surprise_slope.cpu(),
        estimator.intercepts.cpu(),
    ]
    np.testing.assert_allclose(actual, initial + delta, rtol=2e-4, atol=2e-5)
    assert not estimator.info.any()
    assert not estimator.grad.any()
    assert not estimator.counts.any()
    assert not estimator.surprise_counts.any()


@pytest.mark.parametrize("parallel", [False, True])
def test_predecessor_surprise_padding_and_slot_reuse(parallel):
    device = current_platform.device_type
    estimator = _make_estimator(device)
    estimator.surprise_slope.fill_(-1.0)
    slots = torch.tensor([7, -1, 2], device=device, dtype=torch.int32)
    temperature = torch.ones(MAX_NUM_REQS, device=device)
    steps = torch.arange(NUM_STEPS, device=device)
    logits = torch.tensor([[2.0, -4.0]], device=device)
    confidence = torch.full((3, NUM_STEPS), -1.0, device=device)

    num_rows = 3 * NUM_STEPS if parallel else 3
    draws = torch.ones(num_rows, device=device, dtype=torch.int64)
    rows = logits.repeat(num_rows, 1)

    def score():
        if parallel:
            estimator.predict(
                rows,
                slots.repeat_interleave(NUM_STEPS),
                steps.repeat(3),
                confidence,
                temperature,
                draws,
            )
        else:
            for k in range(NUM_STEPS):
                estimator.predict(
                    rows,
                    slots,
                    steps[k],
                    confidence,
                    temperature,
                    draws,
                )

    score()
    assert (estimator.predecessor_surprise[7, 1:] == 6.0).all()
    if torch.cuda.is_available():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            score()
        draws.zero_()  # New request reuses the same persistent slot.
        graph.replay()
    else:
        draws.zero_()
        score()
    assert not estimator.predecessor_surprise[7].any()
    assert (confidence[1] == -1.0).all()
    assert not estimator.features[-1].any()
    assert not estimator.surprise[-1].any()
    # The padding row carries bogus verification counts; it must not be graded.
    estimator.step(
        slots,
        torch.ones(3, device=device, dtype=torch.int32),
        torch.full((3,), NUM_STEPS, device=device, dtype=torch.int32),
    )
    assert estimator.counts.tolist() == [2.0, 0.0, 0.0, 0.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA budget kernel")
def test_fixed_budget_admission_does_not_read_current_draw():
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        assign_draft_token_budget,
    )

    device = "cuda"
    estimator = _make_estimator(device)
    estimator.surprise_slope.fill_(-0.5)
    slots = torch.arange(3, device=device, dtype=torch.int32)
    steps = torch.arange(NUM_STEPS, device=device).repeat(3)
    logits = torch.tensor([[2.0, -1.0, -4.0]], device=device).repeat(3 * NUM_STEPS, 1)
    temperature = torch.ones(MAX_NUM_REQS, device=device)
    confidence = torch.empty((3, NUM_STEPS), device=device)
    scratch = torch.empty(3 * NUM_STEPS, dtype=torch.int32, device=device)

    def admit(draws):
        estimator.predict(
            logits,
            slots.repeat_interleave(NUM_STEPS),
            steps,
            confidence,
            temperature,
            draws,
        )
        capacities = torch.full((3,), NUM_STEPS, dtype=torch.int32, device=device)
        assign_draft_token_budget(confidence, slots, capacities, 5, scratch)
        assert capacities.sum().item() == 5
        return capacities

    draws = torch.zeros(3 * NUM_STEPS, dtype=torch.int64, device=device)
    initial = admit(draws)
    for k in range(NUM_STEPS):
        changed = draws.clone()
        changed[k:NUM_STEPS] = 2
        result = admit(changed)
        assert (initial[0] > k) == (result[0] > k)


def test_predecessor_surprise_uses_target_vocab_sample_id():
    device = current_platform.device_type
    estimator = _make_estimator(device)
    estimator.surprise_slope.fill_(-0.5)
    # Draft column 1 corresponds to target ID 9, not target ID 1.
    logits = torch.tensor([[3.0, -2.0, 1.0]], device=device)
    target_logits = torch.full((1, 12), -float("inf"), device=device)
    target_logits[:, [7, 9, 2]] = logits
    slots = torch.tensor([0], dtype=torch.int32, device=device)
    temperature = torch.ones(MAX_NUM_REQS, device=device)
    confidence = torch.empty((1, NUM_STEPS), device=device)
    token = torch.tensor([9], device=device)
    for step in range(2):
        estimator.predict(
            logits,
            slots,
            torch.tensor(step, device=device),
            confidence,
            temperature,
            token,
            target_logits,
        )
    assert estimator.predecessor_surprise[0, 1] == 5.0


@pytest.mark.parametrize("gradient", [float("inf"), float("-inf"), float("nan")])
def test_refit_rejects_non_finite_surprise_step(gradient):
    device = current_platform.device_type
    estimator = _make_estimator(device)
    estimator.REFIT_INTERVAL = 1
    estimator.info[:, 0] = 1.0
    estimator.info[:, 2] = 1.0
    estimator.info[:, 4] = 1.0
    estimator.grad[:, 1] = 0.5
    estimator.grad[0, 2] = gradient
    estimator.counts[:] = estimator.DAMPING_OBSERVATIONS
    estimator.surprise_counts[:] = estimator.DAMPING_OBSERVATIONS
    before = estimator.intercepts.clone()
    empty = torch.empty(0, dtype=torch.int32, device=device)
    estimator.step(empty, empty, empty)
    assert estimator.surprise_slope.item() == 0.0
    torch.testing.assert_close(
        estimator.intercepts, before + 0.25 / (1.0 + estimator.L2)
    )


def test_surprise_step_damping_uses_nonzero_observations():
    """Three surprising draws must not borrow evidence from 1000 top-1 draws."""
    device = current_platform.device_type
    num_reqs = 1003
    estimator = OnlineAcceptanceEstimator(num_reqs, 2, device)
    estimator.REFIT_INTERVAL = 2
    slots = torch.arange(num_reqs, device=device, dtype=torch.int32)
    logits = torch.tensor([[0.0, -0.1], [0.0, 0.0]], device=device).repeat(num_reqs, 1)
    logits[1::2, 1] = -torch.linspace(0.0, 2.0, num_reqs, device=device)
    steps = torch.arange(2, device=device).repeat(num_reqs)
    draws = torch.zeros(num_reqs, 2, dtype=torch.int64, device=device)
    draws[-3:, 0] = 1
    confidence = torch.empty(num_reqs, 2, device=device)
    estimator.predict(
        logits,
        slots.repeat_interleave(2),
        steps,
        confidence,
        torch.ones(num_reqs, device=device),
        draws.flatten(),
    )
    # All requests reach the second position; only the three surprising ones
    # are rejected there.
    accepted = torch.full((num_reqs,), 2, device=device, dtype=torch.int32)
    accepted[-3:] = 1
    estimator.step(slots, accepted + 1, 2 - accepted)
    assert estimator.counts.tolist() == [num_reqs, num_reqs]
    assert estimator.surprise_counts.tolist() == [0, 3]

    # Dense Newton reference, before applying the sparse feature's damping.
    design = np.column_stack(
        (
            estimator.features.cpu().numpy().flatten(),
            estimator.predecessor_surprise.cpu().numpy().flatten(),
            np.tile(np.eye(2), (num_reqs, 1)),
        )
    )
    probs = confidence.cpu().numpy().flatten().astype(np.float64)
    labels = (np.arange(2) < accepted.cpu().numpy()[:, None]).flatten()
    hessian = design.T @ ((probs * (1 - probs))[:, None] * design)
    hessian += estimator.L2 * np.eye(4)
    raw = np.linalg.solve(hessian, design.T @ (labels - probs))[1]
    expected = raw * 3 / (3 + estimator.DAMPING_OBSERVATIONS)
    empty = torch.empty(0, dtype=torch.int32, device=device)
    estimator.step(empty, empty, empty)
    assert estimator.surprise_slope.item() == pytest.approx(expected, rel=2e-4)
    assert not estimator.surprise_counts.any()


def test_refit_rejects_nonpositive_schur_determinant():
    """An indefinite Schur system must not produce a finite reverse step."""
    device = current_platform.device_type
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    estimator.REFIT_INTERVAL = 1
    # Deliberately indefinite shared-slope moments: |m01| > sqrt(m00*m11).
    estimator.info[:, 0] = 1.0
    estimator.info[:, 2] = 1.0
    estimator.info[:, 3] = 2.0
    estimator.info[:, 4] = 1.0
    estimator.grad[:] = 0.5
    estimator.counts[:] = estimator.DAMPING_OBSERVATIONS
    estimator.surprise_counts[:] = estimator.DAMPING_OBSERVATIONS
    before_slope = estimator.slope.clone()
    before_bias = estimator.intercepts.clone()
    empty = torch.empty(0, dtype=torch.int32, device=device)
    estimator.step(empty, empty, empty)
    torch.testing.assert_close(estimator.slope, before_slope, rtol=0, atol=0)
    assert estimator.surprise_slope.item() == 0.0
    torch.testing.assert_close(
        estimator.intercepts, before_bias + 0.25 / (1.0 + estimator.L2)
    )
