# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the statistically-calibrated accuracy gate.

These are pure-CPU tests: they exercise ``accuracy_gate`` only and do not import
``lm_eval`` or launch a server, so they run anywhere. Run them with
``pytest .buildkite/lm-eval-harness/test_accuracy_gate.py``.
"""

import math

import pytest
from accuracy_gate import evaluate_metric_gate, wilson_lower_bound

# --- Wilson lower bound: cross-check against known references ----------------


def test_wilson_matches_textbook_two_sided_95():
    # The one-sided 97.5% lower bound equals the two-sided 95% lower bound.
    # Textbook Wilson 95% interval for 50/100 successes is [0.4038, 0.5962].
    assert wilson_lower_bound(0.5, 100, 0.975) == pytest.approx(0.4038, abs=1e-3)


def test_wilson_lower_never_exceeds_phat():
    for p in (0.1, 0.5, 0.755, 0.9):
        for n in (50, 200, 1000):
            assert wilson_lower_bound(p, n, 0.95) <= p + 1e-12


def test_wilson_tightens_with_n():
    # More samples -> lower bound moves up toward p_hat (less uncertainty).
    small = wilson_lower_bound(0.72, 100, 0.95)
    large = wilson_lower_bound(0.72, 1000, 0.95)
    assert small < large < 0.72


def test_wilson_boundary_phat_one_is_below_one():
    # The Wald approximation collapses to width 0 at p_hat=1; Wilson does not.
    lb = wilson_lower_bound(1.0, 1000, 0.95)
    assert 0.0 < lb < 1.0
    assert lb == pytest.approx(0.9973, abs=1e-3)


def test_wilson_boundary_phat_zero_is_zero():
    assert wilson_lower_bound(0.0, 1000, 0.95) == 0.0


def test_wilson_in_unit_interval():
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        lb = wilson_lower_bound(p, 37, 0.9)
        assert 0.0 <= lb <= 1.0 and math.isfinite(lb)


@pytest.mark.parametrize(
    "p,n,conf",
    [
        (-0.1, 100, 0.95),
        (1.1, 100, 0.95),
        (0.5, 0, 0.95),
        (0.5, -5, 0.95),
        (0.5, 100, 0.0),
        (0.5, 100, 1.0),
    ],
)
def test_wilson_invalid_inputs_raise(p, n, conf):
    with pytest.raises(ValueError):
        wilson_lower_bound(p, n, conf)


# --- rtol gate: must reproduce the legacy decision exactly -------------------


@pytest.mark.parametrize(
    "measured,gt,rtol,expected",
    [
        (0.76, 0.755, 0.08, True),  # above baseline
        (0.70, 0.755, 0.08, True),  # within 8% relative -> legacy PASS
        (0.69, 0.755, 0.08, False),  # below 8% relative -> legacy FAIL
        (0.755, 0.755, 0.0, True),
    ],  # exactly baseline, rtol 0
)
def test_rtol_gate_matches_legacy(measured, gt, rtol, expected):
    res = evaluate_metric_gate(measured, gt, gate="rtol", rtol=rtol)
    assert res.passed is expected
    # equivalent to the original inline comparison
    assert res.passed == (measured >= gt * (1 - rtol))


# --- the whole point: rtol passes but CI fails at small n -------------------


def test_ci_fails_what_rtol_passes_at_small_n():
    measured, gt, rtol = 0.72, 0.755, 0.08
    # legacy fixed-tolerance gate is satisfied (point estimate clears the bar)
    assert evaluate_metric_gate(measured, gt, gate="rtol", rtol=rtol).passed
    # but with only n=100 samples the result is not statistically distinguishable
    # from a regression -> CI gate fails
    assert not evaluate_metric_gate(
        measured, gt, gate="ci", n=100, confidence=0.95, rtol=rtol
    ).passed
    # the same measured value with n=1000 samples clears the CI gate
    assert evaluate_metric_gate(
        measured, gt, gate="ci", n=1000, confidence=0.95, rtol=rtol
    ).passed


def test_ci_clear_pass_and_clear_fail():
    assert evaluate_metric_gate(
        0.82, 0.755, gate="ci", n=1000, confidence=0.95, rtol=0.08
    ).passed
    assert not evaluate_metric_gate(
        0.50, 0.755, gate="ci", n=1000, confidence=0.95, rtol=0.08
    ).passed


def test_ci_pass_implies_rtol_pass():
    # CI gate is strictly at least as strict as the rtol gate.
    for measured in (0.70, 0.72, 0.76, 0.80):
        ci = evaluate_metric_gate(
            measured, 0.755, gate="ci", n=500, confidence=0.95, rtol=0.08
        )
        rtol = evaluate_metric_gate(measured, 0.755, gate="rtol", rtol=0.08)
        if ci.passed:
            assert rtol.passed


# --- CI gate guards: non-proportion metrics and missing/invalid n -----------


def test_ci_rejects_non_proportion_metric():
    with pytest.raises(ValueError):
        evaluate_metric_gate(5.2, 4.8, gate="ci", n=1000)  # e.g. perplexity


@pytest.mark.parametrize("n", [None, 0, -1, 1.5, "1000"])
def test_ci_rejects_invalid_n(n):
    with pytest.raises(ValueError):
        evaluate_metric_gate(0.72, 0.755, gate="ci", n=n, confidence=0.95)


def test_unknown_gate_raises():
    with pytest.raises(ValueError):
        evaluate_metric_gate(0.72, 0.755, gate="bogus")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
