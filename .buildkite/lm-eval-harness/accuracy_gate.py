# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Statistically-calibrated pass/fail gating for lm-eval accuracy metrics.

The default accuracy gate compares a measured metric against a baseline with a
fixed relative tolerance ``rtol`` (``measured >= ground_truth * (1 - rtol)``).
That threshold ignores how many samples produced ``measured``: with a small
``limit`` the measurement is noisy, so a fixed ``rtol`` both lets real
regressions through (false pass) and, at the tails, flags noise (false fail).

This module adds an *opt-in* alternative that accounts for sampling noise via a
one-sided **Wilson score** lower confidence bound on the (binomial) accuracy. It
is pure stdlib (``math`` + ``statistics.NormalDist``) so it can be unit-tested on
CPU without running ``lm_eval`` or a GPU.

It is intentionally backward compatible: callers select the CI gate explicitly
(``gate="ci"``); the default ``gate="rtol"`` path reproduces the existing
decision exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


def wilson_lower_bound(p_hat: float, n: int, confidence: float = 0.95) -> float:
    """One-sided lower confidence bound for a binomial proportion (Wilson score).

    For an observed proportion ``p_hat`` from ``n`` independent Bernoulli trials,
    returns the lower end of the one-sided ``confidence``-level Wilson interval::

        z      = Phi^-1(confidence)            # one-sided normal quantile
        center = (p_hat + z^2/(2n)) / (1 + z^2/n)
        half   = (z / (1 + z^2/n)) * sqrt( p_hat*(1-p_hat)/n + z^2/(4 n^2) )
        lower  = center - half

    The Wilson interval is preferred over the normal (Wald) approximation because
    it stays inside ``[0, 1]`` and remains well-behaved at the tails (e.g. it
    yields a sensible bound when ``p_hat`` is exactly 0 or 1, where Wald collapses
    to zero width). The result is clamped to ``[0.0, 1.0]``.

    Args:
        p_hat: Observed proportion, in ``[0, 1]``.
        n: Number of trials (the eval sample count); must be a positive integer.
        confidence: One-sided confidence level, in ``(0, 1)`` (default ``0.95``).

    Returns:
        The lower confidence bound, in ``[0, 1]``.

    Raises:
        ValueError: If ``p_hat`` is outside ``[0, 1]``, ``n`` is not a positive
            integer, or ``confidence`` is outside ``(0, 1)``.
    """
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError(f"p_hat must be in [0, 1], got {p_hat}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    z = NormalDist().inv_cdf(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n))
    return min(1.0, max(0.0, center - half))


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single metric gate check."""

    passed: bool
    detail: str


def _is_proportion(x: float) -> bool:
    return 0.0 <= x <= 1.0


def evaluate_metric_gate(
    measured_value: float,
    ground_truth: float,
    *,
    gate: str = "rtol",
    n: int | None = None,
    confidence: float = 0.95,
    rtol: float = 0.08,
) -> GateResult:
    """Decide whether a measured metric passes its baseline gate.

    Two modes:

    * ``gate="rtol"`` (default, unchanged behaviour): pass iff
      ``measured_value >= ground_truth * (1 - rtol)``.
    * ``gate="ci"``: pass iff the one-sided Wilson lower bound of
      ``measured_value`` at ``n`` samples and ``confidence`` is at least the same
      baseline threshold ``ground_truth * (1 - rtol)``. Because the lower bound is
      never above ``measured_value``, passing the CI gate implies passing the
      rtol gate — the CI gate only *adds* strictness, requiring the result to
      clear the bar with statistical confidence given the sample count. Set
      ``rtol=0`` to require confidence that the model is at least the baseline.

    The CI gate applies only to proportion-like metrics (accuracies in
    ``[0, 1]`` such as ``exact_match`` / ``acc``); it raises for non-proportion
    metrics (e.g. perplexity) and for a missing/invalid sample count.

    Args:
        measured_value: The metric value produced by the run.
        ground_truth: The baseline metric value from the config.
        gate: ``"rtol"`` (default) or ``"ci"``.
        n: Sample count for the CI gate (the eval ``limit``). Required when
            ``gate="ci"``.
        confidence: One-sided confidence level for the CI gate.
        rtol: Relative tolerance applied to the baseline (default ``0.08``).

    Returns:
        A :class:`GateResult` with the boolean outcome and a printable detail.

    Raises:
        ValueError: For an unknown ``gate``, or, in CI mode, for a non-proportion
            metric or a missing/invalid ``n``.
    """
    threshold = ground_truth * (1.0 - rtol)

    if gate == "rtol":
        passed = measured_value >= threshold
        detail = (
            f"gate=rtol | measured={measured_value:.4f} "
            f">= threshold={threshold:.4f} (ground_truth={ground_truth:.4f}, "
            f"rtol={rtol}) -> {'PASS' if passed else 'FAIL'}"
        )
        return GateResult(passed=passed, detail=detail)

    if gate == "ci":
        if not _is_proportion(measured_value) or not _is_proportion(ground_truth):
            raise ValueError(
                "gate='ci' only applies to proportion metrics in [0, 1] "
                f"(got measured={measured_value}, ground_truth={ground_truth}); "
                "use gate='rtol' for non-proportion metrics such as perplexity"
            )
        if not isinstance(n, int) or n <= 0:
            raise ValueError(
                "gate='ci' requires a positive integer sample count n "
                f"(set an integer 'limit' in the config); got n={n!r}"
            )
        lower = wilson_lower_bound(measured_value, n, confidence)
        passed = lower >= threshold
        detail = (
            f"gate=ci | measured={measured_value:.4f} "
            f"wilson_lower({int(confidence * 100)}%, n={n})={lower:.4f} "
            f">= threshold={threshold:.4f} (ground_truth={ground_truth:.4f}, "
            f"rtol={rtol}) -> {'PASS' if passed else 'FAIL'}"
        )
        return GateResult(passed=passed, detail=detail)

    raise ValueError(f"unknown gate {gate!r}; expected 'rtol' or 'ci'")
