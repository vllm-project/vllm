# SPDX-License-Identifier: Apache-2.0
"""Wasserstein-based numerical regression helpers for P67 + sister kernels.

Background — why Wasserstein, not L2 / max-diff
================================================

Per Golden et al. arXiv 2405.02803 ("Is Flash Attention Stable?"),
fused Flash Attention has documented O(N) drift vs reference attention
on long sequences due to online-softmax rescale ordering. This drift is
**bounded element-wise** but may shift the output distribution.

Naive metrics fall short:
- **L2 / max-diff**: catches gross numerical errors but not distribution
  shifts. P67 split-M is bit-exact to per-query reference (verified in
  v7.34) so L2 should be ~0 — but if distribution shifts due to rescale
  order, L2 won't catch it for outputs with cancellation.
- **rel_avg < 1e-3**: our existing P67 contract; works for golden cases
  but doesn't catch tail-heavy drift (e.g. 99th percentile elements
  diverging while average stays fine).

Wasserstein-1 distance between empirical distributions:
- **Catches distribution-level drift** even when element-wise diffs cancel
- **Standard ML practice** for output regression detection
- **Bounded interpretation**: W1 < threshold means "moving one tensor's
  mass to the other costs less than threshold per element on average"

Usage
=====
```python
from .numerical_regression_helpers import (
    wasserstein_1d, wasserstein_per_row, regression_report
)

# Compare two output tensors
out_ref = run_reference_attention(q, k, v, ...)
out_p67 = run_p67_attention(q, k, v, ...)

report = regression_report(out_ref, out_p67, threshold=1e-3)
assert report.passes, f"P67 drift detected: {report}"
```

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference: Golden et al. arXiv 2405.02803 (Is FA Stable?), 2024.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# ──────────────────────────────────────────────────────────────────
# Core distance functions
# ──────────────────────────────────────────────────────────────────


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Wasserstein-1 distance between empirical distributions of two
    tensors (treated as 1D after flattening).

    Mathematical definition:
        W1(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
    where F is the empirical CDF.

    Computational form (for equal-size tensors):
        W1 = mean(|sort(a) - sort(b)|)

    For unequal sizes: linear interpolation onto common grid. We assume
    equal sizes (P67 outputs always have same shape as reference).

    Args:
        a, b: tensors of same total size (flattened to 1D internally)

    Returns:
        W1 distance as Python float (CPU scalar). ~0 means identical
        distributions; larger means more drift.
    """
    if a.numel() != b.numel():
        raise ValueError(
            f"wasserstein_1d requires equal-size tensors, got "
            f"{a.numel()} vs {b.numel()}"
        )
    if a.numel() == 0:
        return 0.0
    # Flatten + sort to get empirical CDFs
    a_sorted = a.detach().flatten().float().sort().values
    b_sorted = b.detach().flatten().float().sort().values
    # W1 between equal-size empirical distributions = mean of sorted diffs
    diff = (a_sorted - b_sorted).abs()
    return float(diff.mean().item())


def wasserstein_per_row(
    a: torch.Tensor,
    b: torch.Tensor,
    row_dim: int = -1,
) -> torch.Tensor:
    """Per-row Wasserstein-1 distance.

    Useful for per-token output drift analysis. For attention outputs
    of shape [B, T, D], computing W1 per (B, T) gives a [B, T] tensor
    of distances — can detect localized drift (e.g. some tokens drifted
    while others stayed bit-exact).

    Args:
        a, b: tensors of same shape
        row_dim: dimension that defines a "row" — last dim by default.
                 W1 is computed across all OTHER dims, returning a
                 1D tensor of size a.shape[row_dim].

    Returns:
        Tensor of shape (a.shape[row_dim],) with W1 per row.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"wasserstein_per_row requires same shape, got {a.shape} vs {b.shape}"
        )
    # Move row_dim to front, flatten the rest
    a_perm = a.detach().float().movedim(row_dim, 0)
    b_perm = b.detach().float().movedim(row_dim, 0)
    n_rows = a_perm.shape[0]
    a_rows = a_perm.reshape(n_rows, -1).sort(dim=-1).values
    b_rows = b_perm.reshape(n_rows, -1).sort(dim=-1).values
    return (a_rows - b_rows).abs().mean(dim=-1)


# ──────────────────────────────────────────────────────────────────
# Regression report dataclass
# ──────────────────────────────────────────────────────────────────


@dataclass
class RegressionReport:
    """Comprehensive numerical regression report.

    Stores multiple metrics for a single comparison so test failures
    print actionable diagnostics, not just "assertion failed".
    """
    w1: float                      # Wasserstein-1 distance
    max_abs_diff: float            # Element-wise max |a - b|
    rel_diff_avg: float            # mean(|a - b|) / mean(|a|)
    rel_diff_max: float            # max element-wise relative diff
    threshold: float               # Pass threshold (W1)
    a_stats: dict                  # mean / std / min / max of `a`
    b_stats: dict                  # same for `b`

    @property
    def passes(self) -> bool:
        return self.w1 < self.threshold

    def __str__(self) -> str:
        status = "PASS" if self.passes else "FAIL"
        return (
            f"RegressionReport[{status}]:\n"
            f"  W1={self.w1:.3e} (threshold {self.threshold:.3e})\n"
            f"  max_abs_diff={self.max_abs_diff:.3e}\n"
            f"  rel_diff_avg={self.rel_diff_avg:.3e}\n"
            f"  rel_diff_max={self.rel_diff_max:.3e}\n"
            f"  a stats: {self.a_stats}\n"
            f"  b stats: {self.b_stats}"
        )


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """Summary stats for diagnostic logs."""
    if t.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    tf = t.detach().flatten().float()
    return {
        "mean": float(tf.mean().item()),
        "std": float(tf.std().item()) if tf.numel() > 1 else 0.0,
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
    }


def regression_report(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 1e-3,
) -> RegressionReport:
    """Compute full regression report comparing two output tensors.

    Computes 4 metrics + summary stats so test failures are diagnosable.

    Args:
        a: reference output (e.g., upstream kernel)
        b: test output (e.g., P67 split-M)
        threshold: W1 pass threshold. Default 1e-3 matches our existing
                   P67 rel_avg contract from v7.34 split-M validation.

    Returns:
        RegressionReport with `.passes` boolean and full diagnostics.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"regression_report requires same shape: {a.shape} vs {b.shape}"
        )

    # All metrics computed in fp32 for stability
    af = a.detach().flatten().float()
    bf = b.detach().flatten().float()

    # Wasserstein-1
    w1 = wasserstein_1d(a, b)

    # Element-wise max abs diff
    diff = (af - bf).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0

    # Relative diffs (guard div-by-zero)
    a_abs = af.abs()
    a_mean_abs = a_abs.mean().item()
    if a_mean_abs > 1e-30:
        rel_diff_avg = float(diff.mean().item() / a_mean_abs)
    else:
        rel_diff_avg = 0.0

    # Per-element relative diff (more sensitive to outliers)
    a_safe = torch.where(a_abs > 1e-30, a_abs, torch.ones_like(a_abs))
    rel_per_elem = diff / a_safe
    rel_diff_max = float(rel_per_elem.max().item()) if rel_per_elem.numel() > 0 else 0.0

    return RegressionReport(
        w1=w1,
        max_abs_diff=max_abs_diff,
        rel_diff_avg=rel_diff_avg,
        rel_diff_max=rel_diff_max,
        threshold=threshold,
        a_stats=_tensor_stats(a),
        b_stats=_tensor_stats(b),
    )


# ──────────────────────────────────────────────────────────────────
# Higher-level helpers for kernel comparison workflows
# ──────────────────────────────────────────────────────────────────


def assert_attention_outputs_close(
    out_reference: torch.Tensor,
    out_test: torch.Tensor,
    threshold: float = 1e-3,
    name: str = "attention",
) -> None:
    """Assert two attention outputs are within threshold W1 distance.

    Designed for use in pytest tests comparing kernel outputs.
    Failures print full RegressionReport for actionable diagnostics.

    Args:
        out_reference: gold output (e.g., upstream kernel)
        out_test: kernel-under-test output (e.g., P67)
        threshold: max acceptable W1 distance (default 1e-3)
        name: descriptive name for the comparison (printed on failure)

    Raises:
        AssertionError with full report if W1 >= threshold
    """
    report = regression_report(out_reference, out_test, threshold=threshold)
    if not report.passes:
        raise AssertionError(
            f"\nNumerical regression detected in {name}:\n{report}\n"
            f"\nDiagnostic suggestions:\n"
            f"  - If max_abs_diff is huge but w1 small: outliers, check tile\n"
            f"    masking / fully-masked tile handling\n"
            f"  - If w1 is on the order of element magnitudes: kernel\n"
            f"    likely has compute bug, not just rescale ordering\n"
            f"  - If rel_diff_max ≫ rel_diff_avg: localized drift,\n"
            f"    check per-row Wasserstein with `wasserstein_per_row`"
        )


def find_drift_rows(
    out_reference: torch.Tensor,
    out_test: torch.Tensor,
    threshold: float = 1e-3,
    row_dim: int = -1,
) -> list[int]:
    """Identify which rows have W1 drift above threshold.

    Useful for localizing regression — instead of "P67 failed",
    you get "rows [42, 73] of output have drift ~5e-3".

    Args:
        out_reference: gold output
        out_test: kernel-under-test output
        threshold: per-row W1 threshold
        row_dim: which dim defines a "row"

    Returns:
        List of row indices that exceed threshold (sorted).
    """
    per_row_w1 = wasserstein_per_row(out_reference, out_test, row_dim=row_dim)
    return sorted(int(i) for i in (per_row_w1 >= threshold).nonzero().flatten().tolist())
