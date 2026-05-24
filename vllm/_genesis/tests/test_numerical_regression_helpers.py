# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Wasserstein-based numerical regression helpers (T4.4).

CPU-runnable. Does NOT exercise actual P67 kernel (that's in
test_p67_wasserstein_regression.py which requires CUDA + Triton).

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import pytest
import torch


def test_wasserstein_1d_identical_distributions():
    """W1(a, a) == 0 for any tensor."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_1d
    a = torch.randn(100, 50)
    assert wasserstein_1d(a, a) == 0.0


def test_wasserstein_1d_shifted_distributions():
    """W1 detects constant shift accurately."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_1d
    a = torch.randn(1000)
    b = a + 0.5
    w1 = wasserstein_1d(a, b)
    # W1 between sorted versions of a and a+0.5 = 0.5 exactly
    assert abs(w1 - 0.5) < 1e-5


def test_wasserstein_1d_scaled_distributions():
    """W1 detects scale change."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_1d
    torch.manual_seed(42)
    a = torch.randn(10000)
    b = a * 2.0
    w1 = wasserstein_1d(a, b)
    # W1 between sorted(a) and sorted(2a) = mean(|a - 2a|) = mean(|a|)
    expected = a.abs().mean().item()
    assert abs(w1 - expected) < 1e-3


def test_wasserstein_1d_size_mismatch_raises():
    """Different sizes raise clear ValueError."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_1d
    a = torch.randn(10)
    b = torch.randn(20)
    with pytest.raises(ValueError, match="equal-size"):
        wasserstein_1d(a, b)


def test_wasserstein_1d_empty_tensors():
    """Empty tensors return 0."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_1d
    a = torch.tensor([])
    b = torch.tensor([])
    assert wasserstein_1d(a, b) == 0.0


def test_wasserstein_per_row_identical():
    """Per-row W1 of (a, a) is all zeros."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_per_row
    a = torch.randn(8, 64)
    w1 = wasserstein_per_row(a, a, row_dim=0)
    assert w1.shape == (8,)
    assert torch.allclose(w1, torch.zeros(8))


def test_wasserstein_per_row_localized_drift():
    """Drift in only one row is detected per-row."""
    from vllm._genesis.tests.numerical_regression_helpers import wasserstein_per_row
    torch.manual_seed(0)
    a = torch.randn(4, 100)
    b = a.clone()
    b[2] += 1.0  # drift only row 2
    w1 = wasserstein_per_row(a, b, row_dim=0)
    assert w1.shape == (4,)
    # Row 2 should have large W1, others ~0
    assert w1[0] < 1e-5
    assert w1[1] < 1e-5
    assert w1[3] < 1e-5
    assert abs(w1[2].item() - 1.0) < 1e-3  # constant shift = magnitude


def test_regression_report_passes_on_identical():
    """Report.passes for identical tensors."""
    from vllm._genesis.tests.numerical_regression_helpers import regression_report
    a = torch.randn(64, 128)
    r = regression_report(a, a, threshold=1e-3)
    assert r.passes
    assert r.w1 == 0.0
    assert r.max_abs_diff == 0.0


def test_regression_report_fails_on_significant_drift():
    """Report.passes is False when drift exceeds threshold."""
    from vllm._genesis.tests.numerical_regression_helpers import regression_report
    a = torch.randn(64, 128)
    b = a + 0.1  # 0.1 constant shift, threshold 0.01
    r = regression_report(a, b, threshold=0.01)
    assert not r.passes
    assert abs(r.w1 - 0.1) < 1e-3


def test_regression_report_includes_diagnostic_stats():
    """Report includes per-tensor stats for failure diagnosis."""
    from vllm._genesis.tests.numerical_regression_helpers import regression_report
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.1, 2.1, 3.1, 4.1])
    r = regression_report(a, b, threshold=0.01)
    assert "mean" in r.a_stats
    assert "std" in r.a_stats
    assert "min" in r.a_stats
    assert "max" in r.a_stats
    assert abs(r.a_stats["mean"] - 2.5) < 1e-5
    assert abs(r.b_stats["mean"] - 2.6) < 1e-5


def test_regression_report_str_actionable():
    """Report __str__ contains diagnostic-friendly fields."""
    from vllm._genesis.tests.numerical_regression_helpers import regression_report
    a = torch.randn(64, 128)
    b = a + 1.0
    r = regression_report(a, b, threshold=0.001)
    s = str(r)
    assert "FAIL" in s
    assert "W1=" in s
    assert "max_abs_diff" in s
    assert "rel_diff_avg" in s


def test_assert_close_passes_silently():
    """assert_attention_outputs_close passes silently when within threshold."""
    from vllm._genesis.tests.numerical_regression_helpers import (
        assert_attention_outputs_close,
    )
    a = torch.randn(64, 128)
    # Same tensor — must pass
    assert_attention_outputs_close(a, a, threshold=1e-3)


def test_assert_close_raises_with_full_report():
    """Failure raises AssertionError with full diagnostic report."""
    from vllm._genesis.tests.numerical_regression_helpers import (
        assert_attention_outputs_close,
    )
    a = torch.randn(64, 128)
    b = a + 0.5
    with pytest.raises(AssertionError) as exc_info:
        assert_attention_outputs_close(a, b, threshold=0.01, name="P67_test")
    msg = str(exc_info.value)
    assert "P67_test" in msg
    assert "W1=" in msg
    assert "Diagnostic suggestions" in msg


def test_find_drift_rows_localizes_drift():
    """find_drift_rows returns indices of rows above threshold."""
    from vllm._genesis.tests.numerical_regression_helpers import find_drift_rows
    torch.manual_seed(0)
    a = torch.randn(8, 64)
    b = a.clone()
    # Drift rows 2 and 5
    b[2] += 1.0
    b[5] += 0.8
    drifted = find_drift_rows(a, b, threshold=0.1, row_dim=0)
    assert drifted == [2, 5]


def test_find_drift_rows_empty_when_all_clean():
    """No drift = empty list."""
    from vllm._genesis.tests.numerical_regression_helpers import find_drift_rows
    a = torch.randn(8, 64)
    drifted = find_drift_rows(a, a, threshold=1e-3, row_dim=0)
    assert drifted == []


def test_helpers_handle_fp16_inputs():
    """Helpers handle fp16 tensors (computation in fp32 internally)."""
    from vllm._genesis.tests.numerical_regression_helpers import (
        wasserstein_1d, regression_report,
    )
    a = torch.randn(64, 128, dtype=torch.float16)
    b = a.clone()
    assert wasserstein_1d(a, b) == 0.0
    r = regression_report(a, b)
    assert r.passes


def test_w1_threshold_recommendation_for_p67():
    """P67 v7.34 contract: rel_avg < 1e-3 → W1 should be similar order.

    Document expected threshold for P67 vs upstream comparison.
    """
    # Synthesize attention-output-shape tensor
    torch.manual_seed(42)
    out_ref = torch.randn(1, 4, 32, 128) * 5.0  # B=1, K_PLUS_1=4, Hq=32, D=128
    # 1e-4 rel drift simulates v7.34 split-M correctness
    drift = torch.randn_like(out_ref) * 1e-4 * out_ref.abs().mean()
    out_drifted = out_ref + drift

    from vllm._genesis.tests.numerical_regression_helpers import regression_report
    r = regression_report(out_ref, out_drifted, threshold=1e-3)
    # 1e-4 rel drift on attention outputs → W1 around 1e-4 * mean(|out|)
    # → should pass 1e-3 threshold by ~10× margin
    assert r.passes, f"Expected pass for tiny drift: {r}"
    assert r.w1 < 1e-3
