# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that per-shard FP8 scaling preserves precision in fused modules."""

import torch
import unittest

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

GAP_100X = 100
GAP_5000X = 5000
GAP_100KX = 100000

# Thresholds calibrated from FP8 e4m3 round-trip behavior
FP8_INTRINSIC_ERROR = 0.03
RESOLUTION_FLOOR = 200


def scaled_fp8_quant(tensor, scale=None):
    if scale is None:
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = amax / FP8_E4M3_MAX
    q = (tensor / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = q.to(torch.float8_e4m3fn)
    return q, scale


def fp8_dequant(q, scale):
    return q.to(torch.float32) * scale


def single_scale_quant(weight):
    q, s = scaled_fp8_quant(weight)
    return fp8_dequant(q, s)


def per_shard_quant(weight, shard_widths):
    out = torch.empty_like(weight, dtype=torch.float32)
    offset = 0
    for w in shard_widths:
        shard = weight[offset:offset + w]
        q, s = scaled_fp8_quant(shard)
        out[offset:offset + w] = fp8_dequant(q, s)
        offset += w
    return out


def make_fused(gap, seed=42):
    torch.manual_seed(seed)
    rows, cols = 256, 512
    large = torch.randn(rows, cols)
    small = torch.randn(rows, cols) / gap
    fused = torch.cat([large, small], dim=0)
    return fused, small, [rows, rows], rows


def small_shard_rel_error(fused, small_true, offset, quant_fn, widths=None):
    if widths:
        recon = quant_fn(fused, widths)
    else:
        recon = quant_fn(fused)
    small_recon = recon[offset:]
    return ((small_recon - small_true).abs() / (small_true.abs() + 1e-12)).mean().item()


def small_shard_distinct_levels(fused, offset):
    q, _ = scaled_fp8_quant(fused)
    return q[offset:].to(torch.float32).unique().numel()


# 100x gap: SwiGLU gate_up_proj range

class TestGap100x_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_resolution_preserved(self):
        """Small shard loses >40% of FP8 resolution."""
        fused, _, widths, offset = make_fused(GAP_100X)
        distinct = small_shard_distinct_levels(fused, offset)
        self.assertGreater(distinct, RESOLUTION_FLOOR,
            f"Only {distinct} distinct FP8 levels (need {RESOLUTION_FLOOR})")


class TestGap100x_PerShard_PASSES(unittest.TestCase):

    def test_resolution_preserved(self):
        _, small, widths, _ = make_fused(GAP_100X)
        q, _ = scaled_fp8_quant(small)
        distinct = q.to(torch.float32).unique().numel()
        self.assertGreater(distinct, RESOLUTION_FLOOR,
            f"Only {distinct} distinct FP8 levels (need {RESOLUTION_FLOOR})")


# 5000x gap: GDN in_proj_ba range

class TestGap5000x_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_error_bounded(self):
        """10.7% mean relative error on small shard."""
        fused, small, widths, offset = make_fused(GAP_5000X)
        err = small_shard_rel_error(fused, small, offset, single_scale_quant)
        self.assertLess(err, FP8_INTRINSIC_ERROR,
            f"Error {err:.4f} exceeds FP8 intrinsic limit")

    @unittest.expectedFailure
    def test_resolution_preserved(self):
        """Only 54 distinct FP8 levels (need 200)."""
        fused, _, widths, offset = make_fused(GAP_5000X)
        distinct = small_shard_distinct_levels(fused, offset)
        self.assertGreater(distinct, RESOLUTION_FLOOR,
            f"Only {distinct} distinct FP8 levels")


class TestGap5000x_PerShard_PASSES(unittest.TestCase):

    def test_error_bounded(self):
        fused, small, widths, offset = make_fused(GAP_5000X)
        err = small_shard_rel_error(fused, small, offset, per_shard_quant, widths)
        self.assertLess(err, FP8_INTRINSIC_ERROR,
            f"Error {err:.4f} exceeds FP8 intrinsic limit")

    def test_resolution_preserved(self):
        _, small, _, _ = make_fused(GAP_5000X)
        q, _ = scaled_fp8_quant(small)
        distinct = q.to(torch.float32).unique().numel()
        self.assertGreater(distinct, RESOLUTION_FLOOR,
            f"Only {distinct} distinct FP8 levels")


# 100000x gap

class TestGap100kx_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_error_bounded(self):
        """82.5% mean relative error."""
        fused, small, widths, offset = make_fused(GAP_100KX)
        err = small_shard_rel_error(fused, small, offset, single_scale_quant)
        self.assertLess(err, FP8_INTRINSIC_ERROR,
            f"Error {err:.4f} exceeds FP8 intrinsic limit")

    @unittest.expectedFailure
    def test_resolution_preserved(self):
        """Only 5 distinct FP8 levels."""
        fused, _, widths, offset = make_fused(GAP_100KX)
        distinct = small_shard_distinct_levels(fused, offset)
        self.assertGreater(distinct, RESOLUTION_FLOOR,
            f"Only {distinct} distinct FP8 levels")

    @unittest.expectedFailure
    def test_most_values_nonzero(self):
        """Most small-shard FP8 values collapse to zero."""
        fused, _, widths, offset = make_fused(GAP_100KX)
        q, _ = scaled_fp8_quant(fused)
        small_q = q[offset:].to(torch.float32)
        zero_frac = (small_q == 0).float().mean().item()
        self.assertLess(zero_frac, 0.1,
            f"{zero_frac:.1%} of small shard quantized to zero")


class TestGap100kx_PerShard_PASSES(unittest.TestCase):

    def test_error_bounded(self):
        fused, small, widths, offset = make_fused(GAP_100KX)
        err = small_shard_rel_error(fused, small, offset, per_shard_quant, widths)
        self.assertLess(err, FP8_INTRINSIC_ERROR,
            f"Error {err:.4f} exceeds FP8 intrinsic limit")

    def test_resolution_preserved(self):
        _, small, _, _ = make_fused(GAP_100KX)
        q, _ = scaled_fp8_quant(small)
        distinct = q.to(torch.float32).unique().numel()
        self.assertGreater(distinct, RESOLUTION_FLOOR,
            f"Only {distinct} distinct FP8 levels")

    def test_most_values_nonzero(self):
        _, small, _, _ = make_fused(GAP_100KX)
        q, _ = scaled_fp8_quant(small)
        zero_frac = (q.to(torch.float32) == 0).float().mean().item()
        self.assertLess(zero_frac, 0.1,
            f"{zero_frac:.1%} of values quantized to zero")


# Single-scale error grows with gap, per-shard stays flat

class TestSpectrum(unittest.TestCase):

    def test_single_scale_error_monotonic(self):
        errors = []
        for gap in [1, 10, 100, 1000, 10000, 100000]:
            fused, small, widths, offset = make_fused(gap)
            err = small_shard_rel_error(fused, small, offset, single_scale_quant)
            errors.append(err)
        for i in range(1, len(errors)):
            self.assertGreaterEqual(errors[i], errors[i-1] * 0.9,
                f"Error should grow with gap: {errors}")

    def test_per_shard_error_flat(self):
        for gap in [1, 100, 10000, 100000]:
            fused, small, widths, offset = make_fused(gap)
            err = small_shard_rel_error(fused, small, offset,
                                        per_shard_quant, widths)
            self.assertLess(err, FP8_INTRINSIC_ERROR,
                f"Per-shard error {err:.4f} at {gap}x gap exceeds limit")
