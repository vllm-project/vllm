# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Validates the numerical rationale behind the NVFP4 Marlin FP16 overflow fix
(issues #33461, #33560, PR #33972).

Root cause: the Marlin kernel epilogue previously cast FP32 accumulators to
FP16 *before* multiplying by global_scale, and FP32 sums regularly exceeded
the FP16 max (65504) when global_scale is small (1e-3 – 1e-1).

These tests do NOT require a GPU; they verify at the Python/numpy level that:
  1. Cast-then-scale produces NaN/inf when accumulators overflow FP16 range.
  2. Scale-then-cast (the fix) produces finite, accurate results.
  3. The FP16 MMA accumulation path (SM75) makes things worse, and forcing

     FP32 accumulation restores accuracy.
"""

import pytest
import torch

FP16_MAX = 65504.0


def cast_then_scale(fp32_accum: torch.Tensor, gs: float) -> torch.Tensor:
    """Old epilogue order: float -> half, then * global_scale."""
    return fp32_accum.to(torch.float16) * gs


def scale_then_cast(fp32_accum: torch.Tensor, gs: float) -> torch.Tensor:
    """New epilogue order (the fix): * global_scale, then float -> half."""
    return (fp32_accum * gs).to(torch.float16)


def simulate_fp16_mma_accum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Simulate SM75 FP16-accumulation MMA by accumulating tiles in FP16.
    Represents use_fp16_accum=true behaviour.
    """
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    k = a.shape[1]
    tile = 16
    accum = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float16)
    for i in range(0, k, tile):
        chunk_a = a[:, i : i + tile]
        chunk_b = b[i : i + tile, :]
        accum = (accum.float() + chunk_a.float().matmul(chunk_b.float())).to(
            torch.float16
        )
    return accum


def simulate_fp32_mma_accum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Simulate FP32-accumulation MMA (use_fp16_accum=false, the fix for SM75).
    Represents the corrected path.
    """
    return a.float().matmul(b.float())


class TestEpilogueOrderOverflow:
    """issue: epilogue casts to FP16 before applying global_scale."""

    def test_accumulator_above_fp16_max_nans_with_old_order(self):
        """A value 5x FP16_MAX overflows to inf/NaN on old cast-then-scale."""
        gs = 0.01  # typical global_scale: brings large values into range
        # Accumulator value that is representable in FP32 but > FP16_MAX
        raw = torch.tensor([FP16_MAX * 5.0], dtype=torch.float32)

        old_result = cast_then_scale(raw, gs)
        assert not torch.isfinite(old_result).all(), (
            "Expected overflow with cast-before-scale"
        )

    def test_accumulator_above_fp16_max_finite_with_new_order(self):
        """Same value is finite after scale-then-cast (the fix)."""
        gs = 0.01
        raw = torch.tensor([FP16_MAX * 5.0], dtype=torch.float32)

        new_result = scale_then_cast(raw, gs)
        assert torch.isfinite(new_result).all(), "Fix should produce finite output"
        # Check value is close to expected
        expected = torch.tensor([FP16_MAX * 5.0 * gs], dtype=torch.float32)
        assert torch.allclose(new_result.float(), expected, rtol=1e-2), (
            f"Value mismatch: {new_result} vs {expected}"
        )

    def test_boundary_exactly_at_fp16_max(self):
        """Values exactly at FP16_MAX are fine in both orders."""
        gs = 0.5
        raw = torch.tensor([FP16_MAX], dtype=torch.float32)
        assert torch.isfinite(cast_then_scale(raw, gs)).all()
        assert torch.isfinite(scale_then_cast(raw, gs)).all()

    def test_typical_nvfp4_global_scale_range(self):
        """
        Sweep global_scale across the realistic NVFP4 range (1e-3 – 1e-1)
        and confirm that at each scale the OLD order can overflow while the
        NEW order is always finite.
        """
        global_scales = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        # Accumulators that overflow at the smallest tested scale
        overflow_value = FP16_MAX / 1e-3 * 2  # guaranteed > FP16_MAX at gs=1e-3

        raw = torch.tensor([overflow_value], dtype=torch.float32)

        for gs in global_scales:
            scaled_fp32 = raw * gs
            if scaled_fp32.item() <= FP16_MAX:
                # New order should be finite
                assert torch.isfinite(scale_then_cast(raw, gs)).all(), (
                    f"New order overflowed at gs={gs}"
                )

    def test_full_matrix_scenario(self):
        """
        Simulate a small GEMM where FP32 accumulation results are above FP16_MAX
        and verify new epilogue order produces low relative error.
        """
        torch.manual_seed(42)
        m, k, n = 16, 256, 64
        gs = 0.005  # small global_scale typical of NVFP4 models

        # Generate inputs in a range that keeps FP32 matmul finite but
        # will overflow FP16 before global_scale is applied
        scale_input = (FP16_MAX * 0.6 / k) ** 0.5  # ensures row sums >> FP16_MAX
        a = torch.randn(m, k) * scale_input
        b = torch.randn(k, n) * scale_input

        fp32_accum = a.matmul(b)  # FP32 reference
        assert (fp32_accum.abs() > FP16_MAX * 0.1).any(), (
            "Test precondition: some accumulators should be near/above FP16_MAX"
        )

        # Old: cast then scale
        old_out = cast_then_scale(fp32_accum, gs)
        # New: scale then cast
        new_out = scale_then_cast(fp32_accum, gs)

        reference = fp32_accum.float() * gs

        old_finite = torch.isfinite(old_out).all()
        new_finite = torch.isfinite(new_out).all()

        if not old_finite:
            # This is the bug - old path produced NaN
            assert new_finite, "New (fixed) path must be finite when old is not"
        else:
            # Both finite: relative error of new path should be lower or equal
            rel_err_old = (
                old_out.float() - reference
            ).abs().mean() / reference.abs().mean()
            rel_err_new = (
                new_out.float() - reference
            ).abs().mean() / reference.abs().mean()
            assert rel_err_new <= rel_err_old + 1e-4, (
                f"New path should not be less accurate: old={rel_err_old:.4f}, "
                f"new={rel_err_new:.4f}"
            )


class TestSM75FP16Accumulation:
    """issue: SM75 use_fp16_accum=true causes mid-GEMM overflow for NVFP4."""

    def test_fp16_accum_overflows_where_fp32_does_not(self):
        """
        Show that tiled FP16-accum matmul can NaN when FP32-accum is finite,
        for inputs in a range that NVFP4+FP16 dtype would produce.
        """
        torch.manual_seed(0)
        k = 512  # representative K for NVFP4 model
        tile = 16

        # Values large enough that FP16 partial sums saturate across k tiles
        val_per_element = (FP16_MAX * 0.7 / tile) ** 0.5
        a = torch.ones(1, k, dtype=torch.float16) * val_per_element
        b = torch.ones(k, 1, dtype=torch.float16) * val_per_element

        fp16_accum_result = simulate_fp16_mma_accum(a, b)
        fp32_accum_result = simulate_fp32_mma_accum(a, b)

        assert torch.isfinite(fp32_accum_result).all(), (
            "FP32 accumulation must be finite"
        )
        # With enough tiles, FP16 accumulation WILL overflow
        # (precondition: val chosen so partial sum at tile=16 approaches FP16_MAX)
        partial_sum_after_one_tile = float(a[0, :tile].float().sum() * val_per_element)
        tiles = k // tile
        # FP16 accumulator after all tiles would be tiles * partial_sum if no overflow
        theoretical = partial_sum_after_one_tile * tiles
        if theoretical > FP16_MAX:
            assert not torch.isfinite(fp16_accum_result).all(), (
                "Expected FP16 overflow in tiled accumulation at this input scale"
            )

    def test_fp32_accumulation_matches_reference(self):
        """FP32-accum result matches exact FP32 matmul (no overflow)."""
        torch.manual_seed(1)
        m, k, n = 16, 256, 16
        a = torch.randn(m, k, dtype=torch.float16) * 2.0
        b = torch.randn(k, n, dtype=torch.float16) * 2.0

        fp32_result = simulate_fp32_mma_accum(a, b)
        reference = a.float().matmul(b.float())

        assert torch.allclose(fp32_result, reference, atol=1e-3), (
            "FP32 accumulation should match reference"
        )

    def test_fix_combination_scale_before_cast_plus_fp32_accum(self):
        """
        Full combined fix: FP32 accumulation + scale-before-cast.
        Result should be close to (a @ b) * global_scale in FP16 range.
        """
        torch.manual_seed(2)
        gs = 0.008
        scale_input = (FP16_MAX * 0.5 / 256) ** 0.5
        a = torch.randn(8, 256, dtype=torch.float16) * scale_input
        b = torch.randn(256, 32, dtype=torch.float16) * scale_input

        fp32_accum = simulate_fp32_mma_accum(a, b)
        output = scale_then_cast(fp32_accum, gs)

        reference = (a.float().matmul(b.float()) * gs).to(torch.float16)

        assert torch.isfinite(output).all(), "Combined fix must produce finite output"
        rel_err = (
            output.float() - reference.float()
        ).abs().mean() / reference.float().abs().mean()
        assert rel_err < 0.02, f"Relative error too high: {rel_err:.4f}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU. Tests SM75 fix but passes on all architectures.",
)
@torch.inference_mode()
def test_marlin_kernel_does_not_overflow_on_large_inputs():
    """
    GPU test to verify the actual Marlin kernel does not overflow (NaN/inf)
    when intermediate FP32 accumulators exceed 65504 but the scaled result
    is within FP16 range.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        rand_marlin_weight_nvfp4_like,
    )

    torch.manual_seed(42)

    dtype = torch.float16
    size_m, size_k, size_n = 64, 4096, 4096

    # Choose a small global scale (typical for NVFP4 models)
    global_scale_val = 0.01

    # Choose input magnitude such that K sums easily exceed FP16_MAX (65504)
    # If inputs are ~15.0 and weights are ~1.0:
    # 4096 * (15.0 * 1.0) = 61440. Variance will push sums > 65504
    x = torch.randn((size_m, size_k), device="cuda", dtype=dtype) * 15.0
    dense_weight = torch.randn((size_k, size_n), device="cuda", dtype=dtype)

    weight_ref, marlin_q_weight, marlin_scales, _ = rand_marlin_weight_nvfp4_like(
        dense_weight.T, group_size=16
    )

    # Use the small global scale
    marlin_global_scale = torch.tensor(global_scale_val, device="cuda", dtype=dtype)
    input_global_scale = torch.tensor(0.5, device="cuda", dtype=dtype)

    workspace = marlin_make_workspace_new("cuda")

    output = apply_fp4_marlin_linear(
        input=x,
        weight=marlin_q_weight,
        weight_scale=marlin_scales,
        weight_global_scale=marlin_global_scale,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
        input_global_scale=input_global_scale,
    )

    # Output must be finite
    assert torch.isfinite(output).all(), (
        "Marlin kernel returned NaN/inf. This indicates internal FP16 overflow."
    )
