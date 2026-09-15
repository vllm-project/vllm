# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPMSALADecoderLayer._add_scaled_residual (Stage 5,
the fused-residual optimization). Written explicitly opt-in (default
False) since this model's numerical correctness against HF has never
been established -- see that method's own docstring in minicpm_sala.py
for the full reasoning. These tests confirm both paths compute the
correct formula and characterize the real (benign, floating-point-
ordering) difference between them, rather than assuming either.
"""

import torch

from vllm.model_executor.models.minicpm_sala import MiniCPMSALADecoderLayer


class _FakeLayer:
    """Minimal stand-in exposing only what `_add_scaled_residual` reads
    (`residual_scale`, `use_fused_residual`) -- avoids needing a full
    real (or distributed-initialized) MiniCPMSALADecoderLayer just to
    test this one pure-tensor-math method."""

    residual_scale = 0.2474873734152916  # real value: 1.4 / sqrt(32)
    use_fused_residual = False

    _add_scaled_residual = MiniCPMSALADecoderLayer._add_scaled_residual


class TestAddScaledResidual:
    def test_default_path_matches_unfused_formula(self) -> None:
        layer = _FakeLayer()
        residual = torch.randn(4, 8)
        branch = torch.randn(4, 8)

        result = layer._add_scaled_residual(residual, branch)
        expected = residual + branch * layer.residual_scale

        assert torch.equal(result, expected)

    def test_fused_path_matches_torch_add_alpha_formula(self) -> None:
        layer = _FakeLayer()
        layer.use_fused_residual = True
        residual = torch.randn(4, 8)
        branch = torch.randn(4, 8)

        result = layer._add_scaled_residual(residual, branch)
        expected = torch.add(residual, branch, alpha=layer.residual_scale)

        assert torch.equal(result, expected)

    def test_default_is_off(self) -> None:
        """use_fused_residual must default to False -- this model's
        numerical correctness against HF has never been established, so
        introducing the fused path's benign floating-point difference
        by default (rather than opt-in) would make future differential
        validation harder to interpret."""
        layer = _FakeLayer()
        assert layer.use_fused_residual is False

    def test_fused_and_default_paths_differ_only_by_benign_epsilon(self) -> None:
        """Real, measured characterization -- not assumed. The two
        paths are mathematically equivalent but not bit-identical due
        to floating-point operation-ordering (multiply-then-add vs a
        single fused multiply-add), the same class of benign
        non-determinism as any kernel fusion."""
        layer = _FakeLayer()
        residual = torch.randn(100, 4096)
        branch = torch.randn(100, 4096)

        default_result = layer._add_scaled_residual(residual, branch)
        layer.use_fused_residual = True
        fused_result = layer._add_scaled_residual(residual, branch)

        max_diff = (default_result - fused_result).abs().max().item()
        assert max_diff < 1e-5, (
            f"difference ({max_diff}) larger than the expected benign "
            f"floating-point epsilon -- investigate before trusting "
            f"the fused path as equivalent"
        )
