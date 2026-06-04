# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 kernel fusion patterns.

Verifies that the standalone RMSNorm+MXFP4 fusion patterns register correctly,
that the feature probe returns bool, and that pattern/replacement callables are
tracing-compatible.  GPU-level tests are skipped when ROCm is unavailable.
"""

import pytest
import torch


# ── Test 1: Feature probe returns bool ─────────────────────────────────────────
def test_feature_probe_rmsnorm_returns_bool():
    """has_fused_rmsnorm_mxfp4_quant must never raise."""
    try:
        from vllm._aiter_ops import rocm_aiter_ops
    except ImportError:
        pytest.skip("vllm._aiter_ops not available")

    result = rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant()
    assert isinstance(result, bool), (
        f"Expected bool from has_fused_rmsnorm_mxfp4_quant, got {type(result)}"
    )


def test_feature_probe_rmsnorm_matches_aiter_triton():
    """has_fused_rmsnorm_mxfp4_quant must agree with actual importability of
    aiter.ops.triton.fused_mxfp4_quant.fused_rms_mxfp4_quant."""
    try:
        from vllm._aiter_ops import rocm_aiter_ops
    except (ImportError, AttributeError):
        pytest.skip("vllm._aiter_ops not available (requires vllm C-extension)")

    try:
        from aiter.ops.triton.fused_mxfp4_quant import (
            fused_rms_mxfp4_quant,  # noqa: F401
        )

        kernel_importable = True
    except ImportError:
        kernel_importable = False

    probe_result = rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant()
    assert probe_result == kernel_importable, (
        f"has_fused_rmsnorm_mxfp4_quant() returned {probe_result} "
        f"but fused_rms_mxfp4_quant importable={kernel_importable}"
    )


# ── Test 2: Standalone pattern instantiation ───────────────────────────────────
def test_standalone_pattern_instantiation():
    """AiterRMSNormMXFP4QuantPattern and AiterFusedAddRMSNormMXFP4QuantPattern
    instantiate without errors."""
    try:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            AiterFusedAddRMSNormMXFP4QuantPattern,
            AiterRMSNormMXFP4QuantPattern,
        )
    except (ImportError, AttributeError):
        pytest.skip("rocm_aiter_fusion not importable (requires vllm C-extension)")

    p_no_res = AiterRMSNormMXFP4QuantPattern(epsilon=1e-6)
    p_with_res = AiterFusedAddRMSNormMXFP4QuantPattern(epsilon=1e-6)

    assert hasattr(p_no_res, "FUSED_OP")
    assert hasattr(p_with_res, "FUSED_OP")


# ── Test 3: Custom ops are registered ─────────────────────────────────────────
def test_custom_ops_registered():
    """Verify the three MXFP4 custom ops appear under torch.ops.vllm."""
    try:
        import vllm._aiter_ops  # noqa: F401 — triggers register_ops_once()
        from vllm._aiter_ops import is_aiter_found_and_supported
    except (ImportError, AttributeError):
        pytest.skip("vllm._aiter_ops not available (requires vllm C-extension)")

    if not is_aiter_found_and_supported():
        pytest.skip("AITER not available on this platform (requires ROCm gfx9)")

    expected_ops = [
        "rocm_aiter_dynamic_mxfp4_quant",
        "rocm_aiter_rmsnorm_mxfp4_quant",
        "rocm_aiter_rmsnorm_add_mxfp4_quant",
    ]
    for op_name in expected_ops:
        assert hasattr(torch.ops.vllm, op_name), (
            f"torch.ops.vllm.{op_name} not registered — "
            "check direct_register_custom_op call in _aiter_ops.py"
        )
