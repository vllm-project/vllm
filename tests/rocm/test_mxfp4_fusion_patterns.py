# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 kernel fusion patterns.

Verifies that the MXFP4 AllReduce and standalone RMSNorm fusion patterns
register correctly, that feature probes return bool, and that pattern/
replacement callables are tracing-compatible.  GPU-level end-to-end tests
are skipped when ROCm is unavailable.
"""

import pytest
import torch


# ── Test 1: Feature probes return bool ───────────────────────────────────────
def test_feature_probe_allreduce_returns_bool():
    """has_fused_allreduce_rmsnorm_mxfp4_quant must never raise — returns False
    gracefully when the fused AITER kernel is absent."""
    try:
        from vllm._aiter_ops import rocm_aiter_ops
    except ImportError:
        pytest.skip("vllm._aiter_ops not available")

    result = rocm_aiter_ops.has_fused_allreduce_rmsnorm_mxfp4_quant()
    assert isinstance(result, bool), (
        f"Expected bool from has_fused_allreduce_rmsnorm_mxfp4_quant, "
        f"got {type(result)}"
    )


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
    aiter.ops.triton.fused_mxfp4_quant.fused_rms_mxfp4_quant.

    This test passes even without ROCm — it only checks that the probe
    faithfully reflects what AITER exports, not that a GPU is present.
    """
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


# ── Test 2: AR Pattern A instantiation (no residual) ─────────────────────────
def test_ar_pattern_a_instantiation():
    """AiterAllreduceFusedRMSNormMXFP4QuantPattern instantiates and exposes
    callable pattern/replacement with correct get_inputs() length."""
    try:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            AiterAllreduceFusedRMSNormMXFP4QuantPattern,
        )
    except (ImportError, AttributeError):
        pytest.skip("allreduce_rms_fusion not importable (requires vllm C-extension)")

    p = AiterAllreduceFusedRMSNormMXFP4QuantPattern(
        epsilon=1e-6,
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert callable(p.pattern), "pattern must be callable"
    assert callable(p.replacement), "replacement must be callable"

    inputs = p.get_inputs()
    assert len(inputs) == 2, (
        f"Pattern A (no residual) needs 2 inputs: input_, weight; got {len(inputs)}"
    )
    assert inputs[0].dtype == torch.bfloat16
    assert inputs[1].shape == (16,)


# ── Test 3: AR Pattern B instantiation (with residual) ───────────────────────
def test_ar_pattern_b_instantiation():
    """AiterAllreduceFusedAddRMSNormMXFP4QuantPattern instantiates and
    get_inputs() returns 3 tensors."""
    try:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            AiterAllreduceFusedAddRMSNormMXFP4QuantPattern,
        )
    except (ImportError, AttributeError):
        pytest.skip("allreduce_rms_fusion not importable (requires vllm C-extension)")

    p = AiterAllreduceFusedAddRMSNormMXFP4QuantPattern(
        epsilon=1e-6,
        dtype=torch.bfloat16,
        device="cpu",
    )
    inputs = p.get_inputs()
    assert len(inputs) == 3, (
        f"Pattern B (with residual) needs 3 inputs: residual, input_, weight; "
        f"got {len(inputs)}"
    )
    assert all(t.dtype == torch.bfloat16 for t in inputs)


# ── Test 4: Standalone pattern instantiation ─────────────────────────────────
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


# ── Test 5: Custom ops are registered ────────────────────────────────────────
def test_custom_ops_registered():
    """Verify that the six new MXFP4 custom ops appear under torch.ops.vllm
    after _aiter_ops is imported and AITER is available."""
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
        "rocm_aiter_fused_allreduce_rmsnorm_mxfp4_quant",
        "rocm_aiter_fused_allreduce_add_rmsnorm_mxfp4_quant",
    ]
    for op_name in expected_ops:
        assert hasattr(torch.ops.vllm, op_name), (
            f"torch.ops.vllm.{op_name} not registered — "
            "check direct_register_custom_op call in _aiter_ops.py"
        )


# ── Test 6: AR pattern registration order ────────────────────────────────────
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires ROCm GPU to initialise allreduce communicator",
)
def test_ar_pattern_registration_order():
    """Pattern B (with residual, larger) must be registered before Pattern A
    (no residual, smaller) in RocmAiterAllReduceFusionPass.

    Greedy matching depends on this ordering: Pattern B fires for layers
    1..N (has residual) and Pattern A fires only for layer 0 (no residual).
    """
    try:
        from vllm._aiter_ops import rocm_aiter_ops
    except (ImportError, AttributeError):
        pytest.skip("vllm._aiter_ops not available (requires vllm C-extension)")

    if not rocm_aiter_ops.has_fused_allreduce_rmsnorm_mxfp4_quant():
        pytest.skip("MXFP4 fused AR kernel not available in this AITER build")

    try:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            AiterAllreduceFusedAddRMSNormMXFP4QuantPattern,
            AiterAllreduceFusedRMSNormMXFP4QuantPattern,
            RocmAiterAllReduceFusionPass,
        )
        from vllm.config import VllmConfig
    except (ImportError, AttributeError):
        pytest.skip("allreduce_rms_fusion not importable (requires vllm C-extension)")

    cfg = VllmConfig()
    fusion_pass = RocmAiterAllReduceFusionPass(cfg)

    registered_names = [type(p).__name__ for p in fusion_pass._patterns]

    idx_b = next(
        (
            i
            for i, name in enumerate(registered_names)
            if name == AiterAllreduceFusedAddRMSNormMXFP4QuantPattern.__name__
        ),
        None,
    )
    idx_a = next(
        (
            i
            for i, name in enumerate(registered_names)
            if name == AiterAllreduceFusedRMSNormMXFP4QuantPattern.__name__
        ),
        None,
    )

    assert idx_b is not None, "Pattern B (with residual) not registered"
    assert idx_a is not None, "Pattern A (no residual) not registered"
    assert idx_b < idx_a, (
        f"Pattern B must be registered before Pattern A for greedy matching. "
        f"Got B at index {idx_b}, A at index {idx_a}"
    )
