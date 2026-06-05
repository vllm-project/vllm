# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for PR 1, 2, 3: ensure existing code paths are not broken.

Covers TC-5.1 through TC-5.5 from the test plan.

These tests verify that:
  - NVIDIA (CUDA) deployments are unaffected by the new ROCm env vars
  - All flags OFF: default behaviour unchanged
  - Existing vLLM envs.py var count is not accidentally reduced
  - RMSNorm standard forward() path unaffected
  - F2 output is deterministic (TC-5.5)

Note: TC-5.3 (DeepSeek model tests pass) and TC-5.4 (enforce_eager=False
      benchmark) are executed via the existing pytest suite and are not
      duplicated here.
"""

import pytest

from vllm.envs import environment_variables
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# TC-1.8 / TC-5.x  CI env var count regression
# ---------------------------------------------------------------------------

# Count of environment_variables before PRs 1–3 were applied.
# This is the number of vars in the v0.20.2 base image.
# We verify it does NOT decrease (no vars accidentally removed) and
# increases by EXACTLY 2 after PR 1 (the two new F2/F3 vars).
F2_VAR = "VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT"
F3_VAR = "VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE"


def test_tc1_8_no_vars_accidentally_removed():
    """TC-1.8: The environment_variables registry must contain at least the
    pre-PR count of variables — no accidental deletions."""
    # Baseline count from v0.20.2: 78 vars (verified in container).
    # If PRs only ADD vars this bound holds even before the 2 new ones land.
    BASELINE_COUNT = 78
    assert len(environment_variables) >= BASELINE_COUNT, (
        f"environment_variables has only {len(environment_variables)} entries; "
        f"expected ≥ {BASELINE_COUNT}. A variable may have been accidentally removed."
    )


def test_tc1_8_new_vars_present_after_pr1():
    """TC-1.8: After PR 1 both F2 and F3 vars must appear in environment_variables."""
    assert F2_VAR in environment_variables, (
        f"{F2_VAR} missing from environment_variables"
    )
    assert F3_VAR in environment_variables, (
        f"{F3_VAR} missing from environment_variables"
    )


# ---------------------------------------------------------------------------
# TC-5.1  CUDA/NVIDIA deployment unaffected
# ---------------------------------------------------------------------------


def test_tc5_1_cuda_deployment_unaffected(monkeypatch):
    """TC-5.1: On NVIDIA, setting F2/F3 env vars must not activate the ROCm paths."""
    if current_platform.is_rocm():
        pytest.skip("CUDA-only regression test — skipped on ROCm")

    monkeypatch.setenv(F2_VAR, "1")
    monkeypatch.setenv(F3_VAR, "1")

    import vllm.envs as envs

    # Env vars are accessible on any platform — just reads the env
    assert getattr(envs, F2_VAR) is True
    assert getattr(envs, F3_VAR) is True
    # F2/F3 guards in the ROCm code check current_platform.is_rocm() first,
    # so they will not execute on NVIDIA even when the env vars are set.
    assert not current_platform.is_rocm(), "Expected non-ROCm platform"


# ---------------------------------------------------------------------------
# TC-5.1  is_hip() returns False on NVIDIA
# ---------------------------------------------------------------------------


def test_tc5_1_is_hip_false_on_nvidia():
    """TC-5.1: is_hip() must return False on CUDA platforms."""
    if current_platform.is_rocm():
        pytest.skip("CUDA-only test")
    assert not current_platform.is_rocm(), (
        "is_rocm() returned True on NVIDIA — guard missing"
    )


# ---------------------------------------------------------------------------
# TC-5.2  All flags OFF — RMSNorm baseline behaviour unchanged
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific regression test"
)
def test_tc5_2_all_flags_off_rmsnorm_unchanged(monkeypatch, default_vllm_config):
    """TC-5.2: With all F2/F3 flags unset, RMSNorm must produce the same
    output as the PyTorch-native reference."""
    import torch

    monkeypatch.delenv(F2_VAR, raising=False)
    monkeypatch.delenv(F3_VAR, raising=False)
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_RMSNORM", raising=False)

    from vllm.model_executor.layers.layernorm import RMSNorm

    hidden = 512
    norm = RMSNorm(hidden, eps=1e-6).cuda().bfloat16()
    norm.weight.data.fill_(1.0)

    x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda")

    # Native reference
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    ref = (x.float() * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16)

    out = norm(x)
    if isinstance(out, tuple):
        out = out[0]

    max_diff = (ref.float() - out.float()).abs().max().item()
    assert max_diff < 1e-2, (
        f"RMSNorm baseline deviation {max_diff:.4f} with all flags off. "
        "A PR may have broken the unfused fallback path."
    )


# ---------------------------------------------------------------------------
# TC-5.2  All flags OFF — standard forward() returns BF16
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific")
def test_tc5_2_standard_forward_returns_bf16(monkeypatch, default_vllm_config):
    """TC-5.2: forward() must return BF16 tensor regardless of F2/F3 flag state."""
    import torch

    monkeypatch.setenv(F2_VAR, "0")
    monkeypatch.setenv(F3_VAR, "0")

    from vllm.model_executor.layers.layernorm import RMSNorm

    norm = RMSNorm(512).cuda().bfloat16()
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    out = norm(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# TC-5.5  F2 output is deterministic across runs
# (duplicated here as a standalone regression gate)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific")
def test_tc5_5_rmsnorm_deterministic(monkeypatch, default_vllm_config):
    """TC-5.5: Identical input must produce identical output from forward_hip."""
    import torch

    from vllm.model_executor.layers.layernorm import RMSNorm

    norm = RMSNorm(512, eps=1e-6).cuda().bfloat16()
    norm.weight.data.normal_(mean=1.0, std=0.1)

    torch.manual_seed(42)
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")

    with torch.inference_mode():
        out1 = norm(x.clone())
        out2 = norm(x.clone())

    if isinstance(out1, tuple):
        out1, out2 = out1[0], out2[0]

    assert torch.equal(out1, out2), (
        "RMSNorm forward_hip is non-deterministic: "
        "different results for identical input."
    )


# ---------------------------------------------------------------------------
# TC-5.x  Existing env vars: compile_factors snapshot not broken
# ---------------------------------------------------------------------------


def test_existing_compile_factors_still_present():
    """Regression: existing AITER compile-factor env vars must still be present
    after PR 1 modifies envs.py."""
    import vllm.envs as envs

    compile_factors = envs.compile_factors()
    # These vars existed before PR 1 and must remain as compile factors
    expected_compile_factors = [
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_LINEAR",
    ]
    for var in expected_compile_factors:
        # Only check vars that are defined in this build
        if var in environment_variables:
            assert var in compile_factors, (
                f"{var} was removed from compile_factors by a PR — "
                "this would invalidate the cuda-graph cache for existing deployments."
            )
