# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the QuarkOCP_MX_MoEMethod emulate dispatch logic.

The emulate flag determines whether native CK / Triton MXFP4 kernels are
used or whether the computation falls back to high-precision emulation.
A Boolean-logic regression in this flag (PR #29008) caused gibberish
output on MI350X (Issue #36337) because the fallback was silently
disabled on MX-capable hardware.

These tests verify the flag is set correctly for every relevant
combination of (hardware_support × scheme × aiter_enabled × backend).
No GPU is required — all platform / env-var dependencies are mocked.
"""

import pytest


def _compute_emulate(
    supports_mx: bool,
    ocp_mx_scheme: str | None,
    use_rocm_aiter_moe: bool,
    mxfp4_backend_available: bool,
) -> bool:
    """Mirror the emulate logic from QuarkOCP_MX_MoEMethod.__init__.

    See vllm/model_executor/layers/quantization/quark/quark_moe.py,
    around line 733.
    """
    can_use_native_ck = (
        supports_mx
        and ocp_mx_scheme is not None
        and ocp_mx_scheme.startswith("w_mxfp4")
        and ocp_mx_scheme.endswith("a_mxfp4")
        and use_rocm_aiter_moe
    )
    can_use_mxfp4_backend = mxfp4_backend_available

    return not (can_use_native_ck or can_use_mxfp4_backend)


# ── Native CK path tests ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "supports_mx, scheme, aiter_enabled, backend, expected_emulate",
    [
        # All conditions met → native CK → no emulation
        (True, "w_mxfp4_a_mxfp4", True, False, False),
        # w_mxfp4 but activation is NOT a_mxfp4 → CK requires both → emulate
        (True, "w_mxfp4_a_fp8", True, False, True),
        (True, "w_mxfp4", True, False, True),
        # AITER disabled (VLLM_ROCM_USE_AITER_MOE=0) → must emulate
        (True, "w_mxfp4_a_mxfp4", False, False, True),
        # Hardware doesn't support MX → must emulate
        (False, "w_mxfp4_a_mxfp4", True, False, True),
        (False, "w_mxfp4_a_mxfp4", False, False, True),
        # Non-mxfp4 scheme → must emulate (no backend either)
        (True, "w_mxfp6_e3m2", True, False, True),
        (True, "w_mxfp6_e3m2_a_mxfp6_e3m2", True, False, True),
        (False, "w_mxfp6_e3m2", True, False, True),
        # scheme is None → must emulate
        (True, None, True, False, True),
    ],
    ids=[
        "mi350x-w4a4-aiter_on",
        "mi350x-w4afp8-no_ck_needs_a_mxfp4",
        "mi350x-w4_only-no_ck_needs_a_mxfp4",
        "mi350x-w4a4-aiter_off",
        "no_mx-w4a4-aiter_on",
        "no_mx-w4a4-aiter_off",
        "mi350x-mxfp6-aiter_on",
        "mi350x-mxfp6_sym-aiter_on",
        "no_mx-mxfp6-aiter_on",
        "mi350x-none_scheme-aiter_on",
    ],
)
def test_emulate_native_ck_path(
    supports_mx: bool,
    scheme: str | None,
    aiter_enabled: bool,
    backend: bool,
    expected_emulate: bool,
):
    result = _compute_emulate(supports_mx, scheme, aiter_enabled, backend)
    assert result == expected_emulate, (
        f"emulate should be {expected_emulate} for "
        f"supports_mx={supports_mx}, scheme={scheme!r}, "
        f"aiter_enabled={aiter_enabled}, backend={backend}"
    )


# ── Triton mxfp4 backend tests ────────────────────────────────────────


@pytest.mark.parametrize(
    "supports_mx, scheme, aiter_enabled, backend, expected_emulate",
    [
        # Backend available → no emulation, even without CK
        (False, "w_mxfp4", False, True, False),
        (True, "w_mxfp4", False, True, False),
        # Backend available + CK also available → still no emulation
        (True, "w_mxfp4_a_mxfp4", True, True, False),
    ],
    ids=[
        "no_mx-backend_on-aiter_off",
        "mi350x-backend_on-aiter_off",
        "mi350x-backend_on-aiter_on",
    ],
)
def test_emulate_mxfp4_backend_path(
    supports_mx: bool,
    scheme: str | None,
    aiter_enabled: bool,
    backend: bool,
    expected_emulate: bool,
):
    result = _compute_emulate(supports_mx, scheme, aiter_enabled, backend)
    assert result == expected_emulate


# ── Regression test for Issue #36337 ──────────────────────────────────


def test_regression_issue_36337_aiter_disabled_forces_emulation():
    """On MI350X (supports_mx=True) with w_mxfp4_a_mxfp4 scheme,
    setting VLLM_ROCM_USE_AITER_MOE=0 (aiter_enabled=False) MUST
    result in emulate=True so the user can fall back to the safe
    emulation path when AITER CK kernels are incompatible.

    The old logic (PR #29008) evaluated to emulate=False here because:
        (not True or not True) and (...) → (False) and (...) → False
    """
    result = _compute_emulate(
        supports_mx=True,
        ocp_mx_scheme="w_mxfp4_a_mxfp4",
        use_rocm_aiter_moe=False,
        mxfp4_backend_available=False,
    )
    assert result is True, (
        "emulate must be True when AITER is disabled on MI350X — "
        "this is the exact regression from Issue #36337"
    )
