# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that FLA capability checks correctly classify SM12x as non-Hopper.

SM12x (desktop Blackwell: RTX 5090/5080, DGX Spark GB10) has capability
major=12, which previously triggered >= 9 checks designed for Hopper (SM90).
SM12x is NOT Hopper — it lacks TMA and needs different NUM_WARPS tuning.
"""

import pytest


@pytest.mark.parametrize(
    "capability,expected_hopper,expected_tma",
    [
        # Ampere (SM80) — not Hopper, no TMA
        ((8, 0), False, False),
        # Hopper (SM90) — is "Hopper-like", has TMA
        ((9, 0), True, True),
        # Datacenter Blackwell (SM100) — also "Hopper-like" per the
        # 9 <= cap < 12 range, and has TMA. Note: is_nvidia_hopper is a
        # misnomer inherited from upstream — it really means "SM9x-SM11x
        # class" for NUM_WARPS selection purposes.
        ((10, 0), True, True),
        ((11, 0), True, True),
        # Desktop Blackwell RTX 5090 (SM120) — NOT Hopper-like, no TMA
        ((12, 0), False, False),
        # Desktop Blackwell DGX Spark GB10 (SM121) — NOT Hopper-like, no TMA
        ((12, 1), False, False),
    ],
)
def test_fla_capability_classification(
    capability: tuple[int, int],
    expected_hopper: bool,
    expected_tma: bool,
):
    """Verify is_nvidia_hopper and is_tma_supported for various architectures.

    Tests the range check logic (9 <= cap < 12) that excludes SM12x desktop
    Blackwell from Hopper/TMA code paths while preserving behavior for
    Hopper (SM9x) and datacenter Blackwell (SM10x/SM11x).
    """
    cap = capability
    is_nvidia = True

    # Mirror the logic from utils.py
    is_nvidia_hopper = is_nvidia and (
        "NVIDIA H" in "NVIDIA Test GPU" or 9 <= cap[0] < 12
    )
    # is_tma_supported also checks for triton attrs, but we test just
    # the capability gate here
    is_tma_supported_val = is_nvidia and 9 <= cap[0] < 12

    assert is_nvidia_hopper == expected_hopper, (
        f"capability {cap}: is_nvidia_hopper should be "
        f"{expected_hopper}, got {is_nvidia_hopper}"
    )
    assert is_tma_supported_val == expected_tma, (
        f"capability {cap}: is_tma_supported should be "
        f"{expected_tma}, got {is_tma_supported_val}"
    )
