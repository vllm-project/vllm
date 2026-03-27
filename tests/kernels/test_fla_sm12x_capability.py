# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that FLA capability and SMEM checks correctly classify SM12x.

SM12x (desktop Blackwell: RTX 5090/5080, DGX Spark GB10) has capability
major=12, which previously triggered >= 9 checks designed for Hopper (SM90).
SM12x needs different NUM_WARPS tuning and has insufficient shared memory
for TMA code paths.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fla.ops.utils import (
    MIN_SMEM_FOR_TMA,
    check_nvidia_hopper,
    check_tma_supported,
)


def _make_platform(capability, device_name):
    """Create a mock platform object with the given capability and name."""
    major, minor = capability
    cap_int = major * 10 + minor

    return SimpleNamespace(
        get_device_name=lambda device_id=0: device_name,
        has_device_capability=lambda cap, device_id=0: (
            (major, minor) >= cap if isinstance(cap, tuple)
            else cap_int >= cap
        ),
        is_device_capability_family=lambda cap, device_id=0: (
            cap_int // 10 == cap // 10
        ),
    )


@pytest.mark.parametrize(
    "capability,device_name,expected_hopper",
    [
        # Ampere (SM80) — not Hopper
        ((8, 0), "NVIDIA A100", False),
        # Hopper (SM90)
        ((9, 0), "NVIDIA H100", True),
        # Datacenter Blackwell (SM100/SM110) — "Hopper-like" for warp tuning
        ((10, 0), "NVIDIA B200", True),
        ((11, 0), "NVIDIA B300", True),
        # Desktop Blackwell (SM120) — NOT Hopper-like
        ((12, 0), "NVIDIA GeForce RTX 5090", False),
        # DGX Spark GB10 (SM121)
        ((12, 1), "Orin (nvgpu)", False),
    ],
)
def test_check_nvidia_hopper(capability, device_name, expected_hopper):
    platform = _make_platform(capability, device_name)
    assert check_nvidia_hopper(platform) == expected_hopper, (
        f"capability {capability} ({device_name}): check_nvidia_hopper "
        f"should be {expected_hopper}"
    )


@pytest.mark.parametrize(
    "smem_bytes,expected_tma",
    [
        # SM12x desktop: 101KB — below threshold
        (101376, False),
        # Hopper/datacenter Blackwell: 228KB — above threshold
        (232448, True),
        # Exactly at threshold
        (MIN_SMEM_FOR_TMA, True),
        # Just below threshold
        (MIN_SMEM_FOR_TMA - 1, False),
    ],
)
def test_check_tma_supported(smem_bytes, expected_tma):
    platform = _make_platform((9, 0), "NVIDIA H100")
    # Mock triton to have TMA descriptor support
    with patch(
        "vllm.model_executor.layers.fla.ops.utils.triton"
    ) as mock_triton:
        mock_triton.language = SimpleNamespace(
            make_tensor_descriptor=lambda: None,
        )
        assert check_tma_supported(platform, smem_bytes) == expected_tma, (
            f"smem={smem_bytes}: check_tma_supported "
            f"should be {expected_tma}"
        )


def test_check_tma_no_triton_support():
    """TMA should be False even with enough SMEM if triton lacks descriptors."""
    platform = _make_platform((9, 0), "NVIDIA H100")
    with patch(
        "vllm.model_executor.layers.fla.ops.utils.triton"
    ) as mock_triton:
        mock_triton.language = SimpleNamespace()  # no TMA attrs
        assert not check_tma_supported(platform, 232448)
