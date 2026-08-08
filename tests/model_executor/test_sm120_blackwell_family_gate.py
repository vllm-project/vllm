# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SM120 (consumer Blackwell) inclusion in family(100) capability gates.

Several kernel-selection code paths used ``is_device_capability_family(100)``
to detect Blackwell GPUs, but that check matches only major==10 (SM100/SM103
datacenter). SM120 (RTX 5090, major==12) was silently excluded from
Blackwell-optimized kernels.

These tests verify:
1. Source invariant: patched files check both family(100) and family(120).
2. Runtime invariant: the family logic matches SM120 on real hardware.
"""
from __future__ import annotations

import importlib.util
import os
from unittest.mock import MagicMock

import pytest


class FakeCapability:
    """Mimics vllm.platforms.DeviceCapability."""

    def __init__(self, major: int, minor: int = 0):
        self.major = major
        self.minor = minor

    def to_int(self) -> int:
        return self.major * 10 + self.minor


def _make_platform(capability: FakeCapability) -> MagicMock:
    """Create a mock current_platform that responds to family checks."""
    platform = MagicMock()
    platform.get_device_capability.return_value = capability

    def is_family(family_val: int) -> bool:
        return (capability.to_int() // 10) == (family_val // 10)

    def is_capability(pair: tuple[int, int]) -> bool:
        return capability.major == pair[0] and capability.minor == pair[1]

    platform.is_device_capability_family.side_effect = is_family
    platform.is_device_capability.side_effect = is_capability
    platform.is_cuda.return_value = True
    return platform


SM100 = FakeCapability(10, 0)
SM103 = FakeCapability(10, 3)
SM120 = FakeCapability(12, 0)
SM121 = FakeCapability(12, 1)
SM90 = FakeCapability(9, 0)
SM89 = FakeCapability(8, 9)

ALL_BLACKWELL = [SM100, SM103, SM120, SM121]
NON_BLACKWELL = [SM90, SM89]

# Files patched to include family(120) alongside family(100).
PATCHED_FILES = [
    "model_executor/layers/fused_moe/router/gate_linear.py",
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "model_executor/layers/mamba/ops/replayssm_config.py",
    "model_executor/layers/mamba/mamba_mixer2.py",
    "model_executor/layers/quantization/online/nvfp4.py",
]


def _get_vllm_source_root() -> str:
    """Find the vllm source directory (installed package)."""
    spec = importlib.util.find_spec("vllm")
    assert spec is not None, "vllm not installed"
    assert spec.origin is not None, "vllm has no origin"
    return os.path.dirname(spec.origin)


@pytest.mark.parametrize("rel_path", PATCHED_FILES)
def test_patched_file_checks_family120(rel_path: str) -> None:
    """Invariant: every file that checks family(100) also checks family(120).

    Prevents regressions where someone adds a new family(100) gate without
    including consumer Blackwell (SM120/SM121).
    """
    root = _get_vllm_source_root()
    full_path = os.path.join(root, rel_path)
    assert os.path.exists(full_path), f"File not found: {full_path}"

    with open(full_path) as f:
        source = f.read()

    f100 = source.count("is_device_capability_family(100)")
    f120 = source.count("is_device_capability_family(120)")

    if f100 > 0:
        assert f120 > 0, (
            f"{rel_path} checks family(100) {f100}x but never checks"
            f" family(120) — consumer Blackwell (SM120) is excluded"
        )


@pytest.mark.parametrize("cap", ALL_BLACKWELL, ids=["SM100", "SM103", "SM120", "SM121"])
def test_is_blackwell_on_all_blackwell(cap: FakeCapability) -> None:
    """The corrected is_blackwell expression must be True for all Blackwell."""
    platform = _make_platform(cap)
    is_blackwell = (
        platform.is_device_capability_family(100)
        or platform.is_device_capability_family(120)
    )
    assert is_blackwell is True


@pytest.mark.parametrize("cap", NON_BLACKWELL, ids=["SM90", "SM89"])
def test_not_blackwell_on_non_blackwell(cap: FakeCapability) -> None:
    """The corrected is_blackwell expression must be False for non-Blackwell."""
    platform = _make_platform(cap)
    is_blackwell = (
        platform.is_device_capability_family(100)
        or platform.is_device_capability_family(120)
    )
    assert is_blackwell is False


def test_runtime_sm120_family() -> None:
    """On real SM120 hardware, family(120) must return True.

    Skipped if no GPU or not SM120.
    """
    try:
        from vllm.platforms import current_platform
    except Exception:
        pytest.skip("vllm not importable")

    cap = current_platform.get_device_capability()
    if cap is None:
        pytest.skip("No GPU detected")
    if cap.major != 12:
        pytest.skip(f"This test requires SM120, got SM{cap.major}{cap.minor}")

    assert current_platform.is_device_capability_family(120) is True
    # SM120 must NOT match family(100) — that's the bug we're fixing
    assert current_platform.is_device_capability_family(100) is False
    # The combined check is what makes it work
    combined = (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    )
    assert combined is True
