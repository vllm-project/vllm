# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for is_blackwell_class() Blackwell-family GPU detection.

Verifies that the unified Blackwell-class check correctly identifies
SM10x, SM11x, and SM12x devices while excluding non-Blackwell GPUs.
"""

import importlib.util

import pytest

from vllm.platforms.interface import DeviceCapability, Platform


def _has_vllm_c() -> bool:
    """Check if compiled vllm._C extension is available."""
    return importlib.util.find_spec("vllm._C") is not None


class _FakePlatform(Platform):
    """Minimal Platform subclass for testing capability methods."""

    _capability: DeviceCapability | None = None

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        return cls._capability


# ── is_blackwell_class parametrized tests ──────────────────────────


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        # Pre-Blackwell architectures → False
        (7, 0, False),  # Volta (V100)
        (7, 5, False),  # Turing (RTX 2080)
        (8, 0, False),  # Ampere (A100)
        (8, 6, False),  # Ampere (RTX 3060)
        (8, 9, False),  # Ada Lovelace (RTX 4090)
        (9, 0, False),  # Hopper (H100)
        # Blackwell-class architectures → True
        (10, 0, True),  # B100/B200
        (10, 1, True),  # B200 variant
        (10, 3, True),  # B200 variant
        (11, 0, True),  # Future Blackwell
        (12, 0, True),  # GB10/DGX Spark (SM120)
        (12, 1, True),  # GB10/DGX Spark (SM121)
        # Future / post-Blackwell → False
        (13, 0, False),
        (15, 0, False),
    ],
    ids=lambda v: str(v),
)
def test_is_blackwell_class(major: int, minor: int, expected: bool):
    _FakePlatform._capability = DeviceCapability(major=major, minor=minor)
    assert _FakePlatform.is_blackwell_class() is expected


def test_is_blackwell_class_none_capability():
    """is_blackwell_class returns False when no capability is available."""
    _FakePlatform._capability = None
    assert _FakePlatform.is_blackwell_class() is False


# ── is_blackwell_capability staticmethod tests ─────────────────────


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        (9, 0, False),
        (10, 0, True),
        (11, 0, True),
        (12, 1, True),
        (13, 0, False),
    ],
    ids=lambda v: str(v),
)
def test_is_blackwell_capability_static(major: int, minor: int, expected: bool):
    """Staticmethod works directly on DeviceCapability without device query."""
    cap = DeviceCapability(major=major, minor=minor)
    assert Platform.is_blackwell_capability(cap) is expected


def test_is_blackwell_capability_consistency():
    """Staticmethod and classmethod agree for all Blackwell variants."""
    for major in (10, 11, 12):
        cap = DeviceCapability(major=major, minor=0)
        _FakePlatform._capability = cap
        assert (
            Platform.is_blackwell_capability(cap) is _FakePlatform.is_blackwell_class()
        )


# ── is_device_capability_family consistency check ──────────────────


@pytest.mark.parametrize(
    ("major", "minor", "family"),
    [
        (10, 0, 100),
        (10, 3, 100),
        (11, 0, 110),
        (12, 0, 120),
        (12, 1, 120),
    ],
)
def test_blackwell_class_covers_all_families(major: int, minor: int, family: int):
    """Every Blackwell family (100, 110, 120) is also blackwell_class."""
    _FakePlatform._capability = DeviceCapability(major=major, minor=minor)
    assert _FakePlatform.is_device_capability_family(family) is True
    assert _FakePlatform.is_blackwell_class() is True


# ── Backend priority integration (mocked) ─────────────────────────


@pytest.mark.skipif(
    not _has_vllm_c(),
    reason="Requires compiled vllm._C extension",
)
def test_backend_priorities_sm121():
    """SM121 should get Blackwell backend priorities (FlashInfer first)."""
    from vllm.platforms.cuda import _get_backend_priorities

    cap = DeviceCapability(major=12, minor=1)
    # Non-MLA path: Blackwell should get FlashInfer first
    priorities = _get_backend_priorities(cap, use_mla=False)
    backend_names = [b.name for b in priorities]
    assert "FLASHINFER" in backend_names
    # FlashInfer should be before FlashAttn for Blackwell
    fi_idx = backend_names.index("FLASHINFER")
    if "FLASH_ATTN" in backend_names:
        fa_idx = backend_names.index("FLASH_ATTN")
        assert fi_idx < fa_idx, (
            f"FlashInfer ({fi_idx}) should come before FlashAttn ({fa_idx}) "
            f"for SM121 Blackwell-class GPU"
        )


@pytest.mark.skipif(
    not _has_vllm_c(),
    reason="Requires compiled vllm._C extension",
)
def test_backend_priorities_sm100_unchanged():
    """SM100 (B200) should still get Blackwell backend priorities."""
    from vllm.platforms.cuda import _get_backend_priorities

    cap = DeviceCapability(major=10, minor=0)
    priorities = _get_backend_priorities(cap, use_mla=False)
    backend_names = [b.name for b in priorities]
    assert "FLASHINFER" in backend_names


@pytest.mark.skipif(
    not _has_vllm_c(),
    reason="Requires compiled vllm._C extension",
)
def test_backend_priorities_hopper_not_blackwell():
    """SM90 (Hopper) should NOT get Blackwell backend priorities."""
    from vllm.platforms.cuda import _get_backend_priorities

    cap = DeviceCapability(major=9, minor=0)
    priorities = _get_backend_priorities(cap, use_mla=False)
    backend_names = [b.name for b in priorities]
    # Hopper gets FlashAttn first, not FlashInfer
    if "FLASH_ATTN" in backend_names and "FLASHINFER" in backend_names:
        fa_idx = backend_names.index("FLASH_ATTN")
        fi_idx = backend_names.index("FLASHINFER")
        assert fa_idx < fi_idx, (
            f"FlashAttn ({fa_idx}) should come before FlashInfer ({fi_idx}) "
            f"for SM90 Hopper GPU"
        )
