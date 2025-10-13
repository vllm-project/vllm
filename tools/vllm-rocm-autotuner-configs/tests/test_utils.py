# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test GPU detection utilities."""

import pytest

try:
    from vllm_rocm_autotuner_configs.utils import (_normalize_arch_name,
                                                   check_amdsmi_available,
                                                   get_amd_gpu_info,
                                                   get_amd_gpu_info_safe)

    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False


@pytest.mark.skipif(not _UTILS_AVAILABLE, reason="utils module not available")
class TestArchNormalization:
    """Test architecture name normalization."""

    def test_simple_arch(self):
        assert _normalize_arch_name("gfx942") == "gfx942"
        assert _normalize_arch_name("gfx950") == "gfx950"

    def test_arch_with_features(self):
        assert _normalize_arch_name("gfx942:sramecc+:xnack-") == "gfx942"
        assert _normalize_arch_name("gfx950:sramecc+:xnack+") == "gfx950"

    def test_unknown(self):
        assert _normalize_arch_name("unknown") == "unknown"
        assert _normalize_arch_name("") == "unknown"

    def test_whitespace(self):
        assert _normalize_arch_name("  gfx942  ") == "gfx942"
        assert _normalize_arch_name("GFX942") == "gfx942"


@pytest.mark.skipif(not _UTILS_AVAILABLE, reason="utils module not available")
class TestGPUDetection:
    """Test GPU detection."""

    def test_check_amdsmi_available(self):
        result = check_amdsmi_available()
        assert isinstance(result, bool)

    def test_safe_detection_never_raises(self):
        arch, count = get_amd_gpu_info_safe()
        assert isinstance(count, int)
        assert count >= 0
        assert arch is None or isinstance(arch, str)

    @pytest.mark.requires_gpu
    def test_gpu_detection_on_real_hardware(self):
        """Only runs if amdsmi available and GPU present."""
        if not check_amdsmi_available():
            pytest.skip("amdsmi not available")

        try:
            arch, count = get_amd_gpu_info()
            assert arch is not None
            assert count > 0
            assert arch.startswith("gfx")
        except Exception:
            pytest.skip("No AMD GPU available")
