# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import mock_open, patch

import pytest

from vllm.platforms import _is_amd_zen_cpu, resolve_current_platform_cls_qualname


def test_is_amd_zen_cpu_detects_amd_with_avx512():
    cpuinfo = "vendor_id: AuthenticAMD\nflags: avx avx2 avx512f avx512bw"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cpuinfo)),
    ):
        assert _is_amd_zen_cpu()


def test_is_amd_zen_cpu_returns_false_for_amd_without_avx512():
    cpuinfo = "vendor_id: AuthenticAMD\nflags: avx avx2"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cpuinfo)),
    ):
        assert not _is_amd_zen_cpu()


def test_is_amd_zen_cpu_returns_false_for_intel_with_avx512():
    cpuinfo = "vendor_id: GenuineIntel\nflags: avx avx2 avx512f"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cpuinfo)),
    ):
        assert not _is_amd_zen_cpu()


def test_is_amd_zen_cpu_returns_false_when_cpuinfo_missing():
    with patch("os.path.exists", return_value=False):
        assert not _is_amd_zen_cpu()


def test_cpu_target_selects_cpu_platform_from_non_cpu_wheel(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")

    with (
        patch("vllm.platforms.vllm_version_matches_substr") as version_matches,
        patch("vllm.platforms._is_amd_zen_cpu", return_value=False),
        patch("vllm.platforms.rocm_platform_plugin") as rocm_plugin,
    ):
        assert (
            resolve_current_platform_cls_qualname() == "vllm.platforms.cpu.CpuPlatform"
        )

    # An explicit target does not depend on the installed wheel's version
    # suffix or host accelerators (a native CI job can reuse a ROCm wheel).
    version_matches.assert_not_called()
    rocm_plugin.assert_not_called()
