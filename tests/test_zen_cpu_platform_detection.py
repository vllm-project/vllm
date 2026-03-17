# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import mock_open, patch

from vllm.platforms import _is_amd_zen_cpu


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
