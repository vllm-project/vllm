# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import builtins
import logging
from unittest.mock import mock_open, patch

import pytest

from vllm.platforms import (
    _is_amd_zen_cpu,
    cpu_platform_plugin,
    resolve_current_platform_cls_qualname,
)


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


def test_platform_detection_logs_zentorch_import_failure(caplog):
    original_import = builtins.__import__

    def import_with_broken_zentorch(name, *args, **kwargs):
        if name == "zentorch":
            raise OSError("incompatible shared library")
        return original_import(name, *args, **kwargs)

    with (
        patch("vllm.platforms.envs.VLLM_TARGET_DEVICE", "cuda"),
        patch.dict(
            "vllm.platforms.builtin_platform_plugins",
            {"cpu": cpu_platform_plugin},
            clear=True,
        ),
        patch("vllm.platforms.load_plugins_by_group", return_value={}),
        patch("vllm.platforms.vllm_version_matches_substr", return_value=True),
        patch("vllm.platforms._is_amd_zen_cpu", return_value=True),
        patch.object(builtins, "__import__", side_effect=import_with_broken_zentorch),
        caplog.at_level(logging.DEBUG, logger="vllm.platforms"),
    ):
        platform = resolve_current_platform_cls_qualname()

    assert platform == "vllm.platforms.interface.UnspecifiedPlatform"
    assert "Platform plugin cpu failed during detection" in caplog.text
    assert "OSError: incompatible shared library" in caplog.text
