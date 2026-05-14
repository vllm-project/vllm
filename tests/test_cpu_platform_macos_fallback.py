# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for macOS source-checkout platform detection.

When running from a source checkout (no ``pip install``), the call to
``vllm_version_matches_substr("cpu")`` raises ``PackageNotFoundError``.
The macOS (darwin) fallback must still be reachable so that
``cpu_platform_plugin()`` returns ``CpuPlatform`` instead of ``None``
(which would resolve to ``UnspecifiedPlatform`` with an empty
``device_type``, crashing any ``torch.device()`` call).
"""

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from vllm.platforms import cpu_platform_plugin


def test_cpu_platform_plugin_macos_source_checkout():
    """cpu_platform_plugin returns CpuPlatform on macOS even when
    package metadata is missing (source checkout)."""
    with (
        patch(
            "vllm.platforms.vllm_version_matches_substr",
            side_effect=PackageNotFoundError("vllm"),
        ),
        patch("sys.platform", "darwin"),
    ):
        result = cpu_platform_plugin()
    assert result == "vllm.platforms.cpu.CpuPlatform", (
        f"Expected CpuPlatform on macOS source checkout, got {result!r}"
    )


def test_cpu_platform_plugin_linux_source_checkout():
    """cpu_platform_plugin returns None on Linux when package metadata
    is missing and no GPU is present (not a CPU build)."""
    with (
        patch(
            "vllm.platforms.vllm_version_matches_substr",
            side_effect=PackageNotFoundError("vllm"),
        ),
        patch("sys.platform", "linux"),
    ):
        result = cpu_platform_plugin()
    assert result is None, (
        f"Expected None on Linux source checkout without CPU build, got {result!r}"
    )


def test_cpu_platform_plugin_installed_cpu_build():
    """cpu_platform_plugin returns CpuPlatform when the installed
    package version contains 'cpu'."""
    with patch(
        "vllm.platforms.vllm_version_matches_substr",
        return_value=True,
    ):
        result = cpu_platform_plugin()
    assert result == "vllm.platforms.cpu.CpuPlatform", (
        f"Expected CpuPlatform for installed CPU build, got {result!r}"
    )
