# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Regression test for vllm_version_matches_substr() when vllm is not installed.

Without the fix, vllm_version_matches_substr() re-raised PackageNotFoundError
when the vllm package was not found. This caused cuda_platform_plugin() to
raise, which was silently swallowed by resolve_current_platform_cls_qualname(),
leaving current_platform as UnspecifiedPlatform with device_type = ''.
Later, DeviceConfig(device='') called torch.device('') and crashed with:
  RuntimeError: Device string must not be empty

This regression test verifies that vllm_version_matches_substr() returns False
(instead of raising) when the vllm package is not installed.
"""

from unittest.mock import patch


def test_vllm_version_matches_substr_returns_false_when_not_installed():
    """
    When the vllm package is not installed, vllm_version_matches_substr()
    should return False instead of raising PackageNotFoundError.
    """
    from importlib.metadata import PackageNotFoundError

    # Simulate vllm not being installed by patching importlib.metadata.version.
    # The patch works because vllm_version_matches_substr() dynamically imports
    # `version` from `importlib.metadata` within its own scope on each call,
    # so the patch is active when the function executes.
    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("vllm"),
    ):
        from vllm.platforms import vllm_version_matches_substr

        # Should return False, not raise
        result = vllm_version_matches_substr("cpu")
        assert result is False, (
            "vllm_version_matches_substr should return False when vllm is not "
            "installed, but it raised or returned a non-False value."
        )


def test_vllm_version_matches_substr_returns_true_when_matches():
    """
    When the vllm package is installed and the version contains the substr,
    vllm_version_matches_substr() should return True.
    """
    with patch("importlib.metadata.version", return_value="0.7.0.cpu"):
        from vllm.platforms import vllm_version_matches_substr

        assert vllm_version_matches_substr("cpu") is True


def test_vllm_version_matches_substr_returns_false_when_no_match():
    """
    When the vllm package is installed but the version does not contain the
    substr, vllm_version_matches_substr() should return False.
    """
    with patch("importlib.metadata.version", return_value="0.7.0"):
        from vllm.platforms import vllm_version_matches_substr

        assert vllm_version_matches_substr("cpu") is False
