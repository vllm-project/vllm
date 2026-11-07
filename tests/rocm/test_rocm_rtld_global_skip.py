# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HSA_TOOLS_LIB early-exit in _maybe_promote_torch_symbols_for_rocm.

Verifies that RTLD_GLOBAL promotion of libtorch_cpu.so is skipped when
HSA_TOOLS_LIB is set, and proceeds as normal when it is unset.
"""

import ctypes
from unittest.mock import MagicMock, patch

# Import functions by name after the module is already loaded to avoid
# re-triggering module-level side effects.
import vllm.env_override as env_override_mod
from vllm.env_override import _maybe_promote_torch_symbols_for_rocm


class TestMaybePromoteTorchSymbolsHsaToolsLibSkip:
    """Test the HSA_TOOLS_LIB guard added to _maybe_promote_torch_symbols_for_rocm."""

    def test_skips_rtld_global_when_hsa_tools_lib_set(self, monkeypatch, tmp_path):
        """CDLL must not be called when HSA_TOOLS_LIB is set.

        This is the core invariant of the fix: loading libtorch_cpu.so with
        RTLD_GLOBAL pulls libhsa-runtime64 into the global symbol scope and
        prevents an external HSA tool's OnLoad callback from firing.
        """
        monkeypatch.setenv("HSA_TOOLS_LIB", "/path/to/external_tool.so")
        fake_lib = tmp_path / "libtorch_cpu.so"
        fake_lib.touch()
        with (
            patch.object(
                env_override_mod, "_get_torch_version_attr", return_value=MagicMock()
            ),
            patch.object(
                env_override_mod, "_get_torch_root", return_value=str(tmp_path)
            ),
            patch("ctypes.CDLL") as mock_cdll,
        ):
            _maybe_promote_torch_symbols_for_rocm()
        mock_cdll.assert_not_called()

    def test_promotes_when_hsa_tools_lib_unset(self, monkeypatch, tmp_path):
        """CDLL is called with RTLD_GLOBAL when HSA_TOOLS_LIB is absent.

        Default behaviour must be unchanged when no external HSA tool is active.
        """
        monkeypatch.delenv("HSA_TOOLS_LIB", raising=False)
        # The function looks for <torch_root>/lib/libtorch_cpu.so
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        fake_lib = lib_dir / "libtorch_cpu.so"
        fake_lib.touch()
        with (
            patch.object(
                env_override_mod, "_get_torch_version_attr", return_value=MagicMock()
            ),
            patch.object(
                env_override_mod, "_get_torch_root", return_value=str(tmp_path)
            ),
            patch("ctypes.CDLL") as mock_cdll,
        ):
            _maybe_promote_torch_symbols_for_rocm()
        mock_cdll.assert_called_once_with(str(fake_lib), mode=ctypes.RTLD_GLOBAL)

    def test_skips_entirely_when_not_rocm(self, monkeypatch):
        """Pre-existing guard: nothing happens on non-ROCm builds.

        Applies regardless of HSA_TOOLS_LIB value.
        """
        monkeypatch.delenv("HSA_TOOLS_LIB", raising=False)
        with (
            patch.object(
                env_override_mod, "_get_torch_version_attr", return_value=None
            ),
            patch("ctypes.CDLL") as mock_cdll,
        ):
            _maybe_promote_torch_symbols_for_rocm()
        mock_cdll.assert_not_called()
