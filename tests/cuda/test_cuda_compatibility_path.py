# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CUDA forward compatibility path logic in env_override.py.

Verifies the opt-in LD_LIBRARY_PATH manipulation for CUDA compat libs,
including env var parsing, path detection, and deduplication.
"""

import os
from unittest.mock import patch

import pytest

# Import the functions directly (they're module-level in env_override)
# We must import them without triggering the module-level side effects,
# so we import the functions by name after the module is already loaded.
from vllm.env_override import (
    _get_torch_cuda_version,
    _maybe_set_cuda_compatibility_path,
)


class TestCudaCompatibilityEnvParsing:
    """Test VLLM_ENABLE_CUDA_COMPATIBILITY env var parsing."""

    def test_disabled_by_default(self, monkeypatch):
        """Compat path is NOT set when env var is absent."""
        monkeypatch.delenv("VLLM_ENABLE_CUDA_COMPATIBILITY", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        _maybe_set_cuda_compatibility_path()
        assert (
            "LD_LIBRARY_PATH" not in os.environ
            or os.environ.get("LD_LIBRARY_PATH", "") == ""
        )

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", ""])
    def test_disabled_values(self, monkeypatch, value):
        """Various falsy values should not activate compat path."""
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", value)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        _maybe_set_cuda_compatibility_path()
        # LD_LIBRARY_PATH should not be set (or remain empty)
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        assert "compat" not in ld_path

    @pytest.mark.parametrize("value", ["1", "true", "True", " 1 ", " TRUE "])
    def test_enabled_values_with_valid_path(self, monkeypatch, tmp_path, value):
        """Truthy values activate compat path when a valid path exists."""
        compat_dir = tmp_path / "compat"
        compat_dir.mkdir()
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", value)
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", str(compat_dir))
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        _maybe_set_cuda_compatibility_path()
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        assert str(compat_dir) in ld_path


class TestCudaCompatibilityPathDetection:
    """Test path detection: custom override, conda, default."""

    def test_custom_path_override(self, monkeypatch, tmp_path):
        """VLLM_CUDA_COMPATIBILITY_PATH takes highest priority."""
        custom_dir = tmp_path / "my-compat"
        custom_dir.mkdir()
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", str(custom_dir))
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        _maybe_set_cuda_compatibility_path()
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        assert ld_path.startswith(str(custom_dir))

    def test_conda_prefix_fallback(self, monkeypatch, tmp_path):
        """Falls back to $CONDA_PREFIX/cuda-compat if custom not set."""
        conda_dir = tmp_path / "conda-env"
        compat_dir = conda_dir / "cuda-compat"
        compat_dir.mkdir(parents=True)
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.delenv("VLLM_CUDA_COMPATIBILITY_PATH", raising=False)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_dir))
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        _maybe_set_cuda_compatibility_path()
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        assert str(compat_dir) in ld_path

    def test_no_valid_path_does_nothing(self, monkeypatch):
        """When enabled but no valid path exists, LD_LIBRARY_PATH unchanged."""
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", "/nonexistent/path")
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with patch("vllm.env_override._get_torch_cuda_version", return_value=None):
            _maybe_set_cuda_compatibility_path()
        assert os.environ.get("LD_LIBRARY_PATH", "") == ""

    def test_default_cuda_path_fallback(self, monkeypatch, tmp_path):
        """Falls back to /usr/local/cuda-{ver}/compat via torch version."""
        fake_cuda = tmp_path / "cuda-12.8" / "compat"
        fake_cuda.mkdir(parents=True)
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.delenv("VLLM_CUDA_COMPATIBILITY_PATH", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with (
            patch("vllm.env_override._get_torch_cuda_version", return_value="12.8"),
            patch(
                "vllm.env_override.os.path.isdir",
                side_effect=lambda p: p == "/usr/local/cuda-12.8/compat"
                or os.path.isdir(p),
            ),
        ):
            _maybe_set_cuda_compatibility_path()
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        assert "/usr/local/cuda-12.8/compat" in ld_path


class TestCudaCompatibilityLdPathManipulation:
    """Test LD_LIBRARY_PATH prepend and deduplication logic."""

    def test_prepends_to_empty_ld_path(self, monkeypatch, tmp_path):
        """Compat path is set when LD_LIBRARY_PATH is empty."""
        compat_dir = tmp_path / "compat"
        compat_dir.mkdir()
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", str(compat_dir))
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        _maybe_set_cuda_compatibility_path()
        assert os.environ["LD_LIBRARY_PATH"] == str(compat_dir)

    def test_prepends_to_existing_ld_path(self, monkeypatch, tmp_path):
        """Compat path is prepended before existing entries."""
        compat_dir = tmp_path / "compat"
        compat_dir.mkdir()
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", str(compat_dir))
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib:/other/lib")
        _maybe_set_cuda_compatibility_path()
        ld_path = os.environ["LD_LIBRARY_PATH"]
        parts = ld_path.split(os.pathsep)
        assert parts[0] == str(compat_dir)
        assert "/usr/lib" in parts
        assert "/other/lib" in parts

    def test_deduplicates_existing_compat_path(self, monkeypatch, tmp_path):
        """If compat path already in LD_LIBRARY_PATH, move to front."""
        compat_dir = tmp_path / "compat"
        compat_dir.mkdir()
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", str(compat_dir))
        monkeypatch.setenv(
            "LD_LIBRARY_PATH",
            f"/usr/lib:{compat_dir}:/other/lib",
        )
        _maybe_set_cuda_compatibility_path()
        ld_path = os.environ["LD_LIBRARY_PATH"]
        parts = ld_path.split(os.pathsep)
        assert parts[0] == str(compat_dir)
        assert parts.count(str(compat_dir)) == 1

    def test_already_at_front_is_noop(self, monkeypatch, tmp_path):
        """If compat path is already first, don't modify LD_LIBRARY_PATH."""
        compat_dir = tmp_path / "compat"
        compat_dir.mkdir()
        original = f"{compat_dir}:/usr/lib"
        monkeypatch.setenv("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
        monkeypatch.setenv("VLLM_CUDA_COMPATIBILITY_PATH", str(compat_dir))
        monkeypatch.setenv("LD_LIBRARY_PATH", original)
        _maybe_set_cuda_compatibility_path()
        assert os.environ["LD_LIBRARY_PATH"] == original


class TestGetTorchCudaVersion:
    """Test _get_torch_cuda_version() helper."""

    def test_returns_string_when_torch_available(self):
        """Should return a CUDA version string like '12.8'."""
        version = _get_torch_cuda_version()
        # torch is installed in vllm's environment
        assert version is None or isinstance(version, str)

    def test_returns_none_when_torch_missing(self):
        """Should return None when torch is not importable."""
        with patch(
            "vllm.env_override.importlib.util.find_spec",
            return_value=None,
        ):
            assert _get_torch_cuda_version() is None
