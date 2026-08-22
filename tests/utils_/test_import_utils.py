# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

import vllm.utils.import_utils as import_utils
from vllm.utils.import_utils import (
    PlaceholderModule,
    _has_module,
    cpu_supports_avx,
    has_nixl_ep,
)


def _raises_module_not_found():
    return pytest.raises(ModuleNotFoundError, match="No module named")


def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule("placeholder_1234")

    with _raises_module_not_found():
        int(placeholder)

    with _raises_module_not_found():
        placeholder()

    with _raises_module_not_found():
        _ = placeholder.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __name attribute
        _ = placeholder.name

    # OK to print the placeholder or use it in a f-string
    _ = repr(placeholder)
    _ = str(placeholder)

    # No error yet; only error when it is used downstream
    placeholder_attr = placeholder.placeholder_attr("attr")

    with _raises_module_not_found():
        int(placeholder_attr)

    with _raises_module_not_found():
        placeholder_attr()

    with _raises_module_not_found():
        _ = placeholder_attr.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __module attribute
        _ = placeholder_attr.module


class TestHasModule:
    """Tests for _has_module with trial import verification."""

    def setup_method(self):
        # Clear the @cache between tests so each test gets a fresh call
        _has_module.cache_clear()

    def test_returns_true_for_importable_stdlib_module(self):
        assert _has_module("json") is True

    def test_returns_false_for_nonexistent_module(self):
        assert _has_module("nonexistent_module_xyz_12345") is False

    def test_returns_false_when_find_spec_succeeds_but_import_fails(self):
        """Simulate a native extension whose shared library is missing.

        ``find_spec`` finds the package on disk, but the actual import
        raises ``ImportError`` (e.g. missing ``libcudart.so``).
        """
        fake_spec = MagicMock()

        with (
            patch(
                "vllm.utils.import_utils.importlib.util.find_spec",
                return_value=fake_spec,
            ),
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ImportError(
                    "libcudart.so.12: cannot open shared object file"
                ),
            ),
        ):
            assert _has_module("fake_native_ext") is False

    def test_returns_false_when_find_spec_raises(self):
        """``find_spec`` itself can raise for dotted names whose parent package
        fails to import. This should be treated as the module being unavailable.
        """
        with patch(
            "vllm.utils.import_utils.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'fake_parent'"),
        ):
            assert _has_module("fake_parent.child") is False

    def test_result_is_cached(self):
        """Verify the @cache decorator prevents repeated imports."""
        _has_module("json")  # prime the cache

        with patch("vllm.utils.import_utils.importlib.util.find_spec") as mock_spec:
            result = _has_module("json")  # should hit cache
            mock_spec.assert_not_called()
            assert result is True


class TestCpuSupportsAvx:
    """Tests for cpu_supports_avx, the guard for AVX-compiled UCX in NIXL."""

    def setup_method(self):
        # Clear the @cache between tests so each test gets a fresh call
        cpu_supports_avx.cache_clear()

    def _patch_cpuinfo(self, monkeypatch, tmp_path, flags: str) -> None:
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(f"processor\t: 0\nflags\t\t: {flags}\n")
        monkeypatch.setattr(import_utils, "_CPUINFO_PATH", str(cpuinfo))

    def test_non_x86_cpu_always_supported(self, monkeypatch, tmp_path):
        self._patch_cpuinfo(monkeypatch, tmp_path, "fpu tsc")  # no avx
        monkeypatch.setattr(import_utils.platform, "machine", lambda: "aarch64")
        assert cpu_supports_avx() is True

    def test_x86_with_avx_flag(self, monkeypatch, tmp_path):
        self._patch_cpuinfo(monkeypatch, tmp_path, "fpu vme avx avx2 ae rdrand")
        monkeypatch.setattr(import_utils.platform, "machine", lambda: "x86_64")
        assert cpu_supports_avx() is True

    def test_x86_without_avx_flag(self, monkeypatch, tmp_path):
        # avx2/avx512f substrings must not be mistaken for the avx flag.
        self._patch_cpuinfo(monkeypatch, tmp_path, "fpu vme mmx sse sse2 avx512f")
        monkeypatch.setattr(import_utils.platform, "machine", lambda: "x86_64")
        assert cpu_supports_avx() is False

    def test_x86_without_cpuinfo_falls_back_to_torch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            import_utils, "_CPUINFO_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr(import_utils.platform, "machine", lambda: "x86_64")
        with patch("torch.backends.cpu.get_cpu_capability", return_value="AVX2"):
            assert cpu_supports_avx() is True
        cpu_supports_avx.cache_clear()
        with patch("torch.backends.cpu.get_cpu_capability", return_value="DEFAULT"):
            assert cpu_supports_avx() is False

    def test_result_is_cached(self, monkeypatch):
        # cpu_supports_avx is on the hot path of every lazy NIXL attribute
        # access; the result must be resolved once per process.
        monkeypatch.setattr(import_utils.platform, "machine", lambda: "aarch64")
        assert cpu_supports_avx() is True
        # A different machine type must not change the cached answer.
        monkeypatch.setattr(import_utils.platform, "machine", lambda: "x86_64")
        assert cpu_supports_avx() is True


class TestHasNixlEp:
    """has_nixl_ep must never import nixl_ep on CPUs without AVX."""

    def test_unavailable_without_avx(self, monkeypatch):
        monkeypatch.setattr(import_utils, "cpu_supports_avx", lambda: False)
        # Spy on the logger directly — vllm's logger does not propagate to
        # root, so caplog can't see it.
        warnings: list[str] = []
        monkeypatch.setattr(
            import_utils.logger, "warning_once", lambda msg: warnings.append(msg)
        )
        with patch.object(import_utils, "_has_module") as mock_has_module:
            assert has_nixl_ep() is False
            # The skip must be visible to users instead of a silent downgrade.
            assert any("Skipping nixl_ep" in msg for msg in warnings)
            # The trial import must be skipped entirely: importing nixl_ep
            # loads UCX, which terminates the process on non-AVX x86 CPUs.
            mock_has_module.assert_not_called()

    def test_delegates_to_has_module_with_avx(self, monkeypatch):
        monkeypatch.setattr(import_utils, "cpu_supports_avx", lambda: True)
        with patch.object(
            import_utils, "_has_module", return_value=True
        ) as mock_has_module:
            assert has_nixl_ep() is True
            mock_has_module.assert_called_once_with("nixl_ep")
