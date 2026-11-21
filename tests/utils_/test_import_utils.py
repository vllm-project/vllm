# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.utils.import_utils import (
    PlaceholderModule,
    VllmExtensionImportError,
    _has_module,
    import_plugin,
    import_vllm_extension,
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

        `find_spec` finds the package on disk, but the actual import
        raises `ImportError` (e.g. missing `libcudart.so`).
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
        """`find_spec` itself can raise for dotted names whose parent package
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


class TestImportPlugin:
    def test_importing_from_site_packages(self):
        import json

        result = import_plugin("json")
        assert result is json

    def test_importing_from_file(self, tmp_path):
        plugin_file = tmp_path / "my_test_plugin.py"
        plugin_file.write_text("VALUE = 42\n")

        try:
            result = import_plugin(str(plugin_file))
            assert result is not None
            assert result.VALUE == 42
        finally:
            sys.modules.pop("my_test_plugin", None)

    def test_returns_none_when_both_attempts_fail(self):
        with (
            patch(
                "vllm.utils.import_utils.import_from_path",
                side_effect=FileNotFoundError("no such file"),
            ),
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ModuleNotFoundError("no such module"),
            ),
        ):
            result = import_plugin("nonexistent_plugin_xyz")
            assert result is None


class TestImportVllmExtension:
    """Tests for import_vllm_extension's diagnostic translation.

    Mirrors the failure in https://github.com/vllm-project/vllm/issues/52300:
    a vLLM wheel built for CUDA 13 loaded on a CUDA 12.6 stack dies with a
    bare ``libcudart.so.13: cannot open shared object file``.
    """

    _LOADER_ERROR = (
        "libcudart.so.13: cannot open shared object file: No such file or directory"
    )

    def test_success_returns_the_module(self):
        sentinel = MagicMock()
        with patch(
            "vllm.utils.import_utils.importlib.import_module",
            return_value=sentinel,
        ) as mock_import:
            assert import_vllm_extension("vllm._C_stable_libtorch") is sentinel
        mock_import.assert_called_once_with("vllm._C_stable_libtorch")

    def test_missing_shared_library_raises_actionable_error(self, monkeypatch):
        monkeypatch.setattr(torch.version, "cuda", "12.6", raising=False)
        with (
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ImportError(self._LOADER_ERROR),
            ),
            pytest.raises(VllmExtensionImportError) as excinfo,
        ):
            import_vllm_extension("vllm._C_stable_libtorch")

        message = str(excinfo.value)
        # Names the missing library, the required CUDA major version, the
        # local torch CUDA, and the fix.
        assert "libcudart.so.13" in message
        assert "CUDA 13" in message
        assert "12.6" in message
        assert "wheels.vllm.ai" in message
        assert self._LOADER_ERROR in message
        # Stays an ImportError subclass for existing handlers.
        assert isinstance(excinfo.value, ImportError)
        # Preserves the original loader error for debugging.
        assert isinstance(excinfo.value.__cause__, ImportError)

    def test_missing_library_without_cuda_tag_still_actionable(self):
        with (
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ImportError(
                    "libnccl.so.2: cannot open shared object file: "
                    "No such file or directory"
                ),
            ),
            pytest.raises(VllmExtensionImportError, match="libnccl.so.2"),
        ):
            import_vllm_extension("vllm._moe_C_stable_libtorch")

    def test_other_import_errors_propagate_unchanged(self):
        original = ModuleNotFoundError("No module named 'vllm._C_stable_libtorch'")
        with (
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=original,
            ),
            pytest.raises(ModuleNotFoundError) as excinfo,
        ):
            import_vllm_extension("vllm._C_stable_libtorch")
        assert excinfo.value is original
        assert not isinstance(excinfo.value, VllmExtensionImportError)
