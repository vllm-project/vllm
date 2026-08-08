# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import builtins
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils import import_utils
from vllm.utils.import_utils import PlaceholderModule, _has_module


def _clear_import_utils_caches():
    import_utils._has_module.cache_clear()
    if hasattr(import_utils.has_cutedsl, "cache_clear"):
        import_utils.has_cutedsl.cache_clear()


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


def test_has_cutedsl_requires_importable_cutlass(monkeypatch: pytest.MonkeyPatch):
    real_find_spec = importlib.util.find_spec
    real_import = builtins.__import__

    def fake_find_spec(name, *args, **kwargs):
        if name == "cutlass":
            return object()
        return real_find_spec(name, *args, **kwargs)

    def fake_import(name, *args, **kwargs):
        if name == "cutlass":
            raise ImportError("broken CUTLASS DSL")
        return real_import(name, *args, **kwargs)

    _clear_import_utils_caches()
    monkeypatch.setattr(import_utils.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        assert import_utils.has_cutedsl() is False
    finally:
        _clear_import_utils_caches()
