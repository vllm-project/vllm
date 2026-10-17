# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils import import_utils
from vllm.utils.import_utils import PlaceholderModule, _has_module, import_plugin


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


class TestImportPlugin:
    @pytest.mark.parametrize("nested", [False, True])
    def test_missing_dependency_keeps_the_original_import_error(
        self, tmp_path, monkeypatch, nested
    ):
        name = "vllm_test_broken_named_plugin"
        if nested:
            package = tmp_path / name
            package.mkdir()
            (package / "__init__.py").write_text("")
            plugin = package / "plugin.py"
            requested = f"{name}.plugin"
        else:
            plugin = tmp_path / f"{name}.py"
            requested = name
        dependency = "vllm_test_missing_plugin_dependency"
        plugin.write_text(f"import {dependency}\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        errors = []
        monkeypatch.setattr(
            import_utils.logger,
            "exception",
            lambda *args: errors.append(sys.exc_info()[1]),
        )
        try:
            assert import_plugin(requested) is None
            assert len(errors) == 1
            assert isinstance(errors[0], ModuleNotFoundError)
            assert errors[0].name == dependency
            assert requested not in sys.modules
        finally:
            sys.modules.pop(requested, None)
            sys.modules.pop(name, None)

    def test_importing_from_site_packages(self):
        import json

        result = import_plugin("json")
        assert result is json

    @pytest.mark.parametrize(
        "filename, relative",
        [("my_test_plugin.py", False), ("vllm_missing_parent.plugin.py", True)],
    )
    def test_importing_from_file(self, tmp_path, monkeypatch, filename, relative):
        plugin_file = tmp_path / filename
        plugin_file.write_text("VALUE = 42\n")
        if relative:
            monkeypatch.chdir(tmp_path)

        try:
            result = import_plugin(filename if relative else str(plugin_file))
            assert result is not None
            assert result.VALUE == 42
        finally:
            sys.modules.pop(plugin_file.stem, None)

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
