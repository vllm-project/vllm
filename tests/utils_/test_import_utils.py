# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import builtins
import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.import_utils import (
    PlaceholderModule,
    _has_module,
    check_torchcodec_available,
    import_plugin,
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


class TestCheckTorchcodecAvailable:
    """Tests for check_torchcodec_available with an unusable torchcodec."""

    _LIB_PATH = (
        "/opt/venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_image.so"
    )

    @staticmethod
    def _import_raising(exc: Exception):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torchcodec":
                raise exc
            return real_import(name, *args, **kwargs)

        return patch.object(builtins, "__import__", fake_import)

    def test_load_library_oserror_is_reported_without_the_library_path(self):
        """An installed torchcodec whose shared objects cannot be loaded raises
        OSError from torch.ops.load_library, not RuntimeError. It must surface as
        a RuntimeError that does not echo the host library path.
        """
        exc = OSError(f"Could not load this library: {self._LIB_PATH}")

        with self._import_raising(exc), pytest.raises(RuntimeError) as exc_info:
            check_torchcodec_available()

        assert self._LIB_PATH not in str(exc_info.value)

    def test_missing_ffmpeg_runtimeerror_is_still_trimmed(self):
        """The existing RuntimeError path keeps its message trimming."""
        marker = (
            "The following exceptions were raised as we tried to load libtorchcodec:"
        )
        exc = RuntimeError(f"Could not load libtorchcodec.\n{marker}\n{self._LIB_PATH}")

        with self._import_raising(exc), pytest.raises(RuntimeError) as exc_info:
            check_torchcodec_available()

        assert str(exc_info.value) == "Could not load libtorchcodec."
