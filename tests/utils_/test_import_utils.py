# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import builtins
import importlib.util

import pytest

from vllm.utils import import_utils
from vllm.utils.import_utils import PlaceholderModule


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
