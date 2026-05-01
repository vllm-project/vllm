# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util

import pytest

from vllm.utils.import_utils import PlaceholderModule, _has_module

pytestmark = pytest.mark.skip_global_cleanup


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


def test_has_module_returns_false_for_missing_parent_package(monkeypatch):
    original_find_spec = importlib.util.find_spec
    _has_module.cache_clear()

    def fake_find_spec(module_name):
        if module_name == "triton.language.target_info":
            raise ModuleNotFoundError("No module named 'triton'")
        return original_find_spec(module_name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert _has_module("triton.language.target_info") is False
    _has_module.cache_clear()
