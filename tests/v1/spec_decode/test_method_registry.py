# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration must work before importing model configuration or GPU modules."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import get_args

import pytest


@pytest.fixture
def registry():
    # Load the actual dependency-free module in a fresh namespace, as in a new
    # engine process, without importing vLLM's device initialization.
    path = Path(__file__).resolve().parents[3] / "vllm/v1/spec_decode/registry.py"
    name = "_test_speculative_method_registry"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[name]


def test_registration_is_lazy_and_idempotent(registry):
    spec = registry.SpeculativeMethodSpec("uninstalled_plugin:Speculator")
    registry.register_speculative_method("plugin", spec)
    registry.register_speculative_method("plugin", spec)
    assert registry.get_speculative_method("plugin") == spec
    assert registry.registered_speculative_methods() == ["plugin"]
    with pytest.raises(ModuleNotFoundError, match="uninstalled_plugin"):
        spec.resolve_speculator_cls()


def test_conflicting_registration_requires_explicit_replacement(registry):
    first = registry.SpeculativeMethodSpec("collections:Counter")
    replacement = registry.SpeculativeMethodSpec("collections:OrderedDict")
    registry.register_speculative_method("plugin", first)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_speculative_method("plugin", replacement)
    assert registry.get_speculative_method("plugin") == first
    registry.register_speculative_method("plugin", replacement, replace=True)
    assert registry.get_speculative_method("plugin") == replacement
    registry.unregister_speculative_method("plugin")
    assert registry.get_speculative_method("plugin") is None


def test_plugin_cannot_shadow_any_builtin_method(registry):
    spec = registry.SpeculativeMethodSpec("collections:Counter")
    for name in get_args(registry.SpeculativeMethod):
        with pytest.raises(ValueError, match="built-in"):
            registry.register_speculative_method(name, spec, replace=True)
    assert registry.registered_speculative_methods() == []


@pytest.mark.parametrize("name", [None, 1, [], "", "  "])
def test_invalid_registration_name_is_rejected(registry, name):
    with pytest.raises(ValueError, match="non-empty string"):
        registry.register_speculative_method(
            name, registry.SpeculativeMethodSpec("collections:Counter")
        )


@pytest.mark.parametrize("path", ["collections:Counter", "collections.Counter"])
def test_resolve_registered_class(registry, path):
    assert registry.SpeculativeMethodSpec(path).resolve_speculator_cls() is Counter


def test_non_class_speculator_is_rejected(registry):
    with pytest.raises(TypeError, match="must be a class"):
        registry.SpeculativeMethodSpec("math:sqrt").resolve_speculator_cls()


def test_unknown_and_non_string_lookup_returns_none(registry):
    assert registry.get_speculative_method("missing") is None
    assert registry.get_speculative_method(None) is None
    assert registry.get_speculative_method([]) is None
