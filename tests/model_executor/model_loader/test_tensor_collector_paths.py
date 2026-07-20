# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for grammar-aware resolve_path/set_by_path in tensor_collector.

Covers AC-2 closure/partial/bracket path resolution and mutation,
including required warning semantics for failure cases.
"""
import ctypes
import functools
import logging
import types
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.reload.tensor_collector import (
    _parse_path,
    copy_back_extra_tensors,
    resolve_path,
    set_by_path,
)


# ---------------------------------------------------------------------------
# _parse_path tests
# ---------------------------------------------------------------------------

class TestParsePath:
    def test_simple_attr(self):
        assert _parse_path("weight") == [("attr", "weight")]

    def test_dotted_attrs(self):
        assert _parse_path("quant_method.moe_kernel.b_strides1") == [
            ("attr", "quant_method"),
            ("attr", "moe_kernel"),
            ("attr", "b_strides1"),
        ]

    def test_integer_index(self):
        assert _parse_path("items[0]") == [("attr", "items"), ("index", 0)]

    def test_string_key_single_quote(self):
        assert _parse_path("data['key']") == [("attr", "data"), ("key", "key")]

    def test_string_key_double_quote(self):
        assert _parse_path('data["key"]') == [("attr", "data"), ("key", "key")]

    def test_closure_path(self):
        assert _parse_path("fn.__closure__[0]") == [
            ("attr", "fn"), ("attr", "__closure__"), ("index", 0),
        ]

    def test_partial_keywords(self):
        assert _parse_path("fn.keywords['scale']") == [
            ("attr", "fn"), ("attr", "keywords"), ("key", "scale"),
        ]

    def test_partial_args(self):
        assert _parse_path("fn.args[0]") == [
            ("attr", "fn"), ("attr", "args"), ("index", 0),
        ]

    def test_mixed_path(self):
        tokens = _parse_path("quant_method.lookup['experts'][2].weight")
        assert tokens == [
            ("attr", "quant_method"), ("attr", "lookup"),
            ("key", "experts"), ("index", 2), ("attr", "weight"),
        ]


# ---------------------------------------------------------------------------
# resolve_path tests
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_simple_attr(self):
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        layer.extra = t
        assert resolve_path(layer, "extra") is t

    def test_nested_attr(self):
        layer = nn.Linear(4, 4)
        class Inner: pass
        obj = Inner()
        obj.scale = torch.randn(2)
        layer.quant = obj
        assert resolve_path(layer, "quant.scale") is obj.scale

    def test_list_index(self):
        layer = nn.Linear(4, 4)
        t = torch.randn(5)
        layer.items = [torch.randn(1), t]
        assert resolve_path(layer, "items[1]") is t

    def test_dict_key(self):
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        layer.lookup = {"scale": t}
        assert resolve_path(layer, "lookup['scale']") is t

    def test_closure_cell(self):
        t = torch.randn(4)
        def make_fn():
            captured = t
            return lambda: captured
        layer = nn.Linear(4, 4)
        layer.fn = make_fn()
        assert resolve_path(layer, "fn.__closure__[0]") is t

    def test_partial_keywords(self):
        t = torch.randn(3)
        layer = nn.Linear(4, 4)
        layer.fn = functools.partial(lambda x, scale=None: x, scale=t)
        assert resolve_path(layer, "fn.keywords['scale']") is t

    def test_partial_args(self):
        t = torch.randn(3)
        layer = nn.Linear(4, 4)
        layer.fn = functools.partial(lambda x, y: x, t)
        assert resolve_path(layer, "fn.args[0]") is t

    def test_broken_path_returns_none(self):
        layer = nn.Linear(4, 4)
        assert resolve_path(layer, "nonexistent.foo") is None

    def test_empty_closure_returns_none(self):
        layer = nn.Linear(4, 4)
        layer.fn = lambda: 42
        assert resolve_path(layer, "fn.__closure__[0]") is None

    def test_non_function_at_closure_path_returns_none(self):
        """Non-function object where a closure is expected returns None."""
        layer = nn.Linear(4, 4)
        layer.fn = 42  # not a function
        assert resolve_path(layer, "fn.__closure__[0]") is None

    def test_closure_cell_non_tensor_returns_none(self):
        """Closure cell containing non-tensor returns None from resolve."""
        def make_fn():
            captured = "not a tensor"
            return lambda: captured
        layer = nn.Linear(4, 4)
        layer.fn = make_fn()
        # resolve_path returns None because cell_contents is not a tensor
        assert resolve_path(layer, "fn.__closure__[0]") is None


# ---------------------------------------------------------------------------
# set_by_path tests
# ---------------------------------------------------------------------------

class TestSetByPath:
    def test_simple_attr(self):
        layer = nn.Linear(4, 4)
        layer.extra = torch.randn(3)
        new_t = torch.randn(3)
        assert set_by_path(layer, "extra", new_t) is True
        assert layer.extra is new_t

    def test_nested_attr(self):
        layer = nn.Linear(4, 4)
        class Inner: pass
        obj = Inner()
        obj.scale = torch.randn(2)
        layer.quant = obj
        new_t = torch.randn(2)
        assert set_by_path(layer, "quant.scale", new_t) is True
        assert layer.quant.scale is new_t

    def test_list_index(self):
        layer = nn.Linear(4, 4)
        layer.items = [torch.randn(1), torch.randn(1)]
        new_t = torch.randn(1)
        assert set_by_path(layer, "items[1]", new_t) is True
        assert layer.items[1] is new_t

    def test_dict_key(self):
        layer = nn.Linear(4, 4)
        layer.lookup = {"scale": torch.randn(3)}
        new_t = torch.randn(3)
        assert set_by_path(layer, "lookup['scale']", new_t) is True
        assert layer.lookup["scale"] is new_t

    def test_closure_cell_mutation(self):
        original = torch.randn(4)
        def make_fn():
            captured = original
            return lambda: captured
        layer = nn.Linear(4, 4)
        layer.fn = make_fn()
        new_t = torch.randn(4)
        assert set_by_path(layer, "fn.__closure__[0]", new_t) is True
        assert layer.fn.__closure__[0].cell_contents is new_t

    def test_partial_keywords_mutation(self):
        t = torch.randn(3)
        layer = nn.Linear(4, 4)
        layer.fn = functools.partial(lambda x, scale=None: x, scale=t)
        new_t = torch.randn(3)
        assert set_by_path(layer, "fn.keywords['scale']", new_t) is True
        assert layer.fn.keywords["scale"] is new_t

    def test_partial_args_warns_and_fails(self, caplog):
        """Direct partial.args[N] warns and returns False."""
        t = torch.randn(3)
        layer = nn.Linear(4, 4)
        layer.fn = functools.partial(lambda x, y: x, t)
        with caplog.at_level(logging.WARNING):
            result = set_by_path(layer, "fn.args[0]", torch.randn(3))
        assert result is False
        assert "immutable" in caplog.text

    def test_closure_nested_partial_args_warns_and_fails(self, caplog):
        """partial.args reached through closure cell also warns."""
        t = torch.randn(3)
        p = functools.partial(lambda x, y: x, t)
        def make_fn():
            captured = p
            return lambda: captured
        layer = nn.Linear(4, 4)
        layer.fn = make_fn()
        # Path: fn.__closure__[0].args[0]
        with caplog.at_level(logging.WARNING):
            result = set_by_path(layer, "fn.__closure__[0].args[0]",
                                 torch.randn(3))
        assert result is False
        assert "immutable" in caplog.text

    def test_broken_path_returns_false(self):
        layer = nn.Linear(4, 4)
        assert set_by_path(layer, "nonexistent.foo", torch.randn(1)) is False


# ---------------------------------------------------------------------------
# copy_back_extra_tensors warning tests
# ---------------------------------------------------------------------------

class TestCopyBackWarnings:
    def test_broken_path_warns(self, caplog):
        """resolve_path returning None emits WARNING."""
        layer = nn.Linear(4, 4)
        old_t = torch.randn(3, device="cpu")
        slots = [("nonexistent.path", old_t)]
        with caplog.at_level(logging.WARNING):
            copy_back_extra_tensors(layer, slots)
        assert "broken after reload" in caplog.text

    def test_non_function_at_closure_path_warns(self, caplog):
        """Non-function replacing a closure owner emits WARNING."""
        layer = nn.Linear(4, 4)
        layer.fn = 42  # replaced by non-function during PWAL
        old_t = torch.randn(3, device="cpu")
        slots = [("fn.__closure__[0]", old_t)]
        with caplog.at_level(logging.WARNING):
            copy_back_extra_tensors(layer, slots)
        assert "broken after reload" in caplog.text or "path broken" in caplog.text

    def test_failed_set_by_path_warns(self, caplog):
        """set_by_path returning False emits WARNING about restore failure."""
        t = torch.randn(3, device="cpu")
        p = functools.partial(lambda x, y: x, t)
        layer = nn.Linear(4, 4)
        layer.fn = p
        # partial.args is immutable → set_by_path will fail
        slots = [("fn.args[0]", t)]
        with caplog.at_level(logging.WARNING):
            copy_back_extra_tensors(layer, slots)
        assert "restore failed" in caplog.text or "immutable" in caplog.text

    def test_empty_closure_cell_skips(self, caplog):
        """Empty/unset closure cell skips gracefully."""
        layer = nn.Linear(4, 4)
        # Create a function with an empty cell
        def make_fn():
            if False:
                x = None  # noqa: F841
            return lambda: x  # x is unbound → empty cell
        try:
            layer.fn = make_fn()
        except UnboundLocalError:
            pytest.skip("Cannot create empty cell in this Python version")
        old_t = torch.randn(3, device="cpu")
        slots = [("fn.__closure__[0]", old_t)]
        with caplog.at_level(logging.WARNING):
            copy_back_extra_tensors(layer, slots)
        # Should warn about broken path (cell_contents raises ValueError
        # so resolve_path returns None → "path broken after reload")
        assert "broken" in caplog.text or "path broken" in caplog.text

    def test_copy_failure_warns_and_continues(self, caplog):
        """RuntimeError from copy_() is caught and warned, not raised."""
        layer = nn.Linear(4, 4)
        # Create two tensors with same shape/dtype but make copy fail
        old_t = torch.randn(3, device="cpu")
        layer.data_a = torch.randn(3, device="cpu")
        # Also create a valid tensor to verify copy-back continues
        old_b = torch.randn(2, device="cpu")
        layer.data_b = torch.randn(2, device="cpu")
        new_b_data = layer.data_b.clone()

        slots = [("data_a", old_t), ("data_b", old_b)]

        # Monkey-patch old_t.data.copy_ to raise RuntimeError
        original_copy = old_t.data.copy_

        def failing_copy(src):
            raise RuntimeError("simulated device error")

        old_t.data.copy_ = failing_copy

        with caplog.at_level(logging.WARNING):
            # Should NOT raise — best-effort for walk tensors
            copy_back_extra_tensors(layer, slots)

        # Verify: warning emitted for failed copy
        assert "copy failed" in caplog.text
        assert "data_a" in caplog.text

        # Verify: the second tensor was still restored (copy-back continued)
        assert layer.data_b is old_b
        assert torch.equal(old_b, new_b_data)


# ---------------------------------------------------------------------------
# Integration: copy_back_extra_tensors with closure/partial/bracket paths
# ---------------------------------------------------------------------------

class TestCopyBackIntegration:
    def test_closure_tensor_copy_back(self):
        original = torch.randn(4, device="cpu")
        def make_fn():
            captured = original
            return lambda: captured
        layer = nn.Linear(4, 4)
        layer.fn = make_fn()
        slots = [("fn.__closure__[0]", original)]

        # Simulate PWAL: replace closure content
        new_tensor = torch.randn(4, device="cpu")
        cell = layer.fn.__closure__[0]
        ctypes.pythonapi.PyCell_Set(
            ctypes.py_object(cell), ctypes.py_object(new_tensor))

        copy_back_extra_tensors(layer, slots)
        restored = layer.fn.__closure__[0].cell_contents
        assert restored is original
        assert torch.equal(restored, new_tensor)

    def test_dict_tensor_copy_back(self):
        layer = nn.Linear(4, 4)
        original = torch.randn(3, device="cpu")
        layer.lookup = {"scale": original}
        slots = [("lookup['scale']", original)]
        layer.lookup["scale"] = torch.randn(3, device="cpu")
        new_data = layer.lookup["scale"].clone()
        copy_back_extra_tensors(layer, slots)
        assert layer.lookup["scale"] is original
        assert torch.equal(original, new_data)

    def test_partial_keywords_copy_back(self):
        layer = nn.Linear(4, 4)
        original = torch.randn(2, device="cpu")
        layer.fn = functools.partial(lambda x, scale=None: x, scale=original)
        slots = [("fn.keywords['scale']", original)]
        new_tensor = torch.randn(2, device="cpu")
        layer.fn.keywords["scale"] = new_tensor
        copy_back_extra_tensors(layer, slots)
        assert layer.fn.keywords["scale"] is original
        assert torch.equal(original, new_tensor)

    def test_dot_only_backward_compat(self):
        layer = nn.Linear(4, 4)
        class Inner: pass
        obj = Inner()
        obj.scale = torch.randn(3, device="cpu")
        layer.quant = obj
        original = obj.scale
        slots = [("quant.scale", original)]
        layer.quant.scale = torch.randn(3, device="cpu")
        new_data = layer.quant.scale.clone()
        copy_back_extra_tensors(layer, slots)
        assert layer.quant.scale is original
        assert torch.equal(original, new_data)
