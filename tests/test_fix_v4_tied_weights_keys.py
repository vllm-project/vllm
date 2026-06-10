# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _fix_v4_tied_weights_keys (tests/conftest.py)."""

import logging

import pytest
from conftest import _fix_v4_tied_weights_keys


def _make_cls(**attrs) -> type:
    return type("FakeModel", (), attrs)


def test_list_converted_to_dict_and_keys_no_longer_crash():
    cls = _make_cls(_tied_weights_keys=["lm_head.weight"])
    _fix_v4_tied_weights_keys(cls)
    keys = cls._tied_weights_keys  # type: ignore[attr-defined]
    assert isinstance(keys, dict)
    _ = keys.keys()
    _ = keys.values()
    assert keys == {"lm_head.weight": "model.embed_tokens.weight"}


def test_dict_format_left_unchanged():
    original = {"lm_head.weight": "model.embed_tokens.weight"}
    cls = _make_cls(_tied_weights_keys=original)
    _fix_v4_tied_weights_keys(cls)
    assert cls._tied_weights_keys is original  # type: ignore[attr-defined]


def test_missing_attribute_is_noop():
    cls = _make_cls()
    _fix_v4_tied_weights_keys(cls)
    assert not hasattr(cls, "_tied_weights_keys")


def test_empty_list_is_noop():
    cls = _make_cls(_tied_weights_keys=[])
    _fix_v4_tied_weights_keys(cls)
    assert cls._tied_weights_keys == []  # type: ignore[attr-defined]


def test_unknown_key_warns_and_leaves_attr_unchanged(caplog):
    cls = _make_cls(_tied_weights_keys=["encoder.embed.weight"])
    with caplog.at_level(logging.WARNING):
        _fix_v4_tied_weights_keys(cls)
    assert isinstance(cls._tied_weights_keys, list)  # type: ignore[attr-defined]
    assert "encoder.embed.weight" in caplog.text


def test_live_plamo2_class_patched_correctly():
    pytest.importorskip("huggingface_hub")
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_cls = get_class_from_dynamic_module(
            "modeling_plamo.Plamo2ForCausalLM", "pfnet/plamo-2-1b"
        )
    except Exception as exc:
        pytest.skip(f"Could not load plamo-2-1b dynamic module: {exc}")

    assert isinstance(model_cls._tied_weights_keys, list), (
        "pre-condition: class still has v4 list format"
    )
    _fix_v4_tied_weights_keys(model_cls)
    assert isinstance(model_cls._tied_weights_keys, dict)
    assert "lm_head.weight" in model_cls._tied_weights_keys
