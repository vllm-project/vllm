# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for resolve_enable_thinking (#43728)."""

from vllm.parser.utils import resolve_enable_thinking


def test_default_when_kwargs_missing() -> None:
    assert resolve_enable_thinking(None, default=True) is True
    assert resolve_enable_thinking({}, default=True) is True
    assert resolve_enable_thinking({}, default=False) is False


def test_enable_thinking_canonical() -> None:
    assert resolve_enable_thinking({"enable_thinking": True}) is True
    assert resolve_enable_thinking({"enable_thinking": False}) is False


def test_thinking_alias() -> None:
    assert resolve_enable_thinking({"thinking": True}) is True
    assert resolve_enable_thinking({"thinking": False}) is False


def test_either_true_enables() -> None:
    assert resolve_enable_thinking({"thinking": True, "enable_thinking": False}) is True
    assert resolve_enable_thinking({"thinking": False, "enable_thinking": True}) is True


def test_both_false_disables() -> None:
    assert (
        resolve_enable_thinking({"thinking": False, "enable_thinking": False}) is False
    )
