# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import swap_dict_values


def test_swap_dict_values_basic():
    d = {"a": 1, "b": 2}
    swap_dict_values(d, "a", "b")
    assert d == {"a": 2, "b": 1}


def test_swap_dict_values_explicit_none_in_first():
    d = {"a": None, "b": 2}
    swap_dict_values(d, "a", "b")
    # explicit None must be preserved (not dropped)
    assert d == {"a": 2, "b": None}


def test_swap_dict_values_explicit_none_in_second():
    d = {"a": 1, "b": None}
    swap_dict_values(d, "a", "b")
    assert d == {"a": None, "b": 1}


def test_swap_dict_values_both_none():
    d = {"a": None, "b": None}
    swap_dict_values(d, "a", "b")
    assert d == {"a": None, "b": None}


def test_swap_dict_values_first_key_absent():
    d = {"b": 5}
    swap_dict_values(d, "a", "b")
    # original behaviour: absent key keeps the other value moved in
    assert d == {"a": 5}


def test_swap_dict_values_second_key_absent():
    d = {"a": 7}
    swap_dict_values(d, "a", "b")
    assert d == {"b": 7}
