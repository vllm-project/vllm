# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import LazyDict, common_prefix, swap_dict_values


@pytest.mark.parametrize(
    ("inputs", "expected_output"),
    [
        ([""], ""),
        (["a"], "a"),
        (["a", "b"], ""),
        (["a", "ab"], "a"),
        (["a", "ab", "b"], ""),
        (["abc", "a", "ab"], "a"),
        (["aba", "abc", "ab"], "ab"),
    ],
)
def test_common_prefix(inputs, expected_output):
    assert common_prefix(inputs) == expected_output


@pytest.mark.parametrize(
    ("obj", "key1", "key2"),
    [
        # Tests for both keys exist
        ({1: "a", 2: "b"}, 1, 2),
        # Tests for one key does not exist
        ({1: "a", 2: "b"}, 1, 3),
        # Tests for both keys do not exist
        ({1: "a", 2: "b"}, 3, 4),
    ],
)
def test_swap_dict_values(obj, key1, key2):
    original_obj = obj.copy()

    swap_dict_values(obj, key1, key2)

    if key1 in original_obj:
        assert obj[key2] == original_obj[key1]
    else:
        assert key2 not in obj
    if key2 in original_obj:
        assert obj[key1] == original_obj[key2]
    else:
        assert key1 not in obj


def test_lazy_dict_is_lazy_and_caches():
    calls = []

    def factory():
        calls.append(1)
        return "value"

    d = LazyDict({"k": factory})
    assert calls == []  # not evaluated until accessed
    assert d["k"] == "value"
    assert d["k"] == "value"
    assert len(calls) == 1  # evaluated once, then cached


def test_lazy_dict_setitem_replaces_cached_value():
    """Overwriting a key must drop the value the old factory produced."""
    d = LazyDict({"k": lambda: "first"})
    assert d["k"] == "first"
    d["k"] = lambda: "second"
    assert d["k"] == "second"


def test_lazy_dict_setitem_before_access():
    d = LazyDict({"k": lambda: "first"})
    d["k"] = lambda: "second"
    assert d["k"] == "second"


def test_lazy_dict_new_key_and_missing_key():
    d = LazyDict({})
    d["new"] = lambda: "x"
    assert d["new"] == "x"
    assert len(d) == 1
    assert list(d) == ["new"]
    with pytest.raises(KeyError):
        d["missing"]
