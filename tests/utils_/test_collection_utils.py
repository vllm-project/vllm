# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import (
    LazyDict,
    as_list,
    chunk_list,
    common_prefix,
    full_groupby,
    is_list_of_numbers,
    swap_dict_values,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ([], True),
        ([1, -2, 2**1024], True),
        ([1, 0.5], True),
        ([1, True], False),
        ([1, float("nan")], False),
        ([1, float("inf")], False),
        ([1, -float("inf")], False),
        ([1, "2"], False),
        ([[1]], False),
        ((1, 2), False),
        (None, False),
    ],
)
def test_is_list_of_numbers_checks_all_finite_non_boolean_items(value, expected):
    assert is_list_of_numbers(value) is expected


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


def test_as_list_returns_lists_unchanged():
    lst = [1, 2, 3]
    assert as_list(lst) is lst


@pytest.mark.parametrize(
    "value,expected",
    [
        ((1, 2), [1, 2]),
        (range(3), [0, 1, 2]),
    ],
)
def test_as_list_converts_other_iterables(value, expected):
    assert as_list(value) == expected


@pytest.mark.parametrize(
    "lst,chunk_size,expected",
    [
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([1, 2], 5, [[1, 2]]),
        ([], 3, []),
    ],
)
def test_chunk_list(lst, chunk_size, expected):
    assert list(chunk_list(lst, chunk_size)) == expected


def test_full_groupby_groups_non_contiguous_values():
    values = [1, 2, 1, 3, 2]
    assert dict(full_groupby(values, key=lambda x: x % 2)) == {
        1: [1, 1, 3],
        0: [2, 2],
    }


def test_full_groupby_empty_input():
    assert dict(full_groupby([], key=lambda x: x)) == {}


def test_lazy_dict_evaluates_lazily_and_caches():
    calls = []
    lazy = LazyDict({"a": lambda: calls.append("a") or 1})
    assert calls == []
    assert lazy["a"] == 1
    assert lazy["a"] == 1
    assert calls == ["a"]


def test_lazy_dict_raises_key_error_for_unknown_key():
    lazy = LazyDict({"a": lambda: 1})
    with pytest.raises(KeyError):
        lazy["missing"]


def test_lazy_dict_len_iter_and_setitem():
    lazy = LazyDict({"a": lambda: 1})
    assert len(lazy) == 1
    assert list(lazy) == ["a"]
    lazy["b"] = lambda: 2
    assert len(lazy) == 2
    assert lazy["b"] == 2
