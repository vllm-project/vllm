# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import (
    common_prefix,
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
