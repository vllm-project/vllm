# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import common_prefix, swap_dict_values


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
    ("obj", "key1", "key2", "expected"),
    [
        # Tests for both keys exist
        ({1: "a", 2: "b"}, 1, 2, {1: "b", 2: "a"}),
        # Tests for one key does not exist
        ({1: "a", 2: "b"}, 1, 3, {2: "b", 3: "a"}),
        # Tests for both keys do not exist
        ({1: "a", 2: "b"}, 3, 4, {1: "a", 2: "b"}),
        # Tests for values that are present but None
        ({1: None, 2: "b"}, 1, 2, {1: "b", 2: None}),
        ({1: None, 2: "b"}, 1, 3, {2: "b", 3: None}),
        ({1: None, 2: "b"}, 1, 1, {1: None, 2: "b"}),
    ],
)
def test_swap_dict_values(obj, key1, key2, expected):
    swap_dict_values(obj, key1, key2)
    assert obj == expected
