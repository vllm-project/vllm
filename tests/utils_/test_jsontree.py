# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.utils.jsontree import json_count_leaves


def test_json_count_leaves():
    """Test json_count_leaves function from jsontree utility."""

    # Single leaf values
    assert json_count_leaves(42) == 1
    assert json_count_leaves("hello") == 1
    assert json_count_leaves(None) == 1

    # Empty containers
    assert json_count_leaves([]) == 0
    assert json_count_leaves({}) == 0
    assert json_count_leaves(()) == 0

    # Flat structures
    assert json_count_leaves([1, 2, 3]) == 3
    assert json_count_leaves({"a": 1, "b": 2}) == 2
    assert json_count_leaves((1, 2, 3)) == 3

    # Nested structures
    nested_dict = {"a": 1, "b": {"c": 2, "d": 3}}
    assert json_count_leaves(nested_dict) == 3

    nested_list = [1, [2, 3], 4]
    assert json_count_leaves(nested_list) == 4

    mixed_nested = {"list": [1, 2], "dict": {"x": 3}, "value": 4}
    assert json_count_leaves(mixed_nested) == 4
