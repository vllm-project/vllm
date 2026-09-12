# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for deterministic tool-dict key ordering.

Regression test for https://github.com/vllm-project/vllm/issues/53089 —
non-deterministic dict-key ordering in ``tools`` causes P/D disaggregation
block-count assertion failures.
"""
import json

from vllm.renderers.online_renderer import _sort_tool_keys

# -- fixtures --------------------------------------------------------

TOOL_A = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


def _reversed_keys(d):
    """Return a new dict (recursively) with keys in reverse order."""
    if isinstance(d, dict):
        return {k: _reversed_keys(d[k]) for k in reversed(list(d.keys()))}
    if isinstance(d, list):
        return [_reversed_keys(v) for v in d]
    return d


# -- tests -----------------------------------------------------------


class TestSortToolKeys:
    def test_sorted_output_is_deterministic(self):
        original = _sort_tool_keys(TOOL_A)
        reversed_input = _reversed_keys(TOOL_A)
        from_reversed = _sort_tool_keys(reversed_input)
        assert json.dumps(original) == json.dumps(from_reversed)

    def test_deeply_nested_dicts_are_sorted(self):
        nested = {"z": {"b": 1, "a": 2}, "a": {"y": {"c": 3, "a": 4}}}
        result = _sort_tool_keys(nested)
        keys_outer = list(result.keys())
        keys_z = list(result["z"].keys())
        keys_a_y = list(result["a"]["y"].keys())
        assert keys_outer == ["a", "z"]
        assert keys_z == ["a", "b"]
        assert keys_a_y == ["a", "c"]

    def test_lists_are_preserved(self):
        value = [{"b": 1, "a": 2}, {"d": 3, "c": 4}]
        result = _sort_tool_keys(value)
        assert list(result[0].keys()) == ["a", "b"]
        assert list(result[1].keys()) == ["c", "d"]

    def test_scalars_pass_through(self):
        assert _sort_tool_keys(42) == 42
        assert _sort_tool_keys("hello") == "hello"
        assert _sort_tool_keys(None) is None
        assert _sort_tool_keys(True) is True

    def test_tool_list_produces_identical_json(self):
        tools = [TOOL_A, {"type": "function", "function": {"name": "b"}}]
        reversed_tools = [_reversed_keys(t) for t in tools]
        assert json.dumps([_sort_tool_keys(t) for t in tools]) == json.dumps(
            [_sort_tool_keys(t) for t in reversed_tools]
        )

    def test_empty_dict_and_list(self):
        assert _sort_tool_keys({}) == {}
        assert _sort_tool_keys([]) == []

    def test_tuple_treated_as_list(self):
        result = _sort_tool_keys(({"b": 1, "a": 2},))
        assert isinstance(result, list)
        assert list(result[0].keys()) == ["a", "b"]
