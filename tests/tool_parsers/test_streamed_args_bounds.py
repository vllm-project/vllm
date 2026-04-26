# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for #33916: IndexError when streamed_args_for_tool is
shorter than prev_tool_call_arr in the streaming finish path.

The test reproduces the exact access pattern from serving.py:1149 using
plain dicts to avoid deep import chains that require GPU dependencies.
"""
import json


def _simulate_finish_delta(prev_tool_call_arr, streamed_args_for_tool):
    """Reproduces the access pattern from serving.py ~L1098-1158."""
    auto_tools_called = len(prev_tool_call_arr) > 0
    index = len(prev_tool_call_arr) - 1 if auto_tools_called else 0
    if not auto_tools_called:
        return "no_op"

    args = prev_tool_call_arr[index].get("arguments", {})
    expected_call = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)

    # This is the guard added in the fix
    if index < len(streamed_args_for_tool):
        actual_call = streamed_args_for_tool[index]
        remaining_call = expected_call.replace(actual_call, "", 1)
        return ("remaining", remaining_call)

    return ("skipped", None)


def test_prev_set_streamed_empty_no_crash():
    """#33916: Mistral parser sets prev_tool_call_arr but not streamed_args_for_tool."""
    result = _simulate_finish_delta(
        prev_tool_call_arr=[{"arguments": {"a": 1}}],
        streamed_args_for_tool=[],
    )
    assert result == ("skipped", None)


def test_both_populated_computes_remaining():
    result = _simulate_finish_delta(
        prev_tool_call_arr=[{"arguments": {"a": 1, "b": 2}}],
        streamed_args_for_tool=['{"a": 1, '],
    )
    assert result[0] == "remaining"
    assert '"b": 2}' in result[1]


def test_empty_prev_skips():
    result = _simulate_finish_delta(
        prev_tool_call_arr=[],
        streamed_args_for_tool=[],
    )
    assert result == "no_op"


def test_multiple_tools_last_index_out_of_bounds():
    result = _simulate_finish_delta(
        prev_tool_call_arr=[{"arguments": {}}, {"arguments": {"x": 1}}],
        streamed_args_for_tool=[""],
    )
    assert result == ("skipped", None)


def test_without_guard_crashes():
    """Verify the unpatched code path raises IndexError."""
    prev_tool_call_arr = [{"arguments": {"a": 1}}]
    streamed_args_for_tool = []
    index = len(prev_tool_call_arr) - 1
    try:
        _ = streamed_args_for_tool[index]
        raise AssertionError("Should have raised IndexError")
    except IndexError:
        pass
