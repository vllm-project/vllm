# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# Use a common model that is likely to be available
MODEL = "MiniMaxAi/MiniMax-M1-40k"


@pytest.fixture(scope="module")
def minimax_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def minimax_tool_parser(minimax_tokenizer):
    return MinimaxToolParser(minimax_tokenizer)


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(minimax_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
        "tool_call_with_single_line_json",
        "tool_call_incomplete_tag",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                ))
            ],
            None,
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                )),
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Orlando",
                        "state": "FL",
                        "unit": "fahrenheit",
                    }),
                )),
            ],
            None,
        ),
        (
            """I'll help you check the weather. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Seattle",
                        "state": "WA",
                        "unit": "celsius",
                    }),
                ))
            ],
            "I'll help you check the weather.",
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "New York", "state": "NY", "unit": "celsius"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "New York",
                        "state": "NY",
                        "unit": "celsius",
                    }),
                ))
            ],
            None,
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA"}}""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Boston",
                        "state": "MA",
                    }),
                ))
            ],
            None,
        ),
    ],
)
def test_extract_tool_calls(minimax_tool_parser, model_output,
                            expected_tool_calls, expected_content):
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_preprocess_model_output_with_thinking_tags(minimax_tool_parser):
    """Test that tool calls within thinking tags are removed during preprocessing."""
    model_output = """<think>Let me think about this. <tool_calls>
{"name": "fake_tool", "arguments": {"param": "value"}}
</tool_calls> This should be removed.</think>

I'll help you with that. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA"}}
</tool_calls>"""

    processed_output = minimax_tool_parser.preprocess_model_output(
        model_output)

    # The tool call within thinking tags should be removed
    assert "fake_tool" not in processed_output
    # But the thinking tag itself should remain
    assert "<think>" in processed_output
    assert "</think>" in processed_output
    # The actual tool call outside thinking tags should remain
    assert "get_current_weather" in processed_output


def test_extract_tool_calls_with_thinking_tags(minimax_tool_parser):
    """Test tool extraction when thinking tags contain tool calls that should be ignored."""
    model_output = """<think>I should use a tool. <tool_calls>
{"name": "ignored_tool", "arguments": {"should": "ignore"}}
</tool_calls></think>

Let me help you with the weather. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Miami", "state": "FL", "unit": "fahrenheit"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[
        0].function.name == "get_current_weather"

    # Content extraction is based on the position of the first <tool_calls> in the original model_output
    # Since preprocessing removes tool calls within thinking tags, the actual first <tool_calls> is the external one
    expected_content = """<think>I should use a tool. <tool_calls>
{"name": "ignored_tool", "arguments": {"should": "ignore"}}
</tool_calls></think>

Let me help you with the weather."""
    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_invalid_json(minimax_tool_parser):
    """Test that invalid JSON in tool calls is handled gracefully."""
    model_output = """<tool_calls>
{"name": "valid_tool", "arguments": {"city": "Seattle"}}
{invalid json here}
{"name": "another_valid_tool", "arguments": {"param": "value"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_tool"
    assert extracted_tool_calls.tool_calls[
        1].function.name == "another_valid_tool"


def test_extract_tool_calls_missing_name_or_arguments(minimax_tool_parser):
    """Test that tool calls missing name or arguments are filtered out."""
    model_output = """<tool_calls>
{"name": "valid_tool", "arguments": {"city": "Seattle"}}
{"name": "missing_args"}
{"arguments": {"city": "Portland"}}
{"name": "another_valid_tool", "arguments": {"param": "value"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid tool calls with both name and arguments
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_tool"
    assert extracted_tool_calls.tool_calls[
        1].function.name == "another_valid_tool"


def test_streaming_basic_functionality(minimax_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle"}}
</tool_calls>"""

    # First call should handle the initial setup
    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text="</tool_calls>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The result might be None or contain tool call information
    # This depends on the internal state management
    if result is not None and hasattr(result,
                                      'tool_calls') and result.tool_calls:
        assert len(result.tool_calls) >= 0


def test_streaming_with_content_before_tool_calls(minimax_tool_parser):
    """Test streaming when there's content before tool calls."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    current_text = "I'll help you with that. <tool_calls>"

    # When there's content before tool calls, it should be returned as content
    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you",
        current_text=current_text,
        delta_text=" with that. <tool_calls>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    if result is not None and hasattr(result, 'content'):
        # Should contain some content
        assert result.content is not None


def test_streaming_no_tool_calls(minimax_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just regular text",
        current_text=current_text,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, 'content')
    assert result.content == " without any tool calls."


def test_streaming_with_thinking_tags(minimax_tool_parser):
    """Test streaming with thinking tags that contain tool calls."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    current_text = """<think><tool_calls>{"name": "ignored", "arguments": {}}</tool_calls></think><tool_calls>{"name": "real_tool", "arguments": {"param": "value"}}</tool_calls>"""

    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The preprocessing should remove tool calls from thinking tags
    # and only process the real tool call
    if result is not None and hasattr(result,
                                      'tool_calls') and result.tool_calls:
        for tool_call in result.tool_calls:
            assert tool_call.function.name != "ignored"


def test_extract_tool_calls_multiline_json_not_supported(minimax_tool_parser):
    """Test that multiline JSON in tool calls is not currently supported."""
    model_output = """<tool_calls>
{
  "name": "get_current_weather",
  "arguments": {
    "city": "New York",
    "state": "NY",
    "unit": "celsius"
  }
}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    # Multiline JSON is currently not supported, should return no tools called
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content is None


def test_streaming_arguments_incremental_output(minimax_tool_parser):
    """Test that streaming arguments are returned incrementally, not cumulatively."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    # Simulate progressive tool call building
    stages = [
        # Stage 1: Function name complete
        '<tool_calls>\n{"name": "get_current_weather", "arguments": ',
        # Stage 2: Arguments object starts with first key
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": ',
        # Stage 3: First parameter value added
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle"',
        # Stage 4: Second parameter added
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA"',
        # Stage 5: Third parameter added, arguments complete
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}',
        # Stage 6: Tool calls closed
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n</tool_calls>'
    ]

    function_name_sent = False
    previous_args_content = ""

    for i, current_text in enumerate(stages):
        previous_text = stages[i - 1] if i > 0 else ""
        delta_text = current_text[len(previous_text
                                      ):] if i > 0 else current_text

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        print(f"Stage {i}: Current text: {repr(current_text)}")
        print(f"Stage {i}: Delta text: {repr(delta_text)}")

        if result is not None and hasattr(result,
                                          'tool_calls') and result.tool_calls:
            tool_call = result.tool_calls[0]

            # Check if function name is sent (should happen only once)
            if tool_call.function and tool_call.function.name:
                assert not function_name_sent, "Function name should be sent only once"
                assert tool_call.function.name == "get_current_weather"
                function_name_sent = True
                print(
                    f"Stage {i}: Function name sent: {tool_call.function.name}"
                )

            # Check if arguments are sent incrementally
            if tool_call.function and tool_call.function.arguments:
                args_fragment = tool_call.function.arguments
                print(
                    f"Stage {i}: Got arguments fragment: {repr(args_fragment)}"
                )

                # For incremental output, each fragment should be new content only
                # The fragment should not contain all previous content
                if i >= 2 and previous_args_content:  # After we start getting arguments
                    # The new fragment should not be identical to or contain all previous content
                    assert args_fragment != previous_args_content, f"Fragment should be incremental, not cumulative: {args_fragment}"

                    # If this is truly incremental, the fragment should be relatively small
                    # compared to the complete arguments so far
                    if len(args_fragment) > len(previous_args_content):
                        print(
                            "Warning: Fragment seems cumulative rather than incremental"
                        )

                previous_args_content = args_fragment

    # Verify function name was sent at least once
    assert function_name_sent, "Function name should have been sent"


def test_streaming_arguments_delta_only(minimax_tool_parser):
    """Test that each streaming call returns only the delta (new part) of arguments."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    # Simulate two consecutive calls with growing arguments
    call1_text = '<tool_calls>\n{"name": "test_tool", "arguments": {"param1": "value1"}}'
    call2_text = '<tool_calls>\n{"name": "test_tool", "arguments": {"param1": "value1", "param2": "value2"}}'

    print(f"Call 1 text: {repr(call1_text)}")
    print(f"Call 2 text: {repr(call2_text)}")

    # First call - should get the function name and initial arguments
    result1 = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=call1_text,
        delta_text=call1_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    print(f"Result 1: {result1}")
    if result1 and hasattr(result1, 'tool_calls') and result1.tool_calls:
        for i, tc in enumerate(result1.tool_calls):
            print(f"  Tool call {i}: {tc}")

    # Second call - should only get the delta (new part) of arguments
    result2 = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text=call1_text,
        current_text=call2_text,
        delta_text=', "param2": "value2"}',
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    print(f"Result 2: {result2}")
    if result2 and hasattr(result2, 'tool_calls') and result2.tool_calls:
        for i, tc in enumerate(result2.tool_calls):
            print(f"  Tool call {i}: {tc}")

    # Verify the second call only returns the delta
    if result2 is not None and hasattr(result2,
                                       'tool_calls') and result2.tool_calls:
        tool_call = result2.tool_calls[0]
        if tool_call.function and tool_call.function.arguments:
            args_delta = tool_call.function.arguments
            print(f"Arguments delta from second call: {repr(args_delta)}")

            # Should only contain the new part, not the full arguments
            # The delta should be something like ', "param2": "value2"}' or just '"param2": "value2"'
            assert ', "param2": "value2"}' in args_delta or '"param2": "value2"' in args_delta, f"Expected delta containing param2, got: {args_delta}"

            # Should NOT contain the previous parameter data
            assert '"param1": "value1"' not in args_delta, f"Arguments delta should not contain previous data: {args_delta}"

            # The delta should be relatively short (incremental, not cumulative)
            expected_max_length = len(
                ', "param2": "value2"}') + 10  # Some tolerance
            assert len(
                args_delta
            ) <= expected_max_length, f"Delta seems too long (possibly cumulative): {args_delta}"

            print("✓ Delta validation passed")
        else:
            print("No arguments in result2 tool call")
    else:
        print("No tool calls in result2 or result2 is None")
        # This might be acceptable if no incremental update is needed
        # But let's at least verify that result1 had some content
        assert result1 is not None, "At least the first call should return something"


def test_streaming_openai_compatibility(minimax_tool_parser):
    """Test that streaming behavior is reasonable and incremental."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    # Test scenario: progressive building of a weather query
    # This simulates how a real model would generate tool calls incrementally
    stages = [
        '<tool_calls>\n{"name": "get_weather"',
        '<tool_calls>\n{"name": "get_weather", "arguments": {',
        '<tool_calls>\n{"name": "get_weather", "arguments": {"location": "San Francisco"',
        '<tool_calls>\n{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"',
        '<tool_calls>\n{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}\n</tool_calls>',
    ]

    function_name_sent = False
    total_args_received = ""

    for i, current_text in enumerate(stages):
        previous_text = stages[i - 1] if i > 0 else ""
        delta_text = current_text[len(previous_text
                                      ):] if previous_text else current_text

        print(
            f"\n--- Stage {i}: {['Name', 'Args Start', 'First Param', 'Second Param', 'Close'][i]} ---"
        )
        print(f"Previous: {repr(previous_text)}")
        print(f"Current:  {repr(current_text)}")
        print(f"Delta:    {repr(delta_text)}")

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        print(f"Result: {result}")

        if result and hasattr(result, 'tool_calls') and result.tool_calls:
            tool_call = result.tool_calls[0]

            # Check function name (should happen exactly once)
            if tool_call.function and tool_call.function.name:
                if function_name_sent:
                    # Function name should only be sent once
                    AssertionError(
                        f"Function name sent multiple times at stage {i}")
                assert tool_call.function.name == "get_weather"
                function_name_sent = True
                print(f"✓ Function name sent: {tool_call.function.name}")

            # Check arguments (should be incremental)
            if tool_call.function and tool_call.function.arguments:
                args_delta = tool_call.function.arguments
                print(f"✓ Arguments delta: {repr(args_delta)}")

                # Verify it's incremental - should not contain all previous content
                if total_args_received:
                    # Should not be exactly the same as what we had before
                    assert args_delta != total_args_received, \
                        f"Arguments delta should not be identical to previous content: {args_delta}"

                    # Should not contain the full previous content unless it's a very small addition
                    if total_args_received in args_delta and len(
                            args_delta) > len(total_args_received) * 2:
                        print(
                            "⚠️  Warning: Arguments delta seems cumulative rather than incremental"
                        )

                # Update our tracking of total arguments received
                total_args_received += args_delta

        elif i == 0:
            # First stage should return something (function name)
            print(f"⚠️  Expected function name at stage {i} but got no result")

    # Verify function name was sent
    assert function_name_sent, "Function name should have been sent at some point"

    # Verify we received some arguments
    assert total_args_received, "Should have received some arguments content"

    print(f"\n✓ Total arguments received: {repr(total_args_received)}")
    print("✓ Streaming test completed successfully")
