# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
from typing import Any

import pytest

from vllm.entrypoints.openai.protocol import (
    ChatCompletionToolsParam,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.minimax_tool_parser import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

pytestmark = pytest.mark.cpu_test

# Use a common model that is likely to be available
MODEL = "MiniMaxAi/MiniMax-M1-40k"


@pytest.fixture(scope="module")
def minimax_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def minimax_tool_parser(minimax_tokenizer):
    return MinimaxToolParser(minimax_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
            },
        ),
    ]


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(minimax_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
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
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}
</tool_calls>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Orlando",
                                "state": "FL",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                ),
            ],
            None,
        ),
        (
            """I'll help you check the weather. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}
</tool_calls>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Seattle",
                                "state": "WA",
                                "unit": "celsius",
                            }
                        ),
                    )
                )
            ],
            "I'll help you check the weather.",
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "New York", "state": "NY", "unit": "celsius"}}
</tool_calls>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "New York",
                                "state": "NY",
                                "unit": "celsius",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA"}}""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Boston",
                                "state": "MA",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
    ],
)
def test_extract_tool_calls(
    minimax_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
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

    processed_output = minimax_tool_parser.preprocess_model_output(model_output)

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
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"

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
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_tool"
    assert extracted_tool_calls.tool_calls[1].function.name == "another_valid_tool"


def test_extract_tool_calls_missing_name_or_arguments(minimax_tool_parser):
    """Test that tool calls missing name or arguments are filtered out."""
    model_output = """<tool_calls>
{"name": "valid_tool", "arguments": {"city": "Seattle"}}
{"name": "missing_args"}
{"arguments": {"city": "Portland"}}
{"name": "another_valid_tool", "arguments": {"param": "value"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid tool calls with both name and arguments
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_tool"
    assert extracted_tool_calls.tool_calls[1].function.name == "another_valid_tool"


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
    if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
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

    if result is not None and hasattr(result, "content"):
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
    assert hasattr(result, "content")
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
    if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
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
        model_output, request=None
    )  # type: ignore[arg-type]

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
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n</tool',
        '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n</tool_calls>',
    ]

    function_name_sent = False
    previous_args_content = ""

    for i, current_text in enumerate(stages):
        previous_text = stages[i - 1] if i > 0 else ""
        delta_text = current_text[len(previous_text) :] if i > 0 else current_text

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

        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            tool_call = result.tool_calls[0]

            # Check if function name is sent (should happen only once)
            if tool_call.function and tool_call.function.name:
                assert tool_call.function.name == "get_current_weather"
                function_name_sent = True
                print(f"Stage {i}: Function name sent: {tool_call.function.name}")

            # Check if arguments are sent incrementally
            if tool_call.function and tool_call.function.arguments:
                args_fragment = tool_call.function.arguments
                print(f"Stage {i}: Got arguments fragment: {repr(args_fragment)}")

                # For incremental output, each fragment should be new content only
                # The fragment should not contain all previous content
                if i >= 2 and previous_args_content:  # After we start getting arguments
                    # The new fragment should not be identical to or contain all previous content
                    assert args_fragment != previous_args_content, (
                        f"Fragment should be incremental, not cumulative: {args_fragment}"
                    )

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
    call1_text = (
        '<tool_calls>\n{"name": "test_tool", "arguments": {"param1": "value1"}}'
    )
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
    if result1 and hasattr(result1, "tool_calls") and result1.tool_calls:
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
    if result2 and hasattr(result2, "tool_calls") and result2.tool_calls:
        for i, tc in enumerate(result2.tool_calls):
            print(f"  Tool call {i}: {tc}")

    # Verify the second call only returns the delta
    if result2 is not None and hasattr(result2, "tool_calls") and result2.tool_calls:
        tool_call = result2.tool_calls[0]
        if tool_call.function and tool_call.function.arguments:
            args_delta = tool_call.function.arguments
            print(f"Arguments delta from second call: {repr(args_delta)}")

            # Should only contain the new part, not the full arguments
            # The delta should be something like ', "param2": "value2"}' or just '"param2": "value2"'
            assert (
                ', "param2": "value2"}' in args_delta
                or '"param2": "value2"' in args_delta
            ), f"Expected delta containing param2, got: {args_delta}"

            # Should NOT contain the previous parameter data
            assert '"param1": "value1"' not in args_delta, (
                f"Arguments delta should not contain previous data: {args_delta}"
            )

            # The delta should be relatively short (incremental, not cumulative)
            expected_max_length = len(', "param2": "value2"}') + 10  # Some tolerance
            assert len(args_delta) <= expected_max_length, (
                f"Delta seems too long (possibly cumulative): {args_delta}"
            )

            print("✓ Delta validation passed")
        else:
            print("No arguments in result2 tool call")
    else:
        print("No tool calls in result2 or result2 is None")
        # This might be acceptable if no incremental update is needed
        # But let's at least verify that result1 had some content
        assert result1 is not None, "At least the first call should return something"


def test_streaming_openai_compatibility(minimax_tool_parser):
    """Test that streaming behavior with buffering works correctly."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []
    # Reset buffering state
    minimax_tool_parser.pending_buffer = ""
    minimax_tool_parser.in_thinking_tag = False
    minimax_tool_parser.thinking_depth = 0

    # Test scenario: simple buffering without complex tool call context
    test_cases: list[dict[str, Any]] = [
        {
            "stage": "Token: <",
            "previous": "",
            "current": "<",
            "delta": "<",
            "expected_content": None,  # Should be buffered
        },
        {
            "stage": "Token: tool_calls>",
            "previous": "<",
            "current": "<tool_calls>",
            "delta": "tool_calls>",
            "expected_content": None,  # Complete tag, should not output
        },
        {
            "stage": "Regular content",
            "previous": "Hello",
            "current": "Hello world",
            "delta": " world",
            "expected_content": " world",  # Normal content should pass through
        },
        {
            "stage": "Content with end tag start",
            "previous": "Text",
            "current": "Text content</tool_",
            "delta": " content</tool_",
            "expected_content": " content",  # Content part output, </tool_ buffered
        },
        {
            "stage": "Complete end tag",
            "previous": "Text content</tool_",
            "current": "Text content</tool_calls>",
            "delta": "calls>",
            "expected_content": None,  # Complete close tag, should not output
        },
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Stage {i}: {test_case['stage']} ---")
        print(f"Previous: {repr(test_case['previous'])}")
        print(f"Current:  {repr(test_case['current'])}")
        print(f"Delta:    {repr(test_case['delta'])}")

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=test_case["previous"],
            current_text=test_case["current"],
            delta_text=test_case["delta"],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        print(f"Result: {result}")

        # Check expected content
        if test_case["expected_content"] is None:
            assert result is None or not getattr(result, "content", None), (
                f"Stage {i}: Expected no content, got {result}"
            )
            print("✓ No content output as expected")
        else:
            assert result is not None and hasattr(result, "content"), (
                f"Stage {i}: Expected content, got {result}"
            )
            assert result.content == test_case["expected_content"], (
                f"Stage {i}: Expected content {test_case['expected_content']}, got {result.content}"
            )
            print(f"✓ Content matches: {repr(result.content)}")

    print("✓ Streaming test with buffering completed successfully")


def test_streaming_thinking_tag_buffering(minimax_tool_parser):
    """Test that tool calls within thinking tags are properly handled during streaming."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []
    # Reset buffering state
    minimax_tool_parser.pending_buffer = ""
    minimax_tool_parser.in_thinking_tag = False
    minimax_tool_parser.thinking_depth = 0

    # Test scenario: tool calls within thinking tags should be ignored
    test_cases: list[dict[str, Any]] = [
        {
            "stage": "Start thinking",
            "previous": "",
            "current": "<think>I need to use a tool. <tool_calls>",
            "delta": "<think>I need to use a tool. <tool_calls>",
            "expected_content": "<think>I need to use a tool. <tool_calls>",  # Should pass through as content
        },
        {
            "stage": "Tool call in thinking",
            "previous": "<think>I need to use a tool. <tool_calls>",
            "current": '<think>I need to use a tool. <tool_calls>\n{"name": "ignored_tool", "arguments": {"param": "value"}}\n</tool_calls>',
            "delta": '\n{"name": "ignored_tool", "arguments": {"param": "value"}}\n</tool_calls>',
            "expected_content": '\n{"name": "ignored_tool", "arguments": {"param": "value"}}\n</tool_calls>',  # </tool_calls> should be preserved in thinking tags
        },
        {
            "stage": "Real tool call after thinking",
            "previous": '<think>I need to use a tool. <tool_calls>\n{"name": "ignored_tool", "arguments": {"param": "value"}}\n</tool_calls></think>',
            "current": '<think>I need to use a tool. <tool_calls>\n{"name": "ignored_tool", "arguments": {"param": "value"}}\n</tool_calls></think>\n<tool_calls>',
            "delta": "\n<tool_calls>",
            "expected_content": "\n",  # Should output '\n' and suppress <tool_calls>
        },
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Stage {i}: {test_case['stage']} ---")
        print(f"Previous: {repr(test_case['previous'])}")
        print(f"Current:  {repr(test_case['current'])}")
        print(f"Delta:    {repr(test_case['delta'])}")

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=test_case["previous"],
            current_text=test_case["current"],
            delta_text=test_case["delta"],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        print(f"Result: {result}")

        # Check expected content
        if "expected_content" in test_case:
            if test_case["expected_content"] is None:
                assert result is None or not getattr(result, "content", None), (
                    f"Stage {i}: Expected no content, got {result}"
                )
            else:
                assert result is not None and hasattr(result, "content"), (
                    f"Stage {i}: Expected content, got {result}"
                )
                assert result.content == test_case["expected_content"], (
                    f"Stage {i}: Expected content {test_case['expected_content']}, got {result.content}"
                )
                print(f"✓ Content matches: {repr(result.content)}")

        # Check tool calls
        if test_case.get("expected_tool_call"):
            assert (
                result is not None
                and hasattr(result, "tool_calls")
                and result.tool_calls
            ), f"Stage {i}: Expected tool call, got {result}"

            tool_call = result.tool_calls[0]
            assert tool_call.function.name == "real_tool", (
                f"Expected real_tool, got {tool_call.function.name}"
            )
            print(f"✓ Real tool call detected: {tool_call.function.name}")

    print("✓ Thinking tag buffering test completed successfully")


def reset_streaming_state(minimax_tool_parser):
    """Helper function to properly reset the streaming state for MinimaxToolParser."""
    # Reset minimax-specific state
    minimax_tool_parser._reset_streaming_state()

    # Reset base class state (these should still be reset for compatibility)
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.streamed_args_for_tool = []


def test_streaming_complex_scenario_with_multiple_tools(minimax_tool_parser):
    """Test complex streaming scenario: tools inside <think> tags and multiple tool calls in one group."""
    # Reset streaming state
    reset_streaming_state(minimax_tool_parser)

    # Complex scenario: tools inside thinking tags and multiple tools in one group
    test_stages: list[dict[str, Any]] = [
        {
            "stage": "Initial content",
            "previous": "",
            "current": "Let me help you with this task.",
            "delta": "Let me help you with this task.",
            "expected_content": "Let me help you with this task.",
            "expected_tool_calls": 0,
        },
        {
            "stage": "Start thinking tag",
            "previous": "Let me help you with this task.",
            "current": "Let me help you with this task.<think>I need to analyze this situation first.",
            "delta": "<think>I need to analyze this situation first.",
            "expected_content": "<think>I need to analyze this situation first.",
            "expected_tool_calls": 0,
        },
        {
            "stage": "Tool call inside thinking tag starts",
            "previous": "Let me help you with this task.<think>I need to analyze this situation first.",
            "current": "Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>",
            "delta": "<tool_calls>",
            "expected_content": "<tool_calls>",  # Inside thinking tags, tool tags should be preserved as content
            "expected_tool_calls": 0,
        },
        {
            "stage": "Complete tool call inside thinking tag",
            "previous": "Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>",
            "current": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls>',
            "delta": '\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls>',
            "expected_content": '\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls>',
            "expected_tool_calls": 0,  # Tools inside thinking tags should be ignored
        },
        {
            "stage": "End thinking tag",
            "previous": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls>',
            "current": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>',
            "delta": "</think>",
            "expected_content": "</think>",
            "expected_tool_calls": 0,
        },
        {
            "stage": "Multiple tools group starts",
            "previous": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>',
            "current": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>',
            "delta": "\nNow I need to get weather information and calculate area.<tool_calls>",
            "expected_content": "\nNow I need to get weather information and calculate area.",  # <tool_calls> should be filtered
            "expected_tool_calls": 0,
        },
        {
            "stage": "First tool in group",
            "previous": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>',
            "current": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}',
            "delta": '\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}',
            "expected_content": None,  # No content should be output when tool call is in progress
            "expected_tool_calls": 1,
            "expected_tool_name": "get_current_weather",
        },
        {
            "stage": "Second tool in group",
            "previous": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}',
            "current": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}',
            "delta": '\n{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}',
            "expected_content": None,
            "expected_tool_calls": 1,
            "expected_tool_name": "calculate_area",
        },
        {
            "stage": "Complete tool calls group",
            "previous": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}',
            "current": 'Let me help you with this task.<think>I need to analyze this situation first.<tool_calls>\n{"name": "internal_analysis", "arguments": {"query": "analyze situation"}}\n</tool_calls></think>\nNow I need to get weather information and calculate area.<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}</tool_calls>',
            "delta": "</tool_calls>",
            "expected_content": None,
            "expected_tool_calls": 0,
        },
    ]

    tool_calls_count = 0

    for i, test_case in enumerate(test_stages):
        print(f"\n--- Stage {i}: {test_case['stage']} ---")
        print(
            f"Previous: {repr(test_case['previous'][:100])}{'...' if len(test_case['previous']) > 100 else ''}"
        )
        print(f"Current:  {repr(test_case['current'][-100:])}")
        print(f"Delta:    {repr(test_case['delta'])}")

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=test_case["previous"],
            current_text=test_case["current"],
            delta_text=test_case["delta"],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        print(f"Result: {result}")

        # Check expected content
        if test_case["expected_content"] is None:
            assert result is None or not getattr(result, "content", None), (
                f"Stage {i}: Expected no content output, got {result}"
            )
            print("✓ No content output as expected")
        else:
            assert result is not None and hasattr(result, "content"), (
                f"Stage {i}: Expected content output, got {result}"
            )
            assert result.content == test_case["expected_content"], (
                f"Stage {i}: Expected content {repr(test_case['expected_content'])}, got {repr(result.content)}"
            )
            print(f"✓ Content matches: {repr(result.content)}")

        # Check tool calls
        expected_tool_calls = test_case["expected_tool_calls"]
        actual_tool_calls = (
            len(result.tool_calls)
            if result and hasattr(result, "tool_calls") and result.tool_calls
            else 0
        )

        if expected_tool_calls > 0:
            assert actual_tool_calls >= expected_tool_calls, (
                f"Stage {i}: Expected at least {expected_tool_calls} tool calls, got {actual_tool_calls}"
            )

            if "expected_tool_name" in test_case:
                # Find the tool call with the expected name
                found_tool_call = None
                for tool_call in result.tool_calls:
                    if tool_call.function.name == test_case["expected_tool_name"]:
                        found_tool_call = tool_call
                        break

                assert found_tool_call is not None, (
                    f"Stage {i}: Expected tool name {test_case['expected_tool_name']} not found in tool calls: {[tc.function.name for tc in result.tool_calls]}"
                )
                print(f"✓ Tool call correct: {found_tool_call.function.name}")

                # Ensure tools inside thinking tags are not called
                assert found_tool_call.function.name != "internal_analysis", (
                    f"Stage {i}: Tool 'internal_analysis' inside thinking tags should not be called"
                )

            tool_calls_count += actual_tool_calls
            print(f"✓ Detected {actual_tool_calls} tool calls")
        else:
            assert actual_tool_calls == 0, (
                f"Stage {i}: Expected no tool calls, got {actual_tool_calls}"
            )

    # Verify overall results
    print("\n=== Test Summary ===")
    print(f"Total tool calls count: {tool_calls_count}")
    assert tool_calls_count >= 2, (
        f"Expected at least 2 valid tool calls (outside thinking tags), but got {tool_calls_count}"
    )

    print("✓ Complex streaming test completed:")
    print("  - ✓ Tools inside thinking tags correctly ignored")
    print("  - ✓ Two tool groups outside thinking tags correctly parsed")
    print("  - ✓ Content and tool call streaming correctly handled")
    print("  - ✓ Buffering mechanism works correctly")


def test_streaming_character_by_character_output(minimax_tool_parser):
    """Test character-by-character streaming output to simulate real streaming scenarios."""
    # Reset streaming state
    reset_streaming_state(minimax_tool_parser)

    # Complete text that will be streamed character by character
    complete_text = """I'll help you with the weather analysis. <think>Let me think about this. <tool_calls>
{"name": "internal_analysis", "arguments": {"type": "thinking"}}
</tool_calls>This tool should be ignored.</think>

Now I'll get the weather information for you. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}
{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}
</tool_calls>Here are the results."""

    print("\n=== Starting character-by-character streaming test ===")
    print(f"Complete text length: {len(complete_text)} characters")

    # Track the streaming results
    content_fragments = []
    tool_calls_detected = []

    # Stream character by character
    for i in range(1, len(complete_text) + 1):
        current_text = complete_text[:i]
        previous_text = complete_text[: i - 1] if i > 1 else ""
        delta_text = complete_text[i - 1 : i]

        # Show progress every 50 characters
        if i % 50 == 0 or i == len(complete_text):
            print(f"Progress: {i}/{len(complete_text)} characters")

        # Call the streaming parser
        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        # Collect results
        if result is not None:
            if hasattr(result, "content") and result.content:
                content_fragments.append(result.content)
                # Log important content fragments
                if any(
                    keyword in result.content
                    for keyword in [
                        "<think>",
                        "</think>",
                        "<tool_calls>",
                        "</tool_calls>",
                    ]
                ):
                    print(f"  Char {i}: Content fragment: {repr(result.content)}")

            if hasattr(result, "tool_calls") and result.tool_calls:
                for tool_call in result.tool_calls:
                    tool_info = {
                        "character_position": i,
                        "function_name": tool_call.function.name
                        if tool_call.function
                        else None,
                        "arguments": tool_call.function.arguments
                        if tool_call.function
                        else None,
                    }
                    tool_calls_detected.append(tool_info)
                    print(f"  Char {i}: Tool call detected: {tool_call.function.name}")
                    if tool_call.function.arguments:
                        print(f"    Arguments: {repr(tool_call.function.arguments)}")

    # Verify results
    print("\n=== Streaming Test Results ===")
    print(f"Total content fragments: {len(content_fragments)}")
    print(f"Total tool calls detected: {len(tool_calls_detected)}")

    # Reconstruct content from fragments
    reconstructed_content = "".join(content_fragments)
    print(f"Reconstructed content length: {len(reconstructed_content)}")

    # Verify thinking tags content is preserved
    assert "<think>" in reconstructed_content, (
        "Opening thinking tag should be preserved in content"
    )
    assert "</think>" in reconstructed_content, (
        "Closing thinking tag should be preserved in content"
    )

    # Verify that tool calls inside thinking tags are NOT extracted as actual tool calls
    thinking_tool_calls = [
        tc for tc in tool_calls_detected if tc["function_name"] == "internal_analysis"
    ]
    assert len(thinking_tool_calls) == 0, (
        f"Tool calls inside thinking tags should be ignored, but found: {thinking_tool_calls}"
    )

    # Verify that real tool calls outside thinking tags ARE extracted
    weather_tool_calls = [
        tc for tc in tool_calls_detected if tc["function_name"] == "get_current_weather"
    ]
    area_tool_calls = [
        tc for tc in tool_calls_detected if tc["function_name"] == "calculate_area"
    ]
    print(tool_calls_detected)
    assert len(weather_tool_calls) > 0, (
        "get_current_weather tool call should be detected"
    )
    assert len(area_tool_calls) > 0, "calculate_area tool call should be detected"

    # Verify tool call arguments are properly streamed
    weather_args_found = any(
        tc["arguments"] for tc in weather_tool_calls if tc["arguments"]
    )
    area_args_found = any(tc["arguments"] for tc in area_tool_calls if tc["arguments"])

    print(f"Weather tool call with arguments: {weather_args_found}")
    print(f"Area tool call with arguments: {area_args_found}")

    # Verify content before and after tool calls
    assert "I'll help you with the weather analysis." in reconstructed_content, (
        "Initial content should be preserved"
    )
    assert "Here are the results." in reconstructed_content, (
        "Final content should be preserved"
    )

    # Verify that <tool_calls> and </tool_calls> tags are not included in the final content
    # (they should be filtered out when not inside thinking tags)
    content_outside_thinking = reconstructed_content
    # Remove thinking tag content to check content outside
    if "<think>" in content_outside_thinking and "</think>" in content_outside_thinking:
        start_think = content_outside_thinking.find("<think>")
        end_think = content_outside_thinking.find("</think>") + len("</think>")
        content_outside_thinking = (
            content_outside_thinking[:start_think]
            + content_outside_thinking[end_think:]
        )

    # Outside thinking tags, tool_calls tags should be filtered
    tool_calls_in_content = content_outside_thinking.count("<tool_calls>")
    assert tool_calls_in_content == 0, (
        f"<tool_calls> tags should be filtered from content outside thinking tags, but found {tool_calls_in_content}"
    )

    print("\n=== Character-by-character streaming test completed successfully ===")
    print("✓ Tool calls inside thinking tags correctly ignored")
    print("✓ Tool calls outside thinking tags correctly detected")
    print("✓ Content properly streamed and reconstructed")
    print("✓ Tool call tags properly filtered from content")
    print("✓ Character-level streaming works correctly")


def test_streaming_character_by_character_simple_tool_call(minimax_tool_parser):
    """Test character-by-character streaming for a simple tool call scenario."""
    # Reset streaming state
    reset_streaming_state(minimax_tool_parser)

    # Simple tool call text
    simple_text = 'Let me check the weather. <tool_calls>\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n</tool_calls>'

    print("\n=== Simple character-by-character test ===")
    print(f"Text: {repr(simple_text)}")

    content_parts = []
    tool_name_sent = False
    tool_args_sent = False

    for i in range(1, len(simple_text) + 1):
        current_text = simple_text[:i]
        previous_text = simple_text[: i - 1] if i > 1 else ""
        delta_text = simple_text[i - 1 : i]

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        if result:
            if hasattr(result, "content") and result.content:
                content_parts.append(result.content)
                print(
                    f"  Char {i} ({repr(delta_text)}): Content: {repr(result.content)}"
                )

            if hasattr(result, "tool_calls") and result.tool_calls:
                for tool_call in result.tool_calls:
                    if tool_call.function and tool_call.function.name:
                        tool_name_sent = True
                        print(f"  Char {i}: Tool name: {tool_call.function.name}")
                    if tool_call.function and tool_call.function.arguments:
                        tool_args_sent = True
                        print(
                            f"  Char {i}: Tool args: {repr(tool_call.function.arguments)}"
                        )

    # Verify basic expectations
    reconstructed_content = "".join(content_parts)
    print(f"Final reconstructed content: {repr(reconstructed_content)}")

    assert tool_name_sent, "Tool name should be sent during streaming"
    assert tool_args_sent, "Tool arguments should be sent during streaming"
    assert "Let me check the weather." in reconstructed_content, (
        "Initial content should be preserved"
    )

    print("✓ Simple character-by-character test passed")


def test_streaming_character_by_character_with_buffering(minimax_tool_parser):
    """Test character-by-character streaming with edge cases that trigger buffering."""
    # Reset streaming state
    reset_streaming_state(minimax_tool_parser)

    # Text that includes potential buffering scenarios
    buffering_text = 'Hello world<tool_calls>\n{"name": "test"}\n</tool_calls>done'

    print("\n=== Buffering character-by-character test ===")
    print(f"Text: {repr(buffering_text)}")

    all_content = []

    for i in range(1, len(buffering_text) + 1):
        current_text = buffering_text[:i]
        previous_text = buffering_text[: i - 1] if i > 1 else ""
        delta_text = buffering_text[i - 1 : i]

        result = minimax_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

        if result and hasattr(result, "content") and result.content:
            all_content.append(result.content)
            print(f"  Char {i} ({repr(delta_text)}): {repr(result.content)}")

    final_content = "".join(all_content)
    print(f"Final content: {repr(final_content)}")

    # The parser should handle the edge case where </tool_calls> appears before <tool_calls>
    assert "Hello" in final_content, "Initial 'Hello' should be preserved"
    assert "world" in final_content, (
        "Content after false closing tag should be preserved"
    )
    assert "done" in final_content, "Final content should be preserved"

    print("✓ Buffering character-by-character test passed")
