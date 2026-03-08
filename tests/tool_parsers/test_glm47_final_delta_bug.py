# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for GLM4-7 tool parser bug where the final token returns complete arguments
instead of incremental delta.

Bug description:
When streaming tool calls, the last token should return only the delta (new part)
of the arguments, but GLM4-7 parser returns the complete arguments, causing
parsing errors in applications expecting incremental updates.
"""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer):
    return Glm47MoeModelToolParser(glm47_tokenizer)


@pytest.fixture
def mock_request() -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
    ]
    request.tool_choice = "auto"
    return request


def _reset_streaming_state(parser):
    """Helper to reset parser streaming state."""
    parser._buffer = ""
    parser._in_tool_call = False
    parser.current_tool_name_sent = False
    parser._current_tool_name = None
    parser._pending_key = None
    parser._streaming_string_value = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._args_started = []
    parser._args_closed = []
    parser._seen_keys = []


def test_streaming_final_delta_not_cumulative(glm47_tool_parser, mock_request):
    """Test final streaming delta returns only new content, not complete args.

    This reproduces the bug where the last token returns complete arguments
    instead of just the delta, causing applications to fail when parsing
    incremental updates.
    """
    _reset_streaming_state(glm47_tool_parser)

    # Simulate progressive tool call building
    stages = [
        # Stage 1: Tool call starts with function name
        "<tool_call>get_weather\n",
        # Stage 2: First argument key-value pair starts
        "<tool_call>get_weather\n<arg_key>city</arg_key>",
        # Stage 3: First argument value
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
        # Stage 4: Second argument key
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key>",
        # Stage 5: Second argument value (partial)
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2025-08",
        # Stage 6: Second argument value (complete) - THIS IS WHERE THE BUG OCCURS
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2025-08-01</arg_value>",
        # Stage 7: Tool call ends
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2025-08-01</arg_value></tool_call>",
    ]

    collected_args_fragments = []
    previous_cumulative_args = ""

    for i, current_text in enumerate(stages):
        previous_text = stages[i - 1] if i > 0 else ""
        delta_text = current_text[len(previous_text) :] if i > 0 else current_text

        print(f"\n=== Stage {i} ===")
        print(f"Delta text: {repr(delta_text)}")

        result = glm47_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            for tc in result.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    func = tc.function
                    args = (
                        func.get("arguments")
                        if isinstance(func, dict)
                        else getattr(func, "arguments", None)
                    )
                    if args:
                        print(f"Arguments fragment: {repr(args)}")
                        collected_args_fragments.append(args)

                        # Build cumulative args to compare
                        current_cumulative_args = previous_cumulative_args + args

                        # CRITICAL CHECK: Fragment should be incremental
                        # not cumulative. Verify it's truly incremental
                        if i >= 2 and previous_cumulative_args:
                            # Fragment should NOT start new JSON object
                            # when one is already being built. This would
                            # indicate cumulative (complete) arguments
                            if (
                                args.strip().startswith("{")
                                and previous_cumulative_args
                            ):
                                pytest.fail(
                                    f"Stage {i}: Fragment appears to be "
                                    f"cumulative (starts new object), not "
                                    f"incremental delta. Fragment: "
                                    f"{repr(args)}, Previous cumulative: "
                                    f"{repr(previous_cumulative_args)}"
                                )

                            # Additional check: if fragment is parseable
                            # as complete JSON with content, and we
                            # already have partial JSON, it's cumulative
                            try:
                                parsed_fragment = json.loads(args)
                                if (
                                    isinstance(parsed_fragment, dict)
                                    and len(parsed_fragment) >= 1
                                ):
                                    # Complete JSON object, not a delta
                                    pytest.fail(
                                        f"Stage {i}: Fragment is complete "
                                        f"JSON object (cumulative), not "
                                        f"incremental delta. Fragment: "
                                        f"{repr(args)}"
                                    )
                            except json.JSONDecodeError:
                                # Good - fragment is not complete JSON, it's incremental
                                pass

                        previous_cumulative_args = current_cumulative_args

    # Verify we got multiple fragments (incremental streaming)
    assert len(collected_args_fragments) > 0, "Should have collected argument fragments"

    # Reconstruct complete arguments from fragments
    complete_args_str = "".join(collected_args_fragments)
    print(f"\nComplete arguments: {complete_args_str}")

    # Verify the complete arguments are valid JSON
    parsed_args = json.loads(complete_args_str)
    assert parsed_args["city"] == "Beijing"
    assert parsed_args["date"] == "2025-08-01"

    print(
        "\n✓ Test passed: Streaming returns incremental deltas, "
        "not cumulative arguments"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
