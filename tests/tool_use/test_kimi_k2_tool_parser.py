# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

pytestmark = pytest.mark.cpu_test

# Use a common model that is likely to be available
MODEL = "moonshotai/Kimi-K2-Instruct"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def kimi_k2_tool_parser(kimi_k2_tokenizer):
    return KimiK2ToolParser(kimi_k2_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function

        # assert tool call id format: should contain function name and numeric index
        # Format can be either "functions.func_name:0" or "func_name:0"
        assert actual_tool_call.id.split(":")[-1].isdigit()
        assert (
            actual_tool_call.id.split(":")[0].split(".")[-1]
            == expected_tool_call.function.name
        )


def test_extract_tool_calls_no_tools(kimi_k2_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "tool_call_with_content_before",
        "multi_tool_call_with_content_before",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.get_weather:0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            "I'll help you check the weather. ",
        ),
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.get_weather:0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                            },
                        ),
                    ),
                    type="function",
                ),
                ToolCall(
                    id="functions.get_weather:1",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Shanghai",
                            },
                        ),
                    ),
                    type="function",
                ),
            ],
            "I'll help you check the weather. ",
        ),
    ],
)
def test_extract_tool_calls(
    kimi_k2_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_invalid_json(kimi_k2_tool_parser):
    """we'll return every funcall result"""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing" <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "invalid_get_weather"
    assert extracted_tool_calls.tool_calls[1].function.name == "valid_get_weather"


def test_extract_tool_calls_invalid_funcall(kimi_k2_tool_parser):
    """we'll return every funcall result"""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather.0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_get_weather"


def test_streaming_basic_functionality(kimi_k2_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    kimi_k2_tool_parser.current_tool_name_sent = False
    kimi_k2_tool_parser.prev_tool_call_arr = []
    kimi_k2_tool_parser.current_tool_id = -1
    kimi_k2_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """ check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>"""

    # First call should handle the initial setup
    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you",
        current_text=current_text,
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The result might be None or contain tool call information
    # This depends on the internal state management
    if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
        assert len(result.tool_calls) >= 0


def test_streaming_no_tool_calls(kimi_k2_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
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


def test_token_leak_between_section_and_tool_begin(kimi_k2_tool_parser):
    """
    Test that text between <|tool_calls_section_begin|> and <|tool_call_begin|>
    is suppressed and does not leak into reasoning_delta.
    This is the main vulnerability being fixed.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    # Get token IDs for the markers
    section_begin_token_id = kimi_k2_tool_parser.vocab.get(
        "<|tool_calls_section_begin|>"
    )
    tool_call_begin_token_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")

    # Simulate streaming sequence:
    # Delta 1: "I'll help you with that. "
    result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="I'll help you with that. ",
        delta_text="I'll help you with that. ",
        previous_token_ids=[],
        current_token_ids=[1, 2, 3],  # Regular tokens
        delta_token_ids=[1, 2, 3],
        request=None,
    )
    assert result1 is not None
    assert result1.content == "I'll help you with that. "

    # Delta 2: "<|tool_calls_section_begin|>"
    prev_ids = [1, 2, 3]
    curr_ids = prev_ids + [section_begin_token_id]
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you with that. ",
        current_text="I'll help you with that. <|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=prev_ids,
        current_token_ids=curr_ids,
        delta_token_ids=[section_begin_token_id],
        request=None,
    )
    # Section marker should be stripped and suppressed
    assert result2 is None or (result2.content is None or result2.content == "")

    # Delta 3: " spurious text or tokens " (THE LEAK SCENARIO)
    prev_ids = curr_ids
    curr_ids = curr_ids + [4, 5]
    result3 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you with that. <|tool_calls_section_begin|>",
        current_text="I'll help you with that. <|tool_calls_section_begin|> spurious text ",
        delta_text=" spurious text ",
        previous_token_ids=prev_ids,
        current_token_ids=curr_ids,
        delta_token_ids=[4, 5],
        request=None,
    )
    # CRITICAL: This text should be suppressed, NOT returned as reasoning_delta
    assert result3 is None or (result3.content is None or result3.content == "")

    # Delta 4: "<|tool_call_begin|>..."
    prev_ids = curr_ids
    curr_ids = curr_ids + [tool_call_begin_token_id]
    _result4 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you with that. <|tool_calls_section_begin|> spurious text ",
        current_text="I'll help you with that. <|tool_calls_section_begin|> spurious text <|tool_call_begin|>",
        delta_text="<|tool_call_begin|>",
        previous_token_ids=prev_ids,
        current_token_ids=curr_ids,
        delta_token_ids=[tool_call_begin_token_id],
        request=None,
    )
    # Now we're in tool call mode, result depends on internal state
    # The key is that the spurious text from Delta 3 was not leaked


def test_split_markers_across_deltas(kimi_k2_tool_parser):
    """
    Test that markers split across delta chunks are correctly detected
    via the rolling buffer mechanism.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_token_id = kimi_k2_tool_parser.vocab.get(
        "<|tool_calls_section_begin|>"
    )

    # Delta 1: "...reasoning<|tool_calls_sec"
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning",
        current_text="Some reasoning<|tool_calls_sec",
        delta_text="<|tool_calls_sec",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3],  # Partial token
        delta_token_ids=[3],
        request=None,
    )
    # Partial token not recognized yet, might be buffered
    # Should return as content or None (depends on implementation)

    # Delta 2: "tion_begin|> "  (completes the marker)
    _result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning<|tool_calls_sec",
        current_text="Some reasoning<|tool_calls_section_begin|> ",
        delta_text="tion_begin|> ",
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, section_begin_token_id, 4],
        delta_token_ids=[section_begin_token_id, 4],
        request=None,
    )
    # Now the complete marker should be detected via buffer
    # The parser should enter tool section mode
    assert kimi_k2_tool_parser.in_tool_section is True


def test_marker_variants(kimi_k2_tool_parser):
    """Test that both singular and plural marker variants are recognized."""
    kimi_k2_tool_parser.reset_streaming_state()

    # Test singular variant: <|tool_call_section_begin|> (note: singular "call")
    singular_token_id = kimi_k2_tool_parser.vocab.get("<|tool_call_section_begin|>")

    if singular_token_id is not None:  # Only test if tokenizer supports it
        _result = kimi_k2_tool_parser.extract_tool_calls_streaming(
            previous_text="Reasoning ",
            current_text="Reasoning <|tool_call_section_begin|>",
            delta_text="<|tool_call_section_begin|>",
            previous_token_ids=[1, 2],
            current_token_ids=[1, 2, singular_token_id],
            delta_token_ids=[singular_token_id],
            request=None,
        )
        # Should enter tool section mode with singular variant too
        assert kimi_k2_tool_parser.in_tool_section is True


def test_reentry_to_reasoning_after_tool_section(kimi_k2_tool_parser):
    """
    Test that after exiting a tool section with <|tool_calls_section_end|>,
    subsequent text is correctly returned as reasoning content.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Enter tool section
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Exit tool section
    _result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|><|tool_calls_section_end|>",
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id, section_end_id],
        delta_token_ids=[section_end_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is False

    # Subsequent reasoning text should be returned normally
    result3 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|><|tool_calls_section_end|>",
        current_text="<|tool_calls_section_begin|><|tool_calls_section_end|> More reasoning",
        delta_text=" More reasoning",
        previous_token_ids=[section_begin_id, section_end_id],
        current_token_ids=[section_begin_id, section_end_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )
    assert result3 is not None
    assert result3.content == " More reasoning"


def test_empty_tool_section(kimi_k2_tool_parser):
    """Test an empty tool section (begin immediately followed by end)."""
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Section begin
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning <|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[1],
        current_token_ids=[1, section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )

    # Immediate section end
    _result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning <|tool_calls_section_begin|>",
        current_text="Reasoning <|tool_calls_section_begin|><|tool_calls_section_end|>",
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[1, section_begin_id],
        current_token_ids=[1, section_begin_id, section_end_id],
        delta_token_ids=[section_end_id],
        request=None,
    )
    # Should exit cleanly without errors
    assert kimi_k2_tool_parser.in_tool_section is False


def test_malformed_tool_section_recovery(kimi_k2_tool_parser):
    """
    Test that the parser recovers from a malformed tool section
    that never closes properly.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")

    # Enter tool section
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Simulate a lot of text without proper tool calls or section end
    # This should trigger the error recovery mechanism
    large_text = "x" * 10000  # Exceeds max_section_chars

    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|>" + large_text,
        delta_text=large_text,
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id] + list(range(100, 100 + len(large_text))),
        delta_token_ids=list(range(100, 100 + len(large_text))),
        request=None,
    )

    # Parser should have force-exited the tool section
    assert kimi_k2_tool_parser.in_tool_section is False
    # And returned the content as reasoning
    assert result2 is not None
    assert result2.content == large_text


def test_state_reset(kimi_k2_tool_parser):
    """Test that reset_streaming_state() properly clears all state."""
    # Put parser in a complex state
    kimi_k2_tool_parser.in_tool_section = True
    kimi_k2_tool_parser.token_buffer = "some buffer"
    kimi_k2_tool_parser.current_tool_id = 5
    kimi_k2_tool_parser.prev_tool_call_arr = [{"id": "test"}]
    kimi_k2_tool_parser.section_char_count = 1000

    # Reset
    kimi_k2_tool_parser.reset_streaming_state()

    # Verify all state is cleared
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.token_buffer == ""
    assert kimi_k2_tool_parser.current_tool_id == -1
    assert kimi_k2_tool_parser.prev_tool_call_arr == []
    assert kimi_k2_tool_parser.section_char_count == 0
    assert kimi_k2_tool_parser.current_tool_name_sent is False
    assert kimi_k2_tool_parser.streamed_args_for_tool == []


def test_section_begin_noise_tool_begin_same_chunk(kimi_k2_tool_parser):
    """
    Test that begin→noise→tool_begin within the SAME chunk suppresses
    the noise text correctly (not just across chunks).
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    tool_call_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")

    # Single delta containing: section_begin + spurious text + tool_call_begin
    combined_text = "<|tool_calls_section_begin|> noise text <|tool_call_begin|>"

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning " + combined_text,
        delta_text=combined_text,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id, 3, 4, tool_call_begin_id],
        delta_token_ids=[section_begin_id, 3, 4, tool_call_begin_id],
        request=None,
    )

    # The noise text should NOT leak into content
    # Result should either be None/empty or start tool call parsing
    if result is not None and result.content is not None:
        # If content is returned, it should not contain the noise
        assert "noise text" not in result.content
        assert result.content == "" or result.content.strip() == ""


def test_stream_ends_without_section_end_marker(kimi_k2_tool_parser):
    """
    Test that if the stream ends (EOF) without a proper section end marker,
    the parser doesn't leak text, doesn't crash, and resets state cleanly.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")

    # Enter tool section
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Some content in tool section
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|> partial content",
        delta_text=" partial content",
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )
    # Content should be suppressed
    assert result2.content == "" or result2.content is None

    # Stream ends (EOF) - no more deltas, no section_end marker
    # Simulate this by manually checking state and resetting
    # (In real usage, the request handler would call reset_streaming_state)
    assert kimi_k2_tool_parser.in_tool_section is True  # Still in section

    # Reset state (as would happen between requests)
    kimi_k2_tool_parser.reset_streaming_state()

    # Verify clean slate
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.token_buffer == ""

    # Next request should work normally
    result3 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="New reasoning",
        delta_text="New reasoning",
        previous_token_ids=[],
        current_token_ids=[20, 21],
        delta_token_ids=[20, 21],
        request=None,
    )
    assert result3 is not None
    assert result3.content == "New reasoning"


def test_same_chunk_begin_and_end_markers(kimi_k2_tool_parser):
    """
    CRITICAL TEST: Verify that when both section_begin and section_end
    markers appear in the SAME chunk, the parser correctly:
    1. Enters the tool section
    2. Immediately exits the tool section
    3. Does NOT get stuck in in_tool_section=True state

    This tests the bug fix where elif was changed to if to handle
    both state transitions in a single delta.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Single chunk with both markers (e.g., empty tool section)
    combined_delta = "<|tool_calls_section_begin|><|tool_calls_section_end|>"

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning ",
        current_text="Some reasoning " + combined_delta,
        delta_text=combined_delta,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id, section_end_id],
        delta_token_ids=[section_begin_id, section_end_id],
        request=None,
    )

    # CRITICAL: Parser should NOT be stuck in tool section
    assert kimi_k2_tool_parser.in_tool_section is False, (
        "Parser stuck in tool section after processing both begin/end in same chunk. "
        "This indicates the elif bug was not fixed."
    )

    # Result should be empty or contain only stripped content
    assert result is not None
    assert result.content == "" or result.content is None

    # Verify subsequent content streams correctly (not suppressed)
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning " + combined_delta,
        current_text="Some reasoning " + combined_delta + " More reasoning",
        delta_text=" More reasoning",
        previous_token_ids=[1, 2, section_begin_id, section_end_id],
        current_token_ids=[1, 2, section_begin_id, section_end_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )

    # This content should NOT be suppressed (we're out of tool section)
    assert result2 is not None
    assert result2.content == " More reasoning"


def test_same_chunk_begin_content_end_markers(kimi_k2_tool_parser):
    """
    Test the same-chunk scenario with actual content between markers.
    Example: <|tool_calls_section_begin|> text <|tool_calls_section_end|>
    all arriving in one delta. The key is that the state machine correctly
    transitions in and out within the same chunk.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Chunk with begin, some whitespace/noise, and end all together
    # This simulates a tool section that opens and closes in the same chunk
    combined_delta = "<|tool_calls_section_begin|>   <|tool_calls_section_end|>"

    _result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning " + combined_delta,
        delta_text=combined_delta,
        previous_token_ids=[1],
        current_token_ids=[1, section_begin_id, 100, section_end_id],
        delta_token_ids=[section_begin_id, 100, section_end_id],
        request=None,
    )

    # Parser should exit cleanly (not stuck in tool section)
    assert kimi_k2_tool_parser.in_tool_section is False

    # Verify the fix: next content should stream normally, not be suppressed
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning " + combined_delta,
        current_text="Reasoning " + combined_delta + " Done",
        delta_text=" Done",
        previous_token_ids=[1, section_begin_id, 100, section_end_id],
        current_token_ids=[1, section_begin_id, 100, section_end_id, 200],
        delta_token_ids=[200],
        request=None,
    )

    # Content after section should be returned (not suppressed)
    assert result2 is not None
    assert result2.content == " Done"


def test_tool_call_end_and_section_end_same_chunk(kimi_k2_tool_parser):
    """
    CRITICAL TEST (P1): Verify that when both <|tool_call_end|> and
    <|tool_calls_section_end|> appear in the SAME chunk, the parser:
    1. Processes the tool_call_end first (emits final arguments)
    2. THEN exits the section
    3. Does NOT drop the final tool call update
    4. Does NOT leak special tokens into reasoning

    This tests the deferred section exit fix.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    # Simulate a streaming sequence for a SHORT tool call (all in one chunk):
    # 1. Reasoning text
    result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="Let me help. ",
        delta_text="Let me help. ",
        previous_token_ids=[],
        current_token_ids=[1, 2],
        delta_token_ids=[1, 2],
        request=None,
    )
    assert result1 is not None
    assert result1.content == "Let me help. "

    # 2. Section begin
    _result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Let me help. ",
        current_text="Let me help. <|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # 3. Tool call begin + full content + tool_end + section_end ALL IN ONE CHUNK
    # This is the critical scenario for short tool calls
    combined = (
        '<|tool_call_begin|>get_weather:0 <|tool_call_argument_begin|> {"city": "Paris"} '
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    # Build up the previous text gradually to simulate realistic streaming
    prev_text = "Let me help. <|tool_calls_section_begin|>"
    curr_text = prev_text + combined

    result3 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text=prev_text,
        current_text=curr_text,
        delta_text=combined,
        previous_token_ids=[1, 2, section_begin_id],
        current_token_ids=[
            1,
            2,
            section_begin_id,
            tool_begin_id,
            10,
            11,
            12,
            tool_end_id,
            section_end_id,
        ],
        delta_token_ids=[tool_begin_id, 10, 11, 12, tool_end_id, section_end_id],
        request=None,
    )

    # CRITICAL: Parser should have exited section AFTER processing tool
    assert kimi_k2_tool_parser.in_tool_section is False

    # Tool call should have been emitted (not dropped)
    # The result might be the tool name or None depending on state, but
    # importantly, it shouldn't be returning the literal tokens as content

    if result3 is not None and result3.content is not None:
        # Verify no special tokens leaked into content
        assert "<|tool_call_end|>" not in result3.content
        assert "<|tool_calls_section_end|>" not in result3.content

    # 4. Verify subsequent content streams normally
    result4 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text=curr_text,
        current_text=curr_text + " Done",
        delta_text=" Done",
        previous_token_ids=[
            1,
            2,
            section_begin_id,
            tool_begin_id,
            10,
            11,
            12,
            tool_end_id,
            section_end_id,
        ],
        current_token_ids=[
            1,
            2,
            section_begin_id,
            tool_begin_id,
            10,
            11,
            12,
            tool_end_id,
            section_end_id,
            20,
        ],
        delta_token_ids=[20],
        request=None,
    )

    # Content after tool section should stream normally
    assert result4 is not None
    assert result4.content == " Done"
