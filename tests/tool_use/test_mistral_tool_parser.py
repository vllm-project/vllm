# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the token-based Mistral tool parser (v11+ models only).

Tests cover:
1. Non-streaming extraction for v11+ tokenizers
2. Streaming extraction with proper token-based parsing
3. Edge cases like content before tool calls, multiple tools, etc.

Note: Pre-v11 models (Mistral-7B-Instruct-v0.1/v0.2/v0.3) are not supported.
"""

import json
from collections.abc import Generator

import partial_json_parser
import pytest
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import (
    FunctionCall as MistralFunctionCall,
)
from mistral_common.protocol.instruct.tool_calls import ToolCall
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import DeltaMessage, DeltaToolCall
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolParser,
)
from vllm.tokenizers import MistralTokenizer, TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally


@pytest.fixture(scope="module")
def mistral_tokenizer():
    """V11+ tokenizer using mistral-common format."""
    MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    return get_tokenizer(tokenizer_name=MODEL, tokenizer_mode="mistral")


@pytest.fixture
def mistral_tool_parser(mistral_tokenizer):
    return MistralToolParser(mistral_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall] | list[DeltaToolCall],
    expected_tool_calls: list[ToolCall],
):
    """Assert that actual tool calls match expected ones."""
    assert len(actual_tool_calls) == len(expected_tool_calls), (
        f"Expected {len(expected_tool_calls)} tool calls, got {len(actual_tool_calls)}"
    )

    for actual, expected in zip(actual_tool_calls, expected_tool_calls):
        # Check ID format
        assert isinstance(actual.id, str), (
            f"Tool call ID should be string, got {type(actual.id)}"
        )
        assert len(actual.id) == 9, (
            f"Tool call ID should be 9 chars, got {len(actual.id)}"
        )
        assert actual.id.isalnum(), (
            f"Tool call ID should be alphanumeric, got {actual.id}"
        )

        # Check function
        assert actual.function is not None
        # Handle both Pydantic model and dict-like access
        func = actual.function
        actual_name = getattr(func, "name", None) or (
            func.get("name") if isinstance(func, dict) else None
        )
        actual_args = getattr(func, "arguments", None) or (
            func.get("arguments") if isinstance(func, dict) else None
        )

        assert actual_name == expected.function.name, (
            f"Expected function name '{expected.function.name}', got '{actual_name}'"
        )
        assert actual_args == expected.function.arguments, (
            f"Expected arguments '{expected.function.arguments}', got '{actual_args}'"
        )


def fix_tool_call_tokenization(
    tokens: list[int],
    mistral_tool_parser: MistralToolParser,
    mistral_tokenizer: TokenizerLike,
) -> list[int]:
    """
    Replace textual token sequences for special tokens with their IDs.

    This is needed because encoding free text may produce the textual tokens
    for "[TOOL_CALLS]", "[ARGS]", etc. rather than the single special token.
    """
    # Build mapping of textual sequences to special token IDs
    replacements: list[tuple[list[int], int]] = []

    # [TOOL_CALLS] token
    textual_tool_call_ids = mistral_tokenizer.encode(
        text=mistral_tool_parser.bot_token,
        add_special_tokens=False,
    )
    if mistral_tool_parser.bot_token_id is not None:
        replacements.append((textual_tool_call_ids, mistral_tool_parser.bot_token_id))

    # [ARGS] token
    if mistral_tool_parser._args_token_id is not None:
        textual_args_ids = mistral_tokenizer.encode(
            text="[ARGS]",
            add_special_tokens=False,
        )
        replacements.append((textual_args_ids, mistral_tool_parser._args_token_id))

    if not tokens or not replacements:
        return tokens

    result_tokens = list(tokens)

    # Apply each replacement (longest first to avoid partial matches)
    replacements.sort(key=lambda x: -len(x[0]))

    for textual_ids, special_id in replacements:
        target_len = len(textual_ids)
        new_result = []
        i = 0
        while i < len(result_tokens):
            if result_tokens[i : i + target_len] == textual_ids:
                new_result.append(special_id)
                i += target_len
            else:
                new_result.append(result_tokens[i])
                i += 1
        result_tokens = new_result

    return result_tokens


def stream_delta_message_generator(
    mistral_tool_parser: MistralToolParser,
    mistral_tokenizer: TokenizerLike,
    tools: list[tuple[str, str]],
) -> Generator[DeltaMessage, None, None]:
    """
    Generate streaming delta messages by tokenizing and processing one token at a time.

    Uses encode_instruct to get proper tokenization with special tokens.
    """
    assert isinstance(mistral_tokenizer, MistralTokenizer)

    # Use encode_instruct to get proper special tokens
    assistant_msg = AssistantMessage(
        tool_calls=[
            ToolCall(
                function=MistralFunctionCall(
                    name=name,
                    arguments=arg,
                )
            )
            for (name, arg) in tools
        ],
    )
    request = InstructRequest(messages=[assistant_msg])
    all_token_ids = mistral_tokenizer.instruct.encode_instruct(request).tokens

    # Stream tokens one at a time
    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0

    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[: i + 1]

        # Detokenize incrementally
        (new_tokens, delta_text, new_prefix_offset, new_read_offset) = (
            detokenize_incrementally(
                tokenizer=mistral_tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = mistral_tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request=None,  # type: ignore[arg-type]
        )

        if delta_message:
            yield delta_message

        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


# =============================================================================
# Non-streaming extraction tests
# =============================================================================


class TestExtractToolCallsNoTools:
    """Test extraction when no tools are called."""

    def test_no_tool_call_token(self, mistral_tool_parser):
        model_output = "This is a test response without any tool calls."
        result = mistral_tool_parser.extract_tool_calls(model_output, request=None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output


class TestExtractToolCallsV11Plus:
    """Test non-streaming extraction for v11+ tokenizers."""

    @pytest.mark.parametrize(
        "model_output,expected_tool_calls,expected_content",
        [
            # Single tool (v11+ format: name{args})
            (
                '[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add_this_and_that",
                            arguments=json.dumps({"a": 3.5, "b": 4}),
                        )
                    )
                ],
                None,
            ),
            # Weather tool
            (
                "[TOOL_CALLS]get_current_weather"
                '{"city": "San Francisco", "state": "CA", "unit": "celsius"}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "San Francisco",
                                    "state": "CA",
                                    "unit": "celsius",
                                }
                            ),
                        )
                    )
                ],
                None,
            ),
            # Multiple tool calls
            (
                '[TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add",
                            arguments=json.dumps({"a": 3.5, "b": 4}),
                        )
                    ),
                    ToolCall(
                        function=MistralFunctionCall(
                            name="multiply",
                            arguments=json.dumps({"a": 3, "b": 6}),
                        )
                    ),
                ],
                None,
            ),
        ],
        ids=["single_tool_add", "single_tool_weather", "multiple_tool_calls"],
    )
    def test_extract_tool_calls(
        self,
        mistral_tool_parser,
        model_output,
        expected_tool_calls,
        expected_content,
    ):
        result = mistral_tool_parser.extract_tool_calls(model_output, request=None)
        assert result.tools_called
        assert_tool_calls(result.tool_calls, expected_tool_calls)
        assert result.content == expected_content


# =============================================================================
# Streaming extraction tests
# =============================================================================


def _test_extract_tool_calls_streaming(
    tool_parser,
    tokenizer,
    tools,
    expected_tool_calls,
    expected_content,
):
    """
    Helper function to test streaming extraction.

    Collects all streamed deltas and verifies the final result matches expected.
    """
    other_content: str = ""
    function_names: list[str] = []
    function_args_strs: list[str] = []
    tool_call_idx: int = -1
    tool_call_ids: list[str | None] = []

    for delta_message in stream_delta_message_generator(tool_parser, tokenizer, tools):
        # Role should never be streamed from tool parser
        assert not delta_message.role

        if delta_message.content:
            other_content += delta_message.content

        streamed_tool_calls = delta_message.tool_calls

        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            # Only one tool call delta per message
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            # Verify prev_tool_call_arr is set (for finish_reason detection)
            assert len(tool_parser.prev_tool_call_arr) > 0

            # If new tool, set up tracking
            if tool_call.index != tool_call_idx:
                tool_call_idx = tool_call.index
                function_args_strs.append("")
                tool_call_ids.append(None)

            # Track tool call ID (should be set once per tool)
            if tool_call.id and not tool_call_ids[tool_call.index]:
                tool_call_ids[tool_call.index] = tool_call.id

            # Track function parts
            if tool_call.function:
                # DeltaFunctionCall may be a Pydantic model or dict-like
                func = tool_call.function
                func_name = getattr(func, "name", None) or (
                    func.get("name") if isinstance(func, dict) else None
                )
                func_args = getattr(func, "arguments", None) or (
                    func.get("arguments") if isinstance(func, dict) else None
                )

                if func_name:
                    function_names.append(func_name)

                if func_args:
                    function_args_strs[tool_call.index] += func_args

    # Verify content
    assert other_content == expected_content

    # Build actual tool calls from collected data
    actual_tool_calls = [
        ToolCall(
            id=tool_call_id,
            function=MistralFunctionCall(
                name=function_name,
                arguments=partial_json_parser.ensure_json(
                    function_args_str, Allow.OBJ | Allow.STR
                ),
            ),
        )
        for tool_call_id, function_name, function_args_str in zip(
            tool_call_ids, function_names, function_args_strs
        )
    ]
    assert_tool_calls(actual_tool_calls, expected_tool_calls)


class TestStreamingExtractionV11Plus:
    """Test streaming extraction for v11+ tokenizers."""

    @pytest.mark.parametrize(
        "tools,expected_tool_calls,expected_content",
        [
            # Single tool
            (
                [("add", '{"a": 3, "b": 4}')],
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add",
                            arguments=json.dumps({"a": 3, "b": 4}),
                        )
                    )
                ],
                "",
            ),
            # String arguments
            (
                [("add_two_strings", '{"a": "3", "b": "4"}')],
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add_two_strings",
                            arguments=json.dumps({"a": "3", "b": "4"}),
                        )
                    )
                ],
                "",
            ),
            # Multiple tools
            (
                [
                    ("add", '{"a": 3.5, "b": 4}'),
                    (
                        "get_current_weather",
                        '{"city": "San Francisco", "state": "CA", "unit": "celsius"}',
                    ),
                ],
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add",
                            arguments=json.dumps({"a": 3.5, "b": 4}),
                        )
                    ),
                    ToolCall(
                        function=MistralFunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "San Francisco",
                                    "state": "CA",
                                    "unit": "celsius",
                                }
                            ),
                        )
                    ),
                ],
                "",
            ),
        ],
        ids=["single_tool_add", "single_tool_add_strings", "multiple_tools"],
    )
    def test_streaming_extraction(
        self,
        mistral_tool_parser,
        mistral_tokenizer,
        tools,
        expected_tool_calls,
        expected_content,
    ):
        _test_extract_tool_calls_streaming(
            mistral_tool_parser,
            mistral_tokenizer,
            tools,
            expected_tool_calls,
            expected_content,
        )


class TestStreamingOneChunk:
    """Test streaming when all tokens arrive in a single chunk."""

    @pytest.mark.parametrize(
        "model_output,expected_tool_calls,expected_content",
        [
            # Single tool - v11 format includes [ARGS] between name and JSON
            (
                '[TOOL_CALLS]add_this_and_that[ARGS]{"a": 3.5, "b": 4}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add_this_and_that",
                            arguments=json.dumps({"a": 3.5, "b": 4}),
                        )
                    )
                ],
                "",
            ),
            # Weather tool
            (
                "[TOOL_CALLS]get_current_weather[ARGS]"
                '{"city": "San Francisco", "state": "CA", "unit": "celsius"}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "San Francisco",
                                    "state": "CA",
                                    "unit": "celsius",
                                }
                            ),
                        )
                    )
                ],
                "",
            ),
            # Multiple tools - NOTE: This case is tricky because BPE tokenization
            # may merge the closing } of the first tool with the [ of the next
            # [TOOL_CALLS], making it hard to detect the second tool call when
            # encoding from free text. In real inference, the model generates
            # special tokens directly, avoiding this issue.
            pytest.param(
                '[TOOL_CALLS]add[ARGS]{"a": 3.5, "b": 4}'
                '[TOOL_CALLS]multiply[ARGS]{"a": 3, "b": 6}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add",
                            arguments=json.dumps({"a": 3.5, "b": 4}),
                        )
                    ),
                    ToolCall(
                        function=MistralFunctionCall(
                            name="multiply",
                            arguments=json.dumps({"a": 3, "b": 6}),
                        )
                    ),
                ],
                "",
                marks=pytest.mark.xfail(reason="BPE tokenization merges }[ tokens"),
            ),
            # Content before tool
            (
                'bla[TOOL_CALLS]add_this_and_that[ARGS]{"a": 3.5, "b": 4}',
                [
                    ToolCall(
                        function=MistralFunctionCall(
                            name="add_this_and_that",
                            arguments=json.dumps({"a": 3.5, "b": 4}),
                        )
                    )
                ],
                "bla",
            ),
        ],
        ids=[
            "single_tool_add",
            "single_tool_weather",
            "multiple_tool_calls",
            "content_before_tool",
        ],
    )
    def test_streaming_one_chunk_v11(
        self,
        mistral_tool_parser,
        mistral_tokenizer,
        model_output,
        expected_tool_calls,
        expected_content,
    ):
        """Test v11+ streaming with all tokens in one chunk.

        When all tokens arrive at once, we still produce streaming-style
        output with multiple DeltaToolCall objects. We need to aggregate
        these to verify the final result.
        """
        if isinstance(mistral_tokenizer, MistralTokenizer):
            all_token_ids = mistral_tokenizer.encode(model_output)
        else:
            all_token_ids = mistral_tokenizer.encode(
                model_output, add_special_tokens=False
            )

        all_token_ids = fix_tool_call_tokenization(
            all_token_ids, mistral_tool_parser, mistral_tokenizer
        )

        delta_message = mistral_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=model_output,
            delta_text=model_output,
            previous_token_ids=[],
            current_token_ids=all_token_ids,
            delta_token_ids=all_token_ids,
            request=None,
        )

        assert isinstance(delta_message, DeltaMessage)

        # Aggregate streaming deltas into final tool calls
        # Each tool call starts with a name delta, followed by argument deltas
        tool_call_data: dict[int, dict] = {}  # index -> {id, name, arguments}

        for tc in delta_message.tool_calls or []:
            idx = tc.index
            if idx not in tool_call_data:
                tool_call_data[idx] = {"id": None, "name": "", "arguments": ""}

            if tc.id:
                tool_call_data[idx]["id"] = tc.id

            func = tc.function
            func_name = getattr(func, "name", None) or (
                func.get("name") if isinstance(func, dict) else None
            )
            func_args = getattr(func, "arguments", None) or (
                func.get("arguments") if isinstance(func, dict) else None
            )

            if func_name:
                tool_call_data[idx]["name"] = func_name
            if func_args:
                tool_call_data[idx]["arguments"] += func_args

        # Verify we got the expected number of tool calls
        assert len(tool_call_data) == len(expected_tool_calls), (
            f"Expected {len(expected_tool_calls)} tool calls, got {len(tool_call_data)}"
        )

        # Verify each tool call
        for i, expected in enumerate(expected_tool_calls):
            actual = tool_call_data[i]
            assert actual["name"] == expected.function.name, (
                f"Expected name '{expected.function.name}', got '{actual['name']}'"
            )
            assert actual["arguments"] == expected.function.arguments, (
                f"Expected args '{expected.function.arguments}', "
                f"got '{actual['arguments']}'"
            )
            assert actual["id"] is not None and len(actual["id"]) == 9

        # Verify content before tool calls is preserved
        actual_content = delta_message.content or ""
        assert actual_content == expected_content, (
            f"Expected content '{expected_content}', got '{actual_content}'"
        )


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tool_call_id_format(self, mistral_tool_parser):
        """Verify generated tool call IDs are valid."""
        from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
            MistralToolCall,
        )

        for _ in range(100):
            tool_id = MistralToolCall.generate_random_id()
            assert len(tool_id) == 9
            assert tool_id.isalnum()
            assert MistralToolCall.is_valid_id(tool_id)

    def test_invalid_tool_id_validation(self):
        """Test tool ID validation."""
        from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
            MistralToolCall,
        )

        assert not MistralToolCall.is_valid_id("")
        assert not MistralToolCall.is_valid_id("12345678")  # Too short
        assert not MistralToolCall.is_valid_id("1234567890")  # Too long
        assert not MistralToolCall.is_valid_id("abc-def-g")  # Contains hyphen

    def test_empty_model_output(self, mistral_tool_parser):
        """Test handling of empty output."""
        result = mistral_tool_parser.extract_tool_calls("", request=None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == ""


class TestTokenBasedDetection:
    """Test that token-based detection works correctly."""

    def test_bot_token_id_exists(self, mistral_tool_parser):
        """Verify bot token ID is properly set."""
        assert mistral_tool_parser.bot_token_id is not None
        assert isinstance(mistral_tool_parser.bot_token_id, int)

    def test_args_token_id_exists(self, mistral_tool_parser):
        """Verify [ARGS] token ID is properly set for v11+."""
        assert mistral_tool_parser._args_token_id is not None
        assert isinstance(mistral_tool_parser._args_token_id, int)

    def test_streaming_uses_token_ids(self, mistral_tool_parser, mistral_tokenizer):
        """Test that streaming correctly uses token IDs for detection."""
        # Content without tool call
        content_text = "Hello, how can I help you?"
        content_tokens = mistral_tokenizer.encode(
            content_text, add_special_tokens=False
        )

        delta_message = mistral_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=content_text,
            delta_text=content_text,
            previous_token_ids=[],
            current_token_ids=content_tokens,
            delta_token_ids=content_tokens,
            request=None,
        )

        assert delta_message is not None
        assert delta_message.content == content_text
        assert not delta_message.tool_calls


class TestParserInitialization:
    """Test parser initialization and validation."""

    def test_rejects_pre_v11_tokenizer(self):
        """Test that parser rejects pre-v11 MistralTokenizer."""
        # Get a pre-v11 MistralTokenizer (Mistral-7B-v0.3 uses version 3)
        pre_v11_tokenizer = get_tokenizer(
            tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
            tokenizer_mode="mistral",
        )

        assert isinstance(pre_v11_tokenizer, MistralTokenizer)
        assert pre_v11_tokenizer.version < 11

        with pytest.raises(
            RuntimeError, match="requires tokenizer version 11 or higher"
        ):
            MistralToolParser(pre_v11_tokenizer)
