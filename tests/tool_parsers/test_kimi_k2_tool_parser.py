# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser

MODEL = "moonshotai/Kimi-K2-Instruct"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def parser(kimi_k2_tokenizer):
    return KimiK2ToolParser(kimi_k2_tokenizer)


SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
TOOL_BEGIN = "<|tool_call_begin|>"
TOOL_END = "<|tool_call_end|>"
ARG_BEGIN = "<|tool_call_argument_begin|>"


def _tool(tool_id: str, args: str) -> str:
    return f"{TOOL_BEGIN}{tool_id} {ARG_BEGIN}{args}{TOOL_END}"


def _wrap(*tool_strs: str) -> str:
    return SECTION_BEGIN + "".join(tool_strs) + SECTION_END


class TestExtractToolCalls:
    def test_no_tools(self, parser):
        content, tool_calls = run_tool_extraction(
            parser, "This is a test", streaming=False
        )
        assert content == "This is a test"
        assert tool_calls == []

    @pytest.mark.parametrize(
        "model_output, expected_names, expected_args_list, expected_content",
        [
            pytest.param(
                "I'll check. "
                + _wrap(_tool("functions.get_weather:0", '{"city": "Beijing"}')),
                ["get_weather"],
                [{"city": "Beijing"}],
                "I'll check. ",
                id="single_tool_call",
            ),
            pytest.param(
                "Compare weather. "
                + _wrap(
                    _tool("functions.get_weather:0", '{"city": "Beijing"}'),
                    _tool("functions.get_weather:1", '{"city": "Shanghai"}'),
                ),
                ["get_weather", "get_weather"],
                [{"city": "Beijing"}, {"city": "Shanghai"}],
                "Compare weather. ",
                id="parallel_tool_calls",
            ),
            pytest.param(
                "Multiple tasks. "
                + _wrap(
                    _tool("functions.get_weather:0", '{"city": "New York"}'),
                    _tool("functions.get_news:1", '{"topic": "technology"}'),
                    _tool(
                        "functions.send_email:2",
                        '{"to": "user@example.com", "subject": "Daily Update"}',
                    ),
                ),
                ["get_weather", "get_news", "send_email"],
                [
                    {"city": "New York"},
                    {"topic": "technology"},
                    {"to": "user@example.com", "subject": "Daily Update"},
                ],
                "Multiple tasks. ",
                id="three_tool_calls",
            ),
            pytest.param(
                "Process HTML. "
                + _wrap(
                    _tool("functions.process_html:0", '{"html": "<div>content</div>"}')
                ),
                ["process_html"],
                [{"html": "<div>content</div>"}],
                "Process HTML. ",
                id="angle_brackets_in_json",
            ),
            pytest.param(
                "Formatted. "
                + _wrap(
                    _tool(
                        "functions.process_data:0",
                        '{\n  "name": "test",\n  "value": 123\n}',
                    )
                ),
                ["process_data"],
                [{"name": "test", "value": 123}],
                "Formatted. ",
                id="multiline_json",
            ),
            pytest.param(
                "No prefix. " + _wrap(_tool("get_weather:0", '{"city": "Tokyo"}')),
                ["get_weather"],
                [{"city": "Tokyo"}],
                "No prefix. ",
                id="no_functions_prefix",
            ),
            pytest.param(
                "Empty args. " + _wrap(_tool("functions.test:0", "{}")),
                ["test"],
                [{}],
                "Empty args. ",
                id="empty_arguments",
            ),
        ],
    )
    def test_extract_tool_calls(
        self, parser, model_output, expected_names, expected_args_list, expected_content
    ):
        content, tool_calls = run_tool_extraction(parser, model_output, streaming=False)
        assert content == expected_content
        assert len(tool_calls) == len(expected_names)
        for tc, name, expected_args in zip(
            tool_calls, expected_names, expected_args_list
        ):
            assert tc.type == "function"
            assert tc.function.name == name
            assert json.loads(tc.function.arguments) == expected_args
            # id format: "something:digit"
            assert tc.id.split(":")[-1].isdigit()

    def test_invalid_json_still_extracted(self, parser):
        """Tool calls with invalid JSON are still returned (arguments as-is)."""
        model_output = (
            "Help. "
            + SECTION_BEGIN
            + _tool("functions.bad:0", '{"city": "Beijing"')
            + _tool("functions.good:1", '{"city": "Shanghai"}')
            + SECTION_END
        )
        content, tool_calls = run_tool_extraction(parser, model_output, streaming=False)
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "bad"
        assert tool_calls[1].function.name == "good"

    def test_invalid_funcall_id_skipped(self, parser):
        """Tool calls with malformed id (no colon+digit) are skipped."""
        model_output = (
            "Help. "
            + SECTION_BEGIN
            + _tool("functions.invalid.0", '{"city": "Beijing"}')
            + _tool("functions.valid:1", '{"city": "Shanghai"}')
            + SECTION_END
        )
        content, tool_calls = run_tool_extraction(parser, model_output, streaming=False)
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "valid"

    def test_native_id_extracted(self, parser):
        """Regression: parser extracts native ID onto ToolCall (PR #32768)."""
        model_output = "Checking weather. " + _wrap(
            _tool("functions.get_weather:0", '{"city": "Tokyo"}')
        )
        content, tool_calls = run_tool_extraction(parser, model_output, streaming=False)
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "functions.get_weather:0"
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "Tokyo"}

    def test_multi_turn_native_id_continuity(self, kimi_k2_tokenizer):
        """Regression: native IDs from turn 1 preserved across turns (PR #32768)."""
        turn1_parser = KimiK2ToolParser(kimi_k2_tokenizer)
        turn1_output = "Let me check. " + _wrap(
            _tool("functions.get_weather:0", '{"city": "Beijing"}')
        )
        _, turn1_tools = run_tool_extraction(
            turn1_parser, turn1_output, streaming=False
        )
        assert len(turn1_tools) == 1
        assert turn1_tools[0].id == "functions.get_weather:0"

        # Fresh parser for turn 2
        turn2_parser = KimiK2ToolParser(kimi_k2_tokenizer)
        turn2_output = "Now let me get news. " + _wrap(
            _tool("functions.get_news:0", '{"topic": "weather in Beijing"}')
        )
        _, turn2_tools = run_tool_extraction(
            turn2_parser, turn2_output, streaming=False
        )
        assert len(turn2_tools) == 1
        assert turn2_tools[0].id == "functions.get_news:0"


def _split_tool_output_to_deltas(
    content: str, tool_strs: list[tuple[str, str]]
) -> list[str]:
    """Build a list of string deltas with special tokens as separate chunks.

    Args:
        content: text before tool section
        tool_strs: list of (tool_id, args_json)
    """
    deltas = [content, SECTION_BEGIN]
    for tool_id, args_json in tool_strs:
        deltas.extend(
            [
                TOOL_BEGIN,
                f"{tool_id} ",
                ARG_BEGIN,
                f"{args_json} ",
                TOOL_END,
            ]
        )
    deltas.append(SECTION_END)
    return deltas


class TestStreamingHappyPath:
    def test_single_tool_call(self, parser):
        """Verify DeltaToolCall output: name, id, arguments for one tool."""
        deltas = _split_tool_output_to_deltas(
            "I'll help. ",
            [("functions.get_weather:0", '{"city": "Beijing"}')],
        )
        rec = run_tool_extraction_streaming(parser, deltas)

        assert len(rec.tool_calls) == 1
        tc = rec.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.id == "functions.get_weather:0"
        assert json.loads(tc.function.arguments) == {"city": "Beijing"}

    def test_multiple_tool_calls(self, parser):
        """Two tool calls emitted with correct indices, names, arguments."""
        deltas = _split_tool_output_to_deltas(
            "Compare weather. ",
            [
                ("functions.get_weather:0", '{"city": "Tokyo"}'),
                ("functions.get_weather:1", '{"city": "NYC"}'),
            ],
        )
        rec = run_tool_extraction_streaming(parser, deltas)

        assert len(rec.tool_calls) == 2
        assert rec.tool_calls[0].function.name == "get_weather"
        assert rec.tool_calls[0].id == "functions.get_weather:0"
        assert json.loads(rec.tool_calls[0].function.arguments) == {"city": "Tokyo"}

        assert rec.tool_calls[1].function.name == "get_weather"
        assert rec.tool_calls[1].id == "functions.get_weather:1"
        assert json.loads(rec.tool_calls[1].function.arguments) == {"city": "NYC"}

    def test_content_before_tools(self, parser):
        """Content before section is streamed; markers/args don't leak."""
        deltas = _split_tool_output_to_deltas(
            "I'll check the weather. ",
            [("functions.get_weather:0", '{"city": "Tokyo"}')],
        )
        rec = run_tool_extraction_streaming(parser, deltas)

        assert "check the weather" in rec.other_content
        # No markers or tool content leaked
        for marker in [SECTION_BEGIN, SECTION_END, TOOL_BEGIN, TOOL_END, ARG_BEGIN]:
            assert marker not in rec.other_content
        assert "get_weather" not in rec.other_content
        assert "Tokyo" not in rec.other_content

    def test_no_tool_calls(self, parser):
        """Plain text streaming returns content only."""
        deltas = ["This is just ", "regular text ", "without tools."]
        rec = run_tool_extraction_streaming(parser, deltas)

        assert rec.other_content == "This is just regular text without tools."
        assert rec.tool_calls == []

    def test_incremental_arguments(self, parser):
        """Arguments split across small chunks accumulate correctly."""
        deltas = [
            "Help. ",
            SECTION_BEGIN,
            TOOL_BEGIN,
            "functions.get_weather:0 ",
            ARG_BEGIN,
            '{"ci',
            'ty": "Be',
            'ijing"}',
            " ",
            TOOL_END,
            SECTION_END,
        ]
        rec = run_tool_extraction_streaming(parser, deltas)

        assert len(rec.tool_calls) == 1
        assert rec.tool_calls[0].function.name == "get_weather"
        assert json.loads(rec.tool_calls[0].function.arguments) == {"city": "Beijing"}

    @pytest.mark.parametrize(
        "model_output",
        [
            pytest.param(
                "Single. "
                + _wrap(_tool("functions.get_weather:0", '{"city": "Beijing"}')),
                id="single_tool",
            ),
            pytest.param(
                "Multi. "
                + _wrap(
                    _tool("functions.get_weather:0", '{"city": "Tokyo"}'),
                    _tool("functions.get_news:1", '{"topic": "tech"}'),
                ),
                id="parallel_tools",
            ),
            pytest.param(
                "No prefix id. " + _wrap(_tool("get_weather:0", '{"city": "NYC"}')),
                id="no_functions_prefix",
            ),
        ],
    )
    def test_streaming_matches_nonstreaming(self, parser, model_output):
        """Streaming reconstruction matches non-streaming extraction."""
        content_non, tools_non = run_tool_extraction(
            parser, model_output, streaming=False
        )
        content_stream, tools_stream = run_tool_extraction(
            parser, model_output, streaming=True
        )

        assert len(tools_non) == len(tools_stream)
        for tc_non, tc_stream in zip(tools_non, tools_stream):
            assert tc_non.function.name == tc_stream.function.name
            assert json.loads(tc_non.function.arguments) == json.loads(
                tc_stream.function.arguments
            )


class TestStreamingEdgeCases:
    def test_marker_suppression(self, parser):
        """No special-token markers appear in reconstructed content."""
        deltas = _split_tool_output_to_deltas(
            "I'll check. ",
            [("functions.get_weather:0", '{"city": "Tokyo"}')],
        )
        rec = run_tool_extraction_streaming(parser, deltas)

        forbidden = [SECTION_BEGIN, SECTION_END, TOOL_BEGIN, TOOL_END, ARG_BEGIN]
        for marker in forbidden:
            assert marker not in rec.other_content, (
                f"Marker leaked: {marker!r} in {rec.other_content!r}"
            )

    def test_noise_between_markers_suppressed(self, parser):
        """Text between section_begin and tool_call_begin doesn't leak."""
        deltas = [
            "Reasoning. ",
            SECTION_BEGIN,
            " spurious noise ",
            TOOL_BEGIN,
            "functions.test:0 ",
            ARG_BEGIN,
            '{"k": "v"} ',
            TOOL_END,
            SECTION_END,
        ]
        rec = run_tool_extraction_streaming(parser, deltas)

        assert "spurious" not in rec.other_content
        assert "noise" not in rec.other_content

    def test_empty_tool_section(self, parser):
        """Empty section (begin immediately followed by end) doesn't crash."""
        deltas = ["Reasoning. ", SECTION_BEGIN, SECTION_END]
        rec = run_tool_extraction_streaming(parser, deltas)

        assert rec.tool_calls == []

    def test_three_different_tools(self, parser):
        """Three tool calls with different functions stream correctly."""
        deltas = _split_tool_output_to_deltas(
            "Multiple tasks. ",
            [
                ("functions.get_weather:0", '{"city": "NYC"}'),
                ("functions.get_news:1", '{"topic": "tech"}'),
                ("functions.send_email:2", '{"to": "a@b.com"}'),
            ],
        )
        rec = run_tool_extraction_streaming(parser, deltas)

        assert len(rec.tool_calls) == 3
        names = [tc.function.name for tc in rec.tool_calls]
        assert names == ["get_weather", "get_news", "send_email"]
        ids = [tc.id for tc in rec.tool_calls]
        assert len(set(ids)) == 3  # unique ids


class TestAdjustRequest:
    def test_sets_skip_special_tokens_false(self, parser):
        request = MagicMock(spec=ChatCompletionRequest)
        request.tools = [{"type": "function", "function": {"name": "test"}}]
        request.tool_choice = "auto"
        request.skip_special_tokens = True

        result = parser.adjust_request(request)
        assert result.skip_special_tokens is False

    def test_no_change_when_tool_choice_none(self, parser):
        request = MagicMock(spec=ChatCompletionRequest)
        request.tools = [{"type": "function", "function": {"name": "test"}}]
        request.tool_choice = "none"
        request.skip_special_tokens = True

        result = parser.adjust_request(request)
        assert result.skip_special_tokens is True

    def test_no_change_when_no_tools(self, parser):
        request = MagicMock(spec=ChatCompletionRequest)
        request.tools = None
        request.tool_choice = "auto"
        request.skip_special_tokens = True

        result = parser.adjust_request(request)
        assert result.skip_special_tokens is True


def _chunk_tokenized_deltas(tokenizer, text: str, stream_interval: int) -> list[str]:
    """Encode text, group tokens into chunks of stream_interval, decode each."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    deltas = []
    prev = ""
    for i in range(0, len(token_ids), stream_interval):
        decoded = tokenizer.decode(
            token_ids[: i + stream_interval], skip_special_tokens=False
        )
        deltas.append(decoded[len(prev) :])
        prev = decoded
    return deltas


class TestStreamingIntervals:
    """Test streaming at various token-chunk sizes to catch boundary bugs."""

    @pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
    def test_single_tool_call_at_interval(self, kimi_k2_tokenizer, stream_interval):
        text = "Help. " + _wrap(_tool("functions.get_weather:0", '{"city": "Beijing"}'))
        deltas = _chunk_tokenized_deltas(kimi_k2_tokenizer, text, stream_interval)
        parser = KimiK2ToolParser(kimi_k2_tokenizer)
        rec = run_tool_extraction_streaming(
            parser, deltas, assert_one_tool_per_delta=False
        )

        assert len(rec.tool_calls) == 1
        assert rec.tool_calls[0].function.name == "get_weather"
        assert json.loads(rec.tool_calls[0].function.arguments) == {"city": "Beijing"}

    @pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
    def test_content_then_tool_call_at_interval(
        self, kimi_k2_tokenizer, stream_interval
    ):
        text = "Sure, let me check. " + _wrap(
            _tool("functions.get_weather:0", '{"city": "Tokyo"}')
        )
        deltas = _chunk_tokenized_deltas(kimi_k2_tokenizer, text, stream_interval)
        parser = KimiK2ToolParser(kimi_k2_tokenizer)
        rec = run_tool_extraction_streaming(
            parser, deltas, assert_one_tool_per_delta=False
        )

        assert "let me check" in rec.other_content
        assert "get_weather" not in rec.other_content
        assert len(rec.tool_calls) == 1
        assert rec.tool_calls[0].function.name == "get_weather"
        assert json.loads(rec.tool_calls[0].function.arguments) == {"city": "Tokyo"}

    @pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
    def test_multiple_tool_calls_at_interval(self, kimi_k2_tokenizer, stream_interval):
        text = "Compare. " + _wrap(
            _tool("functions.search:0", '{"q": "cats"}'),
            _tool("functions.search:1", '{"q": "dogs"}'),
        )
        deltas = _chunk_tokenized_deltas(kimi_k2_tokenizer, text, stream_interval)
        parser = KimiK2ToolParser(kimi_k2_tokenizer)
        rec = run_tool_extraction_streaming(
            parser, deltas, assert_one_tool_per_delta=False
        )

        assert len(rec.tool_calls) == 2
        assert rec.tool_calls[0].function.name == "search"
        assert json.loads(rec.tool_calls[0].function.arguments) == {"q": "cats"}
        assert rec.tool_calls[1].function.name == "search"
        assert json.loads(rec.tool_calls[1].function.arguments) == {"q": "dogs"}

    @pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
    def test_plain_text_at_interval(self, kimi_k2_tokenizer, stream_interval):
        text = "This is plain text with no tool calling involved."
        deltas = _chunk_tokenized_deltas(kimi_k2_tokenizer, text, stream_interval)
        parser = KimiK2ToolParser(kimi_k2_tokenizer)
        rec = run_tool_extraction_streaming(
            parser, deltas, assert_one_tool_per_delta=False
        )

        assert rec.other_content == text
        assert rec.tool_calls == []

    def test_content_and_tool_call_in_single_chunk(self, kimi_k2_tokenizer):
        """Content + complete tool call in one chunk must both be emitted."""
        text = "Hi! " + _wrap(_tool("functions.get_weather:0", '{"city": "Beijing"}'))
        deltas = _chunk_tokenized_deltas(kimi_k2_tokenizer, text, stream_interval=9999)
        parser = KimiK2ToolParser(kimi_k2_tokenizer)
        rec = run_tool_extraction_streaming(
            parser, deltas, assert_one_tool_per_delta=False
        )

        assert "Hi!" in rec.other_content
        assert "get_weather" not in rec.other_content
        assert len(rec.tool_calls) == 1
        assert rec.tool_calls[0].function.name == "get_weather"
        assert json.loads(rec.tool_calls[0].function.arguments) == {"city": "Beijing"}
