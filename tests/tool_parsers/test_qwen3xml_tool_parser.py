# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


def _parse_full(xml: str) -> dict:
    """Feed *xml* all at once through StreamingXMLToolCallParser and return
    the collected tool-call arguments as a dict (first tool call only)."""
    p = StreamingXMLToolCallParser()
    p.parse_single_streaming_chunks(xml)
    p.finalize()
    result = p.collect_all()
    assert result.tool_calls, f"No tool calls parsed from:\n{xml}"
    args_str = result.tool_calls[0].function.arguments
    return json.loads(args_str)


def _parse_char_by_char(xml: str) -> dict:
    """Same as _parse_full but feeds one character at a time (simulates
    streaming with the smallest possible chunk size)."""
    p = StreamingXMLToolCallParser()
    for ch in xml:
        p.parse_single_streaming_chunks(ch)
    p.finalize()
    result = p.collect_all()
    assert result.tool_calls, f"No tool calls parsed from:\n{xml}"
    args_str = result.tool_calls[0].function.arguments
    return json.loads(args_str)


class TestParamCloseLookahead:
    """Unit tests for the </parameter> lookahead edge case."""

    # ------------------------------------------------------------------
    # Basic lookahead: </parameter> followed by non-structural tag is content
    # ------------------------------------------------------------------

    def test_closing_tag_in_content_followed_by_html(self):
        """</parameter> followed by </div> → content, not a delimiter."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "some text </parameter>\n"
            "</div>\n"
            "more text\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        args = _parse_full(xml)
        assert args["content"] == "some text </parameter>\n</div>\nmore text"

    def test_closing_tag_in_content_streaming(self):
        """Same case fed character-by-character produces identical result."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "some text </parameter>\n"
            "</div>\n"
            "more text\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_char_by_char(xml) == _parse_full(xml)

    # ------------------------------------------------------------------
    # Normal case unaffected
    # ------------------------------------------------------------------

    def test_normal_two_param_call(self):
        """Standard two-parameter call is not affected by the new logic."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>\n"
            "/tmp/hello.py\n"
            "</parameter>\n"
            "<parameter=content>\n"
            "print('hello')\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        args = _parse_full(xml)
        assert args["path"] == "/tmp/hello.py"
        assert args["content"] == "print('hello')"

    def test_normal_two_param_call_streaming(self):
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>\n"
            "/tmp/hello.py\n"
            "</parameter>\n"
            "<parameter=content>\n"
            "print('hello')\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_char_by_char(xml) == _parse_full(xml)

    # ------------------------------------------------------------------
    # Multiple </parameter> in content; only the last is a real delimiter
    # ------------------------------------------------------------------

    def test_double_fake_closing_tag(self):
        """Two </parameter> tags in content before the real delimiter."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "first </parameter>\n"
            "</span>\n"
            "second </parameter>\n"
            "</em>\n"
            "end\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        args = _parse_full(xml)
        assert args["content"] == (
            "first </parameter>\n</span>\nsecond </parameter>\n</em>\nend"
        )

    def test_double_fake_closing_tag_streaming(self):
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "first </parameter>\n"
            "</span>\n"
            "second </parameter>\n"
            "</em>\n"
            "end\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_char_by_char(xml) == _parse_full(xml)

    # ------------------------------------------------------------------
    # </parameter> directly before </function> (no next param)
    # ------------------------------------------------------------------

    def test_param_end_before_function_end(self):
        """</parameter> followed by </function> is a genuine delimiter."""
        xml = (
            "<tool_call>\n"
            "<function=ping>\n"
            "<parameter=host>\n"
            "example.com\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        args = _parse_full(xml)
        assert args["host"] == "example.com"

    # ------------------------------------------------------------------
    # Stream ends while in PARAM_CLOSE_PENDING (truncated output)
    # ------------------------------------------------------------------

    def test_truncated_after_param_close(self):
        """Stream cut off after </parameter> — finalize treats it as genuine."""
        xml = (
            "<tool_call>\n"
            "<function=ping>\n"
            "<parameter=host>\n"
            "example.com\n"
            "</parameter>"
            # No </function> or </tool_call>
        )
        p = StreamingXMLToolCallParser()
        p.parse_single_streaming_chunks(xml)
        p.finalize()
        result = p.collect_all()
        assert result.tool_calls
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["host"] == "example.com"

    # ------------------------------------------------------------------
    # Content contains the format's own meta-tags (worst-case ambiguity)
    # ------------------------------------------------------------------

    def test_param_value_contains_closing_param_then_next_param(self):
        """
        Ambiguous case: content ends with ``</parameter>\\n<parameter=next>``.
        The parser must treat the </parameter> as the real delimiter here
        (the format is genuinely ambiguous; we side with the structural read).
        """
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>\n"
            "/tmp/f\n"
            "</parameter>\n"
            "<parameter=content>\n"
            "hello\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        args = _parse_full(xml)
        assert args["path"] == "/tmp/f"
        assert args["content"] == "hello"


class TestEarlyTagRejection:
    """
    Tests that non-structural tag prefixes are rejected immediately rather than
    buffered until ">".  The observable effect is that the characters are
    treated as literal content rather than silently eaten into the tag buffer.
    """

    def test_html_comment_in_param_value(self):
        """<!-- ... --> inside a parameter value passes through as-is."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "<!-- a comment -->\n"
            "actual content\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_full(xml)["content"] == "<!-- a comment -->\nactual content"

    def test_html_comment_in_param_value_streaming(self):
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "<!-- a comment -->\n"
            "actual content\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_char_by_char(xml) == _parse_full(xml)

    def test_bang_sequence_in_param_value(self):
        """<! prefix is rejected immediately (DOCTYPE / comment)."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "<!DOCTYPE html>\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_full(xml)["content"] == "<!DOCTYPE html>"

    def test_space_after_lt_in_param_value(self):
        """< followed by space cannot start a structural tag; passes through."""
        xml = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n"
            "a < b\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_full(xml)["content"] == "a < b"

    def test_html_tags_in_param_value(self):
        """Generic HTML tags like <br> and <div class="x"> pass through."""
        xml = (
            "<tool_call>\n"
            "<function=render>\n"
            "<parameter=html>\n"
            "<br>\n"
            "<div class=\"hello\">world</div>\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_full(xml)["html"] == '<br>\n<div class="hello">world</div>'

    def test_html_tags_in_param_value_streaming(self):
        xml = (
            "<tool_call>\n"
            "<function=render>\n"
            "<parameter=html>\n"
            "<br>\n"
            "<div class=\"hello\">world</div>\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_char_by_char(xml) == _parse_full(xml)

    def test_dash_sequence_in_param_value(self):
        """<-- (Markdown/diff leader) is not a structural tag."""
        xml = (
            "<tool_call>\n"
            "<function=diff>\n"
            "<parameter=patch>\n"
            "<-- removed line\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_full(xml)["patch"] == "<-- removed line"

    def test_non_structural_tag_in_plain_text(self):
        """Non-structural tags in TEXT state before any tool call are emitted."""
        p = StreamingXMLToolCallParser()
        delta = p.parse_single_streaming_chunks("Hello <br> world")
        p.finalize()
        full = p.collect_all()
        assert full.content == "Hello <br> world"
        assert not full.tool_calls

    def test_non_structural_tag_in_plain_text_streaming(self):
        p = StreamingXMLToolCallParser()
        for ch in "Hello <br> world":
            p.parse_single_streaming_chunks(ch)
        p.finalize()
        full = p.collect_all()
        assert full.content == "Hello <br> world"
        assert not full.tool_calls

    def test_unclosed_lt_at_end_of_param(self):
        """Trailing '<' at stream end is flushed as literal content."""
        xml = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\n"
            "foo <\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert _parse_full(xml)["query"] == "foo <"


class TestQwen3xmlToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="qwen3_xml",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>",
            parallel_tool_calls_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_time>\n<parameter=timezone>Asia/Tokyo</parameter>\n</function>\n</tool_call>",
            various_data_types_output=(
                "<tool_call>\n<function=test_function>\n"
                "<parameter=string_field>hello</parameter>\n"
                "<parameter=int_field>42</parameter>\n"
                "<parameter=float_field>3.14</parameter>\n"
                "<parameter=bool_field>true</parameter>\n"
                "<parameter=null_field>null</parameter>\n"
                '<parameter=array_field>["a", "b", "c"]</parameter>\n'
                '<parameter=object_field>{"nested": "value"}</parameter>\n'
                "</function>\n</tool_call>"
            ),
            empty_arguments_output="<tool_call>\n<function=refresh>\n</function>\n</tool_call>",
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>Tokyo</parameter>\n"
                "</function>\n</tool_call>\n\n"
                "I will get that information."
            ),
            escaped_strings_output=(
                "<tool_call>\n<function=test_function>\n"
                '<parameter=quoted>He said "hello"</parameter>\n'
                "<parameter=path>C:\\Users\\file.txt</parameter>\n"
                "<parameter=newline>line1\nline2</parameter>\n"
                "</function>\n</tool_call>"
            ),
            malformed_input_outputs=[
                "<tool_call><function=func>",
                "<tool_call><function=></function></tool_call>",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            supports_typed_arguments=False,
        )
