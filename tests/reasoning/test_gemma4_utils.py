# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.reasoning.gemma4_utils import (
    _clean_answer,
    _strip_thought_label,
    parse_thinking_output,
)


class TestStripThoughtLabel:
    def test_strips_thought_prefix(self):
        assert _strip_thought_label("thought\nHello world") == "Hello world"

    def test_no_thought_prefix(self):
        assert _strip_thought_label("Hello world") == "Hello world"

    def test_thought_without_newline(self):
        assert (
            _strip_thought_label("thought without newline") == "thought without newline"
        )

    def test_thought_not_at_start(self):
        assert _strip_thought_label("prefix thought\nrest") == "prefix thought\nrest"

    def test_empty_string(self):
        assert _strip_thought_label("") == ""

    def test_only_thought_newline(self):
        assert _strip_thought_label("thought\n") == ""


class TestCleanAnswer:
    def test_strips_turn_end_tag(self):
        assert _clean_answer("Hello<turn|>") == "Hello"

    def test_strips_eos_tag(self):
        assert _clean_answer("Hello<eos>") == "Hello"

    def test_strips_both_tags(self):
        assert _clean_answer("Hello<turn|><eos>") == "Hello"

    def test_strips_whitespace_around_tags(self):
        assert _clean_answer("  Hello  <turn|>  ") == "Hello"

    def test_no_tags(self):
        assert _clean_answer("Hello world") == "Hello world"

    def test_empty_string(self):
        assert _clean_answer("") == ""

    def test_only_tags(self):
        assert _clean_answer("<turn|>") == ""

    def test_tag_in_middle(self):
        assert _clean_answer("Hello<turn|> world") == "Hello<turn|> world"


class TestParseThinkingOutput:
    def test_thinking_enabled_with_tags(self):
        text = "<|channel>I am thinking<channel|>The answer"
        result = parse_thinking_output(text)
        assert result["thinking"] == "I am thinking"
        assert result["answer"] == "The answer"

    def test_thinking_enabled_with_thought_label(self):
        text = "<|channel>thought\nI am thinking<channel|>The answer"
        result = parse_thinking_output(text)
        assert result["thinking"] == "I am thinking"
        assert result["answer"] == "The answer"

    def test_thinking_disabled_spurious_label(self):
        text = "thought\nThe answer"
        result = parse_thinking_output(text)
        assert result["thinking"] is None
        assert result["answer"] == "The answer"

    def test_clean_output_no_thinking(self):
        text = "Just a normal response"
        result = parse_thinking_output(text)
        assert result["thinking"] is None
        assert result["answer"] == "Just a normal response"

    def test_thinking_with_trailing_tags(self):
        text = "<|channel>reasoning<channel|>answer<turn|>"
        result = parse_thinking_output(text)
        assert result["thinking"] == "reasoning"
        assert result["answer"] == "answer"

    def test_thinking_with_eos(self):
        text = "<|channel>reasoning<channel|>answer<eos>"
        result = parse_thinking_output(text)
        assert result["thinking"] == "reasoning"
        assert result["answer"] == "answer"

    def test_multiple_thinking_blocks(self):
        text = "<|channel>first<channel|>middle<|channel>second<channel|>end"
        result = parse_thinking_output(text)
        assert result["thinking"] == "first"
        assert result["answer"] == "middle<|channel>second<channel|>end"

    def test_empty_thinking_block(self):
        text = "<|channel><channel|>answer"
        result = parse_thinking_output(text)
        assert result["thinking"] == ""
        assert result["answer"] == "answer"

    def test_only_end_tag(self):
        text = "text<channel|>answer"
        result = parse_thinking_output(text)
        assert result["thinking"] == "text"
        assert result["answer"] == "answer"

    def test_thought_label_without_thinking(self):
        text = "thought\nJust the answer"
        result = parse_thinking_output(text)
        assert result["thinking"] is None
        assert result["answer"] == "Just the answer"
