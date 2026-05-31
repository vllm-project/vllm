# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for Qwen3 reasoning parser tool-call recovery (Issue #39056).

These tests verify that XML tool-call blocks emitted inside <think> are
correctly promoted into content so the downstream Qwen3CoderToolParser
can parse them — and that any pre-existing response text is preserved.
"""

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

parser_name = "qwen3"

_TOOL_CALL_BLOCK = (
    "<tool_call>\n"
    "<function=Finish>\n"
    "<parameter=answer>\n"
    "204\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


class _FakeQwen3ToolTokenizer:
    """Minimal tokenizer stub sufficient for parser construction."""

    def get_vocab(self) -> dict[str, int]:
        return {
            "<think>": 1,
            "</think>": 2,
            "<tool_call>": 3,
            "</tool_call>": 4,
        }


def _make_parser() -> ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser(parser_name)(
        _FakeQwen3ToolTokenizer()
    )


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test-model")


# ---------------------------------------------------------------------------
# Basic promotion: tool call extracted from reasoning, placed into content
# ---------------------------------------------------------------------------

def test_embedded_tool_call_is_promoted_from_reasoning_into_content():
    """Tool-call block inside <think> must move to content, not stay in reasoning."""
    parser = _make_parser()
    request = _make_request()

    reasoning, content = parser.extract_reasoning(
        "<think>The verification confirms my solution:\n"
        "- s = 2.5 km/h\n"
        "- t = 24 minutes\n"
        "- Total time at speed 3 km/h = 204 minutes\n\n"
        + _TOOL_CALL_BLOCK
        + "\n</think>",
        request=request,
    )

    assert reasoning is not None
    assert "<tool_call>" not in reasoning, "tool call must not remain in reasoning"
    assert content is not None
    assert "<tool_call>" in content, "tool call must be present in content"
    assert "<function=Finish>" in content


# ---------------------------------------------------------------------------
# Ordering fix: existing content text must be preserved after promotion
# ---------------------------------------------------------------------------

def test_existing_content_text_is_preserved_after_tool_call_promotion():
    """
    Pre-existing response text must appear BEFORE the promoted tool-call block
    in content.  The Qwen3CoderToolParser reads content up to the first tool
    marker as the human-readable reply; if the tool block were prepended that
    text would be silently discarded.
    """
    parser = _make_parser()
    request = _make_request()

    # Model emits tool call inside <think>, then text after </think>
    _, content = parser.extract_reasoning(
        "<think>verify result\n"
        + _TOOL_CALL_BLOCK
        + "</think>Here is the answer.",
        request=request,
    )

    assert content is not None
    assert "Here is the answer." in content, "post-</think> text must be preserved"

    # The text must come BEFORE the tool call so the tool parser keeps it
    text_pos = content.index("Here is the answer.")
    tool_pos = content.index("<tool_call>")
    assert text_pos < tool_pos, (
        "existing response text must appear before the promoted tool call "
        f"(text at {text_pos}, tool at {tool_pos})"
    )


# ---------------------------------------------------------------------------
# End-to-end: promoted content must be parseable by Qwen3CoderToolParser
# ---------------------------------------------------------------------------

def test_promoted_tool_call_is_parseable_by_qwen3coder_and_trailing_text_preserved():
    """
    Full pipeline: reasoning parser promotes the tool call, then
    Qwen3CoderToolParser extracts it.  Trailing assistant text must survive.
    """
    parser = _make_parser()
    tool_parser = Qwen3CoderToolParser(_FakeQwen3ToolTokenizer(), tools=None)
    request = _make_request()

    _, content = parser.extract_reasoning(
        "<think>verify result\n"
        + _TOOL_CALL_BLOCK
        + "</think>assistant trailing text",
        request=request,
    )

    assert content is not None
    assert "assistant trailing text" in content

    tool_call_info = tool_parser.extract_tool_calls(content, request=request)

    assert tool_call_info.tools_called is True
    assert len(tool_call_info.tool_calls) == 1
    tool_call = tool_call_info.tool_calls[0]
    assert tool_call.function.name == "Finish"
    assert json.loads(tool_call.function.arguments) == {"answer": "204"}

    # FIX for reviewer comment: verify trailing text is preserved in the
    # final extracted content field, not discarded by the tool parser.
    assert tool_call_info.content is not None, (
        "trailing response text must be preserved by Qwen3CoderToolParser"
    )
    assert "assistant trailing text" in tool_call_info.content


# ---------------------------------------------------------------------------
# Truncated output: no </think>, but tool call still recoverable
# ---------------------------------------------------------------------------

def test_truncated_reasoning_still_recovers_embedded_tool_call():
    """When output is cut off before </think>, embedded tool calls still promote."""
    parser = _make_parser()
    request = _make_request()

    reasoning, content = parser.extract_reasoning(
        "verify result\n" + _TOOL_CALL_BLOCK,
        request=request,
    )

    assert reasoning is not None
    assert "<tool_call>" not in reasoning
    assert content is not None
    assert "<tool_call>" in content


# ---------------------------------------------------------------------------
# Normal reasoning: no tool call → unchanged behaviour
# ---------------------------------------------------------------------------

def test_normal_reasoning_extraction_unchanged():
    """Reasoning without any tool call must pass through unmodified."""
    parser = _make_parser()
    request = _make_request()

    raw_reasoning = "Let me think about this carefully.\nThe answer is 42."
    reasoning, content = parser.extract_reasoning(
        f"<think>{raw_reasoning}</think>The answer is 42.",
        request=request,
    )

    assert reasoning == raw_reasoning
    assert content == "The answer is 42."


# ---------------------------------------------------------------------------
# No regression: post-</think> content preserved without tool call
# ---------------------------------------------------------------------------

def test_post_think_content_preserved_without_tool_call():
    """Content after </think> must be returned verbatim when no tool call."""
    parser = _make_parser()
    request = _make_request()

    _, content = parser.extract_reasoning(
        "<think>some reasoning</think>plain response text",
        request=request,
    )

    assert content == "plain response text"
