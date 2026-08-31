# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #53363, Defect 1.

``tool_choice='required'`` must be enforced when a gemma4 parser is active.
gemma4 emits its native ``<|tool_call>`` syntax which is extracted directly
instead of being constrained by JSON structured output (see
``Gemma4EngineToolParser.adjust_request``). When the model free-generates
prose instead of a tool call, the parser must reject the result rather than
return a prose-only success.

Covers both parser paths:
- the unified engine path (``Gemma4Parser.parse`` / ``parse_delta``), which
  runs when both the reasoning and tool parser are gemma4
- the tool-parser-only path (``Gemma4EngineToolParser.extract_tool_calls``),
  which runs when the parser is composed through ``DelegatingParser``
"""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.gemma4 import (
    Gemma4Parser,
    ToolChoiceRequiredNotEnforcedError,
)
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

# Special token IDs (arbitrary but consistent)
CHANNEL_START_ID = 50  # <|channel>
CHANNEL_END_ID = 51  # <channel|>
TOOL_CALL_START_ID = 48  # <|tool_call>
TOOL_CALL_END_ID = 49  # <tool_call|>
QUOTED_ID = 52  # <|"|>
NEW_TURN_ID = 53  # <|turn>
SPECIAL_TOKEN_MAP = {
    CHANNEL_START_ID: "<|channel>",
    CHANNEL_END_ID: "<channel|>",
    TOOL_CALL_START_ID: "<|tool_call>",
    TOOL_CALL_END_ID: "<tool_call|>",
    QUOTED_ID: '<|"|>',
    NEW_TURN_ID: "<|turn>",
}
SPECIAL_TEXT_TO_ID = {v: k for k, v in SPECIAL_TOKEN_MAP.items()}

_WEATHER_CALL = '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
_PROSE_ONLY = "I cannot call a tool right now, so here is a plain answer."


def _make_tokenizer(extra_tokens: list[tuple[int, str]] | None = None) -> MagicMock:
    """Build a mock tokenizer with gemma4 special tokens plus extra tokens."""
    decode_map: dict[int, str] = dict(SPECIAL_TOKEN_MAP)
    for tid, text in extra_tokens or []:
        decode_map[tid] = text

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(SPECIAL_TEXT_TO_ID)
    tokenizer.encode.return_value = [tid for tid, _ in extra_tokens or []]

    def decode(ids, skip_special_tokens=False):
        parts = []
        for tid in ids:
            if skip_special_tokens and tid in SPECIAL_TOKEN_MAP:
                continue
            parts.append(decode_map.get(tid, f"?{tid}?"))
        return "".join(parts)

    tokenizer.decode.side_effect = decode
    tokenizer.all_special_tokens = list(SPECIAL_TOKEN_MAP.values())
    tokenizer.all_special_ids = list(SPECIAL_TOKEN_MAP.keys())
    return tokenizer


def _make_tool(name: str) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    )


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Split *text* into word-level ``(token_id, text)`` pairs."""
    tokens: list[tuple[int, str]] = []
    for i, word in enumerate(text.split(" ")):
        prefix = " " if i > 0 else ""
        tokens.append((1000 + i, prefix + word))
    return tokens


def _make_request(*, tool_choice: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gemma4-test",
        messages=[{"role": "user", "content": "What is the weather in London?"}],
        tools=[_make_tool("get_weather")],
        tool_choice=tool_choice,
    )


# ---------------------------------------------------------------------------
# Non-streaming engine path (Gemma4Parser.parse)
# ---------------------------------------------------------------------------


class TestGemma4ParserParseRequiredEnforcement:
    def test_required_prose_only_rejected(self):
        tokenizer = _make_tokenizer(_tokenize(_PROSE_ONLY))
        parser = Gemma4Parser(
            tokenizer, chat_template_kwargs={"enable_thinking": False}
        )
        request = _make_request(tool_choice="required")

        with pytest.raises(ToolChoiceRequiredNotEnforcedError):
            parser.parse(_PROSE_ONLY, request)

    def test_required_tool_call_accepted(self):
        tokenizer = _make_tokenizer(
            [
                (2000, "call"),
                (2001, ":"),
                (2002, "get_weather"),
                (2003, "{"),
                (2004, "location"),
                (2005, ":"),
                (2006, "London"),
                (2007, "}"),
            ]
        )
        parser = Gemma4Parser(
            tokenizer, chat_template_kwargs={"enable_thinking": False}
        )
        request = _make_request(tool_choice="required")

        _, content, tool_calls = parser.parse(_WEATHER_CALL, request)

        assert tool_calls
        assert tool_calls[0].name == "get_weather"
        assert content is None

    def test_auto_prose_only_passes_through(self):
        tokenizer = _make_tokenizer(_tokenize(_PROSE_ONLY))
        parser = Gemma4Parser(
            tokenizer, chat_template_kwargs={"enable_thinking": False}
        )
        request = _make_request(tool_choice="auto")

        _, content, tool_calls = parser.parse(_PROSE_ONLY, request)

        assert tool_calls is None
        assert content == _PROSE_ONLY


# ---------------------------------------------------------------------------
# Streaming engine path (Gemma4Parser.parse_delta at finish)
# ---------------------------------------------------------------------------


class TestGemma4ParserStreamingRequiredEnforcement:
    def _stream(self, parser, tokenizer, request, text: str):
        token_ids = tokenizer.encode("", add_special_tokens=False)
        results = []
        for start in range(0, len(token_ids), 3):
            batch = token_ids[start : start + 3]
            delta_text = tokenizer.decode(batch)
            results.append(
                parser.parse_delta(
                    delta_text,
                    batch,
                    request,
                    prompt_token_ids=[],
                    finished=(start + 3 >= len(token_ids)),
                )
            )
        return results

    def test_required_prose_only_rejected_at_finish(self):
        tokenizer = _make_tokenizer(_tokenize(_PROSE_ONLY))
        parser = Gemma4Parser(
            tokenizer, chat_template_kwargs={"enable_thinking": False}
        )
        request = _make_request(tool_choice="required")

        with pytest.raises(ToolChoiceRequiredNotEnforcedError):
            self._stream(parser, tokenizer, request, _PROSE_ONLY)

    def test_required_tool_call_stream_accepted(self):
        tokenizer = _make_tokenizer(
            [
                (TOOL_CALL_START_ID, "<|tool_call>"),
                (2000, "call"),
                (2001, ":"),
                (2002, "get_weather"),
                (2003, "{"),
                (2004, "location"),
                (2005, ":"),
                (2006, "London"),
                (2007, "}"),
                (TOOL_CALL_END_ID, "<tool_call|>"),
            ]
        )
        parser = Gemma4Parser(
            tokenizer, chat_template_kwargs={"enable_thinking": False}
        )
        request = _make_request(tool_choice="required")

        results = self._stream(parser, tokenizer, request, _WEATHER_CALL)

        tool_calls = [tc for r in results if r and r.tool_calls for tc in r.tool_calls]
        assert tool_calls
        assert tool_calls[0].function.name == "get_weather"


# ---------------------------------------------------------------------------
# Tool-parser-only path (Gemma4EngineToolParser.extract_tool_calls)
# ---------------------------------------------------------------------------


class TestGemma4ToolParserRequiredEnforcement:
    def test_required_prose_only_rejected(self):
        tokenizer = _make_tokenizer(_tokenize(_PROSE_ONLY))
        parser = Gemma4EngineToolParser(tokenizer, tools=[_make_tool("get_weather")])
        request = _make_request(tool_choice="required")

        with pytest.raises(ToolChoiceRequiredNotEnforcedError):
            parser.extract_tool_calls(_PROSE_ONLY, request)

    def test_required_tool_call_accepted(self):
        tokenizer = _make_tokenizer(
            [
                (2000, "call"),
                (2001, ":"),
                (2002, "get_weather"),
                (2003, "{"),
                (2004, "location"),
                (2005, ":"),
                (2006, "London"),
                (2007, "}"),
            ]
        )
        parser = Gemma4EngineToolParser(tokenizer, tools=[_make_tool("get_weather")])
        request = _make_request(tool_choice="required")

        result = parser.extract_tool_calls(_WEATHER_CALL, request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"

    def test_auto_prose_only_passes_through(self):
        tokenizer = _make_tokenizer(_tokenize(_PROSE_ONLY))
        parser = Gemma4EngineToolParser(tokenizer, tools=[_make_tool("get_weather")])
        request = _make_request(tool_choice="auto")

        result = parser.extract_tool_calls(_PROSE_ONLY, request)

        assert result.tools_called is False
        assert result.content == _PROSE_ONLY
