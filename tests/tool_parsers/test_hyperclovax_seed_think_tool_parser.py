# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Unit tests for the HyperCLOVAX-SEED-Think tool call parser."""

import json
from unittest.mock import MagicMock, Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.hyperclovax_seed_think_tool_parser import (
    HyperCLOVAXSeedThinkToolParser,
)

PARSER_NAME = "hyperclovax_seed_think"


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    # The parser only touches `self.vocab` (via `tokenizer.get_vocab()`) once it
    # needs token IDs for streaming bookkeeping; otherwise it's text-driven.
    tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
        "<arg_key>": 1002,
        "</arg_key>": 1003,
        "<arg_value>": 1004,
        "</arg_value>": 1005,
        "</think>": 1006,
        "<|im_end|>": 1007,
    }
    return tokenizer


@pytest.fixture
def tool_parser(mock_tokenizer):
    return HyperCLOVAXSeedThinkToolParser(mock_tokenizer)


@pytest.fixture
def mock_request() -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_current_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["location"],
                },
            ),
        ),
    ]
    request.tool_choice = "auto"
    return request


# ---------------------------------------------------------------------------
# Registration / class invariants
# ---------------------------------------------------------------------------


def test_parser_is_registered():
    parser_cls = ToolParserManager.get_tool_parser(PARSER_NAME)
    assert parser_cls is HyperCLOVAXSeedThinkToolParser


def test_supports_required_and_named_is_false():
    """All tool_choice modes must route through this parser."""
    assert HyperCLOVAXSeedThinkToolParser.supports_required_and_named is False


# ---------------------------------------------------------------------------
# Non-streaming: XML <tool_call> format
# ---------------------------------------------------------------------------


class TestExtractToolCallsXML:
    def test_no_tool_call(self, tool_parser, mock_request):
        out = "Just a regular response."
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is False
        assert r.tool_calls == []
        assert r.content == out

    def test_single_tool_call(self, tool_parser, mock_request):
        out = (
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>\"Seoul\"</arg_value>\n"
            "<arg_key>unit</arg_key>\n"
            "<arg_value>\"celsius\"</arg_value>\n"
            "</tool_call>"
        )
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].type == "function"
        assert r.tool_calls[0].function.name == "get_current_weather"
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "location": "Seoul",
            "unit": "celsius",
        }
        assert r.content is None

    def test_arg_value_falls_back_to_raw_string(self, tool_parser, mock_request):
        """Non-JSON arg_value bodies are kept as raw strings."""
        out = (
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>Seoul</arg_value>\n"
            "</tool_call>"
        )
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "location": "Seoul",
        }

    def test_content_before_tool_call_is_preserved(self, tool_parser, mock_request):
        out = (
            "I'll check that for you.<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>\"Seoul\"</arg_value>\n"
            "</tool_call>"
        )
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert r.content == "I'll check that for you."

    def test_reasoning_prefix_stripped_from_content(self, tool_parser, mock_request):
        """Content before <tool_call> after </think> wins over pre-</think> text."""
        out = (
            "thinking aloud</think>\n\nokay, calling:"
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>\"Seoul\"</arg_value>\n"
            "</tool_call>"
        )
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert r.content == "okay, calling:"

    def test_multiple_tool_calls(self, tool_parser, mock_request):
        out = (
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>\"Seoul\"</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>\"Busan\"</arg_value>\n"
            "</tool_call>"
        )
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert len(r.tool_calls) == 2
        assert json.loads(r.tool_calls[0].function.arguments)["location"] == "Seoul"
        assert json.loads(r.tool_calls[1].function.arguments)["location"] == "Busan"

    def test_malformed_tool_call_falls_back_to_content(
        self, tool_parser, mock_request
    ):
        """<tool_call> marker present but body has no parseable function name."""
        out = "<tool_call>\n</tool_call>"
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        # Empty name → no valid tool_calls → fallback to content.
        assert r.tools_called is False
        assert r.content == out


# ---------------------------------------------------------------------------
# Non-streaming: JSON list fallback (tool_choice=required path)
# ---------------------------------------------------------------------------


class TestExtractToolCallsJSON:
    def test_json_list_payload(self, tool_parser, mock_request):
        out = '[{"name": "get_current_weather", "parameters": {"location": "Seoul"}}]'
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert r.tool_calls[0].function.name == "get_current_weather"
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "location": "Seoul"
        }
        assert r.content is None

    def test_json_single_object_payload(self, tool_parser, mock_request):
        """A single JSON object (not wrapped in a list) is treated as one call."""
        out = '{"name": "get_current_weather", "parameters": {"location": "Busan"}}'
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert len(r.tool_calls) == 1
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "location": "Busan"
        }

    def test_json_with_arguments_key(self, tool_parser, mock_request):
        """Accepts `arguments` as an alias for `parameters`."""
        out = '[{"name": "get_current_weather", "arguments": {"location": "Seoul"}}]'
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "location": "Seoul"
        }

    def test_json_after_think_end(self, tool_parser, mock_request):
        """JSON payload preceded by reasoning + </think>."""
        out = (
            "thinking</think>\n\n"
            '[{"name": "get_current_weather", "parameters": {"location": "Seoul"}}]'
        )
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is True
        assert r.tool_calls[0].function.name == "get_current_weather"

    def test_invalid_json_falls_back_to_content(self, tool_parser, mock_request):
        out = "[not, valid, json"
        r = tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called is False
        assert r.content == out


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _simulate_streaming(
    parser: HyperCLOVAXSeedThinkToolParser,
    deltas: list[str],
    request: ChatCompletionRequest,
) -> list[DeltaMessage | None]:
    results: list[DeltaMessage | None] = []
    previous_text = ""
    previous_token_ids: list[int] = []
    vocab = parser.vocab
    for delta_text in deltas:
        current_text = previous_text + delta_text
        delta_token_ids = [tid for tok, tid in vocab.items() if tok in delta_text]
        current_token_ids = previous_token_ids + delta_token_ids
        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )
        results.append(result)
        previous_text = current_text
        previous_token_ids = current_token_ids
    return results


def _collect_streaming_tool_calls(
    results: list[DeltaMessage | None],
) -> list[dict]:
    tool_calls: dict[int, dict] = {}
    for result in results:
        if result is None or not result.tool_calls:
            continue
        for tc in result.tool_calls:
            idx = tc.index
            entry = tool_calls.setdefault(idx, {"name": "", "arguments": ""})
            if tc.function.name:
                entry["name"] += tc.function.name
            if tc.function.arguments:
                entry["arguments"] += tc.function.arguments
    return [tool_calls[i] for i in sorted(tool_calls.keys())]


def _collect_streaming_content(results: list[DeltaMessage | None]) -> str:
    return "".join(r.content for r in results if r is not None and r.content)


class TestStreamingXML:
    def test_plain_content_only(self, tool_parser, mock_request):
        deltas = ["Just ", "plain ", "text."]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        assert _collect_streaming_content(results) == "Just plain text."
        assert _collect_streaming_tool_calls(results) == []

    def test_single_tool_call_chunked(self, tool_parser, mock_request):
        deltas = [
            "<tool_call>",
            "get_current_weather\n",
            "<arg_key>location</arg_key>\n",
            "<arg_value>\"Seoul\"</arg_value>\n",
            "</tool_call>",
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_current_weather"
        assert json.loads(tc[0]["arguments"]) == {"location": "Seoul"}

    def test_content_then_tool_call(self, tool_parser, mock_request):
        deltas = [
            "Let me check.",
            "<tool_call>get_current_weather\n",
            "<arg_key>location</arg_key>\n<arg_value>\"Seoul\"</arg_value>\n",
            "</tool_call>",
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        assert _collect_streaming_content(results) == "Let me check."
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1 and tc[0]["name"] == "get_current_weather"

    def test_multiple_tool_calls(self, tool_parser, mock_request):
        deltas = [
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n<arg_value>\"Seoul\"</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>get_current_weather\n"
            "<arg_key>location</arg_key>\n<arg_value>\"Busan\"</arg_value>\n"
            "</tool_call>",
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 2
        assert json.loads(tc[0]["arguments"]) == {"location": "Seoul"}
        assert json.loads(tc[1]["arguments"]) == {"location": "Busan"}

    def test_split_open_tag_across_delta(self, tool_parser, mock_request):
        """`<tool_call>` itself may straddle deltas — the partial-prefix guard
        must hold any tail that could complete an open tag."""
        deltas = [
            "<tool_",
            "call>get_current_weather\n",
            "<arg_key>location</arg_key>\n<arg_value>\"Seoul\"</arg_value>\n",
            "</tool_call>",
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1 and tc[0]["name"] == "get_current_weather"
        # The "<tool_" prefix must NOT have leaked into content.
        assert "<tool_" not in _collect_streaming_content(results)

    def test_im_end_marker_stripped_from_content(self, tool_parser, mock_request):
        """`<|im_end|>` is an EOS marker that should never reach the user."""
        deltas = ["answer<|im_end|>"]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        assert _collect_streaming_content(results) == "answer"


class TestStreamingJSON:
    def test_json_list_emitted_in_one_message(self, tool_parser, mock_request):
        """JSON list payload (tool_choice=required path)."""
        deltas = [
            '[{"name": "get_current_weather", ',
            '"parameters": {"location": "Seoul"}}]',
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_current_weather"
        assert json.loads(tc[0]["arguments"]) == {"location": "Seoul"}

    def test_json_after_think_end(self, tool_parser, mock_request):
        """`</think>` in the buffer should not block the JSON path."""
        deltas = [
            "</think>",
            "\n\n",
            '[{"name": "get_current_weather", "parameters": {"location": "Busan"}}]',
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1
        assert json.loads(tc[0]["arguments"]) == {"location": "Busan"}

    def test_json_partial_payload_holds(self, tool_parser, mock_request):
        """Incomplete JSON must not be emitted yet."""
        results = _simulate_streaming(
            tool_parser,
            ['[{"name": "get_current_weather", "parameters": '],
            mock_request,
        )
        # Buffer is not yet parseable → no tool_call emission.
        assert _collect_streaming_tool_calls(results) == []

    def test_split_think_end_across_deltas_does_not_leak(
        self, tool_parser, mock_request
    ):
        """Standalone usage: when `</think>` straddles deltas, the partial tail
        must not be emitted as content (regression test for #42366 review)."""
        deltas = [
            "</thi",
            "nk>",
            '[{"name": "get_current_weather", "parameters": {"location": "Seoul"}}]',
        ]
        results = _simulate_streaming(tool_parser, deltas, mock_request)
        content = _collect_streaming_content(results)
        assert "</thi" not in content
        assert "nk>" not in content
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_current_weather"
