# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
from unittest.mock import MagicMock, Mock

import pytest
import xgrammar as xgr
from openai.types.responses import ResponseFunctionToolCall
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.generate.base.protocol import FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.7"


@pytest.fixture(scope="module")
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
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


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer, sample_tools):
    return Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


@pytest.fixture
def namespace_tool_request() -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__computer_use",
                    "description": "Computer use tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_app_state",
                            "description": "Get app state.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "app": {"type": "string"},
                                },
                            },
                        }
                    ],
                }
            ],
        }
    )


class TestGlm47ExtractToolCalls:
    def test_namespace_tool_call_round_trip_to_responses_output(
        self, glm47_tokenizer, namespace_tool_request
    ):
        parser = Glm47MoeModelToolParser(
            glm47_tokenizer, tools=namespace_tool_request.tools
        )
        out = (
            "<tool_call>mcp__computer_use__get_app_state"
            "<arg_key>app</arg_key>"
            "<arg_value>Google Chrome</arg_value>"
            "</tool_call>"
        )

        result = parser.extract_tool_calls(out, request=namespace_tool_request)

        assert result.tools_called
        tool_call = result.tool_calls[0].function
        assert tool_call == FunctionCall(
            name="mcp__computer_use__get_app_state",
            arguments='{"app": "Google Chrome"}',
        )

        output_items = build_response_output_items(
            reasoning=None,
            content=None,
            tool_calls=[tool_call],
            tools=namespace_tool_request.tools,
        )
        output_tool_call = output_items[0]
        assert isinstance(output_tool_call, ResponseFunctionToolCall)
        assert output_tool_call.name == "get_app_state"
        assert output_tool_call.namespace == "mcp__computer_use"

    def test_no_tool_call(self, glm47_tool_parser, mock_request):
        out = "This is a plain response."
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_args_with_newlines(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_whitespace_preserved_in_arg_values(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "  Beijing  "}

    def test_content_before(self, glm47_tool_parser, mock_request):
        out = "Checking.<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, glm47_tool_parser, mock_request):
        out = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        )
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    def test_empty_content_none(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
        out = "  \n  <tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None


def _reset(parser):
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


class TestGlm47Streaming:
    def test_no_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = ["<tool_call>", "get_current_date", "</tool_call>"]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        tool_calls = [
            tool_call for delta in deltas for tool_call in (delta.tool_calls or [])
        ]
        names = [
            tool_call.function.name
            for tool_call in tool_calls
            if tool_call.function and tool_call.function.name
        ]
        arguments = [
            tool_call.function.arguments
            for tool_call in tool_calls
            if tool_call.function and tool_call.function.arguments
        ]
        assert names == ["get_current_date"]
        assert "".join(arguments) == "{}"

    def test_with_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        arguments = [
            tool_call.function.arguments
            for delta in deltas
            for tool_call in (delta.tool_calls or [])
            if tool_call.function and tool_call.function.arguments
        ]
        args = json.loads("".join(arguments))
        assert args["city"] == "Beijing"


def _preferences_tool(strict: bool | None) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="collect_preferences",
            strict=strict,
            parameters={
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "choices": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "tag": {"type": "string"},
                            },
                            "required": ["prompt", "tag"],
                        },
                    },
                },
                "required": ["items"],
            },
        ),
    )


def _json_schema_formats(node) -> list[dict]:
    if isinstance(node, dict):
        found = [node] if node.get("type") == "json_schema" else []
        return found + [f for v in node.values() for f in _json_schema_formats(v)]
    if isinstance(node, list):
        return [f for v in node for f in _json_schema_formats(v)]
    return []


def _call(items: str, note: str | None = None) -> str:
    args = f"<arg_key>items</arg_key><arg_value>{items}</arg_value>"
    if note is not None:
        args += f"<arg_key>note</arg_key><arg_value>{note}</arg_value>"
    return f"<tool_call>collect_preferences{args}</tool_call>"


class TestGlm47StructuralTag:
    """Strict mode must enforce the same required-first key order the prompt
    shows the model: the grammar then guarantees required arguments instead of
    fighting the order the model was primed with."""

    @staticmethod
    def _tag(tool_choice, strict):
        tools = [_preferences_tool(strict)]
        parser = Glm47MoeModelToolParser(MagicMock(), tools=tools)
        request = ChatCompletionRequest(messages=[], model="m", tools=tools)
        request.tool_choice = tool_choice
        tag = parser.get_structural_tag(request)
        # The request keeps the client's schema order.
        assert list(tools[0].function.parameters["properties"]) == ["note", "items"]
        return tag

    def test_auto_without_strict_is_unconstrained(self):
        assert self._tag("auto", strict=None) is None

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "auto",
            "required",
            ChatCompletionNamedToolChoiceParam(
                function=ChatCompletionNamedFunction(name="collect_preferences")
            ),
        ],
    )
    def test_embedded_schema_uses_required_first_order(self, tool_choice):
        formats = _json_schema_formats(self._tag(tool_choice, strict=True).model_dump())
        assert len(formats) == 1
        schema = formats[0]["json_schema"]
        assert list(schema["properties"]) == ["items", "note"]
        assert list(schema["properties"]["items"]["items"]["properties"]) == [
            "prompt",
            "tag",
            "choices",
        ]

    @pytest.mark.parametrize(
        "output",
        [
            _call('[{"prompt": "q", "tag": "t", "choices": ["a"]}]'),
            _call('[{"prompt": "q", "tag": "t"}]', note="n"),
            _call("[]"),
        ],
    )
    def test_grammar_accepts_required_first_output(self, output):
        grammar = xgr.Grammar.from_structural_tag(self._tag("required", strict=True))
        assert _is_grammar_accept_string(grammar, output, require_termination=False)

    @pytest.mark.parametrize(
        "output",
        [
            _call('[{"prompt": "q", "choices": ["a"]}]'),
            "<tool_call>collect_preferences<arg_key>note</arg_key><arg_value>n"
            "</arg_value></tool_call>",
        ],
    )
    def test_grammar_rejects_missing_required_key(self, output):
        grammar = xgr.Grammar.from_structural_tag(self._tag("required", strict=True))
        assert not _is_grammar_accept_string(grammar, output, require_termination=False)
