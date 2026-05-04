# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.5"


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


class TestGlm47ExtractToolCalls:
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
        for chunk in chunks:
            current_text += chunk
            glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
        assert len(glm47_tool_parser.prev_tool_call_arr) >= 1

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
        for chunk in chunks:
            current_text += chunk
            glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
        args = json.loads(glm47_tool_parser.prev_tool_call_arr[0]["arguments"])
        assert args["city"] == "Beijing"


class TestGlm47AdjustRequestForcedToolChoice:
    """Tests for ``Glm47MoeModelToolParser.adjust_request`` on forced
    tool_choice values.

    GLM-4.7 emits XML ``<tool_call>`` markers, so the grandparent
    ``ToolParser.adjust_request`` (which injects a Hermes JSON schema for
    ``tool_choice="required"`` and named-function choices) would force JSON
    output and break extraction. The override on this parser must short-
    circuit before that injection for both Chat Completions and Responses
    API forced-choice forms, while still delegating cleanly for
    ``"auto"`` / ``"none"`` / no-tools.

    Migrated from ``tests/entrypoints/openai/responses/test_serving_responses.py``
    (was ``TestMaybeDemoteUnsupportedToolChoice``) and reframed around the
    parser-side fix instead of the prior serving-layer demotion helper.
    """

    @staticmethod
    def _build_responses_request(*, tool_choice):
        return ResponsesRequest(
            input="What is the weather in Hanoi?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_choice=tool_choice,
        )

    def test_required_skips_schema_injection_responses(self, glm47_tool_parser):
        """tool_choice='required' on /v1/responses must not get a JSON schema
        injected via ``request.text.format``."""
        request = self._build_responses_request(tool_choice="required")

        adjusted = glm47_tool_parser.adjust_request(request)

        # No structured-output schema should be set — would otherwise force
        # JSON and break the XML extractor.
        assert adjusted.text is None or adjusted.text.format is None
        assert adjusted.structured_outputs is None
        # tool_choice itself stays as the user requested; the override fixes
        # the side-effect, not the field.
        assert adjusted.tool_choice == "required"
        # Tool-call XML tokens must remain visible during decoding.
        assert adjusted.skip_special_tokens is False

    def test_named_function_skips_schema_injection_responses(self, glm47_tool_parser):
        """Named-function tool_choice on /v1/responses arrives as
        ``ToolChoiceFunction`` (Pydantic-parsed). The override must catch it
        too, not only the Chat Completions ``ChatCompletionNamedToolChoiceParam``."""
        request = self._build_responses_request(
            tool_choice={"type": "function", "name": "get_weather"},
        )

        adjusted = glm47_tool_parser.adjust_request(request)

        assert adjusted.text is None or adjusted.text.format is None
        assert adjusted.structured_outputs is None
        # Named-function tool_choice is preserved (still a ToolChoiceFunction).
        assert adjusted.tool_choice is not None
        assert getattr(adjusted.tool_choice, "type", None) == "function"
        assert getattr(adjusted.tool_choice, "name", None) == "get_weather"
        assert adjusted.skip_special_tokens is False

    def test_required_skips_schema_injection_chat_completions(self, glm47_tool_parser):
        """Chat Completions ``tool_choice="required"`` is the Glm4 parent's
        primary case. Confirm the GLM-4.7 override doesn't regress it."""
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            tool_choice="required",
        )  # type: ignore

        adjusted = glm47_tool_parser.adjust_request(request)

        assert adjusted.structured_outputs is None
        assert adjusted.skip_special_tokens is False

    def test_auto_delegates_to_super_responses(self, glm47_tool_parser):
        """tool_choice='auto' should fall through to the parent path:
        no schema injected, but ``skip_special_tokens`` still flipped to
        ``False`` so XML markers survive decoding."""
        request = self._build_responses_request(tool_choice="auto")

        adjusted = glm47_tool_parser.adjust_request(request)

        assert adjusted.text is None or adjusted.text.format is None
        assert adjusted.structured_outputs is None
        assert adjusted.tool_choice == "auto"
        assert adjusted.skip_special_tokens is False

    def test_no_tools_is_noop(self, glm47_tool_parser):
        """No tools -> the override delegates to ``super().adjust_request``
        which returns the request untouched. Substitutes for the spec's
        ``no_tools`` case."""
        # ResponsesRequest's validator coerces tool_choice to "none" when
        # tools is empty; build the request that way to stay valid.
        request = ResponsesRequest(input="hi", tools=[], tool_choice="auto")

        adjusted = glm47_tool_parser.adjust_request(request)

        assert adjusted.text is None or adjusted.text.format is None
        assert adjusted.structured_outputs is None
        # No-tools path doesn't toggle skip_special_tokens — left at default.
        assert adjusted.skip_special_tokens is True
