# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tool_choice="required" fallback to tool_parser.

When tool_choice="required" and the model produces non-JSON tool calls
(e.g. XML format from Qwen3), the non-streaming path should fall back
to the configured tool_parser instead of returning a 400 error.

See: https://github.com/vllm-project/vllm/pull/35936
"""

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import ExtractedToolCallInformation

pytestmark = pytest.mark.cpu_test

MODEL = "test-model"

SAMPLE_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    },
                },
                "required": ["city"],
            },
        },
    ),
]

# JSON format tool call (standard)
JSON_TOOL_CALL = json.dumps(
    [{"name": "get_current_weather", "parameters": {"city": "Dallas"}}]
)

# XML format tool call (Qwen3 style)
XML_TOOL_CALL = """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
</function>
</tool_call>"""


class MockToolParser(ToolParser):
    """A minimal tool parser that recognizes XML-style tool calls."""

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(self, model_output, request):
        from vllm.entrypoints.openai.engine.protocol import (
            FunctionCall,
            ToolCall,
        )

        # Simple check: if it contains <function=, parse it
        if "<function=" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        import regex as re

        func_match = re.search(r"<function=(\w+)>", model_output)
        param_matches = re.findall(
            r"<parameter=(\w+)>\n(.*?)\n</parameter>",
            model_output,
            re.DOTALL,
        )

        if func_match:
            name = func_match.group(1)
            args = {k: v.strip() for k, v in param_matches}
            tool_calls = [
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(args),
                    )
                )
            ]
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        raise NotImplementedError


def _make_request(tool_choice="required"):
    return ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=SAMPLE_TOOLS,
        tool_choice=tool_choice,
    )


class TestToolChoiceRequiredNonStreaming:
    """Tests for _parse_tool_calls_from_content with tool_choice='required'."""

    def test_json_content_succeeds_directly(self):
        """Valid JSON tool calls should be parsed without fallback."""
        request = _make_request()
        function_calls, content = OpenAIServing._parse_tool_calls_from_content(
            request=request,
            tokenizer=None,
            enable_auto_tools=False,
            tool_parser_cls=None,
            content=JSON_TOOL_CALL,
        )

        assert function_calls is not None
        assert len(function_calls) == 1
        assert function_calls[0].name == "get_current_weather"
        assert json.loads(function_calls[0].arguments) == {"city": "Dallas"}
        assert content is None  # cleared after tool call

    def test_xml_content_falls_back_to_tool_parser(self):
        """XML tool calls should fail JSON validation, then fall back
        to the configured tool_parser."""
        request = _make_request()
        tokenizer = MagicMock()

        function_calls, content = OpenAIServing._parse_tool_calls_from_content(
            request=request,
            tokenizer=tokenizer,
            enable_auto_tools=True,
            tool_parser_cls=MockToolParser,
            content=XML_TOOL_CALL,
        )

        assert function_calls is not None
        assert len(function_calls) == 1
        assert function_calls[0].name == "get_current_weather"
        assert json.loads(function_calls[0].arguments) == {"city": "Dallas"}
        assert content is None  # cleared after tool call

    def test_xml_content_no_tool_calls_without_tool_parser(self):
        """Without a configured tool_parser, XML content should result
        in no tool calls (graceful degradation)."""
        request = _make_request()

        function_calls, content = OpenAIServing._parse_tool_calls_from_content(
            request=request,
            tokenizer=None,
            enable_auto_tools=False,
            tool_parser_cls=None,
            content=XML_TOOL_CALL,
        )

        assert function_calls is not None
        assert len(function_calls) == 0
        assert content is None  # still cleared

    def test_xml_content_no_tool_calls_without_enable_auto_tools(self):
        """Even with tool_parser_cls, if enable_auto_tools is False,
        the fallback should not activate."""
        request = _make_request()
        tokenizer = MagicMock()

        function_calls, content = OpenAIServing._parse_tool_calls_from_content(
            request=request,
            tokenizer=tokenizer,
            enable_auto_tools=False,
            tool_parser_cls=MockToolParser,
            content=XML_TOOL_CALL,
        )

        assert function_calls is not None
        assert len(function_calls) == 0
        assert content is None  # still cleared

    def test_multiple_json_tool_calls(self):
        """Multiple JSON tool calls should all be parsed."""
        content = json.dumps(
            [
                {"name": "get_current_weather", "parameters": {"city": "Dallas"}},
                {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
            ]
        )
        request = _make_request()

        function_calls, returned_content = (
            OpenAIServing._parse_tool_calls_from_content(
                request=request,
                tokenizer=None,
                enable_auto_tools=False,
                tool_parser_cls=None,
                content=content,
            )
        )

        assert function_calls is not None
        assert len(function_calls) == 2
        assert function_calls[0].name == "get_current_weather"
        assert function_calls[1].name == "get_current_weather"
        assert json.loads(function_calls[0].arguments) == {"city": "Dallas"}
        assert json.loads(function_calls[1].arguments) == {"city": "Berlin"}

    def test_none_content_does_not_crash(self):
        """When content is None (e.g. max_tokens exceeded), should not
        crash (regression test for #36841)."""
        request = _make_request()

        function_calls, content = OpenAIServing._parse_tool_calls_from_content(
            request=request,
            tokenizer=None,
            enable_auto_tools=False,
            tool_parser_cls=None,
            content=None,
        )

        assert function_calls is not None
        assert len(function_calls) == 0
        assert content is None
