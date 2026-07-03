# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility tests for GLM-4.5 using the shared GLM XML parser."""

import json
from typing import Any, TypedDict

from tests.parser.engine.replay_harness import MockTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.5"

_GLM_VOCAB = {
    "<think>": 50,
    "</think>": 51,
    "<tool_call>": 60,
    "</tool_call>": 61,
    "<arg_key>": 62,
    "</arg_key>": 63,
    "<arg_value>": 64,
    "</arg_value>": 65,
}


class _CollectedToolDelta(TypedDict):
    name: str | None
    args_fragments: list[str]


def _mock_tokenizer() -> MockTokenizer:
    return MockTokenizer(vocab=_GLM_VOCAB, tokens=[])


def _tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_current_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                },
            ),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="calculate",
                parameters={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                        "enabled": {"type": "boolean"},
                    },
                },
            ),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_time", parameters={}),
        ),
    ]


def _request(tools: list[ChatCompletionToolsParam]) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=MODEL, messages=[], tools=tools)


def _parser(tools: list[ChatCompletionToolsParam] | None = None):
    return Glm47MoeModelToolParser(_mock_tokenizer(), tools=tools)


def _collect_tool_deltas(deltas: Any) -> dict[int, _CollectedToolDelta]:
    calls: dict[int, _CollectedToolDelta] = {}
    for delta in deltas:
        if delta is None or not delta.tool_calls:
            continue
        for tool_call in delta.tool_calls:
            entry = calls.setdefault(
                tool_call.index,
                {"name": None, "args_fragments": []},
            )
            function = tool_call.function
            if function is None:
                continue
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments")
            else:
                name = function.name
                arguments = function.arguments
            if isinstance(name, str) and name:
                entry["name"] = name
            if isinstance(arguments, str) and arguments:
                entry["args_fragments"].append(arguments)
    return calls


def test_glm45_uses_shared_glm47_parser():
    assert ToolParserManager.get_tool_parser("glm45") is Glm47MoeModelToolParser
    assert ToolParserManager.get_tool_parser("glm47") is Glm47MoeModelToolParser


def test_glm_required_tool_choice_skips_generic_json_schema():
    tools = _tools()
    parser = _parser(tools)
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=tools,
        tool_choice="required",
    )

    parser.adjust_request(request)

    assert request.structured_outputs is None
    assert request.skip_special_tokens is False


def test_glm_required_tool_choice_skips_responses_json_schema():
    tools = [
        {
            "type": "function",
            "name": "get_current_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    parser = _parser()
    request = ResponsesRequest(
        model=MODEL,
        input=[{"role": "user", "content": "What is the weather in Beijing?"}],
        tools=tools,
        tool_choice="required",
        stream=True,
    )

    parser.adjust_request(request)

    assert request.text is None
    assert request.skip_special_tokens is False


def test_extract_tool_calls_with_glm45_newline_format():
    tools = _tools()
    parser = _parser(tools)
    model_output = """I'll check it. <tool_call>get_current_weather
<arg_key>city</arg_key>
<arg_value>Dallas</arg_value>
<arg_key>state</arg_key>
<arg_value>TX</arg_value>
<arg_key>unit</arg_key>
<arg_value>fahrenheit</arg_value>
</tool_call>"""

    extracted = parser.extract_tool_calls(model_output, request=_request(tools))

    assert extracted.tools_called
    assert extracted.content == "I'll check it."
    assert len(extracted.tool_calls) == 1
    tool_call = extracted.tool_calls[0]
    assert tool_call.function.name == "get_current_weather"
    assert json.loads(tool_call.function.arguments) == {
        "city": "Dallas",
        "state": "TX",
        "unit": "fahrenheit",
    }


def test_extract_multiple_tool_calls_with_glm45_newline_format():
    tools = _tools()
    parser = _parser(tools)
    model_output = """<tool_call>get_current_weather
<arg_key>city</arg_key><arg_value>Dallas</arg_value>
</tool_call>
<tool_call>get_current_weather
<arg_key>city</arg_key><arg_value>Orlando</arg_value>
</tool_call>"""

    extracted = parser.extract_tool_calls(model_output, request=_request(tools))

    assert extracted.tools_called
    assert [tc.function.name for tc in extracted.tool_calls] == [
        "get_current_weather",
        "get_current_weather",
    ]
    assert [
        json.loads(tc.function.arguments)["city"] for tc in extracted.tool_calls
    ] == ["Dallas", "Orlando"]


def test_extract_tool_calls_coerces_schema_types():
    tools = _tools()
    parser = _parser(tools)
    model_output = """<tool_call>calculate
<arg_key>operation</arg_key><arg_value>add</arg_value>
<arg_key>a</arg_key><arg_value>42</arg_value>
<arg_key>b</arg_key><arg_value>3.14</arg_value>
<arg_key>enabled</arg_key><arg_value>true</arg_value>
</tool_call>"""

    extracted = parser.extract_tool_calls(model_output, request=_request(tools))

    assert extracted.tools_called
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "operation": "add",
        "a": 42,
        "b": 3.14,
        "enabled": True,
    }


def test_extract_zero_argument_tool_call_with_glm45_newline_format():
    tools = _tools()
    parser = _parser(tools)

    extracted = parser.extract_tool_calls(
        "<tool_call>get_time\n</tool_call>",
        request=_request(tools),
    )

    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_time"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {}


def test_streaming_tool_call_with_glm45_newline_format():
    tools = _tools()
    parser = _parser(tools)
    request = _request(tools)
    chunks = [
        "<tool_call>",
        "get_current_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Bei",
        "jing</arg_value>",
        "</tool_call>",
    ]
    deltas = []
    current_text = ""

    for chunk in chunks:
        current_text += chunk
        deltas.append(
            parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )
        )

    calls = _collect_tool_deltas(deltas)
    assert calls[0]["name"] == "get_current_weather"
    assert json.loads("".join(calls[0]["args_fragments"])) == {"city": "Beijing"}
