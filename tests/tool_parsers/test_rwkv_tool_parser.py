# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers import ToolParser, ToolParserManager


class FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


def _tool(name: str) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "description": f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )


def _request(
    tools: list[ChatCompletionToolsParam] | None = None,
) -> ChatCompletionRequest:
    kwargs: dict[str, Any] = {}
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return ChatCompletionRequest(
        model="rwkv-test",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


def _parser(tools: list[ChatCompletionToolsParam] | None = None) -> ToolParser:
    return ToolParserManager.get_tool_parser("rwkv")(FakeTokenizer(), tools=tools)


def _tool_call(name: str, arguments: dict[str, Any]) -> str:
    return (
        "**Tool Call:**\n"
        "```json\n"
        f"{json.dumps({'name': name, 'arguments': arguments}, indent=2)}\n"
        "```"
    )


def _stream(
    parser: ToolParser,
    text: str,
    request: ChatCompletionRequest,
    chunk_size: int = 7,
) -> list[DeltaMessage]:
    previous_text = ""
    deltas: list[DeltaMessage] = []
    for idx in range(0, len(text), chunk_size):
        delta_text = text[idx : idx + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def _content_and_tool_deltas(
    deltas: list[DeltaMessage],
) -> tuple[str, list[Any]]:
    return (
        "".join(delta.content or "" for delta in deltas),
        [tool_call for delta in deltas for tool_call in (delta.tool_calls or [])],
    )


def test_rwkv_tool_parser_keeps_non_tool_output_as_content() -> None:
    text = "There is no tool call here."
    result = _parser().extract_tool_calls(text, _request())

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == text


def test_rwkv_tool_parser_extracts_markdown_tool_call() -> None:
    tools = [_tool("get_weather")]
    text = _tool_call("get_weather", {"city": "Paris"})

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert result.content is None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.type == "function"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_rwkv_tool_parser_extracts_parallel_markdown_tool_calls() -> None:
    tools = [_tool("get_weather"), _tool("get_forecast")]
    text = "\n".join(
        [
            _tool_call("get_weather", {"city": "Paris"}),
            _tool_call("get_forecast", {"city": "Berlin"}),
        ]
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert [call.function.name for call in result.tool_calls] == [
        "get_weather",
        "get_forecast",
    ]
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Berlin"}


def test_rwkv_tool_parser_extracts_standalone_json_tool_call() -> None:
    tools = [_tool("get_weather")]
    text = (
        "I should check the weather.</think>\n"
        "```json\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": "{\\"city\\": \\"Paris\\"}",\n'
        '  "id": "call_123"\n'
        "}\n"
        "```\n"
        "```json\n"
        '{"name": "unknown_weather", "arguments": {"city": "Berlin"}}\n'
        "```"
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert result.content == "I should check the weather.</think>\n"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_rwkv_tool_parser_extracts_legacy_xml_tool_call() -> None:
    tools = [_tool("read")]
    text = (
        "Need to inspect the file.\n"
        "<tool_call>\n"
        "{'arguments': {'action': 'view_file', 'industryType': 'code', "
        "'filePath': 'calculator.py'}, 'name': 'read', 'type': 'tool_call'}\n"
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert result.content == "Need to inspect the file.\n"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "read"
    assert json.loads(call.function.arguments) == {"path": "calculator.py"}


def test_rwkv_tool_parser_normalizes_legacy_edit_tool_call() -> None:
    tools = [_tool("edit")]
    text = (
        "<tool_call>\n"
        "{'arguments': {'action': 'replace_text', 'filePath': 'calculator.py', "
        "'oldText': 'return a - b', 'newText': 'return a + b'}, "
        "'name': 'edit', 'type': 'tool_call'}\n"
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "edit"
    assert json.loads(call.function.arguments) == {
        "path": "calculator.py",
        "edits": [{"oldText": "return a - b", "newText": "return a + b"}],
    }


def test_rwkv_tool_parser_normalizes_nested_legacy_edit_path() -> None:
    tools = [_tool("edit")]
    text = (
        "<tool_call>\n"
        "{'arguments': {'edits': [{'path': 'calculator.py', "
        "'oldText': 'return a - b', 'newText': 'return a + b'}]}, "
        "'name': 'edit', 'type': 'tool_call'}\n"
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "edit"
    assert json.loads(call.function.arguments) == {
        "path": "calculator.py",
        "edits": [{"oldText": "return a - b", "newText": "return a + b"}],
    }


def test_rwkv_tool_parser_keeps_bash_fence_as_content() -> None:
    tools = [_tool("bash")]
    text = (
        "You can run this command manually:\n"
        "```bash\n"
        "perl -0pi -e 's/return a - b/return a + b/' /tmp/calculator.py\n"
        "```\n"
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == text


def test_rwkv_tool_parser_extracts_bash_json_tool_call() -> None:
    tools = [_tool("bash")]
    text = _tool_call("bash", {"command": "python /tmp/calculator.py"})

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "bash"
    assert json.loads(call.function.arguments) == {
        "command": "python /tmp/calculator.py"
    }


def test_rwkv_tool_parser_rejects_malformed_tool_call() -> None:
    text = (
        "**Tool Call:**\n"
        "```json\n"
        '{ "name": "get_weather", "arguments": {"city": "Paris" }\n'
        "```"
    )

    result = _parser([_tool("get_weather")]).extract_tool_calls(
        text, _request([_tool("get_weather")])
    )

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == text


def test_rwkv_tool_parser_rejects_unknown_tool_call() -> None:
    tools = [_tool("get_weather")]
    text = _tool_call("unknown_weather", {"city": "Paris"})

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == text


def test_rwkv_tool_parser_streams_tool_call_without_leaking_marker() -> None:
    tools = [_tool("get_weather")]
    text = "Checking.\n" + _tool_call("get_weather", {"city": "Paris"})
    parser = _parser(tools)

    deltas = _stream(parser, text, _request(tools), chunk_size=5)

    content, tool_deltas = _content_and_tool_deltas(deltas)
    assert content == "Checking.\n"
    assert tool_deltas[0].function is not None
    assert tool_deltas[0].function.name == "get_weather"
    streamed_arguments = "".join(
        tool_delta.function.arguments or ""
        for tool_delta in tool_deltas
        if tool_delta.function is not None
    )
    assert json.loads(streamed_arguments) == {"city": "Paris"}


def test_rwkv_tool_parser_streams_standalone_json_tool_call() -> None:
    tools = [_tool("get_weather")]
    text = (
        "Checking.\n"
        "```json\n"
        '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}\n'
        "```"
    )
    parser = _parser(tools)

    deltas = _stream(parser, text, _request(tools), chunk_size=5)

    content, tool_deltas = _content_and_tool_deltas(deltas)
    assert content == "Checking.\n"
    assert len(tool_deltas) == 1
    assert tool_deltas[0].function is not None
    assert tool_deltas[0].function.name == "get_weather"
    assert json.loads(tool_deltas[0].function.arguments or "{}") == {"city": "Paris"}


def test_rwkv_tool_parser_streams_legacy_xml_tool_call() -> None:
    tools = [_tool("read")]
    text = (
        "Need to inspect the file.\n"
        "<tool_call>\n"
        "{'arguments': {'filePath': 'calculator.py'}, "
        "'name': 'read', 'type': 'tool_call'}\n"
    )
    parser = _parser(tools)

    deltas = _stream(parser, text, _request(tools), chunk_size=5)

    content, tool_deltas = _content_and_tool_deltas(deltas)
    assert content == "Need to inspect the file.\n"
    assert len(tool_deltas) == 1
    assert tool_deltas[0].function is not None
    assert tool_deltas[0].function.name == "read"
    assert json.loads(tool_deltas[0].function.arguments or "{}") == {
        "path": "calculator.py"
    }


def test_rwkv_tool_parser_streams_bash_fence_as_content() -> None:
    tools = [_tool("bash")]
    text = "Running it.\n```bash\npython /tmp/calculator.py\n```\n"
    parser = _parser(tools)

    deltas = _stream(parser, text, _request(tools), chunk_size=4)

    content, tool_deltas = _content_and_tool_deltas(deltas)
    assert content == text
    assert tool_deltas == []
