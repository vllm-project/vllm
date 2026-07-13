# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

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


@pytest.mark.parametrize(
    ("tools", "text"),
    [
        pytest.param(None, "There is no tool call here.", id="plain-text"),
        pytest.param(
            [_tool("bash")],
            "You can run this command manually:\n"
            "```bash\npython /tmp/calculator.py\n```\n",
            id="bash-fence",
        ),
        pytest.param(
            [_tool("get_weather")],
            "**Tool Call:**\n```json\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}\n```',
            id="malformed-json",
        ),
        pytest.param(
            [_tool("get_weather")],
            _tool_call("unknown_weather", {"city": "Paris"}),
            id="unknown-tool",
        ),
    ],
)
def test_non_tool_output_remains_content(
    tools: list[ChatCompletionToolsParam] | None, text: str
) -> None:
    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert (result.tools_called, result.tool_calls, result.content) == (False, [], text)


@pytest.mark.parametrize(
    ("tools", "text", "content", "name", "arguments"),
    [
        pytest.param(
            [_tool("get_weather")],
            _tool_call("get_weather", {"city": "Paris"}),
            None,
            "get_weather",
            {"city": "Paris"},
            id="markdown",
        ),
        pytest.param(
            [_tool("get_weather")],
            "I should check the weather.</think>\n"
            "```json\n"
            '{"name": "get_weather", '
            '"arguments": "{\\"city\\": \\"Paris\\"}"}\n'
            "```\n"
            "```json\n"
            '{"name": "unknown_weather", "arguments": {"city": "Berlin"}}\n'
            "```",
            "I should check the weather.</think>\n",
            "get_weather",
            {"city": "Paris"},
            id="standalone-json",
        ),
        pytest.param(
            [_tool("read")],
            "Need to inspect the file.\n<tool_call>\n"
            "{'arguments': {'action': 'view_file', 'filePath': 'calculator.py'}, "
            "'name': 'read', 'type': 'tool_call'}\n",
            "Need to inspect the file.\n",
            "read",
            {"path": "calculator.py"},
            id="legacy-xml",
        ),
        pytest.param(
            [_tool("bash")],
            _tool_call("bash", {"command": "python /tmp/calculator.py"}),
            None,
            "bash",
            {"command": "python /tmp/calculator.py"},
            id="bash-json",
        ),
    ],
)
def test_extracts_supported_single_tool_call(
    tools: list[ChatCompletionToolsParam],
    text: str,
    content: str | None,
    name: str,
    arguments: dict[str, Any],
) -> None:
    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert result.content == content
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert (call.type, call.function.name) == ("function", name)
    assert json.loads(call.function.arguments) == arguments


def test_extracts_parallel_markdown_tool_calls() -> None:
    tools = [_tool("get_weather"), _tool("get_forecast")]
    text = "\n".join(
        _tool_call(name, {"city": city})
        for name, city in (("get_weather", "Paris"), ("get_forecast", "Berlin"))
    )

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert [
        (call.function.name, json.loads(call.function.arguments))
        for call in result.tool_calls
    ] == [
        ("get_weather", {"city": "Paris"}),
        ("get_forecast", {"city": "Berlin"}),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>\n"
        "{'arguments': {'action': 'replace_text', 'filePath': 'calculator.py', "
        "'oldText': 'return a - b', 'newText': 'return a + b'}, "
        "'name': 'edit', 'type': 'tool_call'}\n",
        "<tool_call>\n"
        "{'arguments': {'edits': [{'path': 'calculator.py', "
        "'oldText': 'return a - b', 'newText': 'return a + b'}]}, "
        "'name': 'edit', 'type': 'tool_call'}\n",
    ],
    ids=["flat-path", "nested-path"],
)
def test_normalizes_legacy_edit_call(text: str) -> None:
    tools = [_tool("edit")]

    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "edit"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "path": "calculator.py",
        "edits": [{"oldText": "return a - b", "newText": "return a + b"}],
    }


@pytest.mark.parametrize(
    ("tools", "text", "content", "name", "arguments"),
    [
        pytest.param(
            [_tool("get_weather")],
            "Checking.\n" + _tool_call("get_weather", {"city": "Paris"}),
            "Checking.\n",
            "get_weather",
            {"city": "Paris"},
            id="markdown",
        ),
        pytest.param(
            [_tool("get_weather")],
            "Checking.\n```json\n"
            '{"name": "get_weather", '
            '"arguments": "{\\"city\\": \\"Paris\\"}"}\n```',
            "Checking.\n",
            "get_weather",
            {"city": "Paris"},
            id="standalone-json",
        ),
        pytest.param(
            [_tool("read")],
            "Need to inspect the file.\n<tool_call>\n"
            "{'arguments': {'filePath': 'calculator.py'}, "
            "'name': 'read', 'type': 'tool_call'}\n",
            "Need to inspect the file.\n",
            "read",
            {"path": "calculator.py"},
            id="legacy-xml",
        ),
    ],
)
def test_streams_supported_tool_calls(
    tools: list[ChatCompletionToolsParam],
    text: str,
    content: str,
    name: str,
    arguments: dict[str, Any],
) -> None:
    deltas = _stream(_parser(tools), text, _request(tools), chunk_size=5)

    streamed_content, tool_deltas = _content_and_tool_deltas(deltas)
    functions = [delta.function for delta in tool_deltas if delta.function is not None]
    assert streamed_content == content
    assert [function.name for function in functions if function.name] == [name]
    assert json.loads("".join(function.arguments or "" for function in functions)) == (
        arguments
    )


def test_streams_bash_fence_as_content() -> None:
    tools = [_tool("bash")]
    text = "Running it.\n```bash\npython /tmp/calculator.py\n```\n"

    content, tool_deltas = _content_and_tool_deltas(
        _stream(_parser(tools), text, _request(tools), chunk_size=4)
    )

    assert (content, tool_deltas) == (text, [])
