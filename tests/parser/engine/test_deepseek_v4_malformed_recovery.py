# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
    DSML_TOOL_END,
    DeepSeekV4Parser,
)


def _tool(name: str = "get_weather") -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )


def _request(*tools: ChatCompletionToolsParam):
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = list(tools)
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req


def _invoke(name: str = "get_weather", city: str = "Seoul") -> str:
    return (
        f"{DSML_INVOKE_PREFIX}{name}{DSML_INVOKE_NAME_END}\n"
        f'<｜DSML｜parameter name="city" string="true">{city}{DSML_PARAM_CLOSE}\n'
        f"{DSML_INVOKE_END}"
    )


def _content_parser(*tools: ChatCompletionToolsParam) -> DeepSeekV4Parser:
    return DeepSeekV4Parser(
        make_mock_tokenizer({}),
        tools=list(tools),
        chat_template_kwargs={"thinking": False},
    )


def test_missing_start_wrapper_recovers_declared_tool():
    """Default thinking mode also recovers an invoke with no outer wrapper."""
    tool = _tool()
    request = _request(tool)
    parser = DeepSeekV4Parser(make_mock_tokenizer({}), tools=[tool])

    result = parser.extract_tool_calls(_invoke() + DSML_TOOL_END, request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}


def test_corrupted_start_wrapper_still_recovers_invoke():
    """Regression for #51914: tool_calls -> toolcalls must not lose the invoke."""
    tool = _tool()
    request = _request(tool)
    parser = DeepSeekV4Parser(make_mock_tokenizer({}), tools=[tool])
    text = "<｜DSML｜toolcalls>\n" + _invoke() + "\n" + DSML_TOOL_END

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}


def test_undeclared_orphan_invoke_stays_content():
    tool = _tool()
    request = _request(tool)
    parser = _content_parser(tool)
    text = _invoke(name="not_declared") + DSML_TOOL_END

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


def test_orphan_invoke_without_tools_stays_content():
    request = _request()
    parser = _content_parser()
    text = _invoke() + DSML_TOOL_END

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


def test_request_without_tools_does_not_reuse_prior_tool_names():
    tool = _tool()
    parser = _content_parser(tool)
    text = _invoke() + DSML_TOOL_END

    first = parser.extract_tool_calls(text, _request(tool))
    second = parser.extract_tool_calls(text, _request())

    assert first.tools_called is True
    assert second.tools_called is False
    assert second.tool_calls == []
    assert second.content == text


def test_tool_choice_none_disables_recovery():
    tool = _tool()
    request = _request(tool)
    request.tool_choice = "none"
    parser = _content_parser(tool)
    text = _invoke() + DSML_TOOL_END

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


def test_truncated_recovery_candidate_flushes_as_content():
    tool = _tool()
    request = _request(tool)
    parser = _content_parser(tool)
    text = "Docs quote " + DSML_INVOKE_PREFIX + "get_wea"

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


def test_valid_name_without_invoke_end_stays_content():
    """A declared name alone is not enough to commit a recovered call."""
    tool = _tool()
    request = _request(tool)
    parser = _content_parser(tool)
    text = (
        f"{DSML_INVOKE_PREFIX}get_weather{DSML_INVOKE_NAME_END}\n"
        '<｜DSML｜parameter name="city" string="true">Seoul</｜DSML｜parameter>'
    )

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


def test_streaming_orphan_invoke_recovers_after_split_marker():
    tool = _tool()
    request = _request(tool)
    parser = _content_parser(tool)
    chunks = [
        "Checking.\n",
        "<｜DSML",
        '｜invoke name="get_weather">',
        '\n<｜DSML｜parameter name="city" string="true">Seoul</｜DSML｜parameter>\n',
        DSML_INVOKE_END,
        DSML_TOOL_END,
    ]

    results = simulate_tool_streaming(parser, request, chunks)

    assert collect_function_name(results) == "get_weather"
    assert json.loads(collect_tool_arguments(results)) == {"city": "Seoul"}
    assert collect_content(results) == "Checking.\n"
