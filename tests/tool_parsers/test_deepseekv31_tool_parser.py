# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.deepseekv31_tool_parser import (
    DeepSeekV31ToolParser,
)

MODEL = "deepseek-ai/DeepSeek-V3.1"


@pytest.fixture(scope="module")
def deepseekv31_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def parser(deepseekv31_tokenizer):
    return DeepSeekV31ToolParser(deepseekv31_tokenizer)


def test_extract_tool_calls_with_tool(parser):
    model_output = (
        "normal text"
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "foo"
    assert result.tool_calls[0].function.arguments == '{"x":1}'
    assert result.content == "normal text"


def test_extract_tool_calls_with_multiple_tools(parser):
    model_output = (
        "some prefix text"
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
        " some suffix text"
    )

    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called
    assert len(result.tool_calls) == 2

    assert result.tool_calls[0].function.name == "foo"
    assert result.tool_calls[0].function.arguments == '{"x":1}'

    assert result.tool_calls[1].function.name == "bar"
    assert result.tool_calls[1].function.arguments == '{"y":2}'

    # prefix is content
    assert result.content == "some prefix text"


_TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
_TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_TOOL_CALL_END = "<｜tool▁call▁end｜>"
_TOOL_SEP = "<｜tool▁sep｜>"


def _fake_deepseek_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        _TOOL_CALLS_BEGIN: 1,
        _TOOL_CALLS_END: 2,
        _TOOL_CALL_BEGIN: 3,
        _TOOL_CALL_END: 4,
    }
    return tokenizer


def _timeout_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test-model")


def test_regex_timeout_treated_as_no_tool_call():
    parser = DeepSeekV31ToolParser(_fake_deepseek_tokenizer())
    model_output = f"{_TOOL_CALLS_BEGIN}{_TOOL_CALL_BEGIN}incomplete"
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(parser, "tool_call_regex", mock_regex):
        result = parser.extract_tool_calls(model_output, _timeout_request())

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output
    mock_regex.findall.assert_called_once()
    assert (
        mock_regex.findall.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )


def test_streaming_portion_regex_timeout_skips_delta():
    parser = DeepSeekV31ToolParser(_fake_deepseek_tokenizer())
    mock_regex = MagicMock()
    mock_regex.match.side_effect = TimeoutError("Regex timeout")
    current_text = f"{_TOOL_CALLS_BEGIN}{_TOOL_CALL_BEGIN}foo{_TOOL_SEP}" + '{"x":1}'
    token_ids = [1, 3]

    with patch.object(parser, "stream_tool_call_portion_regex", mock_regex):
        result = parser.extract_tool_calls_streaming(
            current_text,
            current_text,
            "",
            token_ids,
            token_ids,
            [],
            _timeout_request(),
        )

    assert result is None
    mock_regex.match.assert_called_once()
    assert (
        mock_regex.match.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )


def test_streaming_name_regex_timeout_skips_delta():
    parser = DeepSeekV31ToolParser(_fake_deepseek_tokenizer())
    portion_regex = MagicMock()
    portion_regex.match.return_value = None
    name_regex = MagicMock()
    name_regex.match.side_effect = TimeoutError("Regex timeout")
    current_text = f"{_TOOL_CALLS_BEGIN}{_TOOL_CALL_BEGIN}foo{_TOOL_SEP}"
    token_ids = [1, 3]

    with (
        patch.object(parser, "stream_tool_call_portion_regex", portion_regex),
        patch.object(parser, "stream_tool_call_name_regex", name_regex),
    ):
        result = parser.extract_tool_calls_streaming(
            current_text,
            current_text,
            "",
            token_ids,
            token_ids,
            [],
            _timeout_request(),
        )

    assert result is None
    name_regex.match.assert_called_once()
    assert (
        name_regex.match.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )
