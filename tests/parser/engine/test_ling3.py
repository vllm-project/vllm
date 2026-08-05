# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from tests.parser.engine.replay_harness import MockTokenizer, _test_request
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager


def _tokenizer() -> MockTokenizer:
    vocab = {
        "<think>": 1,
        "</think>": 2,
        "<tool_call>": 3,
        "</tool_call>": 4,
        "<arg_key>": 5,
        "</arg_key>": 6,
        "<arg_value>": 7,
        "</arg_value>": 8,
    }
    return MockTokenizer(vocab=vocab, tokens=[])


def test_ling3_registered():
    assert ReasoningParserManager.get_reasoning_parser("ling3") is not None
    assert ToolParserManager.get_tool_parser("ling3") is not None


def test_ling3_defaults_thinking_on():
    parser_cls = ReasoningParserManager.get_reasoning_parser("ling3")
    parser = parser_cls(_tokenizer())

    assert parser.reasoning_start_str == "<think>"
    assert parser.reasoning_end_str == "</think>"

    reasoning, content = parser.extract_reasoning(
        "<think>reason</think>answer", _test_request()
    )

    assert reasoning == "reason"
    assert content == "answer"


def test_ling3_disable_thinking_keeps_reasoning_as_content():
    parser_cls = ReasoningParserManager.get_reasoning_parser("ling3")
    parser = parser_cls(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = parser.extract_reasoning(
        "<think>reason</think>answer", _test_request()
    )

    assert reasoning is None
    assert content == "<think>reason</think>answer"


def test_ling3_enable_thinking_keeps_open_reasoning_as_content():
    parser_cls = ReasoningParserManager.get_reasoning_parser("ling3")
    parser = parser_cls(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning, content = parser.extract_reasoning(
        "<think>only reasoning", _test_request()
    )

    assert reasoning is None
    assert content == "only reasoning"


def test_ling3_tool_call_without_newline():
    parser_cls = ToolParserManager.get_tool_parser("ling3")
    parser = parser_cls(_tokenizer())

    result = parser.extract_tool_calls(
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>",
        request=_test_request(),
    )

    assert result.tools_called
    assert result.content is None
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "city": "Beijing",
    }


def test_ling3_tool_call_without_args():
    parser_cls = ToolParserManager.get_tool_parser("ling3")
    parser = parser_cls(_tokenizer())

    result = parser.extract_tool_calls(
        "<tool_call>ping</tool_call>",
        request=_test_request(),
    )

    assert result.tools_called
    assert result.content is None
    assert result.tool_calls[0].function.name == "ping"
    assert json.loads(result.tool_calls[0].function.arguments) == {}
