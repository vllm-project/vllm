# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tool_choice='none' must skip tool extraction and keep original model text.

Regression for https://github.com/vllm-project/vllm/issues/55080
"""

from __future__ import annotations

import json

import pytest

from tests.parser.engine.replay_harness import MockTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

TEXT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'

ENGINE_PARSERS = ("glm45", "qwen3_xml")

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

_QWEN3_VOCAB = {
    "<tool_call>": 100,
    "</tool_call>": 101,
}

GLM45_AUTO_TEXT = """I'll check it. <tool_call>get_current_weather
<arg_key>city</arg_key>
<arg_value>Dallas</arg_value>
</tool_call>"""

QWEN3_XML_AUTO_TEXT = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>Tokyo</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


class _TokenizerStub:
    """Tokenizer stub: the defect is in parser dispatch, not tokenization."""

    def get_vocab(self) -> dict[str, int]:
        return {}

    @property
    def vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, **kwargs) -> list[int]:
        return [ord(c) % 1000 for c in text]

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, tokens) -> int:
        return 90000

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(i) for i in ids]

    @property
    def all_special_tokens(self) -> list[str]:
        return []

    def __len__(self) -> int:
        return 100000


def _none_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="none",
    )


def _auto_request(tool_name: str) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )


def _make_parser(parser_name: str, tokenizer=None):
    parser_cls = ParserManager.get_parser(
        tool_parser_name=parser_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls(tokenizer if tokenizer is not None else _TokenizerStub())


@pytest.mark.parametrize("parser_name", ENGINE_PARSERS)
def test_extract_tool_calls_preserves_content_when_tool_choice_none(parser_name):
    parser = _make_parser(parser_name)
    request = _none_request()

    calls, content = parser._extract_tool_calls(TEXT, request)

    assert calls == []
    assert content == TEXT


@pytest.mark.parametrize("parser_name", ENGINE_PARSERS)
def test_parse_preserves_content_when_tool_choice_none(parser_name):
    parser = _make_parser(parser_name)
    request = _none_request()

    reasoning, content, calls = parser.parse(TEXT, request, enable_auto_tools=True)

    assert reasoning is None
    assert calls == []
    assert content == TEXT


@pytest.mark.parametrize("parser_name", ENGINE_PARSERS)
def test_streaming_preserves_delta_when_tool_choice_none(parser_name):
    parser = _make_parser(parser_name)
    request = _none_request()
    token_ids = parser.model_tokenizer.encode(TEXT)

    delta, function_name_returned = parser._extract_tool_calls_streaming(
        previous_text="",
        current_text=TEXT,
        delta_text=TEXT,
        previous_token_ids=[],
        current_token_ids=token_ids,
        delta_token_ids=token_ids,
        request=request,
    )

    assert function_name_returned is False
    assert delta is not None
    assert delta.content == TEXT
    assert not delta.tool_calls


@pytest.mark.parametrize("parser_name", ENGINE_PARSERS)
def test_streaming_empty_delta_is_none_when_tool_choice_none(parser_name):
    parser = _make_parser(parser_name)
    request = _none_request()

    delta, _ = parser._extract_tool_calls_streaming(
        previous_text="",
        current_text="",
        delta_text="",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert delta is None


def test_glm45_auto_still_extracts_tool_calls():
    parser = _make_parser("glm45", MockTokenizer(vocab=_GLM_VOCAB, tokens=[]))
    request = _auto_request("get_current_weather")

    calls, content = parser._extract_tool_calls(
        GLM45_AUTO_TEXT, request, enable_auto_tools=True
    )

    assert calls
    assert calls[0].name == "get_current_weather"
    assert json.loads(calls[0].arguments) == {"city": "Dallas"}
    assert content == "I'll check it."


def test_qwen3_xml_auto_still_extracts_tool_calls():
    parser = _make_parser("qwen3_xml", MockTokenizer(vocab=_QWEN3_VOCAB, tokens=[]))
    request = _auto_request("get_weather")

    calls, content = parser._extract_tool_calls(
        QWEN3_XML_AUTO_TEXT, request, enable_auto_tools=True
    )

    assert calls
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Tokyo"}
    assert not content or content.strip() == ""
