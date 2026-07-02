# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from types import SimpleNamespace

import pytest

_STRICT_TOOL_CALLING_ENV = "VLLM_ENFORCE_STRICT_TOOL_CALLING"
_STRICT_TOOL_CALLING_ENV_VALUE = os.environ.get(_STRICT_TOOL_CALLING_ENV)
os.environ[_STRICT_TOOL_CALLING_ENV] = "0"

from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest  # noqa: E402
from vllm.parser.abstract_parser import DelegatingParser  # noqa: E402
from vllm.parser.utils import count_history_tool_calls  # noqa: E402
from vllm.reasoning.basic_parsers import (  # noqa: E402
    BaseThinkingReasoningParser,
)
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def restore_strict_tool_calling_env():
    yield
    if _STRICT_TOOL_CALLING_ENV_VALUE is None:
        os.environ.pop(_STRICT_TOOL_CALLING_ENV, None)
    else:
        os.environ[_STRICT_TOOL_CALLING_ENV] = _STRICT_TOOL_CALLING_ENV_VALUE


class ThinkReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"


MODEL_OUTPUT = (
    "<think>let me think about this</think>"
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)

PLAIN_TEXT = "The weather in Dallas is sunny and 75°F."

TOOL_CALL_ONLY = (
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)

TOOL_ARGUMENTS = '{"city": "Dallas"}'


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


def make_request(**overrides):
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest.model_validate(base)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


KIMI_K2_MODEL_CONFIG = SimpleNamespace(
    hf_text_config=SimpleNamespace(model_type="kimi_k2"),
    hf_overrides=None,
)

HISTORY_MESSAGES = [
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "functions.get_current_weather:0",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": "{}",
                },
            },
            {
                "id": "functions.get_forecast:1",
                "type": "function",
                "function": {
                    "name": "get_forecast",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "functions.get_current_weather:0",
        "content": "{}",
    },
    {
        "role": "tool",
        "tool_call_id": "functions.get_forecast:1",
        "content": "{}",
    },
    {"role": "user", "content": "again"},
]


def make_parser(tokenizer, reasoning=False, tool=False, **kwargs):
    class TestParser(DelegatingParser):
        reasoning_parser_cls = ThinkReasoningParser if reasoning else None
        tool_parser_cls = Hermes2ProToolParser if tool else None

    return TestParser(tokenizer, **kwargs)


@pytest.mark.parametrize(
    "reasoning,tool",
    [(False, False), (False, True)],
    ids=["neither", "tool-only"],
)
def test_parse_plain_text_no_reasoning_parser(tokenizer, reasoning, tool):
    parser = make_parser(tokenizer, reasoning=reasoning, tool=tool)
    request = make_request()
    r, content, tool_calls = parser.parse(PLAIN_TEXT, request)

    assert r is None
    assert content == PLAIN_TEXT
    assert tool_calls is not None
    assert len(tool_calls) == 0


@pytest.mark.parametrize(
    "reasoning,tool",
    [(True, False), (True, True)],
    ids=["reasoning-only", "both"],
)
def test_parse_plain_text_with_reasoning_parser(tokenizer, reasoning, tool):
    parser = make_parser(tokenizer, reasoning=reasoning, tool=tool)
    request = make_request()
    r, content, tool_calls = parser.parse(PLAIN_TEXT, request)

    assert r == PLAIN_TEXT
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_both_parsers(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = make_request(tools=TOOLS)
    reasoning, content, tool_calls = parser.parse(
        MODEL_OUTPUT, request, enable_auto_tools=True
    )

    assert reasoning is not None
    assert "let me think about this" in reasoning
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
    assert not content or content.strip() == ""


def test_parse_reasoning_only(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = make_request()
    reasoning, content, tool_calls = parser.parse(MODEL_OUTPUT, request)

    assert reasoning is not None
    assert "let me think about this" in reasoning
    assert content is not None
    assert "<tool_call>" in content
    assert "get_weather" in content
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_tool_only(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(tools=TOOLS)
    reasoning, content, tool_calls = parser.parse(
        MODEL_OUTPUT, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}


def test_parse_named_tool_choice(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    reasoning, content, tool_calls = parser.parse(
        TOOL_ARGUMENTS, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == TOOL_ARGUMENTS


def test_parse_named_tool_choice_with_reasoning(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    model_output = f"<think>thinking</think>{TOOL_ARGUMENTS}"
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    reasoning, content, tool_calls = parser.parse(
        model_output, request, enable_auto_tools=True
    )

    assert reasoning is not None
    assert "thinking" in reasoning
    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == TOOL_ARGUMENTS


def test_parse_required_tool_choice(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    functions_json = json.dumps(
        [
            {"name": "get_weather", "parameters": {"city": "Dallas"}},
            {"name": "get_time", "parameters": {"timezone": "UTC"}},
        ]
    )
    request = make_request(tools=TOOLS, tool_choice="required")
    reasoning, content, tool_calls = parser.parse(
        functions_json, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
    assert tool_calls[1].name == "get_time"
    assert json.loads(tool_calls[1].arguments) == {"timezone": "UTC"}


def test_parse_required_tool_choice_kimi_k2_ids(tokenizer):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    functions_json = json.dumps(
        [
            {"name": "get_current_weather", "parameters": {"city": "Dallas"}},
            {"name": "get_forecast", "parameters": {"city": "Dallas", "days": 2}},
        ]
    )
    request = make_request(tools=TOOLS, tool_choice="required")
    _, content, tool_calls = parser.parse(
        functions_json, request, enable_auto_tools=True
    )

    assert content is None
    assert tool_calls is not None
    assert [tc.id for tc in tool_calls] == [
        "functions.get_current_weather:0",
        "functions.get_forecast:1",
    ]


def test_parse_required_tool_choice_kimi_k2_ids_after_history(tokenizer):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    functions_json = json.dumps(
        [{"name": "get_current_weather", "parameters": {"city": "Dallas"}}]
    )
    request = make_request(
        messages=HISTORY_MESSAGES,
        tools=TOOLS,
        tool_choice="required",
    )
    _, _, tool_calls = parser.parse(functions_json, request, enable_auto_tools=True)

    assert tool_calls is not None
    assert tool_calls[0].id == "functions.get_current_weather:2"


def test_count_history_tool_calls_responses_request():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_0",
                    "name": "get_current_weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_forecast",
                    "arguments": "{}",
                },
            ],
        }
    )

    assert count_history_tool_calls(request) == 2


def test_parse_required_tool_choice_random_ids_deferred(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    functions_json = json.dumps(
        [{"name": "get_current_weather", "parameters": {"city": "Dallas"}}]
    )
    request = make_request(
        messages=HISTORY_MESSAGES,
        tools=TOOLS,
        tool_choice="required",
    )
    _, _, tool_calls = parser.parse(functions_json, request, enable_auto_tools=True)

    assert tool_calls is not None
    assert tool_calls[0].id is None


def test_parse_named_tool_choice_kimi_k2_id(tokenizer):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    _, content, tool_calls = parser.parse(
        TOOL_ARGUMENTS, request, enable_auto_tools=True
    )

    assert content is None
    assert tool_calls is not None
    assert tool_calls[0].id == "functions.get_weather:0"


def test_parse_named_tool_choice_content_none(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    reasoning, content, tool_calls = parser.parse("", request, enable_auto_tools=True)
    assert reasoning is None
    assert content is None
    assert tool_calls is not None


def test_parse_required_tool_choice_content_none(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(tools=TOOLS, tool_choice="required")
    reasoning, content, tool_calls = parser.parse("", request, enable_auto_tools=True)
    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_auto_tools_no_parser(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=False)
    request = make_request()
    reasoning, content, tool_calls = parser.parse(
        TOOL_CALL_ONLY, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content == TOOL_CALL_ONLY
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_auto_tools_no_calls_returns_none(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(tools=TOOLS)
    reasoning, content, tool_calls = parser.parse(
        PLAIN_TEXT, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content == PLAIN_TEXT
    assert tool_calls is None
