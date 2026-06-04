# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.mistral import MistralParser
from vllm.reasoning.mistral_reasoning_parser import MistralReasoningParser
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall, MistralToolParser

PLAIN_TEXT = "The weather in Dallas is sunny."

# Pre-v11: content[TOOL_CALLS] [{"name": ..., "arguments": ...}]
PRE_V11_MODEL_OUTPUT = (
    "[THINK]let me think[/THINK]"
    "Here is the result"
    '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Dallas"}}]'
)

# v11+: content[TOOL_CALLS]name{args}
V11_MODEL_OUTPUT = (
    "[THINK]let me think[/THINK]"
    "Here is the result"
    '[TOOL_CALLS]get_weather{"city": "Dallas"}'
)

NAMED_TOOL_ARGS = '{"city": "Dallas"}'
REQUIRED_TOOL_OUTPUT = (
    '[{"name": "get_weather", "parameters": {"city": "Dallas"}},'
    ' {"name": "get_time", "parameters": {"timezone": "UTC"}}]'
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def make_request(grammar, **overrides):
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    req = ChatCompletionRequest.model_validate(base)
    if grammar:
        req._grammar_from_tool_parser = True
    return req


@pytest.fixture(scope="module")
def pre_v11_tokenizer():
    return get_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")


@pytest.fixture(scope="module")
def v11_tokenizer():
    return get_tokenizer(
        "mistralai/Magistral-Small-2509",
        tokenizer_mode="mistral",
    )


def make_parser(tokenizer, reasoning_parser):
    class _Parser(MistralParser):
        reasoning_parser_cls = MistralReasoningParser if reasoning_parser else None
        tool_parser_cls = MistralToolParser

    return _Parser(tokenizer)


def assert_mistral_id(tool_call):
    assert tool_call.id is not None
    assert len(tool_call.id) == 9
    assert tool_call.id.isalnum()


class TestParse:
    # -- Non-grammar path (pre-v11) --

    def test_plain_text(self, pre_v11_tokenizer):
        parser = make_parser(pre_v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=False)
        reasoning, content, tool_calls = parser.parse(PLAIN_TEXT, request)

        assert reasoning is None
        assert content == PLAIN_TEXT
        assert tool_calls is not None
        assert len(tool_calls) == 0

    def test_no_reasoning_parser_think_tags_in_content(self, pre_v11_tokenizer):
        parser = make_parser(pre_v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=False, tools=TOOLS)
        reasoning, content, tool_calls = parser.parse(
            PRE_V11_MODEL_OUTPUT, request, enable_auto_tools=True
        )

        assert reasoning is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert_mistral_id(tool_calls[0])
        # Without reasoning parser, [THINK] tags remain in content
        assert content is not None
        assert "[THINK]" in content

    def test_with_reasoning_backfills_ids(self, v11_tokenizer):
        parser = make_parser(v11_tokenizer, reasoning_parser=True)
        request = make_request(grammar=False, tools=TOOLS)
        reasoning, content, tool_calls = parser.parse(
            V11_MODEL_OUTPUT, request, enable_auto_tools=True
        )

        assert reasoning == "let me think"
        assert content is not None
        assert "Here is the result" in content
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert_mistral_id(tool_calls[0])

    def test_auto_tool_choice(self, pre_v11_tokenizer):
        parser = make_parser(pre_v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=False, tools=TOOLS)
        reasoning, content, tool_calls = parser.parse(
            PRE_V11_MODEL_OUTPUT, request, enable_auto_tools=True
        )

        assert reasoning is None
        assert content is not None
        assert "[THINK]" in content
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
        assert_mistral_id(tool_calls[0])

    def test_named_tool_choice(self, pre_v11_tokenizer):
        parser = make_parser(pre_v11_tokenizer, reasoning_parser=False)
        request = make_request(
            grammar=False,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
        )
        reasoning, content, tool_calls = parser.parse(
            NAMED_TOOL_ARGS, request, enable_auto_tools=True
        )

        assert reasoning is None
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == NAMED_TOOL_ARGS
        assert_mistral_id(tool_calls[0])

    def test_required_tool_choice(self, pre_v11_tokenizer):
        parser = make_parser(pre_v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=False, tools=TOOLS, tool_choice="required")
        reasoning, content, tool_calls = parser.parse(
            REQUIRED_TOOL_OUTPUT, request, enable_auto_tools=True
        )

        assert reasoning is None
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "get_weather"
        assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
        assert tool_calls[1].name == "get_time"
        assert json.loads(tool_calls[1].arguments) == {"timezone": "UTC"}
        for tc in tool_calls:
            assert_mistral_id(tc)

    # -- Grammar path (v11+) --

    def test_grammar_tool_calls(self, v11_tokenizer):
        parser = make_parser(v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=True, tools=TOOLS)
        reasoning, content, tool_calls = parser.parse(V11_MODEL_OUTPUT, request)

        assert reasoning is None
        assert content is not None
        assert "[THINK]" in content
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
        assert MistralToolCall.is_valid_id(tool_calls[0].id)

    def test_grammar_with_reasoning(self, v11_tokenizer):
        parser = make_parser(v11_tokenizer, reasoning_parser=True)
        request = make_request(grammar=True, tools=TOOLS)
        reasoning, content, tool_calls = parser.parse(V11_MODEL_OUTPUT, request)

        assert reasoning == "let me think"
        assert content is not None
        assert "Here is the result" in content
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}

    def test_grammar_no_tool_calls(self, v11_tokenizer):
        parser = make_parser(v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=True)
        reasoning, content, tool_calls = parser.parse(PLAIN_TEXT, request)

        assert reasoning is None
        assert content == PLAIN_TEXT
        assert tool_calls is None

    def test_grammar_content_with_tools(self, v11_tokenizer):
        parser = make_parser(v11_tokenizer, reasoning_parser=False)
        request = make_request(grammar=True, tools=TOOLS)
        reasoning, content, tool_calls = parser.parse(V11_MODEL_OUTPUT, request)

        assert reasoning is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        # Content before [TOOL_CALLS] is preserved (includes [THINK] tags
        # since no reasoning parser is active).
        assert content is not None
        assert "[THINK]" in content
        assert "Here is the result" in content
