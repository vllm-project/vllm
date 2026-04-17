# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

pytestmark = pytest.mark.skip_global_cleanup


def _build_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 1,
        "</think>": 2,
        "<|action_start|>": 3,
        "<|plugin|>": 4,
        "<|action_end|>": 5,
    }
    return tokenizer


def _build_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={"type": "object", "properties": {}},
            )
        )
    ]


def test_intern_s1_reasoning_parser_registered():
    parser_cls = ReasoningParserManager.get_reasoning_parser("intern-s1")
    assert parser_cls.__name__ == "InternS1ReasoningParser"


def test_intern_s1_reasoning_parser_hoists_action_to_content():
    tokenizer = _build_tokenizer()
    parser = ReasoningParserManager.get_reasoning_parser("intern-s1")(tokenizer)
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "<think>Need tool first.\n"
        '<|action_start|> <|plugin|>\n{"name": "get_weather", '
        '"parameters": {"city": "Tokyo"}}\n<|action_end|>\n'
        "Then continue reasoning.</think>Visible answer.",
        request,
    )

    assert reasoning is not None
    assert "Need tool first." in reasoning
    assert "Then continue reasoning." in reasoning
    assert "<|action_start|>" not in reasoning
    assert content is not None
    assert content.startswith("<|action_start|> <|plugin|>")
    assert content.endswith("Visible answer.")


def test_intern_s1_reasoning_parser_keeps_deepseek_r1_behavior_without_action():
    tokenizer = _build_tokenizer()
    parser = ReasoningParserManager.get_reasoning_parser("intern-s1")(tokenizer)
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "<think>Need to think first.</think>Visible answer.",
        request,
    )

    assert reasoning == "Need to think first."
    assert content == "Visible answer."


def test_intern_s1_reasoning_and_tool_parser_work_together():
    tokenizer = _build_tokenizer()
    tools = _build_tools()
    request = ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=tools,
        tool_choice="auto",
    )

    reasoning_parser = ReasoningParserManager.get_reasoning_parser("intern-s1")(
        tokenizer
    )
    tool_parser = ToolParserManager.get_tool_parser("intern-s1")(tokenizer, tools)

    reasoning, intermediate_content = reasoning_parser.extract_reasoning(
        "<think>Need tool first.\n"
        '<|action_start|> <|plugin|>\n{"name": "get_weather", '
        '"parameters": {"city": "Tokyo"}}\n<|action_end|>\n'
        "Then continue reasoning.</think>Visible answer.",
        request,
    )
    extracted = tool_parser.extract_tool_calls(intermediate_content or "", request)

    assert reasoning is not None
    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert extracted.content == "Visible answer."
