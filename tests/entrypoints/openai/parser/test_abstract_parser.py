# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import DelegatingParser, _WrappedParser
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

MODEL_NAME = "fake-model"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    return tokenizer


def _make_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        **kwargs,
    )


def test_extract_chat_completion_parts_reasoning_only():
    parser = DelegatingParser(_make_tokenizer())
    reasoning_parser = MagicMock()
    tool_parser = MagicMock()
    reasoning_parser.extract_reasoning.return_value = ("thinking", "final answer")
    parser.reasoning_parser = reasoning_parser
    parser.tool_parser = tool_parser

    reasoning, tool_calls, content = parser.extract_chat_completion_parts(
        model_output="ignored",
        request=_make_request(),
    )

    assert reasoning == "thinking"
    assert tool_calls == []
    assert content == "final answer"
    tool_parser.extract_tool_calls.assert_not_called()


def test_extract_chat_completion_parts_named_tool_choice():
    parser = DelegatingParser(_make_tokenizer())
    reasoning_parser = MagicMock()
    reasoning_parser.extract_reasoning.return_value = (None, '{"location": "Rome"}')
    parser.reasoning_parser = reasoning_parser

    reasoning, tool_calls, content = parser.extract_chat_completion_parts(
        model_output="ignored",
        request=_make_request(
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        ),
    )

    assert reasoning is None
    assert tool_calls == [
        FunctionCall(name="get_weather", arguments='{"location": "Rome"}')
    ]
    assert content is None


def test_extract_chat_completion_parts_required_tool_choice():
    parser = DelegatingParser(_make_tokenizer())
    reasoning_parser = MagicMock()
    reasoning_parser.extract_reasoning.return_value = (
        None,
        '[{"name":"get_weather","parameters":{"location":"Rome"}}]',
    )
    parser.reasoning_parser = reasoning_parser

    reasoning, tool_calls, content = parser.extract_chat_completion_parts(
        model_output="ignored",
        request=_make_request(tools=TOOLS, tool_choice="required"),
    )

    assert reasoning is None
    assert tool_calls == [
        FunctionCall(name="get_weather", arguments='{"location": "Rome"}')
    ]
    assert content is None


def test_extract_chat_completion_parts_auto_tool_choice():
    parser = DelegatingParser(_make_tokenizer())
    reasoning_parser = MagicMock()
    tool_parser = MagicMock()
    reasoning_parser.extract_reasoning.return_value = ("thinking", "raw tool output")
    tool_parser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=[
            ToolCall(
                id="call_auto",
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"location": "Rome"}',
                ),
            )
        ],
        content="   ",
    )
    parser.reasoning_parser = reasoning_parser
    parser.tool_parser = tool_parser

    reasoning, tool_calls, content = parser.extract_chat_completion_parts(
        model_output="ignored",
        request=_make_request(tools=TOOLS, tool_choice="auto"),
        enable_auto_tools=True,
    )

    assert reasoning == "thinking"
    assert tool_calls == [
        FunctionCall(
            id="call_auto",
            name="get_weather",
            arguments='{"location": "Rome"}',
        )
    ]
    assert content is None
    tool_parser.extract_tool_calls.assert_called_once()


def test_get_parser_wrapped_parser_forwards_chat_template_kwargs(monkeypatch):
    seen_kwargs: dict[str, object] = {}

    class DummyReasoningParser(ReasoningParser):
        def __init__(self, tokenizer, *args, **kwargs):
            super().__init__(tokenizer, *args, **kwargs)
            seen_kwargs["reasoning"] = kwargs

        def is_reasoning_end(self, input_ids):
            return False

        def extract_content_ids(self, input_ids):
            return input_ids

        def extract_reasoning(self, model_output, request):
            return None, model_output

        def extract_reasoning_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        ):
            return None

    class DummyToolParser(ToolParser):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            seen_kwargs["tool"] = "constructed"

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    monkeypatch.setattr(
        ParserManager,
        "get_parser_internal",
        classmethod(lambda cls, name: (_ for _ in ()).throw(KeyError(name))),
    )
    monkeypatch.setattr(
        ParserManager,
        "get_reasoning_parser",
        classmethod(lambda cls, reasoning_parser_name: DummyReasoningParser),
    )
    monkeypatch.setattr(
        ParserManager,
        "get_tool_parser",
        classmethod(
            lambda cls, tool_parser_name, enable_auto_tools, model_name: DummyToolParser
        ),
    )

    parser_cls = ParserManager.get_parser(
        tool_parser_name="dummy_tool",
        reasoning_parser_name="dummy_reasoning",
        enable_auto_tools=True,
        model_name=MODEL_NAME,
    )

    assert parser_cls is not None
    parser = parser_cls(_make_tokenizer(), chat_template_kwargs={"foo": "bar"})

    assert seen_kwargs["reasoning"] == {"chat_template_kwargs": {"foo": "bar"}}
    assert seen_kwargs["tool"] == "constructed"
    assert parser.reasoning_parser is not None
    assert parser.tool_parser is not None


def test_minimax_m2_parser_forwards_chat_template_kwargs(monkeypatch):
    from vllm.parser import minimax_m2_parser as minimax_m2_parser_module

    seen_kwargs: dict[str, object] = {}

    class DummyReasoningParser(ReasoningParser):
        def __init__(self, tokenizer, *args, **kwargs):
            super().__init__(tokenizer, *args, **kwargs)
            seen_kwargs["reasoning"] = kwargs

        def is_reasoning_end(self, input_ids):
            return False

        def extract_content_ids(self, input_ids):
            return input_ids

        def extract_reasoning(self, model_output, request):
            return None, model_output

        def extract_reasoning_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        ):
            return None

    class DummyToolParser(ToolParser):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            seen_kwargs["tool"] = "constructed"

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    monkeypatch.setattr(
        minimax_m2_parser_module,
        "MiniMaxM2ReasoningParser",
        DummyReasoningParser,
    )
    monkeypatch.setattr(
        minimax_m2_parser_module,
        "MinimaxM2ToolParser",
        DummyToolParser,
    )

    parser = minimax_m2_parser_module.MiniMaxM2Parser(
        _make_tokenizer(),
        chat_template_kwargs={"foo": "bar"},
    )

    assert seen_kwargs["reasoning"] == {"chat_template_kwargs": {"foo": "bar"}}
    assert seen_kwargs["tool"] == "constructed"
    assert parser.reasoning_parser is not None
    assert parser.tool_parser is not None


def test_get_parser_preserves_distinct_reasoning_and_tool_parser_selection():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="minimax_m2",
        reasoning_parser_name="minimax_m2_append_think",
        enable_auto_tools=True,
        model_name=MODEL_NAME,
    )

    assert parser_cls is _WrappedParser
    assert parser_cls.reasoning_parser_cls is ParserManager.get_reasoning_parser(
        "minimax_m2_append_think"
    )
    assert parser_cls.tool_parser_cls is ParserManager.get_tool_parser(
        "minimax_m2",
        enable_auto_tools=True,
        model_name=MODEL_NAME,
    )


def test_get_parser_tool_only_selection_preserves_reasoning_none():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="minimax_m2",
        reasoning_parser_name=None,
        enable_auto_tools=True,
        model_name=MODEL_NAME,
    )

    assert parser_cls is _WrappedParser
    assert parser_cls.reasoning_parser_cls is None
    assert parser_cls.tool_parser_cls is ParserManager.get_tool_parser(
        "minimax_m2",
        enable_auto_tools=True,
        model_name=MODEL_NAME,
    )


def test_get_parser_reasoning_only_selection_preserves_tool_none():
    parser_cls = ParserManager.get_parser(
        tool_parser_name=None,
        reasoning_parser_name="minimax_m2",
        enable_auto_tools=True,
        model_name=MODEL_NAME,
    )

    assert parser_cls is _WrappedParser
    assert parser_cls.reasoning_parser_cls is ParserManager.get_reasoning_parser(
        "minimax_m2"
    )
    assert parser_cls.tool_parser_cls is None
