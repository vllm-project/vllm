# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.entrypoints.openai.api_server import validate_api_server_args
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.run_batch import validate_run_batch_args
from vllm.parser import ParserManager
from vllm.parser.rust_unified_parser import RustUnifiedParser
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.rust_unified_reasoning_parser import (
    RustUnifiedReasoningParser,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.rust_unified_tool_parser import RustUnifiedToolParser

rust_parser = pytest.importorskip("vllm._rust_tool_parser")


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self.all_special_ids = list(vocab.values())

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._vocab[text]]


def request(tool_choice: Any = "auto") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "weather"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice=tool_choice,
        include_reasoning=True,
    )


def parser(
    name: str,
    tokenizer: FakeTokenizer,
) -> RustUnifiedParser:
    parser_cls = ParserManager.get_parser(
        tool_parser_name=name,
        reasoning_parser_name=name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, RustUnifiedParser)
    req = request()
    return parser_cls(tokenizer, req.tools)


def test_registry_generates_capability_wrappers_and_requires_matching_names():
    parser_names = tuple(rust_parser.list_unified_parsers())
    assert parser_names == ("inkling", "minimax_m3")
    reasoning_parser_classes = []
    tool_parser_classes = []
    for name in parser_names:
        reasoning_parser_cls = ReasoningParserManager.reasoning_parsers[name]
        tool_parser_cls = ToolParserManager.tool_parsers[name]
        reasoning_parser_classes.append(reasoning_parser_cls)
        tool_parser_classes.append(tool_parser_cls)

        assert issubclass(reasoning_parser_cls, RustUnifiedReasoningParser)
        assert issubclass(tool_parser_cls, RustUnifiedToolParser)
        assert reasoning_parser_cls.rust_parser_name == name
        assert tool_parser_cls.rust_parser_name == name
        assert reasoning_parser_cls.__name__ == f"RustUnifiedReasoningParser_{name}"
        assert tool_parser_cls.__name__ == f"RustUnifiedToolParser_{name}"
        assert name not in ReasoningParserManager.lazy_parsers
        assert name not in ToolParserManager.lazy_parsers
        assert ReasoningParserManager.get_reasoning_parser(name) is reasoning_parser_cls
        assert ToolParserManager.get_tool_parser(name) is tool_parser_cls

    assert len(set(reasoning_parser_classes)) == len(parser_names)
    assert len(set(tool_parser_classes)) == len(parser_names)
    assert RustUnifiedParser.reasoning_parser_cls is RustUnifiedReasoningParser
    assert RustUnifiedParser.tool_parser_cls is RustUnifiedToolParser
    assert not hasattr(rust_parser, "ToolParser")

    with pytest.raises(ValueError, match="matching reasoning and tool parser names"):
        ParserManager.get_parser(
            tool_parser_name="inkling",
            reasoning_parser_name="minimax_m3",
            enable_auto_tools=True,
        )


def test_rust_pair_requires_enabled_tool_parser():
    with pytest.raises(ValueError, match="matching reasoning and tool parser names"):
        ParserManager.get_parser(
            tool_parser_name="inkling",
            reasoning_parser_name="inkling",
            enable_auto_tools=False,
        )


def test_matching_names_pass_api_server_validation():
    validate_api_server_args(
        SimpleNamespace(
            tool_call_parser="inkling",
            enable_auto_tool_choice=True,
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="inkling",
            ),
        )
    )


def test_matching_names_pass_run_batch_validation():
    validate_run_batch_args(
        SimpleNamespace(
            tool_call_parser="inkling",
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="inkling",
            ),
        )
    )


def test_reasoning_stub_supports_standalone_capability_queries():
    parser_cls = ParserManager.get_reasoning_parser("minimax_m3")
    assert parser_cls is not None
    assert issubclass(parser_cls, RustUnifiedReasoningParser)
    assert parser_cls.rust_parser_name == "minimax_m3"

    reasoner = parser_cls(FakeTokenizer({"<mm:think>": 256, "</mm:think>": 257}))
    assert reasoner.reasoning_start_str == "<mm:think>"
    assert reasoner.reasoning_end_str == "</mm:think>"
    assert not reasoner.is_reasoning_end([256])
    assert reasoner.is_reasoning_end([256, 257])


def test_tokenizer_metadata_is_shared_across_request_parsers():
    tokenizer = FakeTokenizer({"<mm:think>": 256, "</mm:think>": 257})
    first = parser("minimax_m3", tokenizer)
    second = parser("minimax_m3", tokenizer)

    assert first._tokenizer_metadata() is second._tokenizer_metadata()


def test_minimax_m3_complete_parse_uses_prompt_state():
    tokenizer = FakeTokenizer({"<mm:think>": 256, "</mm:think>": 257})
    unified = parser("minimax_m3", tokenizer)
    req = request()

    assert unified.reasoning_start_str == "<mm:think>"
    assert unified.reasoning_end_str == "</mm:think>"
    assert not unified.is_reasoning_end([256])
    assert unified.is_reasoning_end([256, 257])

    reasoning, content, tool_calls = unified.parse(
        "plan</mm:think>answer"
        "]<]minimax[>[<tool_call>"
        ']<]minimax[>[<invoke name="get_weather">'
        "]<]minimax[>[<city>Paris]<]minimax[>[</city>"
        "]<]minimax[>[</invoke>"
        "]<]minimax[>[</tool_call>",
        req,
        enable_auto_tools=True,
        prompt_token_ids=[256],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert tool_calls is not None
    assert [(call.name, call.arguments) for call in tool_calls] == [
        ("get_weather", '{"city":"Paris"}')
    ]
    assert unified.reasoning_parser is None
    assert unified.tool_parser is None
    assert unified.count_reasoning_tokens([101, 102, 257, 103]) == 2


def test_inkling_streaming_emits_reasoning_text_and_tool_calls():
    tokenizer = FakeTokenizer(
        {
            "<|message_model|>": 200001,
            "<|content_text|>": 200004,
            "<|content_thinking|>": 200008,
        }
    )
    unified = parser("inkling", tokenizer)
    req = request()
    chunks = [
        "<|content_thinking|>plan<|end_message|>",
        "<|content_text|>answer<|end_message|>",
        '<|content_invoke_tool_json|>{"name":"get_weather",',
        '"args":{"city":"Paris"}}<|end_message|>',
    ]

    deltas = [
        delta
        for index, chunk in enumerate(chunks)
        if (
            delta := unified.parse_delta(
                chunk,
                [],
                req,
                prompt_token_ids=[],
                finished=index == len(chunks) - 1,
            )
        )
    ]

    assert "".join(delta.reasoning or "" for delta in deltas) == "plan"
    assert "".join(delta.content or "" for delta in deltas) == "answer"
    calls = [call for delta in deltas for call in delta.tool_calls]
    assert calls[0].function is not None
    assert calls[0].function.name == "get_weather"
    assert (
        "".join(
            call.function.arguments or "" for call in calls if call.function is not None
        )
        == '{"city":"Paris"}'
    )


def test_inkling_bare_text_flushes_at_finish():
    tokenizer = FakeTokenizer(
        {
            "<|message_model|>": 200001,
            "<|content_text|>": 200004,
            "<|content_thinking|>": 200008,
            "<|content_model_end_sampling|>": 200006,
        }
    )
    req = request()

    complete = parser("inkling", tokenizer)
    assert complete.parse(
        "OKOKOK",
        req,
        enable_auto_tools=True,
        prompt_token_ids=[200001],
    ) == (None, "OKOKOK", [])

    streaming = parser("inkling", tokenizer)
    assert (
        streaming.parse_delta(
            "OK",
            [],
            req,
            prompt_token_ids=[200001],
            finished=False,
        )
        is None
    )
    final_delta = streaming.parse_delta(
        "OKOK",
        [],
        req,
        finished=True,
    )
    assert final_delta is not None
    assert final_delta.content == "OKOKOK"


def test_inkling_tool_choice_none_keeps_visible_content():
    tokenizer = FakeTokenizer(
        {
            "<|message_model|>": 200001,
            "<|content_text|>": 200004,
            "<|content_thinking|>": 200008,
        }
    )
    unified = parser("inkling", tokenizer)

    reasoning, content, tool_calls = unified.parse(
        "<|content_thinking|>plan<|end_message|>"
        "<|content_text|>answer<|end_message|>"
        '<|content_invoke_tool_json|>{"name":"get_weather",'
        '"args":{"city":"Paris"}}<|end_message|>',
        request(tool_choice="none"),
        enable_auto_tools=True,
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert tool_calls == []
