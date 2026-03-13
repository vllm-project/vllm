# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAIToolParser EBNF grammar and xgrammar validation."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.openai_tool_parser import OpenAIToolParser

MODEL = "gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


@pytest.fixture
def parser(tokenizer):
    return OpenAIToolParser(tokenizer)


# ---------------------------------------------------------------------------
# Grammar generation
# ---------------------------------------------------------------------------


def test_build_grammar_single_tool(parser: OpenAIToolParser) -> None:
    grammar = parser._build_tool_required_grammar(["get_weather"])
    assert '"functions.get_weather"' in grammar
    assert "root ::=" in grammar
    assert "tool_block" in grammar


def test_build_grammar_multiple_tools(parser: OpenAIToolParser) -> None:
    grammar = parser._build_tool_required_grammar(
        ["get_weather", "search", "calculate"]
    )
    assert (
        '"functions.get_weather" | "functions.search" | "functions.calculate"'
        in grammar
    )


def test_build_grammar_no_final_channel(parser: OpenAIToolParser) -> None:
    grammar = parser._build_tool_required_grammar(["f"])
    assert '"final"' not in grammar


def test_build_grammar_rejects_invalid_tool_names(
    parser: OpenAIToolParser,
) -> None:
    with pytest.raises(ValueError, match="invalid for EBNF grammar"):
        parser._build_tool_required_grammar(['get"weather'])
    with pytest.raises(ValueError, match="invalid for EBNF grammar"):
        parser._build_tool_required_grammar(["get\nweather"])


# ---------------------------------------------------------------------------
# adjust_request
# ---------------------------------------------------------------------------


def _make_tools(*names: str) -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name=name,
                description=f"Tool {name}",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        )
        for name in names
    ]


def test_adjust_request_required(parser: OpenAIToolParser) -> None:
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=_make_tools("get_weather", "search"),
        tool_choice="required",
    )
    result = parser.adjust_request(request)
    assert isinstance(result.structured_outputs, StructuredOutputsParams)
    assert result.structured_outputs.grammar is not None
    assert '"functions.get_weather"' in result.structured_outputs.grammar
    assert result.response_format is None


def test_adjust_request_non_required_unchanged(parser: OpenAIToolParser) -> None:
    for tool_choice in ["auto", "none"]:
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=_make_tools("f"),
            tool_choice=tool_choice,
        )
        assert parser.adjust_request(request).structured_outputs is None


# ---------------------------------------------------------------------------
# xgrammar validation (require xgrammar installed)
# ---------------------------------------------------------------------------

xgrammar = pytest.importorskip("xgrammar")

VOCAB = [
    "<|end|>",  # 0
    "<|start|>",  # 1
    "<|channel|>",  # 2
    "<|message|>",  # 3
    "<|return|>",  # 4
    "<|call|>",  # 5
    "assistant",  # 6
    "analysis",  # 7
    "commentary",  # 8
    "final",  # 9
    " to=",  # 10
    "functions.",  # 11
    "get_weather",  # 12
    "search",  # 13
    "I need",  # 14
    " to check",  # 15
    "{",  # 16
    "}",  # 17
    '"',  # 18
    "location",  # 19
    ":",  # 20
    "Tokyo",  # 21
    "<|eos|>",  # 22
    "Let me",  # 23
    " call",  # 24
    " < ",  # 25
    "hello",  # 26
]
V = {s: i for i, s in enumerate(VOCAB)}


@pytest.fixture(scope="module")
def xgr_compiler():
    tokenizer_info = xgrammar.TokenizerInfo(
        encoded_vocab=VOCAB,
        vocab_type=xgrammar.VocabType.RAW,
        vocab_size=len(VOCAB),
        stop_token_ids=[V["<|eos|>"]],
    )
    return xgrammar.GrammarCompiler(tokenizer_info)


def _compile_and_run(compiler, tool_names, token_ids) -> bool:
    grammar = OpenAIToolParser._build_tool_required_grammar(tool_names)
    ctx = compiler.compile_grammar(grammar)
    matcher = xgrammar.GrammarMatcher(ctx)
    return all(matcher.accept_token(tid) for tid in token_ids)


def _bitmask_allowed(bitmask, token_id: int) -> bool:
    return bool(bitmask[0, token_id // 32].item() & (1 << (token_id % 32)))


class TestXgrammarAcceptance:
    def test_direct_tool_call(self, xgr_compiler) -> None:
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V['"'],
            V["location"],
            V['"'],
            V[":"],
            V['"'],
            V["Tokyo"],
            V['"'],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)

    def test_analysis_then_tool_call(self, xgr_compiler) -> None:
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["I need"],
            V[" to check"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)

    def test_two_tool_calls(self, xgr_compiler) -> None:
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["search"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather", "search"], seq)

    def test_content_with_lt_operator(self, xgr_compiler) -> None:
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V[" < "],
            V["hello"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)


class TestXgrammarBlocking:
    def test_final_channel_blocked(self, xgr_compiler) -> None:
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        assert not matcher.accept_token(V["final"])

    def test_wrong_function_name_blocked(self, xgr_compiler) -> None:
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        for tid in [V["commentary"], V[" to="], V["functions."]]:
            assert matcher.accept_token(tid)
        assert not matcher.accept_token(V["search"])


class TestXgrammarTermination:
    def test_eos_blocked_before_tool_call(self, xgr_compiler) -> None:
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        for tid in [V["analysis"], V["<|message|>"], V["hello"], V["<|end|>"]]:
            assert matcher.accept_token(tid)
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert not _bitmask_allowed(bitmask, V["<|eos|>"])

    def test_eos_allowed_after_tool_call(self, xgr_compiler) -> None:
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        for tid in seq:
            assert matcher.accept_token(tid)
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|eos|>"])
