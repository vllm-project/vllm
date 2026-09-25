# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MiniCPM-V family parser."""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import simulate_reasoning_streaming
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.minicpmv import (
    EscapedNewlineNormalizer,
    MiniCPMVOutputNormalizer,
    MiniCPMVParser,
    recover_newlines,
)
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

_VOCAB = {
    "<think>": 10,
    "</think>": 11,
    "<|im_end|>": 12,
    "<tool_call>": 13,
    "</tool_call>": 14,
}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_VOCAB)


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_reasoning_parser_is_registered():
    parser_cls = ReasoningParserManager.get_reasoning_parser("minicpmv")

    assert parser_cls.__name__ == "MiniCPMVParserReasoningAdapter"


def test_tool_parser_is_registered():
    parser_cls = ToolParserManager.get_tool_parser("minicpmv")

    assert parser_cls.__name__ == "MiniCPMVEngineToolParser"


def test_tool_parser_composes_with_reasoning_parser(mock_tokenizer):
    """`--tool-call-parser minicpmv` must work through the serving Parser."""
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
    ]
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="minicpmv",
        reasoning_parser_name="minicpmv",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(mock_tokenizer, request.tools)

    _, _, tool_calls = parser.parse(
        "<tool_call><function=get_weather>"
        "<parameter=city>SF</parameter></function></tool_call>",
        request,
        enable_auto_tools=True,
    )

    assert tool_calls is not None
    assert tool_calls[0].name == "get_weather"
    assert "SF" in tool_calls[0].arguments


class TestNonThinking:
    @pytest.fixture
    def parser(self, mock_tokenizer):
        return MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )

    def test_plain_content_is_unchanged(self, parser):
        reasoning, content = parser.extract_reasoning("The answer is 42.", None)

        assert reasoning is None
        assert content == "The answer is 42."

    def test_reserved_markers_are_removed(self, parser):
        output = (
            "<reserved_12>reasoning<reserved_13><|reserved_14|>tool call<|reserved_15|>"
        )

        reasoning, content = parser.extract_reasoning(output, None)

        assert reasoning is None
        assert content == "reasoningtool call"

    def test_unrelated_reserved_text_is_unchanged(self, parser):
        reasoning, content = parser.extract_reasoning(
            "<reserved_11>content<|reserved_16|>",
            None,
        )

        assert reasoning is None
        assert content == "<reserved_11>content<|reserved_16|>"

    def test_dangling_standard_think_end_is_removed(self, parser):
        output = "first part </think>\n\nfinal part"

        reasoning, content = parser.extract_reasoning(output, None)

        assert reasoning is None
        assert content == "first part \n\nfinal part"

    def test_truncated_standard_reasoning_does_not_leak(self, parser):
        reasoning, content = parser.extract_reasoning(
            "prefix<think>private reasoning",
            None,
        )

        assert reasoning is None
        assert content == "prefix"

    def test_streaming_split_standard_markers(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "<thi",
                "nk>",
                "private reasoning",
                "</thi",
                "nk>",
                "final answer",
            ],
        )

        assert reasoning == ""
        assert content == "final answer"

    def test_streaming_split_reserved_markers(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "<reser",
                "ved_12>",
                "reasoning",
                "<|reserved",
                "_13|>",
                "final answer",
            ],
        )

        assert reasoning == ""
        assert content == "reasoningfinal answer"

    def test_tool_calls_are_parsed(self, parser, request_obj):
        """Thinking is off by default, and tool calls must still be parsed."""
        reasoning, content, tool_calls = parser.parse(
            "<tool_call><function=get_weather>"
            "<parameter=city>SF</parameter></function></tool_call>",
            request_obj,
        )

        assert reasoning is None
        assert not content
        assert tool_calls is not None
        assert tool_calls[0].name == "get_weather"
        assert "SF" in tool_calls[0].arguments


class TestThinking:
    def test_standard_think_tags_are_parsed(self, mock_tokenizer):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )

        reasoning, content = parser.extract_reasoning(
            "<think>private reasoning</think>final answer",
            None,
        )

        assert reasoning == "private reasoning"
        assert content == "final answer"

    def test_reserved_markers_are_removed(self, mock_tokenizer):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )

        reasoning, content = parser.extract_reasoning(
            "<think><reserved_12>private reasoning</think><|reserved_13|>final answer",
            None,
        )

        assert reasoning == "private reasoning"
        assert content == "final answer"

    def test_streaming_im_end_is_removed(self):
        tokenizer = make_mock_tokenizer(_VOCAB, special_tokens=[])
        parser = MiniCPMVParser(
            tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )

        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["Answer: 111", "<|im_end|>"],
        )

        assert reasoning == "Answer: 111"
        assert content == ""


class TestNewlineRecovery:
    """The tokenizer writes a newline as the two characters ``\\`` and ``n``."""

    def test_recover_newlines(self):
        text = (
            "first\\nsecond\\rthird\\r\\nfourth "
            "`code\\nvalue` "
            "```python\\nvalue = '\\\\n'\\n``` "
            "$x\\ny$ $$x\\ny$$ \\(x\\ny\\) \\[x\\ny\\] "
            "escaped \\\\n"
        )

        assert recover_newlines(text) == (
            "first\nsecond\nthird\nfourth "
            "`code\\nvalue` "
            "```python\\nvalue = '\\\\n'\\n``` "
            "$x\\ny$ $$x\\ny$$ \\(x\\ny\\) \\[x\\ny\\] "
            "escaped \\\\n"
        )

    def test_streaming_matches_complete_recovery(self):
        text = (
            "first\\nsecond `code\\nvalue` $x\\ny$ ```text\\nvalue\\n``` last\\r\\nline"
        )
        expected = recover_newlines(text)

        for split in range(len(text) + 1):
            normalizer = EscapedNewlineNormalizer()
            actual = normalizer.feed(text[:split])
            actual += normalizer.feed(text[split:], final=True)
            assert actual == expected

        normalizer = EscapedNewlineNormalizer()
        actual = "".join(normalizer.feed(char) for char in text)
        actual += normalizer.feed("", final=True)
        assert actual == expected

    def test_backslash_runs_stay_consistent_across_splits(self):
        # A run of backslashes can be split over chunks. Its parity decides
        # whether a following `n` is a newline, so streaming and non-streaming
        # must agree for every split point.
        for count in range(1, 7):
            text = "\\" * count + "n"
            expected = recover_newlines(text)

            for split in range(len(text) + 1):
                normalizer = EscapedNewlineNormalizer()
                actual = normalizer.feed(text[:split])
                actual += normalizer.feed(text[split:], final=True)
                assert actual == expected, (count, split)

    def test_streaming_recovers_reasoning_and_content(self):
        normalizer = MiniCPMVOutputNormalizer()

        first = normalizer.normalize_delta(
            DeltaMessage(reasoning="reason\\"),
            finished=False,
        )
        second = normalizer.normalize_delta(
            DeltaMessage(reasoning="nnext\\", content="answer\\"),
            finished=False,
        )
        final = normalizer.normalize_delta(
            DeltaMessage(content="nend"),
            finished=True,
        )

        assert first == DeltaMessage(reasoning="reason")
        assert second == DeltaMessage(reasoning="\nnext\\", content="answer")
        assert final == DeltaMessage(content="\nend")

    def test_streaming_does_not_modify_tool_arguments(self):
        normalizer = MiniCPMVOutputNormalizer()
        arguments = '{"value":"first\\\\nsecond"}'
        delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(arguments=arguments),
                )
            ]
        )

        result = normalizer.normalize_delta(delta, finished=True)

        assert result is not None
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.arguments == arguments

    def test_parse_recovers_content(self, mock_tokenizer, request_obj):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )

        reasoning, content, tool_calls = parser.parse(
            "first\\nsecond",
            request_obj,
        )

        assert reasoning is None
        assert content == "first\nsecond"
        assert tool_calls is None

    def test_parse_delta_recovers_content(self, mock_tokenizer, request_obj):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )

        chunk = parser.parse_delta(
            "first\\",
            [],
            request_obj,
            finished=False,
        )
        rest = parser.parse_delta(
            "nsecond",
            [],
            request_obj,
            finished=True,
        )

        # The trailing half of the escape is held back until it is complete.
        assert chunk is not None
        assert chunk.content == "first"
        assert rest is not None
        assert rest.content == "\nsecond"

    def test_buffered_delta_is_dropped_not_sent_empty(
        self, mock_tokenizer, request_obj
    ):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )
        parser.parse_delta("hello", [], request_obj, finished=False)

        # The lone backtick is held back for the next chunk. Reporting it as an
        # all-None DeltaMessage would reach the client as an empty SSE chunk.
        assert parser.parse_delta("`", [], request_obj, finished=False) is None

    def test_normalizer_drops_empty_delta(self):
        normalizer = MiniCPMVOutputNormalizer()

        assert (
            normalizer.normalize_delta(DeltaMessage(content="`"), finished=False)
            is None
        )

    def test_normalizer_flushes_buffered_content_when_finished(self):
        normalizer = MiniCPMVOutputNormalizer()
        normalizer.normalize_delta(DeltaMessage(content="`"), finished=False)

        final = normalizer.normalize_delta(DeltaMessage(), finished=True)

        assert final is not None
        assert final.content == "`"
