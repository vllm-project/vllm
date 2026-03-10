# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypedDict

import pytest
import regex as re

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "nemotron_v3"


class ReasoningCase(TypedDict):
    output: str
    reasoning: str | None
    content: str | None


class FakeNemotronTokenizer:
    def __init__(self):
        self._vocab = {
            "<think>": 1,
            "</think>": 2,
        }
        self._pattern = re.compile(r"(<think>|</think>)")

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for part in self._pattern.split(text):
            if part:
                tokens.append(part)
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


@pytest.fixture
def tokenizer():
    return FakeNemotronTokenizer()


@pytest.mark.parametrize(
    "streaming,param_dict",
    [
        pytest.param(
            False,
            {
                "output": "This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="without_start_token",
        ),
        pytest.param(
            True,
            {
                "output": "This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="without_start_token_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "<think>This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="with_start_token",
        ),
        pytest.param(
            True,
            {
                "output": "<think>This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="with_start_token_streaming",
        ),
    ],
)
def test_nemotron_v3_reasoning(
    tokenizer: FakeNemotronTokenizer,
    streaming: bool,
    param_dict: ReasoningCase,
):
    output = tokenizer.tokenize(param_dict["output"])
    model_output = [tokenizer.convert_tokens_to_string([token]) for token in output]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, model_output, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


def test_nemotron_v3_without_thinking_returns_content(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is plain content"],
        request=request,
        streaming=False,
    )

    assert reasoning is None
    assert content == "This is plain content"


def test_nemotron_v3_with_thinking_keeps_truncated_reasoning(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is truncated reasoning"],
        request=request,
        streaming=False,
    )

    assert reasoning == "This is truncated reasoning"
    assert content is None
