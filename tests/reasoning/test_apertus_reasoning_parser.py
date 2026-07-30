# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "apertus"

INNER_VOCAB = {
    "<|inner_prefix|>": 32,
    "<|inner_suffix|>": 33,
    "<think>": 500,
    "</think>": 501,
}
THINK_VOCAB = {
    "<think>": 32,
    "</think>": 33,
    "<|inner_prefix|>": 900,
    "<|inner_suffix|>": 901,
}


class MockTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        # Only special tokens matter to the parser; deltas are fed one per token.
        return [text] if text in self._vocab else []


def make_parser(vocab: dict[str, int]) -> ReasoningParser:
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    return parser_cls(MockTokenizer(vocab))


@pytest.mark.parametrize(
    "vocab, start, end",
    [
        (INNER_VOCAB, "<|inner_prefix|>", "<|inner_suffix|>"),
        (THINK_VOCAB, "<think>", "</think>"),
    ],
)
def test_delimiters_follow_the_tokenizer(vocab, start, end):
    """The emitted (lower-id) delimiter pair wins, whichever scheme the
    tokenizer build registers there."""
    parser = make_parser(vocab)

    assert (parser.start_token, parser.end_token) == (start, end)
    assert (parser.start_token_id, parser.end_token_id) == (32, 33)


@pytest.mark.parametrize("streaming", [True, False])
def test_deliberation_block_is_reasoning(streaming: bool):
    parser = make_parser(INNER_VOCAB)
    output = ["<|inner_prefix|>", "Let me think", "<|inner_suffix|>", "The answer"]

    reasoning, content = run_reasoning_extraction(
        reasoning_parser=parser, model_output=output, streaming=streaming
    )

    assert reasoning == "Let me think"
    assert content == "The answer"


def test_output_without_deliberation_is_all_content():
    """A direct tool call carries no inner block; it must reach the tool parser
    as content rather than being swallowed as reasoning."""
    parser = make_parser(INNER_VOCAB)
    output = '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}]<|tools_suffix|>'

    reasoning, content = run_reasoning_extraction(
        reasoning_parser=parser, model_output=[output]
    )

    assert reasoning is None
    assert content == output
