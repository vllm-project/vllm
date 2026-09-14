# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager


class FakeStep3Tokenizer:
    def __init__(self):
        self._vocab = {
            "</think>": 1,
            "answer": 2,
            "step one": 3,
        }
        self._pattern = re.compile(r"(</think>)")

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


def test_step3_streaming_waits_for_buffered_end_token_text():
    parser_cls = ReasoningParserManager.get_reasoning_parser("step3")
    parser = parser_cls(FakeStep3Tokenizer())

    result = parser.extract_reasoning_streaming(
        previous_text="step one",
        current_text="step one",
        delta_text="",
        previous_token_ids=[3],
        current_token_ids=[3, parser.think_end_token_id],
        delta_token_ids=[parser.think_end_token_id],
    )

    assert result is None


def test_step3_streaming_splits_when_buffered_end_token_text_flushes():
    parser_cls = ReasoningParserManager.get_reasoning_parser("step3")
    parser = parser_cls(FakeStep3Tokenizer())

    result = parser.extract_reasoning_streaming(
        previous_text="step one",
        current_text="step one</think>answer",
        delta_text="</think>answer",
        previous_token_ids=[3, parser.think_end_token_id],
        current_token_ids=[3, parser.think_end_token_id, 2],
        delta_token_ids=[2],
    )

    assert isinstance(result, DeltaMessage)
    assert result.reasoning == ""
    assert result.content == "answer"
