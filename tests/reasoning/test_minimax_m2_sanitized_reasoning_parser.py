# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.minimax_m2_sanitized_reasoning_parser import (
    MiniMaxM2SanitizedReasoningParser,
)


class FakeTokenizer:
    def __init__(self):
        self.model_tokenizer = True
        self.vocab = {
            "<think>": 1,
            "</think>": 2,
        }

    def get_vocab(self):
        return self.vocab


def test_sanitized_reasoning_normalizes_non_streaming_content():
    parser = MiniMaxM2SanitizedReasoningParser(FakeTokenizer())
    reasoning, content = parser.extract_reasoning(
        "Looking...</think>scripts/monkey_character. gd", request=None
    )
    assert reasoning == "Looking..."
    assert content == "scripts/monkey_character.gd"
