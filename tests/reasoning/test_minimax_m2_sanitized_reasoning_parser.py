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


def test_sanitized_reasoning_normalizes_streaming_content():
    parser = MiniMaxM2SanitizedReasoningParser(FakeTokenizer())
    delta = parser.extract_reasoning_streaming(
        previous_text="Thinking</think>",
        current_text="Thinking</think>scripts/monkey_character. gd",
        delta_text="scripts/monkey_character. gd",
        previous_token_ids=[2],
        current_token_ids=[2],
        delta_token_ids=[],
    )
    assert delta is not None
    assert delta.content == "scripts/monkey_character.gd"
