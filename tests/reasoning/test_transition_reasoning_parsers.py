# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.reasoning.minimax_m2_reasoning_parser import (
    MiniMaxM2AppendThinkReasoningParser,
)
from vllm.reasoning.step3_reasoning_parser import Step3ReasoningParser


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_identity_reasoning_end_indices_always_return_minus_one():
    parser = IdentityReasoningParser(FakeTokenizer({}))

    assert parser.reasoning_end_index([1, 2, 3]) == -1
    assert parser.reasoning_end_delta_index([1, 2], [3, 4]) == -1


def test_kimi_k2_reasoning_end_indices_support_end_and_tool_tokens():
    parser = KimiK2ReasoningParser(
        FakeTokenizer(
            {
                "<think>": 10,
                "</think>": 11,
                "<|tool_calls_section_begin|>": 12,
            }
        )
    )

    assert parser.reasoning_end_index([10, 101, 11, 102]) == 2
    assert parser.reasoning_end_index([10, 101, 12, 102]) == 2
    assert parser.reasoning_end_delta_index([10, 101], [11, 102]) == 0
    assert parser.reasoning_end_delta_index([10, 101], [12, 102]) == 0


def test_kimi_k2_reasoning_end_indices_fall_back_to_identity_mode():
    parser = KimiK2ReasoningParser(
        FakeTokenizer(
            {
                "<think>": 10,
                "</think>": 11,
                "<|tool_calls_section_begin|>": 12,
            }
        ),
        chat_template_kwargs={"thinking": False},
    )

    assert parser.reasoning_end_index([10, 11, 12]) == -1
    assert parser.reasoning_end_delta_index([10], [11, 12]) == -1


def test_minimax_append_reasoning_end_indices_find_think_end_token():
    parser = MiniMaxM2AppendThinkReasoningParser(
        FakeTokenizer({"<think>": 10, "</think>": 11})
    )

    assert parser.reasoning_end_index([101, 11, 102]) == 1
    assert parser.reasoning_end_delta_index([101], [11, 102]) == 0
    assert parser.reasoning_end_index([101, 102]) == -1


def test_step3_reasoning_end_indices_find_think_end_token():
    parser = Step3ReasoningParser(FakeTokenizer({"</think>": 11}))

    assert parser.reasoning_end_index([101, 11, 102]) == 1
    assert parser.reasoning_end_delta_index([101], [11, 102]) == 0
    assert parser.reasoning_end_delta_index([101], [102, 103]) == -1
