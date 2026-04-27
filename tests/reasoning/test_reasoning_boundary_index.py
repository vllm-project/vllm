# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"<think>": 1, "</think>": 99}


class DummyThinkingParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"


class MultiTokenEndParser(ReasoningParser):
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return len(input_ids) >= 3 and list(input_ids[-3:]) == [7, 8, 9]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids


def test_single_token_end_marker_boundary_uses_delta_fast_path():
    parser = DummyThinkingParser(DummyTokenizer())

    assert parser.find_reasoning_end_index([1, 2], [3, 99, 4]) == 1
    assert parser.find_reasoning_end_index([1, 2, 99], [3, 4]) is None
    assert parser.may_have_reasoning_end_in_delta([3, 99, 4]) is True
    assert parser.may_have_reasoning_end_in_delta([3, 4]) is False


def test_fallback_boundary_detection_crosses_prefix_and_delta():
    parser = MultiTokenEndParser(None)

    assert parser.find_reasoning_end_index([1, 7, 8], [9, 10]) == 0
    assert parser.find_reasoning_end_index([1, 7], [8, 10]) is None
    assert parser.may_have_reasoning_end_in_delta([10]) is True
    assert parser.may_have_reasoning_end_in_delta([]) is False
