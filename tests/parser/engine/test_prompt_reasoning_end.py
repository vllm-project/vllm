# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-level reasoning-end detection must only look at the prompt tail."""

from unittest.mock import MagicMock

import pytest

from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

THINK_START = "<mm:think>"
THINK_END = "</mm:think>"
START_ID = 100
END_ID = 101


def _tokenizer():
    """Mock tokenizer where each marker is a single vocabulary token.

    NOTE: tests.parser.engine.conftest.make_mock_tokenizer cannot be used here
    because it stubs encode() with a constant return value, which would make
    both markers encode to the same ids and silently void these assertions.
    """
    vocab = {THINK_START: START_ID, THINK_END: END_ID}
    id_to_text = {v: k for k, v in vocab.items()}

    def encode(text, add_special_tokens=False):
        if text in vocab:
            return [vocab[text]]
        return [ord(c) for c in text]

    def decode(ids, skip_special_tokens=False):
        return "".join(id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids)

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(vocab)
    tokenizer.all_special_tokens = list(vocab)
    tokenizer.all_special_ids = list(vocab.values())
    tokenizer.encode.side_effect = encode
    tokenizer.decode.side_effect = decode
    return tokenizer


@pytest.fixture
def reasoning_parser():
    return MiniMaxM3ReasoningParser(_tokenizer())


@pytest.fixture
def parser(reasoning_parser):
    obj = DelegatingParser.__new__(DelegatingParser)
    obj._reasoning_parser = reasoning_parser
    obj._tool_parser = None
    return obj


def test_mock_encodes_markers_as_single_tokens(reasoning_parser):
    """Guard the fixture itself: if the markers do not encode to the ids used
    below, every other assertion in this file becomes vacuous."""
    assert reasoning_parser._start_token_ids == (START_ID,)
    assert reasoning_parser._end_token_ids == (END_ID,)


def test_marker_pair_inside_prompt_does_not_end_reasoning(parser):
    """A <mm:think></mm:think> pair coming from the chat template must not be
    mistaken for a completed reasoning block."""
    # instructions ... <mm:think></mm:think> ... </mm:think> ... user turn
    prompt = [1, 2, START_ID, END_ID, 3, 4, END_ID, 5, 6]
    assert parser._prompt_ends_reasoning(prompt) is False


def test_unpaired_end_marker_inside_prompt_does_not_end_reasoning(parser):
    prompt = [1, 2, END_ID, 3, 4, 5]
    assert parser._prompt_ends_reasoning(prompt) is False


def test_prompt_ending_with_end_marker_ends_reasoning(parser):
    prompt = [1, 2, START_ID, 3, 4, END_ID]
    assert parser._prompt_ends_reasoning(prompt) is True


def test_no_reasoning_parser_ends_reasoning(parser):
    parser._reasoning_parser = None
    assert parser._prompt_ends_reasoning([1, 2, 3]) is True
