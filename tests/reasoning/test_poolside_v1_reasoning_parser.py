# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Poolside Laguna force-open reasoning parser tests.

Laguna emits CoT then a bare ``</think>`` with no opening ``<think>``. The
parser must always force-open (independent of chat_template_kwargs.thinking)
and must not leak the close tag into content.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.poolside_v1_reasoning_parser import PoolsideV1ReasoningParser

pytestmark = pytest.mark.skip_global_cleanup


class _FakeTok:
    """Minimal tokenizer with the specials poolside_v1 needs."""

    def __init__(self) -> None:
        self._vocab = {
            "<think>": 1,
            "</think>": 2,
            "<assistant>": 3,
        }
        self.all_special_tokens = list(self._vocab)
        self.all_special_ids = list(self._vocab.values())
        self.additional_special_tokens: list[str] = []
        self.additional_special_tokens_ids: list[int] = []

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        specials = sorted(self._vocab, key=len, reverse=True)
        while i < len(s):
            matched = False
            for t in specials:
                if s.startswith(t, i):
                    out.append(self._vocab[t])
                    i += len(t)
                    matched = True
                    break
            if not matched:
                out.append(100 + (ord(s[i]) % 50))
                i += 1
        return out

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        rev = {v: k for k, v in self._vocab.items()}
        return [rev.get(i, f"t{i}") for i in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def __len__(self) -> int:
        return len(self._vocab)


@pytest.fixture
def tokenizer() -> _FakeTok:
    return _FakeTok()


@pytest.fixture
def parser(tokenizer: _FakeTok) -> PoolsideV1ReasoningParser:
    # No thinking kwargs: must still force-open (this was the Identity leak).
    return PoolsideV1ReasoningParser(tokenizer)


def test_registered_as_poolside_v1():
    assert (
        ReasoningParserManager.get_reasoning_parser("poolside_v1")
        is PoolsideV1ReasoningParser
    )


def test_subclasses_deepseek_r1_not_v3_identity_path(parser):
    assert isinstance(parser, DeepSeekR1ReasoningParser)
    assert not hasattr(parser, "_parser")


def test_force_open_nonstream_without_thinking_kwargs(parser):
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)
    reasoning, content = parser.extract_reasoning(
        "reason...</think>answer", request=request
    )
    assert reasoning == "reason..."
    assert content == "answer"
    assert "</think>" not in (content or "")


def test_force_open_with_optional_open_tag(parser):
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)
    reasoning, content = parser.extract_reasoning(
        "<think>reason...</think>answer", request=request
    )
    assert reasoning == "reason..."
    assert content == "answer"


def test_force_open_streaming_without_thinking_kwargs(parser, tokenizer):
    d1_text = "reason..."
    d1_ids = tokenizer.encode(d1_text)
    d1 = parser.extract_reasoning_streaming(
        "", d1_text, d1_text, [], d1_ids, d1_ids
    )
    assert d1 is not None
    assert d1.reasoning == "reason..."
    assert not d1.content

    prev = d1_text
    delta = "</think>answer"
    curr = prev + delta
    prev_ids = d1_ids
    delta_ids = tokenizer.encode(delta)
    curr_ids = prev_ids + delta_ids
    d2 = parser.extract_reasoning_streaming(
        prev, curr, delta, prev_ids, curr_ids, delta_ids
    )
    assert d2 is not None
    assert d2.content == "answer"
    assert "</think>" not in (d2.content or "")


def test_is_reasoning_end_scoped_to_assistant_turn(parser, tokenizer):
    # Prior turn had </think>; current assistant turn has not closed yet.
    # Sequence: <assistant> prior ... </think> ... user ... <assistant> reason...
    prior_close = (
        tokenizer.encode("<assistant>")
        + tokenizer.encode("old")
        + tokenizer.encode("</think>")
        + tokenizer.encode("<assistant>")
        + tokenizer.encode("new reason")
    )
    assert parser.is_reasoning_end(prior_close) is False

    with_close = prior_close + tokenizer.encode("</think>")
    assert parser.is_reasoning_end(with_close) is True


def test_missing_assistant_token_fails_loud():
    class NoAssistantTok(_FakeTok):
        def __init__(self) -> None:
            super().__init__()
            del self._vocab["<assistant>"]

    with pytest.raises(ValueError, match="<assistant>"):
        PoolsideV1ReasoningParser(NoAssistantTok())
