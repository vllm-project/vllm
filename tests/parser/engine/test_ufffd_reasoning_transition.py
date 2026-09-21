# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for U+FFFD leak at reasoning→content transition.

When byte-fallback tokens span the reasoning/content boundary,
decoding isolated content-side token IDs via tokenizer.decode()
produces U+FFFD (Unicode replacement character).  The fix flushes
the reasoning parser's engine lexer instead.

Reproduces the bug at various chunk sizes and validates that the
fix prevents U+FFFD from leaking into streamed content.
"""

from __future__ import annotations

import pytest

from tests.parser.engine.replay_harness import (
    CHUNK_SIZES,
    MockTokenizer,
    collect_output,
    replay_streaming,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import (
    Glm47MoeParserReasoningAdapter,
    Glm47MoeParserToolAdapter,
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
)


class ByteFallbackMockTokenizer(MockTokenizer):
    """MockTokenizer that returns U+FFFD for specified token IDs.

    Simulates byte-fallback tokenizer behavior where isolated
    partial-byte tokens decode to the Unicode replacement character.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        tokens: list[tuple[int, str]],
        ufffd_token_ids: set[int],
    ) -> None:
        super().__init__(vocab, tokens)
        self._ufffd_token_ids = frozenset(ufffd_token_ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        parts: list[str] = []
        for tid in ids:
            if skip_special_tokens and tid in self._special_ids:
                continue
            if tid in self._ufffd_token_ids:
                parts.append("�")
            else:
                text = self._token_decode_map.get(tid, f"?{tid}?")
                parts.append(text)
        return "".join(parts)


# ── Model-specific DelegatingParser subclasses ───────────────────────


class _Glm47Delegating(DelegatingParser):
    reasoning_parser_cls = Glm47MoeParserReasoningAdapter
    tool_parser_cls = Glm47MoeParserToolAdapter


class _Qwen3Delegating(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = Qwen3ParserToolAdapter


# ── Shared test data ─────────────────────────────────────────────────

_SHARED_TOKENS: list[tuple[int, str]] = [
    (100, "Let me"),
    (101, " think"),
    (102, " about"),
    (103, " Samsung."),
    (51, "</think>"),
    (200, "삼성"),
    (201, "전자의"),
    (202, " 주가를"),
    (203, " 분석합니다."),
]

_SHARED_UFFFD_IDS: set[int] = {200}

EXPECTED_REASONING = "Let me think about Samsung."
EXPECTED_CONTENT = "삼성전자의 주가를 분석합니다."

_MODEL_CONFIGS = [
    pytest.param(
        {
            "<think>": 50,
            "</think>": 51,
            "<tool_call>": 60,
            "</tool_call>": 61,
            "<arg_key>": 62,
            "</arg_key>": 63,
            "<arg_value>": 64,
            "</arg_value>": 65,
        },
        _Glm47Delegating,
        id="glm47",
    ),
    pytest.param(
        {
            "<think>": 50,
            "</think>": 51,
            "<tool_call>": 60,
            "</tool_call>": 61,
        },
        _Qwen3Delegating,
        id="qwen3",
    ),
]


# ── Tests ────────────────────────────────────────────────────────────


class TestUfffdReasoningTransition:
    """U+FFFD must not appear at the reasoning→content transition."""

    @pytest.mark.parametrize("vocab,delegating_cls", _MODEL_CONFIGS)
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_no_ufffd(self, chunk_size, vocab, delegating_cls):
        tokenizer = ByteFallbackMockTokenizer(vocab, _SHARED_TOKENS, _SHARED_UFFFD_IDS)
        parser = delegating_cls(tokenizer)
        deltas = replay_streaming(
            parser,
            _SHARED_TOKENS,
            chunk_size=chunk_size,
            finished_on_last=True,
        )
        output = collect_output(deltas)

        assert "�" not in output.content, (
            f"U+FFFD leaked into content: {output.content!r}"
        )
        assert output.content == EXPECTED_CONTENT
        assert output.reasoning == EXPECTED_REASONING

    def test_byte_fallback_tokenizer_produces_ufffd(self):
        """Validate the fixture: decode() returns U+FFFD for isolated
        byte-fallback token IDs, proving the old code path would leak."""
        vocab = dict(_MODEL_CONFIGS[0].values[0])
        tokenizer = ByteFallbackMockTokenizer(vocab, _SHARED_TOKENS, _SHARED_UFFFD_IDS)
        assert tokenizer.decode([200]) == "�"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_multiple_ufffd_tokens_at_boundary(self, chunk_size):
        """Multiple consecutive byte-fallback tokens at the boundary."""
        tokens: list[tuple[int, str]] = [
            (100, "Reasoning."),
            (51, "</think>"),
            (200, "삼"),
            (201, "성"),
            (202, "전자"),
        ]
        ufffd_ids: set[int] = {200, 201}
        vocab = dict(_MODEL_CONFIGS[0].values[0])

        tokenizer = ByteFallbackMockTokenizer(vocab, tokens, ufffd_ids)
        parser = _Glm47Delegating(tokenizer)
        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
        )
        output = collect_output(deltas)

        assert "�" not in output.content, (
            f"U+FFFD leaked into content: {output.content!r}"
        )
        assert output.content == "삼성전자"
        assert output.reasoning == "Reasoning."
