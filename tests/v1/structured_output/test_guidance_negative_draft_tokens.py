# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guidance backend must not choke on speculative -1 padding.

Speculative drafters (e.g. ngram_gpu) pad unfilled draft slots with a -1
sentinel. llguidance raises ``OverflowError`` on a negative id, so
``GuidanceGrammar.validate_tokens`` must stop at the first sentinel. The mock
matcher below emulates that raise, so each test fails on the unpatched backend
and passes with the guard.
"""

import pytest

from vllm.v1.structured_output.backend_guidance import GuidanceGrammar

pytestmark = pytest.mark.skip_global_cleanup


class _Matcher:
    """Stand-in for ``llguidance.LLMatcher`` that rejects negative ids."""

    def __init__(self) -> None:
        self.tokens: list[int] | None = None

    def is_stopped(self) -> bool:
        return False

    def validate_tokens(self, tokens: list[int]) -> int:
        if any(t < 0 for t in tokens):
            raise OverflowError("llguidance rejects negative token ids")
        self.tokens = tokens
        return len(tokens)

    def get_error(self):
        return None


class _Tokenizer:
    eos_token = 0


def _grammar(matcher: _Matcher) -> GuidanceGrammar:
    return GuidanceGrammar(ll_matcher=matcher, ll_tokenizer=_Tokenizer(), vocab_size=16)


def test_validate_tokens_stops_at_trailing_padding() -> None:
    matcher = _Matcher()
    assert _grammar(matcher).validate_tokens([3, 4, -1, -1]) == [3, 4]
    assert matcher.tokens == [3, 4]


def test_validate_tokens_single_negative_does_not_raise() -> None:
    matcher = _Matcher()
    assert _grammar(matcher).validate_tokens([-1]) == []
    assert matcher.tokens is None


def test_validate_tokens_all_padding_returns_empty() -> None:
    matcher = _Matcher()
    assert _grammar(matcher).validate_tokens([-1, -1]) == []
    assert matcher.tokens is None
