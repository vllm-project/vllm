# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.structured_output.backend_guidance import GuidanceGrammar

pytestmark = pytest.mark.skip_global_cleanup


class _Matcher:
    def __init__(self) -> None:
        self.tokens: list[int] | None = None

    def is_stopped(self) -> bool:
        return False

    def validate_tokens(self, tokens: list[int]) -> int:
        self.tokens = tokens
        return len(tokens)

    def get_error(self):
        return None


class _Tokenizer:
    eos_token = 0


def test_guidance_ignores_negative_draft_padding_before_matcher() -> None:
    matcher = _Matcher()
    grammar = GuidanceGrammar(
        ll_matcher=matcher,
        ll_tokenizer=_Tokenizer(),
        vocab_size=16,
    )

    assert grammar.validate_tokens([3, 4, -1, -1]) == [3, 4]
    assert matcher.tokens == [3, 4]


def test_guidance_skips_matcher_when_draft_starts_with_padding() -> None:
    matcher = _Matcher()
    grammar = GuidanceGrammar(
        ll_matcher=matcher,
        ll_tokenizer=_Tokenizer(),
        vocab_size=16,
    )

    assert grammar.validate_tokens([-1, -1]) == []
    assert matcher.tokens is None
