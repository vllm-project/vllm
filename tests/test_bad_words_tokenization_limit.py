# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError

pytestmark = pytest.mark.skip_global_cleanup


class MockTokenizer:
    max_token_id = 1024

    def __init__(self):
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        return [2] if text.startswith(" ") else [1]


def test_duplicate_bad_words_are_deduplicated_in_order():
    params = SamplingParams(bad_words=["bad", "worse", "bad", "worst"])

    assert params.bad_words == ["bad", "worse", "worst"]


def test_bad_word_tokenization_stops_at_worker_limit():
    params = SamplingParams(bad_words=[f"word-{i}" for i in range(65)])
    tokenizer = MockTokenizer()

    with pytest.raises(VLLMValidationError, match="Too many bad words"):
        params.update_from_tokenizer(tokenizer)

    assert tokenizer.calls == 129


def test_bad_word_tokenization_limit_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_BAD_WORDS", "2")
    params = SamplingParams(bad_words=["bad", "worse"])
    tokenizer = MockTokenizer()

    with pytest.raises(VLLMValidationError, match="The max number is 2"):
        params.update_from_tokenizer(tokenizer)

    assert tokenizer.calls == 3
