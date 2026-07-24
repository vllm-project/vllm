# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(ValueError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


class TestAllowedTokenIdsVocabValidation:
    """Regression tests for GHSA-wr27-mx79-rg6c: allowed_token_ids must be
    validated against model output vocab size, not tokenizer length."""

    def test_accept_last_valid_id(self):
        config = MockModelConfig()
        vocab_size = config.get_vocab_size()
        params = SamplingParams(allowed_token_ids=[vocab_size - 1])
        params.verify(config, None, None, None)

    def test_reject_id_equal_to_vocab_size(self):
        config = MockModelConfig()
        vocab_size = config.get_vocab_size()
        params = SamplingParams(allowed_token_ids=[vocab_size])
        with pytest.raises(ValueError, match="out-of-vocab"):
            params.verify(config, None, None, None)

    def test_reject_id_exceeding_vocab_size(self):
        config = MockModelConfig()
        vocab_size = config.get_vocab_size()
        params = SamplingParams(allowed_token_ids=[vocab_size + 100])
        with pytest.raises(ValueError, match="out-of-vocab"):
            params.verify(config, None, None, None)

    def test_reject_negative_id(self):
        config = MockModelConfig()
        params = SamplingParams(allowed_token_ids=[-1])
        with pytest.raises(ValueError, match="out-of-vocab"):
            params.verify(config, None, None, None)

    def test_reject_mixed_valid_and_invalid(self):
        config = MockModelConfig()
        vocab_size = config.get_vocab_size()
        params = SamplingParams(allowed_token_ids=[0, vocab_size])
        with pytest.raises(ValueError, match="out-of-vocab"):
            params.verify(config, None, None, None)

    def test_reject_empty_list(self):
        config = MockModelConfig()
        params = SamplingParams(allowed_token_ids=[])
        with pytest.raises(ValueError, match="empty"):
            params.verify(config, None, None, None)

    def test_accept_valid_ids(self):
        config = MockModelConfig()
        params = SamplingParams(allowed_token_ids=[0, 1, 512, 1023])
        params.verify(config, None, None, None)
