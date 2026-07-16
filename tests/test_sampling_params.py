# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None
    _vocab_size: int = 1024

    def get_vocab_size(self) -> int:
        return self._vocab_size


@dataclass
class MockTokenizer:
    """Minimal tokenizer mock for update_from_tokenizer tests."""

    _max_token_id: int = 1023
    _vocab: dict[str, list[int]] = field(default_factory=dict)

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._vocab.get(text, [0])


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


class TestBadWordsVocabValidation:
    """Regression tests for GHSA-4hhp-h66f-j5j7.

    Multimodal tokenizers can have input-only tokens (e.g. <|audio|>) whose
    IDs exceed the model's output vocabulary (logits width). Without
    model-vocab-aware validation, such tokens cause out-of-bounds writes
    in the BadWords Triton kernel.
    """

    def test_rejects_token_exceeding_model_vocab(self):
        tokenizer = MockTokenizer(
            _max_token_id=128256,
            _vocab={"<|audio|>": [128256]},
        )
        model_config = MockModelConfig(_vocab_size=128256)

        params = SamplingParams(bad_words=["<|audio|>"])
        with pytest.raises(VLLMValidationError, match="specified as bad"):
            params.update_from_tokenizer(tokenizer, model_config)

    def test_accepts_token_at_model_vocab_boundary(self):
        tokenizer = MockTokenizer(
            _max_token_id=128256,
            _vocab={"<|reserved|>": [128255]},
        )
        model_config = MockModelConfig(_vocab_size=128256)

        params = SamplingParams(bad_words=["<|reserved|>"])
        params.update_from_tokenizer(tokenizer, model_config)
        assert params.bad_words_token_ids is not None
        assert [128255] in params.bad_words_token_ids

    def test_fallback_to_tokenizer_vocab_without_model_config(self):
        tokenizer = MockTokenizer(
            _max_token_id=128256,
            _vocab={"<|audio|>": [128256]},
        )

        params = SamplingParams(bad_words=["<|audio|>"])
        params.update_from_tokenizer(tokenizer)
        assert params.bad_words_token_ids is not None

    def test_rejects_negative_token_ids(self):
        tokenizer = MockTokenizer(
            _max_token_id=1023,
            _vocab={"bad": [-1]},
        )
        model_config = MockModelConfig(_vocab_size=1024)

        params = SamplingParams(bad_words=["bad"])
        with pytest.raises(VLLMValidationError, match="specified as bad"):
            params.update_from_tokenizer(tokenizer, model_config)
