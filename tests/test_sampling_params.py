# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


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
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


@pytest.mark.parametrize("token_id", [-1, 1024])
def test_stop_token_ids_reject_model_out_of_vocab_ids(token_id: int):
    params = SamplingParams(stop_token_ids=[token_id])

    with pytest.raises(VLLMValidationError, match="stop_token_ids"):
        params.verify(MockModelConfig(), None, None, None)


def test_eos_expansion_rejects_model_out_of_vocab_ids():
    params = SamplingParams()
    params.update_from_generation_config(
        {"eos_token_id": [1023, 1024]}, eos_token_id=1023
    )

    with pytest.raises(VLLMValidationError, match="stop_token_ids"):
        params.validate_model_token_ids(MockModelConfig())


def test_allowed_token_ids_use_model_vocab_without_tokenizer():
    params = SamplingParams(allowed_token_ids=[1024])

    with pytest.raises(VLLMValidationError, match="allowed_token_ids"):
        params.verify(MockModelConfig(), None, None, None)


def test_bad_word_token_ids_use_model_vocab():
    class Tokenizer:
        max_token_id = 2047

        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            return [1024]

    params = SamplingParams(bad_words=["bad"])
    params.update_from_tokenizer(Tokenizer())

    with pytest.raises(VLLMValidationError, match="bad_words"):
        params.validate_model_token_ids(MockModelConfig())


def test_bad_word_total_token_limit_uses_effective_environment(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_BAD_WORDS_TOTAL_TOKENS", "1")

    class Tokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2]

    params = SamplingParams(bad_words=["bad"])

    with pytest.raises(VLLMValidationError, match="total bad word tokens"):
        params.update_from_tokenizer(Tokenizer())


def test_bad_word_tokenization_accepts_only_prefixed_nonempty_encoding():
    class Tokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            return [1] if text.startswith(" ") else []

    params = SamplingParams(bad_words=["bad"])
    params.update_from_tokenizer(Tokenizer())

    assert params.bad_words_token_ids == [[1]]
