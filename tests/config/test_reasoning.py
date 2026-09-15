# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest

from vllm.config.model import ModelConfig
from vllm.config.reasoning import ReasoningConfig


class _Tokenizer:
    _vocab = {"<think>": 18, "</think>": 19, "<assistant>": 23}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [self._vocab[text]]


def test_poolside_parser_defaults_to_thinking_for_token_id_initialization(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        "vllm.config.reasoning.cached_tokenizer_from_config",
        lambda model_config: tokenizer,
    )
    config = ReasoningConfig(reasoning_parser="poolside_v1")

    config.initialize_token_ids(cast(ModelConfig, object()))

    assert config.enabled
    assert config.reasoning_start_token_ids == [18]
    assert config.reasoning_end_token_ids == [19]
