# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A half-written reasoning delimiter pair should say so at startup."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.config.reasoning import ReasoningConfig


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2] if text else []


@pytest.fixture
def tokenizer():
    with patch(
        "vllm.config.reasoning.cached_tokenizer_from_config", return_value=_Tokenizer()
    ):
        yield


@pytest.mark.parametrize(
    "kwargs,named",
    [
        ({"reasoning_start_str": "<think>"}, "reasoning_start_str"),
        ({"reasoning_end_str": "</think>"}, "reasoning_end_str"),
    ],
)
def test_a_half_written_pair_is_reported(tokenizer, caplog_vllm, kwargs, named):
    """The request-time error tells the user to set the flag they already set.

    `thinking_token_budget` rejects requests once reasoning is not enabled, so
    leaving startup silent points them at the wrong thing.
    """
    config = ReasoningConfig(**kwargs)

    with caplog_vllm.at_level("WARNING"):
        config.initialize_token_ids(MagicMock())

    assert not config.enabled
    assert named in caplog_vllm.text, caplog_vllm.text


def test_a_complete_pair_initialises_quietly(tokenizer, caplog_vllm):
    config = ReasoningConfig(
        reasoning_start_str="<think>", reasoning_end_str="</think>"
    )

    with caplog_vllm.at_level("WARNING"):
        config.initialize_token_ids(MagicMock())

    assert config.enabled
    assert "reasoning_start_str" not in caplog_vllm.text


def test_a_parser_without_delimiters_stays_quiet(tokenizer, caplog_vllm):
    """Most parsers define none and do not want token IDs.

    gpt-oss extracts reasoning through Harmony, so warning here would report a
    working deployment as broken.
    """
    config = ReasoningConfig(reasoning_parser="openai_gptoss")

    with caplog_vllm.at_level("WARNING"):
        config.initialize_token_ids(MagicMock())

    assert not config.enabled
    assert "reasoning_start_str" not in caplog_vllm.text
