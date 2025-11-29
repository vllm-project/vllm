# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import _get_protocol_attrs  # type: ignore

import pytest
from transformers import PreTrainedTokenizerBase

from vllm.tokenizers import TokenizerLike
from vllm.transformers_utils.tokenizer import get_tokenizer


def _get_missing_attrs(obj: object, target: type):
    return [k for k in _get_protocol_attrs(target) if not hasattr(obj, k)]


def test_tokenizer_like_protocol():
    assert not (
        missing_attrs := _get_missing_attrs(
            get_tokenizer("gpt2", use_fast=False),
            TokenizerLike,
        )
    ), f"Missing attrs: {missing_attrs}"

    assert not (
        missing_attrs := _get_missing_attrs(
            get_tokenizer("gpt2", use_fast=True),
            TokenizerLike,
        )
    ), f"Missing attrs: {missing_attrs}"

    assert not (
        missing_attrs := _get_missing_attrs(
            get_tokenizer(
                "mistralai/Mistral-7B-Instruct-v0.3", tokenizer_mode="mistral"
            ),
            TokenizerLike,
        )
    ), f"Missing attrs: {missing_attrs}"


@pytest.mark.parametrize("tokenizer_name", ["facebook/opt-125m", "gpt2"])
def test_tokenizer_revision(tokenizer_name: str):
    # Assume that "main" branch always exists
    tokenizer = get_tokenizer(tokenizer_name, revision="main")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Assume that "never" branch always does not exist
    with pytest.raises(OSError, match="not a valid git identifier"):
        get_tokenizer(tokenizer_name, revision="never")


@pytest.mark.parametrize("tokenizer_name", ["BAAI/bge-base-en"])
@pytest.mark.parametrize("n_tokens", [510])
def test_special_tokens(tokenizer_name: str, n_tokens: int):
    tokenizer = get_tokenizer(tokenizer_name, revision="main")

    prompts = "[UNK]" * n_tokens
    prompt_token_ids = tokenizer.encode(prompts)
    assert len(prompt_token_ids) == n_tokens + 2
