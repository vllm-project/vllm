# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ScoringIOProcessor post-tokenization helpers."""

from dataclasses import dataclass

import pytest
from transformers import BertTokenizer

from vllm import TokensPrompt
from vllm.entrypoints.pooling.scoring.io_processor import (
    _apply_post_tokenization_to_token_type_ids,
)
from vllm.entrypoints.pooling.scoring.utils import compress_token_type_ids
from vllm.renderers import TokenizeParams

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class _DummyTokenizer:
    truncation_side: str = "left"
    padding_side: str = "right"
    # Outside the range of the prompt ids below, so a test can tell a pad
    # token apart from a real one.
    pad_token_id: int = 99999


def test_token_type_ids_stay_aligned_with_a_truncated_padded_prompt():
    """The cross-encoder segment boundary must survive truncate + pad.

    `token_type_ids` are parallel to `prompt_token_ids` and are reduced to a
    single boundary index by `compress_token_type_ids`. If the two arrays are
    truncated and padded in different orders they no longer describe the same
    positions, and the model is told the query segment is empty.
    """
    tokenizer = _DummyTokenizer()
    num_query, num_doc = 20, 30
    prompt = TokensPrompt(prompt_token_ids=list(range(num_query + num_doc)))
    token_type_ids = [0] * num_query + [1] * num_doc

    tok_params = TokenizeParams(
        max_total_tokens=100,
        pad_prompt_tokens=-1,
        truncate_prompt_tokens=40,
        truncation_side="left",
    )

    prompt = tok_params.apply_post_tokenization(tokenizer, prompt)
    token_type_ids = _apply_post_tokenization_to_token_type_ids(
        tokenizer, tok_params, token_type_ids
    )

    prompt_token_ids = prompt["prompt_token_ids"]
    assert len(token_type_ids) == len(prompt_token_ids)

    # Keeping the last 40 tokens drops the first 10 query tokens, so 10 query
    # tokens survive and the document starts at index 10.
    first_doc = compress_token_type_ids(token_type_ids)
    assert first_doc == 10
    assert prompt_token_ids[:first_doc] == list(range(10, num_query))
    assert prompt_token_ids[first_doc:40] == list(range(num_query, 50))


@pytest.mark.parametrize("keep", [None, 2])
def test_left_padding_token_types_match_hf(keep):
    """Padding has type 0 even when left truncation keeps only document tokens."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", "document"]
    tokenizer = BertTokenizer(
        vocab={token: i for i, token in enumerate(vocab)}, padding_side="left"
    )
    encoded = tokenizer("query", "document")
    expected_input = {
        key: list(value) if keep is None else value[-keep:]
        for key, value in encoded.items()
    }
    expected = tokenizer.pad(expected_input, padding="max_length", max_length=8)
    params = TokenizeParams(
        max_total_tokens=8,
        pad_prompt_tokens=8,
        truncate_prompt_tokens=keep,
        truncation_side="left",
    )
    actual = params.apply_post_tokenization(
        tokenizer, TokensPrompt(prompt_token_ids=encoded["input_ids"])
    )
    types = _apply_post_tokenization_to_token_type_ids(
        tokenizer, params, encoded["token_type_ids"]
    )
    assert actual["prompt_token_ids"] == expected["input_ids"]
    assert types == expected["token_type_ids"]
    boundary = compress_token_type_ids(types)
    assert [int(i >= boundary) for i in range(len(types))] == expected["token_type_ids"]
