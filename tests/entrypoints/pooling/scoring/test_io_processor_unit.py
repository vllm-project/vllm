# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ScoringIOProcessor post-tokenization helpers."""

from dataclasses import dataclass

import pytest

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
    # Outside the range of the prompt ids below, so a test can tell a pad
    # token apart from a real one.
    pad_token_id: int = 99999
    pad_token_type_id: int = 0


def test_token_type_ids_stay_aligned_with_a_truncated_padded_prompt():
    """The cross-encoder segment boundary must survive truncate + pad.

    `token_type_ids` are parallel to `prompt_token_ids` and are reduced to the
    document segment's start and end by `compress_token_type_ids`. If the two
    arrays are truncated and padded in different orders they no longer
    describe the same positions, and the model is told the query segment is
    empty.
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
    doc_start, doc_end = compress_token_type_ids(token_type_ids)
    assert (doc_start, doc_end) == (10, 40)
    assert prompt_token_ids[:doc_start] == list(range(10, num_query))
    assert prompt_token_ids[doc_start:doc_end] == list(range(num_query, 50))
    assert token_type_ids[doc_end:] == [tokenizer.pad_token_type_id] * 60


def test_token_type_padding_uses_the_tokenizer_pad_type():
    tokenizer = _DummyTokenizer()
    tok_params = TokenizeParams(
        max_total_tokens=8,
        pad_prompt_tokens=8,
        truncate_prompt_tokens=None,
        truncation_side="right",
    )

    token_type_ids = _apply_post_tokenization_to_token_type_ids(
        tokenizer, tok_params, [0, 0, 0, 1, 1]
    )

    assert token_type_ids == [0, 0, 0, 1, 1, 0, 0, 0]
    assert compress_token_type_ids(token_type_ids) == (3, 5)


@pytest.mark.parametrize("token_type_ids", [[0, 1, 0, 1], [0, 2]])
def test_compress_token_type_ids_rejects_invalid_sequences(token_type_ids):
    with pytest.raises(ValueError, match="optional trailing zeros"):
        compress_token_type_ids(token_type_ids)
