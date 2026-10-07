# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateLogProb
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import Logprob


def test_top_logprobs_alternatives_have_own_token_ids():
    """Each top_logprobs alternative must carry its own integer token id."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[262],
        top_logprobs=[{262: Logprob(-0.1), 257: Logprob(-1.2), 428: Logprob(-2.3)}],
        num_output_top_logprobs=2,
    )
    token_ids = {e.token_id for e in result.content[0].top_logprobs}
    assert token_ids == {262, 257}, f"got {token_ids}"


def test_sampled_entry_carries_token_id_and_rank():
    """The sampled token is identified by id, and the engine's rank survives."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[262],
        top_logprobs=[
            {262: Logprob(-0.1, rank=1), 257: Logprob(-1.2, rank=2)},
        ],
        num_output_top_logprobs=2,
    )
    entry = result.content[0]
    assert entry.token_id == 262
    assert entry.logprob == -0.1
    assert entry.rank == 1
    assert [(t.token_id, t.rank) for t in entry.top_logprobs] == [(262, 1), (257, 2)]
    # No tokenizer on the generate server: no string token, no bytes.
    assert not hasattr(entry, "token")
    assert not hasattr(entry, "bytes")


def test_sampled_token_absent_from_topk_uses_sentinel():
    """A sampled token missing from the engine's map keeps the -9999.0 sentinel."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[5],
        top_logprobs=[{7: Logprob(-0.9)}],
        num_output_top_logprobs=1,
    )
    entry = result.content[0]
    assert entry.token_id == 5
    assert entry.logprob == -9999.0
    assert entry.rank is None
    assert entry.top_logprobs == []


def test_logprobs_zero_emits_sampled_token():
    """logprobs=0 must still emit 1 entry (the sampled token)."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9), 8: Logprob(-1.1)}],
        num_output_top_logprobs=0,
    )
    assert len(result.content[0].top_logprobs) == 1


def test_logprobs_minus_one_emits_all_tokens():
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9), 8: Logprob(-1.1)}],
        num_output_top_logprobs=-1,
    )
    assert len(result.content[0].top_logprobs) == 2


def test_text_logprobs_without_tokenizer_are_placeholders_without_bytes():
    """--return-tokens-as-token-ids keeps `token_id:N` placeholders in text mode."""
    result = ServingTokens._create_text_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9), 8: Logprob(-1.1)}],
        num_output_top_logprobs=2,
    )
    entry = result.content[0]
    assert entry.token == "token_id:7"
    assert entry.bytes is None
    assert all(t.bytes is None for t in entry.top_logprobs)


def test_resolved_logprobs_use_engine_decoded_tokens_and_bytes():
    tokenizer = MagicMock()
    result = ServingTokens._create_text_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[
            {7: Logprob(-0.9, decoded_token="é"), 8: Logprob(-1.1, decoded_token="e")}
        ],
        num_output_top_logprobs=2,
        tokenizer=tokenizer,
    )
    entry = result.content[0]
    assert (entry.token, entry.bytes) == ("é", [195, 169])
    assert [(t.token, t.bytes) for t in entry.top_logprobs] == [
        ("é", [195, 169]),
        ("e", [101]),
    ]
    tokenizer.decode.assert_not_called()


def test_resolved_logprobs_decode_tokens_the_engine_did_not():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda ids: f"<{ids[0]}>"
    result = ServingTokens._create_text_logprobs(
        None,
        token_ids=[7, 9],
        top_logprobs=[{7: Logprob(-0.9)}, None],
        num_output_top_logprobs=1,
        tokenizer=tokenizer,
    )
    assert [e.token for e in result.content] == ["<7>", "<9>"]
    assert result.content[0].top_logprobs[0].token == "<7>"
    assert result.content[1].bytes == list(b"<9>")


def test_sampled_token_outside_topk_comes_first():
    """The engine puts the sampled token first, then ranks 1..k. When the
    sampled token is outside the top k it takes one of the k slots, so rank k
    is left out (same as the OpenAI endpoints)."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[50],
        top_logprobs=[
            {
                50: Logprob(-3.0, rank=5),
                10: Logprob(-0.2, rank=1),
                20: Logprob(-1.0, rank=2),
            }
        ],
        num_output_top_logprobs=2,
    )
    assert [t.rank for t in result.content[0].top_logprobs] == [5, 1]


def test_logprob_is_required_on_the_wire():
    """A payload missing logprob is rejected rather than read as -9999."""
    with pytest.raises(ValidationError):
        GenerateLogProb.model_validate({"token_id": 1})


def test_rank_zero_from_a_nan_logprob_is_none():
    """The engine reports rank 0 for a sampled token whose logprob is NaN."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(float("nan"), rank=0), 8: Logprob(-0.2, rank=1)}],
        num_output_top_logprobs=1,
    )
    entry = result.content[0]
    assert entry.rank is None
    assert entry.top_logprobs[0].rank is None
    # Clamped like the Rust frontend (max(nan, x) is nan in Python).
    assert entry.logprob == -9999.0
    assert entry.top_logprobs[0].logprob == -9999.0
    json.dumps(result.model_dump(), allow_nan=False)


def test_text_logprobs_clamp_a_nan_logprob():
    result = ServingTokens._create_text_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(float("nan"), rank=0, decoded_token="a")}],
        num_output_top_logprobs=1,
        tokenizer=MagicMock(),
    )
    entry = result.content[0]
    assert entry.logprob == -9999.0
    assert entry.top_logprobs[0].logprob == -9999.0
    json.dumps(result.model_dump(), allow_nan=False)
