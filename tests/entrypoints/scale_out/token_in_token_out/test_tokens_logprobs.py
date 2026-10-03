# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateLogProb
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import Logprob


def test_top_logprobs_alternatives_have_own_token_ids():
    """Each top_logprobs alternative must carry its own integer token id."""
    # Engine dict at logprobs=2 with the sampled token 262 also top-1.
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[262],
        top_logprobs=[{262: Logprob(-0.1), 257: Logprob(-1.2)}],
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
    """logprobs=0: the engine returns only the sampled token, and it is kept."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9)}],
        num_output_top_logprobs=0,
    )
    assert [t.token_id for t in result.content[0].top_logprobs] == [7]


def test_logprobs_minus_one_emits_all_tokens():
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9), 8: Logprob(-1.1)}],
        num_output_top_logprobs=-1,
    )
    assert len(result.content[0].top_logprobs) == 2


def test_sampled_token_outside_topk_comes_first():
    """The engine puts the sampled token first, then ranks 1..k. Generate keeps
    all k + 1 candidates when the sampled token is outside the top k; derender
    applies each endpoint's cut (#59513)."""
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
    assert [t.rank for t in result.content[0].top_logprobs] == [5, 1, 2]


def test_logprob_is_required_on_the_wire():
    """A payload missing logprob is rejected rather than read as -9999."""
    with pytest.raises(ValidationError):
        GenerateLogProb.model_validate({"token_id": 1})
