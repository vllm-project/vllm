# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
