# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import Logprob


def test_top_logprobs_alternatives_have_own_token_ids():
    """Each top_logprobs alternative must carry its own token_id placeholder."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[262],
        top_logprobs=[{262: Logprob(-0.1), 257: Logprob(-1.2), 428: Logprob(-2.3)}],
        num_output_top_logprobs=2,
    )
    tokens = {e.token for e in result.content[0].top_logprobs}
    assert tokens == {"token_id:262", "token_id:257"}, f"got {tokens}"


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


def test_tokens_level_logprobs_have_no_bytes():
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9), 8: Logprob(-1.1)}],
        num_output_top_logprobs=2,
    )
    entry = result.content[0]
    assert entry.bytes is None
    assert all(t.bytes is None for t in entry.top_logprobs)


def test_resolved_logprobs_use_engine_decoded_tokens_and_bytes():
    tokenizer = MagicMock()
    result = ServingTokens._create_tokens_logprobs(
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
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7, 9],
        top_logprobs=[{7: Logprob(-0.9)}, None],
        num_output_top_logprobs=1,
        tokenizer=tokenizer,
    )
    assert [e.token for e in result.content] == ["<7>", "<9>"]
    assert result.content[0].top_logprobs[0].token == "<7>"
    assert result.content[1].bytes == list(b"<9>")
