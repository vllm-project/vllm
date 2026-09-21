# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
