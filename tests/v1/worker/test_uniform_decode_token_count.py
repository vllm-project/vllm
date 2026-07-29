# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`get_uniform_decode_token_count` gates FULL-graph replay: rejecting a
uniform-shaped batch that is actually still prefilling is a correctness
requirement, but rejecting a genuinely uniform decode batch is a silent
performance regression (falls back to PIECEWISE) rather than a crash. Both
directions are pinned here as a pure predicate test, with no model runner.
"""

import numpy as np

from vllm.v1.worker.utils import get_uniform_decode_token_count


def test_uniform_decode_batch_is_classified_as_uniform():
    # Every request has finished its prefill, so the shared query length is
    # a real decode step and must not fall back to PIECEWISE.
    num_reqs = 4
    query_len = 8
    num_computed_tokens = np.array([16, 32, 64, 128], dtype=np.int32)
    prefill_lens = np.array([16, 32, 64, 128], dtype=np.int32)

    assert (
        get_uniform_decode_token_count(
            num_reqs,
            num_reqs * query_len,
            query_len,
            num_computed_tokens,
            prefill_lens,
        )
        == query_len
    )


def test_prompt_chunk_shaped_like_decode_is_not_uniform():
    # A K+1-token prompt chunk has the exact shape of a spec-decode step
    # (uniform query length, K+1 tokens per request) but is not a decode.
    # See https://github.com/vllm-project/vllm/issues/49918.
    num_reqs = 2
    query_len = 8  # K + 1 with K == 7 speculative tokens.
    num_computed_tokens = np.array([0, 0], dtype=np.int32)
    prefill_lens = np.array([40, 40], dtype=np.int32)

    assert (
        get_uniform_decode_token_count(
            num_reqs,
            num_reqs * query_len,
            query_len,
            num_computed_tokens,
            prefill_lens,
        )
        is None
    )


def test_one_still_prefilling_request_rejects_the_whole_batch():
    # Uniform query length, but one request out of several has not reached
    # its prefill length yet -- the whole batch must fall back, not just the
    # prefilling request.
    num_reqs = 3
    query_len = 4
    num_computed_tokens = np.array([16, 16, 12], dtype=np.int32)
    prefill_lens = np.array([16, 16, 20], dtype=np.int32)

    assert (
        get_uniform_decode_token_count(
            num_reqs,
            num_reqs * query_len,
            query_len,
            num_computed_tokens,
            prefill_lens,
        )
        is None
    )


def test_non_uniform_query_length_is_not_uniform_regardless_of_prefill_state():
    # Shape check runs first: a non-uniform batch is rejected even though
    # every request has finished prefilling.
    num_reqs = 2
    num_computed_tokens = np.array([16, 16], dtype=np.int32)
    prefill_lens = np.array([16, 16], dtype=np.int32)

    assert (
        get_uniform_decode_token_count(
            num_reqs,
            12,  # Not max_query_len * num_reqs for any shared query length.
            8,
            num_computed_tokens,
            prefill_lens,
        )
        is None
    )
