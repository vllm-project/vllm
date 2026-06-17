# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Numba-accelerated N-gram matching kernels.

Provides CPU-based, KMP-style string matching to find the longest n-gram suffix
match in each request's token history and propose the following k tokens as
draft tokens.
"""

import numpy as np
from numba import jit, njit, prange


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    """Batch ngram proposal using numba parallelization.

    Args:
        valid_ngram_requests: Indices of requests needing ngram proposals.
        num_tokens_no_spec: Number of non-speculative tokens per request.
        token_ids_cpu: Token IDs for all requests [batch, max_model_len].
        min_n: Minimum n-gram length.
        max_n: Maximum n-gram length.
        max_model_len: Maximum model sequence length.
        k: Number of draft tokens to propose.
        valid_ngram_draft: Output buffer for draft tokens [batch, k].
        valid_ngram_num_drafts: Output buffer for draft counts [batch].
    """
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, extract k tokens following the matched n-gram.

    Uses a KMP (Knuth-Morris-Pratt) variant: flip the token sequence and
    compute the LPS (longest prefix suffix) table for the first max_ngram
    prefix to find all suffix-prefix matches efficiently in O(n) time.
    """
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens — now we search for the longest prefix match of the
    # reversed suffix, which corresponds to the longest suffix match in
    # the original sequence.
    tokens = origin_tokens[::-1]

    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
