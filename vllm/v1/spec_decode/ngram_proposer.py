# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np
from numba import jit

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class NgramProposer:

    def __init__(self, vllm_config: VllmConfig):
        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        fake_req_cnt = 32
        fake_token_cnt = 1024
        self.bulk_propose(
            np.zeros((fake_req_cnt, fake_token_cnt), dtype=np.int32),
            np.zeros(fake_token_cnt, dtype=np.int32),
            [[] for _ in range(fake_req_cnt)])

    def bulk_propose(self, tokens_per_request: np.ndarray,
                     num_tokens_per_request: np.ndarray,
                     result_draft_tokens: list[list[int]]) -> None:
        draft_tokens_per_request = _bulk_find_longest_leftmost_ngram(
            tokens_per_request=tokens_per_request,
            num_tokens_per_request=num_tokens_per_request,
            total_request=len(result_draft_tokens),
            min_n=self.min_n,
            max_n=self.max_n,
            max_model_len=self.max_model_len,
            k=self.k)
        for i, draft_tokens in enumerate(draft_tokens_per_request):
            if draft_tokens is not None:
                result_draft_tokens[i].extend(draft_tokens.tolist())

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@jit(nopython=True)
def _bulk_find_longest_leftmost_ngram(tokens_per_request: np.ndarray,
                                      num_tokens_per_request: np.ndarray,
                                      total_request: int, min_n: int,
                                      max_n: int, max_model_len: int,
                                      k: int) -> list[Optional[np.ndarray]]:
    return [
        _find_longest_leftmost_ngram(
            tokens_per_request[i, :num_tokens_per_request[i]], min_n, max_n,
            max_model_len, k) if num_tokens_per_request[i] > 0 else None
        for i in range(total_request)
    ]


@jit(nopython=True)
def _find_longest_leftmost_ngram(origin_tokens: np.ndarray, min_n: int,
                                 max_n: int, max_model_len: int,
                                 k: int) -> Optional[np.ndarray]:
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_n:
        return None

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return None

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    lps = np.zeros(max_n, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # Fuse the pattern LPS computation and full substring match
    # in a single for loop.
    # 1) When i < max_n, it tries to compute LPS
    # 2) When i >= max_n, it tries to find longest matched ngram
    #    of the prefix with length [min_n, max_n]

    # lps[0] always equal to 0, we starts with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_n:
                # Update LPS
                lps[i] = prev_lps
            elif prev_lps == max_n:
                # i >= max_n,
                prev_lps = lps[prev_lps - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second longest prefix
            # among all suffix of tokens[:i],
            # which is actually the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_n:
        # No valid ngram is found
        return None

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position:start_position + k]
