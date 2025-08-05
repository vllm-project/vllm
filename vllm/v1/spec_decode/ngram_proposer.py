# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np
from numba import jit

from vllm.config import VllmConfig


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
        self.propose(np.zeros(1024, dtype=np.int32))

    def propose(
        self,
        context_token_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on n-gram pattern 
        matching in the context. The function finds matches of the last n 
        tokens in the previous context, and returns k tokens that followed 
        that match.
        
        Args:
            context_token_ids: Numpy array of token IDs representing the 
                               context sequence.

        Returns:
            np.ndarray: The sequence of tokens that followed 
                        the matched n-gram in the context.
            None: If no matching n-gram pattern is found.

        Example:
            If context_token_ids = [1,2,3,4,2,3], min_n = 2, max_n = 3, and
            k = 4:
            - The last 3 (= max_n) tokens [4,2,3] cannot find a match.
            - The last 2 tokens [2,3] will be matched against the previous 
              4 tokens [1,2,3,4].
            - Finding a match of [2,3] would return the tokens that 
              followed that pattern. Here we will return [4,2,3] because 
              we only have three tokens after the match.
        """
        # Do not generate draft tokens is context is shorter than minimum n-gram
        if context_token_ids.shape[0] < self.min_n:
            return None

        # Do not generate draft tokens beyond the max model length.
        k = min(self.k, self.max_model_len - context_token_ids.shape[0])
        if k <= 0:
            return None

        return _find_longest_leftmost_ngram(origin_tokens=context_token_ids,
                                            min_n=self.min_n,
                                            max_n=self.max_n,
                                            k=k)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@jit(nopython=True)
def _find_longest_leftmost_ngram(origin_tokens: np.ndarray, min_n: int,
                                 max_n: int, k: int) -> Optional[np.ndarray]:
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
    total_token = origin_tokens.shape[0]
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
