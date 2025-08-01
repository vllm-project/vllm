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
        # Do not generate draft tokens beyond the max model length.
        k = min(self.k, self.max_model_len - context_token_ids.shape[0])
        if k <= 0:
            return None

        # TODO(woosuk): Optimize this.
        for n in range(self.max_n, self.min_n - 1, -1):
            result = _find_subarray_kmp(context_token_ids, n, k)
            if result is not None:
                return result
        return None

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@jit(nopython=True)
def _kmp_lps_array(pattern: np.ndarray) -> np.ndarray:
    """
    Build the lps (longest proper prefix which is also suffix) 
    array for the pattern.
    """
    lps = np.zeros(len(pattern), dtype=np.int32)
    prev_lps = 0  # length of the previous longest prefix suffix
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[prev_lps]:
            prev_lps += 1
            lps[i] = prev_lps
            i += 1
        else:
            if prev_lps != 0:
                prev_lps = lps[prev_lps - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


@jit(nopython=True)
def _find_subarray_kmp(
    context_token_ids: np.ndarray,
    n: int,
    k: int,
) -> Optional[np.ndarray]:
    context_len = context_token_ids.shape[0]
    assert n > 0

    pattern = context_token_ids[-n:]
    # Precompute lps array for Y
    lps = _kmp_lps_array(pattern)

    i = 0
    j = 0
    # -n because the last n tokens are used as pattern
    while i < context_len - n:
        if context_token_ids[i] == pattern[j]:
            i += 1
            j += 1

            # If we have matched the entire Y
            if j == n:
                # Found pattern in context, gather the next K elements
                return context_token_ids[i:i + k]
        else:
            # Mismatch
            if j != 0:
                # Use the lps array to avoid re-checking elements
                j = lps[j - 1]
            else:
                i += 1

    # Y not found
    return None
