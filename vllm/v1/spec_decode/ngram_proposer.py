# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

from vllm.v1.utils import ConstantList


class NgramProposer:

    def __init__(self):
        pass

    def propose(self, context_token_ids: ConstantList[int], n: int,
                k: int) -> Optional[List[int]]:
        """Proposes the next sequence of tokens based on n-gram pattern 
        matching in the context. The function finds matches of the last n 
        tokens in the previous context, and returns k tokens that followed 
        that match.
        
        Args:
            context_token_ids: List of token IDs representing the 
                               context sequence.
            n: Length of the n-gram to match.
            k: Number of tokens follow the match. If there are less 
               than k tokens follow the match, we will return 
               the maximum amount of tokens until the end.
        
        Returns:
            List[int]: The sequence of tokens that followed 
                       the matched n-gram in the context.
            None: If no matching n-gram pattern is found.
        
        Example:
            If context_token_ids = [1,2,3,4,2,3], n = 2, and k = 4:
            - The last 2 tokens [2,3] will be matched against the previous 
              4 tokens [1,2,3,4].
            - Finding a match of [2,3] would return the tokens that 
              followed that pattern. Here we will return [4,2,3] because 
              we only have three tokens after the match.
        """
        # TODO: Use c++ to implement the _find_subarray_kmp to
        # improve the efficiency
        return self._find_subarray_kmp(context_token_ids, n, k)

    @staticmethod
    def _kmp_lps_array(pattern: List[int]) -> List[int]:
        """
        Build the lps (longest proper prefix which is also suffix) 
        array for the pattern.
        """
        lps = [0] * len(pattern)
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

    @staticmethod
    def _find_subarray_kmp(context_token_ids: ConstantList[int], n: int,
                           k: int) -> Optional[List[int]]:
        context_len = len(context_token_ids)
        assert n > 0

        pattern = context_token_ids[-n:]
        # Precompute lps array for Y
        lps = NgramProposer._kmp_lps_array(pattern)

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
