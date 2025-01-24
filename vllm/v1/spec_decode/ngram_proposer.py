from typing import List, Optional
from vllm.v1.utils import ConstantList

class NgramProposer():
    def __init__(self):
        pass
    
    def _kmp_lps_array(self, pattern: List[int]) -> List[int]:
        """
        Build the lps (longest proper prefix which is also suffix) array for the pattern.
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

    def _find_subarray_kmp(self, 
                           X: List[int], 
                           Y: List[int], 
                           K: int) -> Optional[List[int]]:
        """
        Returns the subarray starting at the first occurrence of Y in X,
        plus K subsequent elements (if available). If not found, returns None.
        """
        N = len(X)
        M = len(Y)

        if M == 0:
            # If Y is empty, 
            # let's define that it matches at index 0
            return X[:K]

        # Precompute lps array for Y
        lps = self._kmp_lps_array(Y)

        i = 0  # index for X
        j = 0  # index for Y

        while i < N:
            if X[i] == Y[j]:
                i += 1
                j += 1

                # If we have matched the entire Y
                if j == M:
                    # Found Y in X, gather the next K elements
                    start_index = i - M # Where the match started
                    return X[start_index : start_index + M + K]
            else:
                # Mismatch
                if j != 0:
                    # Use the lps array to avoid re-checking elements
                    j = lps[j - 1]
                else:
                    i += 1
        
        # Y not found
        return None
        
    def propose(self, 
                context_token_ids: ConstantList[int],
                n: int, k: int) -> Optional[List[int]]:
        ngrams = context_token_ids[-n:]
        lookup_tokens = context_token_ids[:-n]
        match_tokens = self._find_subarray_kmp(lookup_tokens,
                                ngrams, k)
        if match_tokens is None:
            return None
        return match_tokens[n:]