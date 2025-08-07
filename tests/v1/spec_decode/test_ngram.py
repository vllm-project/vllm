# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.ngram_proposer import (NgramProposer,
                                                _find_subarray_kmp,
                                                _kmp_lps_array)


def test_kmp_lps_array():
    np.testing.assert_array_equal(_kmp_lps_array(np.array([])), np.array([]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1])), np.array([0]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1, 1, 1])),
                                  np.array([0, 1, 2]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1, 2, 3, 4])),
                                  np.array([0, 0, 0, 0]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1, 2, 1, 2, 3])),
                                  np.array([0, 0, 1, 2, 0]))


def test_find_subarray_kmp():
    X = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    assert _find_subarray_kmp(X, 2, 2) is None
    X = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(_find_subarray_kmp(X, 2, 3),
                                  np.array([4, 1, 2]))
    np.testing.assert_array_equal(_find_subarray_kmp(X, 2, 2), np.array([4,
                                                                         1]))
    np.testing.assert_array_equal(_find_subarray_kmp(X, 1, 3),
                                  np.array([4, 1, 2]))
    np.testing.assert_array_equal(_find_subarray_kmp(X, 1, 2), np.array([4,
                                                                         1]))
    X = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(_find_subarray_kmp(X, 2, 3),
                                  np.array([4, 1, 2]))
    # Return on the first match
    np.testing.assert_array_equal(_find_subarray_kmp(X, 1, 3),
                                  np.array([6, 2, 3]))


def test_ngram_proposer():

    def ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(model_config=model_config,
                                   speculative_config=SpeculativeConfig(
                                       prompt_lookup_min=min_n,
                                       prompt_lookup_max=max_n,
                                       num_speculative_tokens=k,
                                       method="ngram",
                                   )))

    # No match.
    result = ngram_proposer(
        2, 2, 2).propose(context_token_ids=np.array([1, 2, 3, 4, 5]))
    assert result is None

    # No match for 4-gram.
    result = ngram_proposer(
        4, 4, 2).propose(context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]))
    assert result is None

    # No match for 4-gram but match for 3-gram.
    result = ngram_proposer(
        3, 4, 2).propose(context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]))
    assert np.array_equal(result, np.array([4, 1]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    result = ngram_proposer(3, 4, 2).propose(
        context_token_ids=np.array([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]))
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 1]

    # Match for 2-gram and 3-gram, but not 4-gram.
    result = ngram_proposer(
        2, 4,
        2).propose(context_token_ids=np.array([3, 4, 5, 2, 3, 4, 1, 2, 3, 4]))
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 2]
